import pandas as pd
from typing import List, Union
import numpy as np
from tqdm import tqdm
import scipy
from torch_geometric_temporal.signal import StaticGraphTemporalSignal  # , temporal_signal_split
from src.models.models import CustomTemporalSignal
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data.encode_externals import Weather_container
from datetime import timedelta


def load_csv_dataset(
    path: str,
    time_column: str,
    location_columns: List[str] = None,
    station_column: str = None,
    time_intervals: str = "4h",
) -> pd.DataFrame:
    """
    Load a dataset with rides.

    Args:
        path (str): path to csv file
        time_columns (List[str]): list of time columns
        location_columns (List[str]): list of longitide/lattitude columns

    Returns:
        pd.DataFrame: Parsed DataFrame
    """

    df = pd.read_csv(
        path,
        parse_dates=[time_column],
        infer_datetime_format=True,
    )

    df[time_column] = df[time_column].dt.floor(time_intervals)

    cols_to_select = [time_column]

    if location_columns is not None:
        cols_to_select += location_columns

    if station_column is not None:
        cols_to_select.append(station_column)

    return df[cols_to_select].dropna().sort_values(by=time_column)


def create_grid(df: pd.DataFrame, lng_col: str, lat_col: str, splits: int = 10) -> pd.DataFrame:
    """
    Splits a pd.DataFrame that defines different rides into a grid
    Each area in the grid defines a node in the graph

    Args:
        splits (int, optional): Number of regions created. Defaults to 10.

    Returns:
        pd.DataFrame: Contains "grid_start" and "grid_end"
    """
    min_lng = df[lng_col].min()
    max_lng = df[lng_col].max()
    lng_intervals = np.linspace(min_lng, max_lng, splits + 1)

    bins_lng_col = pd.cut(df[lng_col], lng_intervals, labels=list(range(splits)), include_lowest=True)
    df[lng_col + "_binned"] = bins_lng_col

    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    lat_intervals = np.linspace(min_lat, max_lat, splits + 1)

    bins_lat_col = pd.cut(df[lat_col], lat_intervals, labels=list(range(splits)), include_lowest=True)
    df[lat_col + "_binned"] = bins_lat_col

    return df


def create_grid_ids(df: pd.DataFrame, longitude_col: str, lattitude_col: str) -> List[str]:
    """
    Creates grid ids, which represent nodes in our city
    Every id is indicated by <longitude int.><lattitude int.>

    Args:
        longitude_col (str): Binned longitude column
        lattitude_col (str): Binned lattitude column

    Returns:
        List[str]: list of grid ids
    """
    grid_id = []
    for lng, lat in zip(df[longitude_col], df[lattitude_col]):
        grid_id.append(str(lng) + str(lat))

    return grid_id


def neighbourhood_adjacency_matrix(region_ordering: List[str]):
    """
    Creates adjacency matrix based on neighouring nodes(grids)

    Args:
        df (pd.DataFrame): [description]
        region_ordering (List[str]): [description]
        lng_col (str): [description]
        lat_col (str): [description]
        splits (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """

    neighbourhood_graph = np.zeros((len(region_ordering), len(region_ordering)))
    region_ordering_dict = {i: nodes for i, nodes in enumerate(region_ordering)}
    region_ordering_dict_rev = {nodes: i for i, nodes in enumerate(region_ordering)}

    for i in tqdm(range(len(region_ordering)), total=len(region_ordering)):
        node_name = region_ordering_dict[i]

        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if j == 0 and k == 0:
                    continue
                else:

                    # GRID POINT TO THE RIGHT
                    string_name = f"{int(node_name[0]) + j}{int(node_name[1]) + k}"
                    if string_name in region_ordering_dict_rev:
                        idx_connection = region_ordering_dict_rev[string_name]

                        neighbourhood_graph[i, idx_connection] = 1
                        neighbourhood_graph[idx_connection, i] = 1

    return neighbourhood_graph


def correlation_adjacency_matrix(
    rides_df: pd.DataFrame, region_ordering: List[str], id_col: str, time_col: str, threshold: float = 0.25
) -> pd.DataFrame:
    """
    Creates adjacency matrix, where the correlation of historical rides
    is used as the weights between regions/grid spaces.

    Args:
        rides_df ([type]): [description]
    """

    correlation_graph = np.zeros((len(region_ordering), len(region_ordering)))

    for i, node_base in tqdm.tqdm(enumerate(region_ordering), total=len(region_ordering)):
        for j, node_compare in enumerate(region_ordering):
            if i > j or i == j:
                continue

            df_1 = rides_df[rides_df[id_col] == node_base]
            df_2 = rides_df[rides_df[id_col] == node_compare]

            counts_1 = df_1[time_col].value_counts()
            counts_2 = df_2[time_col].value_counts()

            idx_intersections = counts_1.index.intersection(counts_2.index)

            values_1 = counts_1.loc[idx_intersections].values
            values_2 = counts_2.loc[idx_intersections].values

            corr_coef = np.abs(np.corrcoef(values_1, values_2)[0, 1])
            if corr_coef > threshold:
                correlation_graph[i, j] = 1  # corr_coef
                correlation_graph[j, i] = 1  # corr_coef
    return correlation_graph


def features_targets_and_externals(
    df: pd.DataFrame,
    region_ordering: List[str],
    id_col: str,
    time_col: str,
    time_encoder: OneHotEncoder,
    weather: Weather_container,
    time_interval: str,
    latitude: str,
    longitude: str,
):
    """
    Function that computes the node features (outflows), target values (next step prediction)
    and external data such as time_encoding and weather information

    Args:
        df (pd.DataFrame): [description]
        region_ordering (List[str]): [description]
        id_col (str): [description]
        time_col (str): [description]
        time_encoder (OneHotEncoder): [description]
        weather (Weather_container): [description]

    Returns:
        [type]: [description]
    """

    id_grouped_df = df.groupby(id_col)
    lat_dict = dict()
    lng_dict = dict()
    for node in region_ordering:
        grid_group_df = id_grouped_df.get_group(node)
        lat_dict[node] = grid_group_df[latitude].mean()
        lng_dict[node] = grid_group_df[longitude].mean()

    grouped_df = df.groupby([time_col, id_col])

    dt_range = pd.date_range(df[time_col].min(), df[time_col].max(), freq=time_interval)

    node_inflows = np.zeros((len(dt_range), len(region_ordering), 1))
    lat_vals = np.zeros((len(dt_range), len(region_ordering)))
    lng_vals = np.zeros((len(dt_range), len(region_ordering)))

    targets = np.zeros((len(dt_range) - 1, len(region_ordering)))

    # arrays for external data
    weather_external = np.zeros((len(dt_range), 4))
    num_cats = 0
    for cats in time_encoder.categories_:
        num_cats += len(cats)
    time_external = np.zeros((len(dt_range), num_cats))

    # Loop through every (timestep, node) pair in dataset. For each find number of outflows and set as feature
    # also set the next timestep for the same node as the target.

    for t, starttime in tqdm(enumerate(dt_range), total=len(dt_range)):
        for i, node in enumerate(region_ordering):

            query = (starttime, node)
            try:
                group = grouped_df.get_group(query)
                node_inflows[t, i] = len(group)

            except KeyError:
                node_inflows[t, i] = 0

            lat_vals[t, i] = lat_dict[node]
            lng_vals[t, i] = lng_dict[node]

            # current solution:
            # The target to predict, is the number of inflows at next timestep.
            if t > 0:
                targets[t - 1, i] = node_inflows[t, i]

        time_obj = group[time_col].iloc[0]
        time_external[t, :] = time_encoder.transform(
            np.array([[time_obj.hour, time_obj.weekday(), time_obj.month]])
        ).toarray()

        start_time_dt = pd.Timestamp(starttime).to_pydatetime()

        weather_dat = weather.get_weather_df(start=start_time_dt, end=start_time_dt + timedelta(hours=1))
        weather_dat = np.nan_to_num(weather_dat, copy=False, nan=0.0)
        weather_external[t, :] = weather_dat

    time_external = time_external[:-1, :]
    # normalize weather features
    weather_external = (weather_external - weather_external.mean(axis=0)) / (weather_external.std(axis=0) + 1e-6)
    weather_external = weather_external[:-1, :]

    X = node_inflows[:-1, :, :]
    lng_vals = lng_vals[:-1, :]
    lat_vals = lat_vals[:-1, :]

    feature_scaler = StandardScaler()
    feature_scaler.fit(X[:, :, 0])

    target_scaler = StandardScaler()
    target_scaler.fit(targets)

    return X, lat_vals, lng_vals, targets, time_external, weather_external, feature_scaler, target_scaler


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Creates distance based adjacency matrix.

    TAKEN FROM https://github.com/sshleifer/Graph-WaveNet/blob/master/gen_adj_mx.py
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


class Dataset:
    def __init__(
        self,
        adjacency_matrix: np.array,
        targets: np.array,
        X: np.array,
        weather_information: np.array = None,
        time_encoding: np.array = None,
        feature_scaler: StandardScaler = None,
        target_scaler: StandardScaler = None,
        latitude: np.array = None,
        longitude: np.array = None,
    ):
        self.adjacency_matrix = adjacency_matrix
        self.scipy_graph = scipy.sparse.lil_matrix(adjacency_matrix)
        self.targets = targets
        self.X = X
        self.weather_information = np.expand_dims(weather_information, -1)
        self.time_encoding = np.expand_dims(time_encoding, -1)
        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(self.scipy_graph)
        self.edge_weight = self.edge_weight.type(torch.FloatTensor)
        self.latitude = np.expand_dims(
            (latitude - latitude.mean(axis=-1).reshape(-1, 1)) / latitude.std(axis=-1).reshape(-1, 1), -1
        )
        self.longitude = np.expand_dims(
            (longitude - longitude.mean(axis=-1).reshape(-1, 1)) / longitude.std(axis=-1).reshape(-1, 1), -1
        )

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        self.num_observations, self.num_nodes, self.num_features = self.X.shape

    def _prep_data(self, num_history: int = 4):
        num_timesteps = self.num_observations - num_history

        # INITIALIZE TENSORS WITH SHAPES:
        # (NUM_TIMESTEPS, NUM_HISTORY, FEATURES)
        X_prep = torch.zeros((num_timesteps, *self.X[0].repeat(num_history, 1).T.shape))
        lat_prep = torch.zeros((num_timesteps, *self.latitude[0].repeat(num_history, 1).T.shape))
        lng_prep = torch.zeros((num_timesteps, *self.longitude[0].repeat(num_history, 1).T.shape))
        weather_prep = torch.zeros((num_timesteps, *self.weather_information[0].repeat(num_history, 1).T.shape))
        time_prep = torch.zeros((num_timesteps, *self.time_encoding[0].repeat(num_history, 1).T.shape))
        targets_prep = torch.zeros((num_timesteps, self.num_nodes))

        for i in range(num_history, self.num_observations):
            count = num_history
            graph_tensor = torch.zeros_like(X_prep[0])
            lat_tensor = torch.zeros_like(lat_prep[0])
            lng_tensor = torch.zeros_like(lng_prep[0])
            weather_tensor = torch.zeros_like(weather_prep[0])
            time_tensor = torch.zeros_like(time_prep[0])
            while count > 0:

                # INSERT THE LAST NUM_HISTORY TIME SNAPSHOTS
                graph_tensor[num_history - count, :] = torch.Tensor(self.X[i - count]).squeeze()
                lat_tensor[num_history - count] = torch.Tensor(self.latitude[i - count]).squeeze()
                lng_tensor[num_history - count] = torch.Tensor(self.longitude[i - count]).squeeze()
                weather_tensor[num_history - count] = torch.Tensor(self.weather_information[i - count]).squeeze()
                time_tensor[num_history - count] = torch.Tensor(self.time_encoding[i - count]).squeeze()

                count -= 1

            X_prep[i - num_history] = graph_tensor
            lat_prep[i - num_history] = lat_tensor
            lng_prep[i - num_history] = lng_tensor
            weather_prep[i - num_history] = weather_tensor
            time_prep[i - num_history] = time_tensor

            # TAKE TARGET FROM TIMESTEP i-1 AS THIS TIMESTEP CONTAINS
            # THE DEMANDS OF TIMESTEP i
            targets_prep[i - num_history] = torch.Tensor(self.targets[i - 1])

        return X_prep, lat_prep, lng_prep, weather_prep, time_prep, targets_prep

    def create_temporal_dataset(self, num_history: int = 4, scale: bool = True):
        X_prep, lat_prep, lng_prep, weather_prep, time_prep, targets_prep = self._prep_data(num_history)

        if scale:
            X_prep_shape = X_prep.shape
            X_prep = self.feature_scaler.transform(X_prep.view(-1, X_prep_shape[-1]))
            X_prep = torch.Tensor(X_prep).reshape(X_prep_shape)

            targets_prep = torch.Tensor(self.target_scaler.transform(targets_prep))

        dataset = CustomTemporalSignal(
            weather_information=weather_prep,
            time_encoding=time_prep,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            features=X_prep,
            targets=targets_prep,
            latitude=lat_prep,
            longitude=lng_prep,
        )
        return dataset

    @staticmethod
    def train_test_split(
        dataset: Union[CustomTemporalSignal, "Dataset"],
        num_history: int = 4,
        ratio: float = 0.8,
        scale: bool = True,
        shuffle: bool = True,
    ):
        if isinstance(dataset, CustomTemporalSignal):
            train, test = temporal_signal_split(dataset, train_ratio=ratio, shuffle=shuffle)
        elif isinstance(dataset, Dataset):
            train, test = temporal_signal_split(
                dataset.create_temporal_dataset(num_history), train_ratio=ratio, shuffle=shuffle
            )
        else:
            print("input type is not correct...")

        return train, test


def temporal_signal_split(data_iterator, train_ratio: float = 0.8, shuffle: bool = True):
    np.random.seed(42)
    train_snapshots = int(train_ratio * len(data_iterator))
    # Shuffle test/train otherwise take last datapoints for testing
    if shuffle:
        permutation = np.random.permutation(len(data_iterator))
        print("Shuffling data...")
    else:
        permutation = np.arange(len(data_iterator))
        print("Not shuffling data...")
    train_idx = permutation[0:train_snapshots]
    test_idx = permutation[train_snapshots:]

    train_iterator = CustomTemporalSignal(
        weather_information=data_iterator.weather_information[train_idx],
        time_encoding=data_iterator.time_encoding[train_idx],
        edge_index=data_iterator.edge_index,
        edge_weight=data_iterator.edge_weight,
        features=data_iterator.features[train_idx],
        targets=data_iterator.targets[train_idx],
        latitude=data_iterator.latitude,
        longitude=data_iterator.longitude,
    )

    test_iterator = CustomTemporalSignal(
        weather_information=data_iterator.weather_information[test_idx],
        time_encoding=data_iterator.time_encoding[test_idx],
        edge_index=data_iterator.edge_index,
        edge_weight=data_iterator.edge_weight,
        features=data_iterator.features[test_idx],
        targets=data_iterator.targets[test_idx],
        latitude=data_iterator.latitude,
        longitude=data_iterator.longitude,
    )

    return train_iterator, test_iterator
