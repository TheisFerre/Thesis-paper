import torch
from torch.functional import einsum
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv, DynamicEdgeConv
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomData(Data):
    def __cat_dim__(self, key, value):
        cat_set = {"weather", "time_encoding", "y", "latitude", "longitude"}
        if key in cat_set:
            return None
        else:
            return super().__cat_dim__(key, value)


class CustomTemporalSignal(Dataset):
    def __init__(
        self, weather_information, time_encoding, edge_index, edge_weight, features, targets, latitude, longitude
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)
        self.weather_information = torch.Tensor(weather_information)
        self.time_encoding = torch.Tensor(time_encoding)
        self.feature_scaler = StandardScaler()
        # mean over time and number of examples
        self.features_scaled = self.feature_scaler.fit_transform(self.features.view(-1, self.features.shape[-1]))
        self.features_scaled = self.features_scaled.reshape(self.features.shape)
        self.latitude = torch.Tensor(latitude)
        self.longitude = torch.Tensor(longitude)

        self.target_scaler = StandardScaler()
        self.targets_scaled = self.target_scaler.fit_transform(self.targets)

    def _get_latitude(self, index: int):
        if self.latitude[index] is None:
            return self.latitude[index]
        else:
            return torch.FloatTensor(self.latitude[index])

    def _get_longitude(self, index: int):
        if self.longitude[index] is None:
            return self.longitude[index]
        else:
            return torch.FloatTensor(self.longitude[index])

    def _get_weather(self, index: int):
        if self.weather_information[index] is None:
            return self.weather_information[index]
        else:
            return torch.FloatTensor(self.weather_information[index])

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_time(self, index: int):
        if self.weather_information[index] is None:
            return self.time_encoding[index]
        else:
            return torch.FloatTensor(self.time_encoding[index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            return torch.FloatTensor(self.targets[time_index])

    def _get_target_scaled(self, time_index: int):
        if self.targets_scaled[time_index] is None:
            return self.targets_scaled[time_index]
        else:
            return torch.FloatTensor(self.targets_scaled[time_index])

    def _get_features_scaled(self, time_index: int):
        if self.features_scaled[time_index] is None:
            return self.features_scaled[time_index]
        else:
            return torch.FloatTensor(self.features_scaled[time_index])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # x = self._get_features_scaled(index)
        x = self._get_features(index).unsqueeze(0)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        # y = self._get_target_scaled(index)
        y = self._get_target(index)
        weather = self._get_weather(index)
        time_encoding = self._get_time(index)
        latitude = self._get_latitude(index)
        longitude = self._get_longitude(index)

        snapshot = CustomData(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        snapshot.weather = weather
        snapshot.time_encoding = time_encoding
        snapshot.latitude = latitude
        snapshot.longitude = longitude
        return snapshot

    def __get_item__(self, time_index: int):
        # x = self._get_features_scaled(time_index)
        x = self._get_features(time_index).unsqueeze(0)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        # y = self._get_target_scaled(time_index)
        y = self._get_target(time_index)
        weather = self._get_weather(time_index)
        time_encoding = self._get_time(time_index)
        latitude = self._get_latitude(time_index)
        longitude = self._get_longitude(time_index)

        snapshot = CustomData(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        snapshot.weather = weather
        snapshot.time_encoding = time_encoding
        snapshot.latitude = latitude
        snapshot.longitude = longitude
        return snapshot


class CustomTemporalSignalBatch(StaticGraphTemporalSignal):
    def __init__(self, weather_information, time_encoding, *args, **kwargs):
        super(CustomTemporalSignal, self).__init__(*args, **kwargs)
        self.weather_information = weather_information
        self.time_encoding = time_encoding

    def _get_weather(self, index: int):
        if self.weather_information[index] is None:
            return self.weather_information[index]
        else:
            return torch.FloatTensor(self.weather_information[index])

    def _get_time(self, index: int):
        if self.weather_information[index] is None:
            return self.time_encoding[index]
        else:
            return torch.FloatTensor(self.time_encoding[index])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self._get_features(index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(index)
        weather = self._get_weather(index)
        time_encoding = self._get_time(index)

        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        return snapshot, weather, time_encoding


class ExternalLSTM(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_nodes: int,
        out_features: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        external_feat: str = None,
    ):
        super(ExternalLSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.external_feat = external_feat

        self.lstm = torch.nn.LSTM(
            input_size=self.in_features, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True
        )
        self.embedding_hidden = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_nodes * self.out_features
        )
        self.embedding_state = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_nodes * self.out_features
        )

    def forward(self, data: Data):

        # check if batches are defined, otherwise set batchsize to 1
        x = getattr(data, self.external_feat)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        else:
            x = x

        # x shape should be (BATCHSIZE, SEQ LEN, FEATURES)
        _, (hidden_state, cell_state) = self.lstm(x)

        # only take last hidden state from last layer
        hidden_state = hidden_state[-1, :, :]
        cell_state = cell_state[-1, :, :]

        # reshape hidden_state to size (batch, nodes, features)
        embedding_hidden = self.embedding_hidden(hidden_state)
        embedding_output_hidden = embedding_hidden.reshape(-1, self.num_nodes * self.out_features)

        embedding_state = self.embedding_state(cell_state)
        embedding_output_state = embedding_state.reshape(-1, self.num_nodes * self.out_features)

        return embedding_output_state, embedding_output_hidden


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        node_in_features: int,
        num_nodes: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
        dropout_p: float = 0.3,
        gpu: bool = False
    ):
        super(GraphModel, self).__init__()
        self.node_in_features = node_in_features
        self.num_nodes = num_nodes
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.conv1_sh = GATv2Conv(node_in_features, node_out_features)
        self.conv2_sh = GATv2Conv(node_out_features, node_out_features)
        self.lstm = torch.nn.LSTM(
            input_size=node_out_features * num_nodes,
            hidden_size=self.num_nodes * self.node_out_features,
            batch_first=True,
        )
        self.gpu = gpu

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        lstm_inputs = torch.zeros((batch_size, num_hist, self.node_out_features * self.num_nodes))
        if self.gpu:
            lstm_inputs = lstm_inputs.cuda()

        for i in range(num_hist):
            x, edge_index, edge_weight = data.x[:, i, :], data.edge_index, data.edge_attr
            x = x.reshape(-1, 1)

            x = self.conv1_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_p, training=self.training)
            x = self.conv2_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            lstm_inputs[:, i, :] = x.reshape(batch_size, self.node_out_features * self.num_nodes)
        _, (hidden_state, cell_state) = self.lstm(lstm_inputs)

        # only take last hidden state from last layer
        hidden_state = hidden_state[-1, :, :]
        cell_state = cell_state[-1, :, :]

        return cell_state, hidden_state


class Encoder(torch.nn.Module):
    def __init__(
        self,
        node_in_features: int,
        num_nodes: int,
        time_features: int,
        weather_features: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
        gpu: bool = False
    ):
        super(Encoder, self).__init__()
        self.node_in_features = node_in_features
        self.num_nodes = num_nodes
        self.time_features = time_features
        self.weather_features = weather_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.gpu = gpu

        self.weight_graph_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )
        
        self.weight_graph_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_weather_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_weather_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_time_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_time_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.graph_model = GraphModel(
            node_in_features=self.node_in_features,
            num_nodes=self.num_nodes,
            node_out_features=self.node_out_features,
            hidden_size=self.hidden_size,
            gpu=self.gpu
        )

        self.weather_model = ExternalLSTM(
            in_features=self.weather_features,
            num_nodes=self.num_nodes,
            out_features=self.node_out_features,
            hidden_size=self.hidden_size,
            external_feat="weather",
        )

        self.time_model = ExternalLSTM(
            in_features=self.time_features,
            num_nodes=self.num_nodes,
            out_features=self.node_out_features,
            hidden_size=self.hidden_size,
            external_feat="time_encoding",
        )

    def forward(self, data: Data):

        cell_state_graph, hidden_state_graph = self.graph_model(data)
        cell_state_weather, hidden_state_weather = self.weather_model(data)
        cell_state_time, hidden_state_time = self.time_model(data)

        cell_state_fused = (
            torch.einsum("ab, b -> ab", cell_state_graph, self.weight_graph_cell_state)
            + torch.einsum("ab, b -> ab", cell_state_weather, self.weight_weather_cell_state)
            + torch.einsum("ab, b -> ab", cell_state_time, self.weight_time_cell_state)
        )
        hiden_state_fused = (
            torch.einsum("ab, b -> ab", hidden_state_graph, self.weight_graph_hidden_state)
            + torch.einsum("ab, b -> ab", hidden_state_weather, self.weight_weather_hidden_state)
            + torch.einsum("ab, b -> ab", hidden_state_time, self.weight_time_hidden_state)
        )

        return cell_state_fused, hiden_state_fused


class Decoder(torch.nn.Module):
    def __init__(self, node_out_features, num_nodes):
        super(Decoder, self).__init__()
        self.node_out_features = node_out_features
        self.num_nodes = num_nodes
        self.hidden_size = self.num_nodes * self.node_out_features

        # DESIGN QUESTION HAVE LSTM OUTPUT HIDDEN STATE FOR EACH NODE
        # OR MAKE IT OUTPUT IT FOR THEM ALL TOGETHER AND THEN RESHAPE/SPLIT
        # MANUALLY USING PYTORCH?
        self.lstm = torch.nn.LSTM(
            input_size=self.num_nodes, hidden_size=self.hidden_size, num_layers=1, batch_first=True
        )
        self.linear = torch.nn.Linear(self.hidden_size, self.num_nodes)

    def forward(self, x_input, hidden_state, cell_state):
        lstm_out, (hidden, cell) = self.lstm(x_input, (hidden_state, cell_state))
        output = self.linear(lstm_out)

        return output, (hidden, cell)


class STGNNModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(STGNNModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_nodes = encoder.num_nodes

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        cell_state_fused, hidden_state_fused = self.encoder(data)

        out, (hidden_state, cell_state) = self.decoder(
            data.x[:, -1, :].reshape(batch_size, 1, self.num_nodes),
            hidden_state_fused.unsqueeze(0),
            cell_state_fused.unsqueeze(0),
        )

        return out


class GATLSTM(torch.nn.Module):
    """
    This model is using a stand GNN layer to embed nodes
    It computes features for each node in the graph over q time steps.
    A LSTM model then takes the sequence of q node features (for each node)
    the last output of the LSTM model is then fed into a linear layer
    that also takes into account the weather and time features
    """

    def __init__(
        self,
        node_in_features: int,
        weather_features: int,
        time_features: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
        dropout_p: float = 0.3,
        gpu: bool = False
    ):
        super(GATLSTM, self).__init__()
        self.node_in_features = node_in_features
        self.weather_features = weather_features
        self.time_features = time_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.gpu = gpu
        self.conv1_sh = GATv2Conv(node_in_features, 16)
        self.conv2_sh = GATv2Conv(16, 32)
        self.conv3_sh = GATv2Conv(32, node_out_features)
        self.lstm = torch.nn.LSTM(
            input_size=node_out_features,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size + self.weather_features + self.time_features, 1)

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        # SHAPE (SEQ LENGTH, BATCHSIZE X NUM_NODES, NODE_OUT_FEATURES)
        
        lstm_inputs = torch.zeros((num_hist, batch_size * nodes, self.node_out_features))
        if self.gpu:
            lstm_inputs = lstm_inputs.cuda()

        for i in range(num_hist):
            x, edge_index, edge_weight = data.x[:, i, :], data.edge_index, data.edge_attr
            x = x.reshape(-1, 1)

            x = self.conv1_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_p, training=self.training)
            x = self.conv2_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = self.conv3_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            lstm_inputs[i, :, :] = x

        # only take last output
        out, _ = self.lstm(lstm_inputs)
        out = out[-1]
        weather_repeated = data.weather[:, -1, :].repeat(nodes, 1)
        time_repeated = data.time_encoding[:, -1, :].repeat(nodes, 1)
        out_embedded = torch.cat([out, weather_repeated, time_repeated], dim=1)
        if self.gpu:
            out_embedded = out_embedded.cuda()

        prediction = self.linear(out_embedded)

        prediction = prediction.reshape(batch_size, nodes, 1)

        return prediction


class Edgeconvmodel(torch.nn.Module):
    """
    This model is using the Dynamic Edge Convolution operator
    It computes features for each node in the graph over q time steps.
    A LSTM model then takes the sequence of q node features (for each node)
    the last output of the LSTM model is then fed into a linear layer
    that also takes into account the weather and time features
    """

    def __init__(
        self,
        node_in_features: int,
        weather_features: int,
        time_features: int,
        node_out_features: int = 8,
        hidden_size: int = 32,
        dropout_p: float = 0.5,
        gpu: bool = False,
        k: int = 20
    ):
        super(Edgeconvmodel, self).__init__()
        self.node_in_features = node_in_features
        self.weather_features = weather_features
        self.time_features = time_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.gpu = gpu
        self.k = k
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear((node_in_features + 2) * 2, (node_in_features + 2) * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear((node_in_features + 2) * 4, 32),
        )
        self.edgeconv1 = DynamicEdgeConv(nn=self.lin1, k=self.k)

        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64)
        )
        self.edgeconv2 = DynamicEdgeConv(nn=self.lin2, k=self.k)

        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(64 * 2, 32),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, node_out_features),
        )
        self.edgeconv3 = DynamicEdgeConv(nn=self.lin3, k=self.k)

        self.lstm = torch.nn.LSTM(
            input_size=node_out_features,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.linear = torch.nn.Linear(self.hidden_size + self.weather_features + self.time_features, 1)

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        # SHAPE (SEQ LENGTH, BATCHSIZE X NUM_NODES, NODE_OUT_FEATURES)
        lstm_inputs = torch.zeros((num_hist, batch_size * nodes, self.node_out_features))
        if self.gpu:
            lstm_inputs = lstm_inputs.cuda()

        for i in range(num_hist):
            x = data.x[:, i, :]
            lat, lng = data.latitude[:, i, :], data.longitude[:, i, :]
            lat = lat.reshape(batch_size, nodes, 1)
            lng = lng.reshape(batch_size, nodes, 1)
            x = x.reshape(batch_size, nodes, 1)
            x = torch.cat([x, lat, lng], dim=-1)
            data_list = [Data(x=x[i, :, :]) for i in range(batch_size)]
            batched_data = Batch.from_data_list(data_list)
            if self.gpu:
                batched_data = batched_data.cuda()

            x = self.edgeconv1(batched_data.x, batch=batched_data.batch)
            x = self.edgeconv2(x, batch=batched_data.batch)
            x = self.edgeconv3(x, batch=batched_data.batch)

            lstm_inputs[i, :, :] = x

        # only take last output
        out, _ = self.lstm(lstm_inputs)
        out = out[-1]
        weather_repeated = data.weather[:, -1, :].repeat(nodes, 1)
        time_repeated = data.time_encoding[:, -1, :].repeat(nodes, 1)
        out_embedded = torch.cat([out, weather_repeated, time_repeated], dim=1)

        if self.gpu:
            out_embedded = out_embedded.cuda()

        prediction = self.linear(out_embedded)

        prediction = prediction.reshape(batch_size, nodes, 1)

        return prediction


class BaselineGNNLSTM(torch.nn.Module):
    """
    This model is using a stand GNN layer to embed nodes
    It computes features for each node in the graph over q time steps.
    A LSTM model then takes the sequence of q node features (for each node)
    the last output of the LSTM model is then fed into a linear layer
    that also takes into account the weather and time features
    """

    def __init__(
        self,
        node_in_features: int,
        weather_features: int,
        time_features: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
        dropout_p: float = 0.3,
        graph_hidden_size: int = 32,
        gpu: bool = False
    ):
        super(BaselineGNNLSTM, self).__init__()
        self.node_in_features = node_in_features
        self.weather_features = weather_features
        self.time_features = time_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.gpu = gpu
        self.conv1_sh = GraphConv(node_in_features, graph_hidden_size)
        self.conv2_sh = GraphConv(graph_hidden_size, node_out_features)
        self.lstm = torch.nn.LSTM(
            input_size=node_out_features,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size + self.weather_features + self.time_features, 1)

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        # SHAPE (SEQ LENGTH, BATCHSIZE X NUM_NODES, NODE_OUT_FEATURES)
        
        lstm_inputs = torch.zeros((num_hist, batch_size * nodes, self.node_out_features))
        if self.gpu:
            lstm_inputs = lstm_inputs.cuda()

        for i in range(num_hist):
            x, edge_index, edge_weight = data.x[:, i, :], data.edge_index, data.edge_attr
            x = x.reshape(-1, 1)

            x = self.conv1_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_p, training=self.training)
            x = self.conv2_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)

            lstm_inputs[i, :, :] = x

        # only take last output
        out, _ = self.lstm(lstm_inputs)
        out = out[-1]
        weather_repeated = data.weather[:, -1, :].repeat(nodes, 1)
        time_repeated = data.time_encoding[:, -1, :].repeat(nodes, 1)
        out_embedded = torch.cat([out, weather_repeated, time_repeated], dim=1)
        if self.gpu:
            out_embedded = out_embedded.cuda()

        prediction = self.linear(out_embedded)

        prediction = prediction.reshape(batch_size, nodes, 1)

        return prediction


class BaselineGATLSTM(torch.nn.Module):
    """
    This model is using a stand GNN layer to embed nodes
    It computes features for each node in the graph over q time steps.
    A LSTM model then takes the sequence of q node features (for each node)
    the last output of the LSTM model is then fed into a linear layer
    that also takes into account the weather and time features
    """

    def __init__(
        self,
        node_in_features: int,
        weather_features: int,
        time_features: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
        dropout_p: float = 0.3,
        graph_hidden_size: int = 32,
        gpu: bool = False
    ):
        super(BaselineGATLSTM, self).__init__()
        self.node_in_features = node_in_features
        self.weather_features = weather_features
        self.time_features = time_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.gpu = gpu
        self.conv1_sh = GATv2Conv(node_in_features, graph_hidden_size)
        self.conv2_sh = GATv2Conv(graph_hidden_size, node_out_features)
        self.lstm = torch.nn.LSTM(
            input_size=node_out_features,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size + self.weather_features + self.time_features, 1)

    def forward(self, data: Data):
        batch_size, num_hist, nodes = data.x.shape
        # SHAPE (SEQ LENGTH, BATCHSIZE X NUM_NODES, NODE_OUT_FEATURES)
        
        lstm_inputs = torch.zeros((num_hist, batch_size * nodes, self.node_out_features))
        if self.gpu:
            lstm_inputs = lstm_inputs.cuda()

        for i in range(num_hist):
            x, edge_index, edge_weight = data.x[:, i, :], data.edge_index, data.edge_attr
            x = x.reshape(-1, 1)

            x = self.conv1_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_p, training=self.training)
            x = self.conv2_sh(x=x, edge_index=edge_index)  # , edge_weight=edge_weight)

            lstm_inputs[i, :, :] = x

        # only take last output
        out, _ = self.lstm(lstm_inputs)
        out = out[-1]
        weather_repeated = data.weather[:, -1, :].repeat(nodes, 1)
        time_repeated = data.time_encoding[:, -1, :].repeat(nodes, 1)
        out_embedded = torch.cat([out, weather_repeated, time_repeated], dim=1)
        if self.gpu:
            out_embedded = out_embedded.cuda()

        prediction = self.linear(out_embedded)

        prediction = prediction.reshape(batch_size, nodes, 1)

        return prediction
