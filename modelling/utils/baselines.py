import numpy as np
from typing import Union
import torch
from src.models.models import CustomTemporalSignal
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from src.utils import compute_mae, compute_mape


def historical_average(
    train_data: Union[np.array, torch.Tensor, CustomTemporalSignal],
    test_data: Union[np.array, torch.Tensor, CustomTemporalSignal],
    scaler: StandardScaler = None,
):

    assert type(train_data) == type(test_data), "Have to be type!"

    if train_data.__class__.__name__ == "CustomTemporalSignal":
        if scaler is not None:
            train_targets = scaler.inverse_transform(train_data.targets)
            test_targets = scaler.inverse_transform(test_data.targets)
            HA = train_targets.mean(0)
            HA = np.expand_dims(HA, 0)
            HA = np.repeat(HA, test_data.targets.shape[0], 0)
            MSE = ((HA - test_targets)**2).mean()
            MAE = compute_mae(test_targets, HA)
            MAPE = compute_mape(test_targets, HA)

        else:
            train_targets = train_data.targets
            test_targets = test_data.targets
            HA = train_targets.mean(0)
            HA = HA.repeat(test_data.targets.shape[0], 1)
            MSE = (HA - test_targets).pow(2).mean()
            MAE = compute_mae(test_targets, HA)
            MAPE = compute_mape(test_targets, HA)
    else:
        if scaler is not None:
            train_targets = scaler.inverse_transform(train_data)
            test_targets = scaler.inverse_transform(test_data)
        else:
            train_targets = train_data
            test_targets = test_data
        MSE = ((train_targets - test_targets) ** 2).mean()

    return MSE, MAE, MAPE


def ARIMA(targets: np.array, train_size: float = 0.8):
    """
    Outputs MSE error from ARIMA model where a model has been fitted to timeseries data
    for each node. THis means that for 20 nodes, we fit 20 different ARIMA models.
    We then predict on test data and finally aggregate all predictions together
    and compute MSE against the test data.
    This way we get 1-step predictions of ARIMA model, similar to our DL model

    Args:
        targets (np.array): [description]
        train_size (float, optional): [description]. Defaults to 0.8.
    """

    train_targets, test_targets = train_test_split(targets, train_size=train_size)

    nodes_preds = np.zeros((len(test_targets), train_targets.shape[-1]))
    for node in range(train_targets.shape[-1]):
        train_targets_node = train_targets[:, node]
        test_targets_node = test_targets[:, node]
        model = pm.auto_arima(train_targets_node, seasonal=False, m=1)

        for i in range(len(test_targets_node)):
            fit_data = np.append(train_targets_node, test_targets_node[0:i])
            try:
                model.fit(fit_data)
                y_hat = model.predict(1)
            except Exception as e:
                y_hat = test_targets_node[i]
            nodes_preds[i, node] = y_hat
    
    MSE = np.mean((nodes_preds - test_targets)**2)
    return MSE
    


