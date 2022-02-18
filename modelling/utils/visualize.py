import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_losses(losses: dict):
    fig, ax = plt.subplots()
    if "train_loss" in losses and "test_loss" in losses:
        ax.plot(range(1, len(losses["train_loss"]) + 1), losses["train_loss"], label="Train loss (RMSE)")
        ax.plot(range(1, len(losses["test_loss"]) + 1), losses["test_loss"], label="Test loss (RMSE)")
        ax.legend()
    else:
        for key, value in losses.items():
            ax.plot(range(1, len(value) + 1), value, label=key)
            ax.legend()
    return fig, ax

def plot_errs_pr_node(test_errs, train_errs, num_nodes: int):

    train_errs = np.sqrt(np.mean((train_errs) ** 2, axis=0))
    test_errs = np.sqrt(np.mean((test_errs) ** 2, axis=0))

    stacked_errs = np.stack([train_errs, test_errs], -1)
    node_level_errs_df = pd.DataFrame(stacked_errs, columns=["Train Errors", "Test Errors"])
    node_level_errs_df.plot.bar(xlabel="node (region)", ylabel="RMSE", title="Node-level RMSE", log=True)

    return node_level_errs_df


def plot_moments_pr_node(train_preds, train_targets):

    pred_mu_nodes = train_preds.mean(0)
    pred_std_nodes = train_preds.std(0)

    y_mu_nodes = train_targets.mean(0)
    y_std_nodes = train_targets.std(0)

    stacked_mu = np.stack([y_mu_nodes, pred_mu_nodes], -1)
    node_level_mu_df = pd.DataFrame(stacked_mu, columns=["$\mu$ nodes (True)", "$\mu$ nodes (Predicted)"])
    node_level_mu_df.plot.bar(xlabel="node (region)", ylabel="$\mu$", title="Node-level $\mu$ on test data")

    stacked_std = np.stack([y_std_nodes, pred_std_nodes], -1)
    node_level_std_df = pd.DataFrame(stacked_std, columns=["$\sigma$ nodes (True)", "$\sigma$ nodes (Predicted)"])
    node_level_std_df.plot.bar(xlabel="node (region)", ylabel="$\sigma$", title="Node-level $\sigma$ on test data")

    return node_level_mu_df, node_level_std_df
