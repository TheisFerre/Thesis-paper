from src.models.models import Edgeconvmodel, GATLSTM, Encoder, Decoder, STGNNModel
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import dill
from src.data.process_dataset import Dataset
from torch_geometric.data import DataLoader
import argparse
import datetime
import logging
import os
import json
from distutils.dir_util import copy_tree
import optuna
from optuna.trial import TrialState

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

open_file = open("/home/s163700/Thesis/data/processed/citibike2014-tripdata-regions.pkl", "rb")
dataset = dill.load(open_file)


def objective(trial):

    train_dataset, test_dataset = Dataset.train_test_split(dataset, num_history=12, ratio=0.9421, shuffle=False)

    train_data_list = []
    np.random.seed(42)
    random_idx = np.random.permutation((len(train_dataset)))
    random_idx_subset = random_idx[0:int(len(random_idx) * 0.25)]


    for i in random_idx_subset:
        train_data_list.append(train_dataset[i])
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)

    test_data_list = []
    for i in range(len(test_dataset)):
        test_data_list.append(test_dataset[i])
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=True)

    weather_features = train_dataset.weather_information.shape[-1]
    time_features = train_dataset.time_encoding.shape[-1]

    # GET OPTIMIZATION INFORMATION!

    optimizer_name = "RMSprop"
    node_out_features = trial.suggest_int("node_out_features", 6, 16)
    hidden_size = trial.suggest_int("hidden_size", 12, 40)
    dropout_p = trial.suggest_float("dropout_p", 0.2, 0.6)
    k = trial.suggest_int("K", 15, 35)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    learning_rate = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

    model = Edgeconvmodel(
        node_in_features=1,
        weather_features=weather_features,
        time_features=time_features,
        node_out_features=node_out_features,
        hidden_size=hidden_size,
        dropout_p=dropout_p,
        k=k,
        gpu=True
    )

    criterion = torch.nn.MSELoss()
    model.to(DEVICE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=50)
    train_losses = []
    test_losses = []

    NUM_EPOCHS = 250

    for EPOCH in range(NUM_EPOCHS):
        model.eval()
        test_loss = 0
        num_batch_test = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.to(DEVICE))
                test_loss += criterion(batch.y, out.view(batch.num_graphs, -1)).item()
                num_batch_test += 1

        model.train()
        train_loss = 0
        num_batch_train = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(batch.to(DEVICE))

            loss = criterion(batch.y, out.view(batch.num_graphs, -1))
            train_loss += loss.item()
            num_batch_train += 1

            loss.backward()
            optimizer.step()

        train_loss = train_loss / (num_batch_train)
        train_losses.append(np.sqrt(train_loss))

        test_loss = test_loss / (num_batch_test)
        test_losses.append(np.sqrt(test_loss))

        scheduler.step(test_loss)

        if EPOCH % 25 == 0:
            logger.info(f"Epoch number {EPOCH+1}")
            logger.info(f"Epoch avg RMSE loss (TRAIN): {train_losses[-1]}")
            logger.info(f"Epoch avg RMSE loss (TEST): {test_losses[-1]}")
            logger.info("-" * 10)

        elif EPOCH == NUM_EPOCHS - 1:
            logger.info(f"Epoch number {EPOCH+1}")
            logger.info(f"Epoch avg RMSE loss (TRAIN): {train_losses[-1]}")
            logger.info(f"Epoch avg RMSE loss (TEST): {test_losses[-1]}")
            logger.info("-" * 10)
            torch.cuda.empty_cache()

        trial.report(test_loss, EPOCH)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_loss


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    logging.basicConfig(
        filename="/home/s163700/Thesis/models/edgeonv-logs.log",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    logger = logging.getLogger(__name__)

    logger.info(f"Starting search: {str(start_time)}")
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=40, interval_steps=10)
    )
    study.optimize(objective, n_trials=75)
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info(f"Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    cur_dir = os.getcwd()
    while True:
        split_dir = cur_dir.split("/")
        if "Thesis" not in split_dir:
            break
        else:
            if split_dir[-1] == "Thesis":
                break
            else:
                os.chdir("..")
                cur_dir = os.getcwd()
    os.chdir("models")
    cur_dir = os.getcwd()
    logger.info(f"Saving files to {cur_dir}/edgeconv_hyperopt_{end_time_str}")
    os.mkdir(f"edgeconv_hyperopt_{end_time_str}")

    trial_df = study.trials_dataframe()
    trial_df.to_csv(f"edgeconv_hyperopt_{end_time_str}/trials.csv", index=False)

    logger.info("Files saved successfully")

    os.chdir(f"edgeconv_hyperopt_{end_time_str}")
    os.mkdir(f"logs")

    target_dir = "logs"
    source_dir = f"{os.getenv('HOME')}/.lsbatch"

    copy_tree(source_dir, target_dir)

    for f in os.listdir(target_dir):
        if not f.endswith("err") and not f.endswith("out"):
            os.remove(f"{target_dir}/{f}")

