from src.models.models import BaselineGATLSTM, Edgeconvmodel, GATLSTM, Encoder, Decoder, STGNNModel, BaselineGNNLSTM
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(
    dataset: Dataset,
    num_start_obs: int,
    batch_size: int = 24,
    epochs: int = 150,
    model_path: str = "edgeconv",
    weight_decay: float = 0.000000000001,
    learning_rate: float = 0.0005,
    lr_factor: float = 0.1,
    lr_patience: int = 25,
    hidden_size: int = 46,
    optimizer_name: str = "RMSprop",
    node_out_feature: int = 10, 
    dropout_p: float = 0.2,
    graph_hidden_size: int = 32,
    gpu: bool = False
):

    train_dataset, test_dataset = Dataset.train_test_split(dataset, num_history=12, ratio=0.991, shuffle=False)

    train_data_list = []
    test_data_list = []
    for i in range(len(train_dataset)):
        if i < num_start_obs:
            train_data_list.append(train_dataset[i])
        else:
            test_data_list.append(train_dataset[i])
    if num_start_obs < batch_size:
        train_loader = DataLoader(train_data_list, batch_size=num_start_obs, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)
    elif (num_start_obs + batch_size) > len(train_dataset):
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=1, shuffle=True)
    else:
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

    weather_features = train_dataset.weather_information.shape[-1]
    time_features = train_dataset.time_encoding.shape[-1]

    with open(f"{model_path}/settings.json", "rb") as f:
        edgeconv_params = json.load(f)
        edgeconv_params.pop("data_dir")
        edgeconv_params.pop("train_size")
        edgeconv_params.pop("batch_task_size")
        edgeconv_params.pop("k_shot")
        edgeconv_params.pop("adaptation_steps")
        edgeconv_params.pop("epochs")
        edgeconv_params.pop("adapt_lr")
        edgeconv_params.pop("meta_lr")
        edgeconv_params.pop("log_dir")
        edgeconv_params.pop("exclude")
        edgeconv_params.pop("gpu")

    model = Edgeconvmodel(
        node_in_features=1,
        weather_features=weather_features,
        time_features=time_features,
        node_out_features=node_out_feature,
        gpu=gpu,
        hidden_size=hidden_size,
        dropout_p=dropout_p
    )
    

    criterion = torch.nn.MSELoss()
    model.to(DEVICE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_factor < 1:
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=lr_factor, patience=lr_patience)
    train_losses = []
    test_losses = []
    best_loss = 100
    patience = 0

    for EPOCH in range(epochs):
        model.eval()
        test_loss = 0
        num_batch_test = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.to(DEVICE))
                test_loss += criterion(batch.y, out.view(batch.num_graphs, -1)).item()
                num_batch_test += 1
        torch.cuda.empty_cache()

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

        if lr_factor < 1:
            scheduler.step(test_loss)

        if EPOCH % 5 == 0:

            logger.info(f"Epoch number {EPOCH+1}")
            logger.info(f"Epoch avg RMSE loss (TRAIN): {train_losses[-1]}")
            logger.info(f"Epoch avg RMSE loss (TEST): {test_losses[-1]}")
            logger.info("-" * 10)
        
        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            patience = 0
        else:
            patience += 1

        if patience > 35:
            return model, train_losses, test_losses

    return model, train_losses, test_losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Model training argument parser")
    parser.add_argument("-d", "--data", type=str, help="path to processed data")
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--train_size", type=float, default=0.8, help="Ratio of data to be used for training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batchsize to be used")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Amount of weight decay in optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-sd", "--save_dir", type=str)
    parser.add_argument(
        "-f", "--lr_factor", type=float, default=1, help="factor for reduing learning rate with lr scheduler"
    )
    parser.add_argument("-p", "--lr_patience", type=int, default=100, help="Patience for reducing lr")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam")
    parser.add_argument("-g", "--gpu", action='store_true')

    args = parser.parse_args()
    open_file = open(args.data, "rb")
    dataset = dill.load(open_file)

    start_time = datetime.datetime.now()
    logger.info(str(vars(args)))
    logger.info(open_file)

    # We have 700 observations. We make a model that is trained on all data every 25 datapoint
    # this will give us 28 models. However, we don't need a model that is trained on all the data
    for i in range(28 - 1):
        start_idx = i * 25
        end_idx = (i+1) * 25
        logger.info(f"Fitting model at time: {str(start_time)}")
        model, train_loss, test_loss = train_model(
            dataset=dataset,
            num_start_obs=end_idx,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_path=args.model_path,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            lr_patience=args.lr_patience,
            optimizer_name=args.optimizer,
            gpu=args.gpu
        )
        end_time = datetime.datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        td = end_time - start_time
        minutes = round(td.total_seconds() / 60, 2)
        totsec = td.total_seconds()
        h = int(totsec // 3600)
        m = int((totsec % 3600) // 60)
        sec = int((totsec % 3600) % 60)
        logger.info(f"Total training time: {h}:{m}:{sec}")
        logger.info(f"Average Epoch time: {round(minutes/args.epochs, 2)} minutes")
        if len(args.save_dir) > 0 and os.path.exists(args.save_dir):
            dataset_name = args.data.split("/")[-1].split(".")[0]

            logger.info(f"Saving files to /zhome/2b/7/117471/Thesis/CASESTUDY/metalearn_finetuned_multiple")

            model.to("cpu")
            torch.save(model.state_dict(), f"/zhome/2b/7/117471/Thesis/CASESTUDY/metalearn_finetuned_multiple/{str(end_idx)}-model.pth")

            logger.info(f"model '{str(end_idx)}-model.pth' saved successfully")
            torch.cuda.empty_cache()