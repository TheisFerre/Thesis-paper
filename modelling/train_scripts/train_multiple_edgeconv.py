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
    batch_size: int = 32,
    epochs: int = 200,
    model: str = "edgeconv",
    weight_decay: float = 0,
    learning_rate: float = 0.001,
    lr_factor: float = 1.0,
    lr_patience: int = 100,
    hidden_size: int = 64,
    optimizer_name: str = "Adam",
    node_out_feature: int = 12, 
    dropout_p: float = 0.3,
    k: int = 30,
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

    if model == "edgeconv":
        model = Edgeconvmodel(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=node_out_feature,
            gpu=gpu,
            k=k,
            hidden_size=hidden_size,
            dropout_p=dropout_p
        )
    elif model == "seq2seq-gnn":
        num_nodes = dataset.num_nodes
        encoder = Encoder(
            node_in_features=1,
            num_nodes=num_nodes,
            node_out_features=node_out_feature,
            time_features=time_features,
            weather_features=weather_features,
            hidden_size=hidden_size,
            gpu=gpu
        )
        decoder = Decoder(node_out_features=node_out_feature, num_nodes=num_nodes)
        model = STGNNModel(encoder, decoder)
    elif model == "gatlstm":
        model = GATLSTM(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=node_out_feature,
            gpu=gpu,
            hidden_size=hidden_size,
            dropout_p=dropout_p
        )
    elif model == "baselinegnn":
        model = BaselineGNNLSTM(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=node_out_feature,
            gpu=gpu,
            hidden_size=hidden_size,
            graph_hidden_size=graph_hidden_size,
            dropout_p=dropout_p
        )
    elif model == "baselinegat":
        model = BaselineGATLSTM(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=node_out_feature,
            gpu=gpu,
            hidden_size=hidden_size,
            graph_hidden_size=graph_hidden_size,
            dropout_p=dropout_p
        )
    else:
        assert False, "Please provide a correct model name!"

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
    parser.add_argument(
        "-m", "--model", type=str, default="edgeconv", help="Use either [edgeconv, gatlstm, seq2seq-gnn"
    )
    parser.add_argument("-n", "--num_history", type=int, default=8, help="number of history steps for predicting")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, help="Ratio of data to be used for training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batchsize to be used")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Amount of weight decay in optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "-f", "--lr_factor", type=float, default=1, help="factor for reduing learning rate with lr scheduler"
    )
    parser.add_argument("-p", "--lr_patience", type=int, default=100, help="Patience for reducing lr")
    parser.add_argument("-hd", "--hidden_size", type=int, default=32)
    parser.add_argument("-o", "--optimizer", type=str, default="Adam")
    parser.add_argument("-no", "--node_out_feature", type=int, default=12)
    parser.add_argument("-dp", "--dropout", type=float, default=0.3)
    parser.add_argument("-k", "--k_neighbours", type=int, default=20)
    parser.add_argument("-gh", "--graph_hidden_size", type=int, default=32)
    parser.add_argument("-sd", "--save_dir", type=str, default="")

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
            model=args.model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            optimizer_name=args.optimizer,
            hidden_size=args.hidden_size,
            dropout_p=args.dropout,
            node_out_feature=args.node_out_feature,
            k=args.k_neighbours,
            graph_hidden_size=args.graph_hidden_size,
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

            logger.info(f"Saving files to {args.save_dir}/{dataset_name}")

            model.to("cpu")
            torch.save(model.state_dict(), f"{args.save_dir}/{dataset_name}/{str(end_idx)}-model.pth")

            logger.info(f"model '{str(end_idx)}-model.pth' saved successfully")
            torch.cuda.empty_cache()