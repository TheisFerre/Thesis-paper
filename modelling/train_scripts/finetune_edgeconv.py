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


def finetune_model(
    dataset: Dataset,
    model_path: str,
    data_type: str,
    train_size: float = 0.8,
    batch_size: int = 32,
    epochs: int = 200,
    weight_decay: float = 0,
    learning_rate: float = 0.001,
    lr_factor: float = 1.0,
    lr_patience: int = 100,
    optimizer_name: str = "Adam",
    gpu: bool = False
):


    train_dataset, test_dataset = Dataset.train_test_split(dataset, num_history=12, ratio=train_size, shuffle=True)

    train_data_list = []
    for i in range(len(train_dataset)):
        train_data_list.append(train_dataset[i])
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)

    test_data_list = []
    for i in range(len(test_dataset)):
        test_data_list.append(test_dataset[i])
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
        gpu=gpu,
        **edgeconv_params
    )
    edgeconv_state_dict = torch.load(f"{model_path}/vanilla_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(edgeconv_state_dict)

    criterion = torch.nn.MSELoss()
    model.to(DEVICE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_factor < 1:
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=lr_factor, patience=lr_patience)
    train_losses = []
    test_losses = []

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
        
        if (EPOCH + 1) % 50 == 0:
            model.to("cpu")
            torch.save(model.state_dict(), f"{model_path}/checkpoint_{EPOCH}_{data_type}_finetuned_vanilla_model.pth")
            model.to(DEVICE)

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
    parser.add_argument("-sd", "--save_to_dir", action='store_true')
    parser.add_argument(
        "-f", "--lr_factor", type=float, default=1, help="factor for reduing learning rate with lr scheduler"
    )
    parser.add_argument("-p", "--lr_patience", type=int, default=100, help="Patience for reducing lr")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam")
    parser.add_argument("-g", "--gpu", action='store_true')

    args = parser.parse_args()
    open_file = open(args.data, "rb")
    dataset = dill.load(open_file)
    open_file.close()

    TYPE = args.data.split("/")[-1].split("-")[-1]
    if "GRID" in TYPE:
        TYPE = "GRID"
    else:
        TYPE = "REGION"

    start_time = datetime.datetime.now()
    logger.info(str(vars(args)))
    logger.info(f"Fitting model at time: {str(start_time)}")
    model, train_loss, test_loss = finetune_model(
        dataset=dataset,
        model_path=args.model_path,
        data_type=TYPE,
        train_size=args.train_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_factor=args.lr_factor,
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

    logger.info(f"Saving files to: {args.model_path}")

    if args.save_to_dir:
        args_dict = vars(args)
        args_dict.pop("save_to_dir")
        save_dir = args.data.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{args.model_path}/{save_dir}"):
            os.mkdir(f"{args.model_path}/{save_dir}")
        with open(f"{args.model_path}/{save_dir}/finetune_{TYPE}_vanilla_settings.json", "w") as outfile:
            json.dump(args_dict, outfile)
        
        model.to("cpu")
        torch.save(model.state_dict(), f"{args.model_path}/{save_dir}/finetuned_{TYPE}_vanilla_model.pth")

        losses_dict = {"train_loss": train_loss, "test_loss": test_loss}
        outfile = open(f"{args.model_path}/{save_dir}/finetune_{TYPE}_vanilla_losses.pkl", "wb")
        dill.dump(losses_dict, outfile)
        outfile.close()
    else:
        args_dict = vars(args)
        args_dict.pop("save_to_dir")
        with open(f"{args.model_path}/finetune_{TYPE}_vanilla_settings.json", "w") as outfile:
            json.dump(args_dict, outfile)
        
        model.to("cpu")
        torch.save(model.state_dict(), f"{args.model_path}/finetuned_{TYPE}_vanilla_model.pth")

        losses_dict = {"train_loss": train_loss, "test_loss": test_loss}
        outfile = open(f"{args.model_path}/finetune_{TYPE}_vanilla_losses.pkl", "wb")
        dill.dump(losses_dict, outfile)
        outfile.close()



