from modelling.models import BaselineGATLSTM, Edgeconvmodel, GATLSTM, Encoder, Decoder, STGNNModel, BaselineGNNLSTM
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import dill
from data_processing.process_dataset import Dataset
from torch_geometric.loader import DataLoader
import argparse
import datetime
import logging
import os
import json
from distutils.dir_util import copy_tree
import learn2learn as l2l
import random
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_to_dataloader(data, k_shots=6):
    data_list = []
    for i in range(len(data)):
        data_list.append(data[i])
    loader = DataLoader(data_list, batch_size=k_shots, shuffle=True)
    return loader


def train_model(
    train_datasets: dict,
    test_datasets: dict,
    epochs: int = 200,
    adapt_lr: float = 0.001,
    batch_task_size: int = -1,
    meta_lr: float = 0.001,
    adaptation_steps: int = 5,
    weather_features: int = 4,
    time_features: int = 43,
    log_dir: str = None,
    dropout: float = 0.2,
    hidden_size: int = 32,
    node_out_features: int = 10,
    gpu: bool = False
):

    model = Edgeconvmodel(
        node_in_features=1,
        weather_features=weather_features,
        time_features=time_features,
        node_out_features=node_out_features,
        gpu=gpu,
        hidden_size=hidden_size,
        dropout_p=dropout
    )

    model.to(DEVICE)

    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True)

    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = torch.nn.MSELoss(reduction='mean')

    if batch_task_size == -1 or batch_task_size > len(train_datasets.keys()):
        batch_task_size = len(train_datasets.keys())

    writer = SummaryWriter(log_dir=log_dir)
    step_dict = {f_name: 0 for f_name in train_datasets.keys()}
    for epoch in range(epochs):
        opt.zero_grad()
        meta_train_loss = 0.0

        # num_evals = 0
        for f_name, task in random.sample(train_datasets.items(), batch_task_size):
            learner = maml.clone()

            support_data = next(iter(task)).to(DEVICE)
            query_data = next(iter(task)).to(DEVICE)

            for _ in range(adaptation_steps):  # adaptation_steps

                support_preds = learner(support_data)
                support_loss = lossfn(support_data.y, support_preds.view(support_data.num_graphs, -1))
                learner.adapt(support_loss)

    

            query_preds = learner(query_data)
            query_loss = lossfn(query_data.y, query_preds.view(query_data.num_graphs, -1))
            writer.add_scalar(tag=f"{f_name}/query_loss", scalar_value=query_loss.item(), global_step=step_dict[f_name])
            step_dict[f_name] += 1

           
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / batch_task_size

        if epoch % 1 == 0:
            logger.info(f"Epoch: {epoch+1}")
            logger.info(f"Meta Train Loss: {meta_train_loss.item()}")
            logger.info(8 * "#")
        
        writer.add_scalar(tag=f"Meta/loss", scalar_value=meta_train_loss.item(), global_step=epoch)

        meta_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(maml.parameters(), 1)

        opt.step()
    
    return model




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Model training argument parser")

    parser.add_argument("-d", "--data_dir", type=str, help="Directory of datasets")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, help="Ratio of data to be used for training")
    parser.add_argument("-b", "--batch_task_size", type=int, default=-1, help="number of tasks to sample")
    parser.add_argument("-k", "--k_shot", type=int, default=5, help="shots to be used")
    parser.add_argument("-a", "--adaptation_steps", type=int, default=5, help="Number of adaptation steps")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-alr", "--adapt_lr", type=float, default=0.001, help="Adaptation learning rate")
    parser.add_argument("-mlr", "--meta_lr", type=float, default=0.001, help="Meta learning rate")
    parser.add_argument("-ld", "--log_dir", type=str, default=None, help="directory to log stuff")
    parser.add_argument("-ex", "--exclude", type=str, default="", help="comma seperated list of datasets to exclude")
    parser.add_argument("-hs", "--hidden_size", type=int, default=32)
    parser.add_argument("-dp", "--dropout_p", type=float, default=0.2)
    parser.add_argument("-no", "--node_out_features", type=int, default=10)
    parser.add_argument("-g", "--gpu", action='store_true')

    args = parser.parse_args()
    train_dataloader_dict = {}
    test_dataloader_dict = {}

    exclude_list = args.exclude.split(",")

    for f in os.listdir(args.data_dir):
        abs_path = os.path.join(args.data_dir, f)
        CONTINUE_FLAG=False
        for exclude_file in exclude_list:
            if f.startswith(exclude_file) and len(exclude_file) > 0:
                CONTINUE_FLAG = True

        if CONTINUE_FLAG:
            continue
        
        with open(abs_path, "rb") as infile:
            logger.info(abs_path)
            data = dill.load(infile)
            train_data, test_data = Dataset.train_test_split(data, num_history=12, shuffle=True, ratio=args.train_size)

            train_data_dataloader = convert_to_dataloader(train_data, k_shots=args.k_shot)
            test_data_dataloader = convert_to_dataloader(test_data, k_shots=args.k_shot)

            f_name = f.split("/")[-1].replace(".pkl", "")

            train_dataloader_dict[f_name] = train_data_dataloader
            test_dataloader_dict[f_name] = test_data_dataloader
    
    logger.info(str(train_dataloader_dict))
    
    WEATHER_FEATURES = train_data.weather_information.shape[-1]
    TIME_FEATURES = train_data.time_encoding.shape[-1]

    start_time = datetime.datetime.now()
    logger.info(f"Fitting model at time: {str(start_time)}")
    if args.log_dir is not None:
        log_dir = f"{args.log_dir}/{start_time.isoformat()}"
    else:
        log_dir = None

    model = train_model(
        train_datasets=train_dataloader_dict,
        test_datasets=test_dataloader_dict,
        adaptation_steps=args.adaptation_steps,
        batch_task_size=args.batch_task_size,
        epochs=args.epochs,
        adapt_lr=args.adapt_lr,
        meta_lr=args.meta_lr,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        hidden_size=args.hidden_size,
        dropout=args.dropout_p,
        node_out_features=args.node_out_features,
        log_dir=log_dir,
        gpu=args.gpu
    )

    model.to("cpu")
    torch.save(model.state_dict(), f"{log_dir}/model.pth")

    args_dict = vars(args)
    with open(f"{log_dir}/settings.json", "w") as outfile:
        json.dump(args_dict, outfile)
