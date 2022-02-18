from logging import exception
import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from src.data.process_dataset import Dataset

import os
from src.models.finetune_meta import finetune_model
from src.models.models import Edgeconvmodel
from torch_geometric.data import DataLoader
import dill
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from src.visualization.visualize import plot_losses
import logging
plt.rcParams["figure.figsize"] = (20,5)

DATASET_FOLDER = "/zhome/2b/7/117471/Thesis/data/processed/loss_study"
EDGECONV_FOLDER = "/zhome/2b/7/117471/Thesis/models/loss_study"
WEATHER_FEATURES = 4
TIME_FEATURES = 43

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_edgeconv_model(path, trained=True):
    with open(f"{path}/settings.json", "rb") as f:
        edgeconv_params = json.load(f)

    if not trained:
        edgeconv_params_optim = {
            "optimizer": edgeconv_params["optimizer"],
            "weight_decay": edgeconv_params["weight_decay"],
            "lr": edgeconv_params["learning_rate"]
        }

    edgeconv_params.pop("data")
    edgeconv_params.pop("model")
    edgeconv_params.pop("train_size")
    edgeconv_params.pop("batch_size")
    edgeconv_params.pop("epochs")
    edgeconv_params.pop("num_history")
    edgeconv_params.pop("weight_decay")
    edgeconv_params.pop("learning_rate")
    edgeconv_params.pop("lr_factor")
    edgeconv_params.pop("lr_patience")
    edgeconv_params.pop("gpu")
    edgeconv_params.pop("optimizer")
    edgeconv_params.pop("k_neighbours")
    edgeconv_params.pop("graph_hidden_size")
    if "save_dir" in edgeconv_params:
        edgeconv_params.pop("save_dir")
    edgeconv_params["node_out_features"] = edgeconv_params.pop("node_out_feature")
    edgeconv_params["dropout_p"] = edgeconv_params.pop("dropout")

    edgeconv = Edgeconvmodel(
        node_in_features=1,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=True,
        **edgeconv_params
    )

    if trained:
        edgeconv_state_dict = torch.load(f"{path}/model.pth", map_location=torch.device('cpu'))
        edgeconv.load_state_dict(edgeconv_state_dict)
        return edgeconv
    
    else:
        optimizer = getattr(torch.optim, edgeconv_params_optim.pop("optimizer"))(
            edgeconv.parameters(), 
            **edgeconv_params_optim
        )
        return edgeconv, optimizer



def eval_model(edgeconv_model,datapath):
    loss_dict = dict()
    for dataset_file in os.listdir(DATASET_FOLDER):
        dataset_abs_path = os.path.abspath(os.path.join(DATASET_FOLDER, dataset_file))

        logger.info(f"Evaluating on: {dataset_abs_path}")
        open_file = open(dataset_abs_path, "rb")
        dataset = dill.load(open_file)
        open_file.close()
        train_dataset, test_dataset = Dataset.train_test_split(dataset, ratio=0.9, num_history=12, shuffle=True)
        torch.manual_seed(0)
        test_data_list = []
        for i in range(len(test_dataset)):
            test_data_list.append(test_dataset[i])
        

        test_loader_eval = DataLoader(test_data_list, batch_size=20, shuffle=True)
        lossfn = nn.MSELoss(reduction='mean')

        edgeconv_model.eval()
        model_loss = 0
        num_batch_test = 0
        with torch.no_grad():
            for batch in test_loader_eval:
                query_preds_trained = edgeconv_model(batch.to(DEVICE))
                model_loss += lossfn(batch.y, query_preds_trained.view(batch.num_graphs, -1)).item()
                num_batch_test += 1
        
        model_loss = model_loss / num_batch_test
        logger.info(str(model_loss))
        loss_dict[dataset_file.split(".")[0]] = model_loss
    
    logger.info(str(loss_dict))
    return loss_dict


for dataset in os.listdir(DATASET_FOLDER):
    dataset_abs_path = os.path.abspath(os.path.join(DATASET_FOLDER, dataset))

    for edgeconv_path in os.listdir(EDGECONV_FOLDER):
        if not edgeconv_path.startswith("edgeconv"):
            continue
        edgeconv_abs_path = os.path.abspath(os.path.join(EDGECONV_FOLDER, edgeconv_path))
        with open(f"{edgeconv_abs_path}/settings.json") as settings_json:
            edgeconv_settings = json.load(settings_json)
        
        if edgeconv_settings["data"].split("/")[-1] == dataset:
            edgeconv_model = load_edgeconv_model(edgeconv_abs_path, trained=True)
            break

    logger.info(f"Reference model: {dataset_abs_path}")
    loss_dict = eval_model(
        edgeconv_model=edgeconv_model.to(DEVICE),
        datapath=dataset_abs_path
    )
    dataset_save_name = dataset.split(".")[0]
    with open(f"{edgeconv_abs_path}/{dataset_save_name}-compare_losses", "wb") as outfile:
        dill.dump(loss_dict, outfile)

