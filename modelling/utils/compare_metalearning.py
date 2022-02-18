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

DATASET_FOLDER = "/zhome/2b/7/117471/Thesis/data/processed/metalearning"
METAMODEL_FOLDER = "/zhome/2b/7/117471/Thesis/metalearning/NOT-BIKES/not-augmented/finetuned_models"
EDGECONV_FOLDER = "/zhome/2b/7/117471/Thesis/models/metalearning_non-augmented"
WEATHER_FEATURES = 4
TIME_FEATURES = 43

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finetuned_model(path, data_type):
    with open(f"{path}/settings.json", "rb") as f:
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

    edgeconv = Edgeconvmodel(
        node_in_features=1,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=True,
        **edgeconv_params
    )

    edgeconv_state_dict = torch.load(f"{path}/finetuned_{data_type}_model.pth", map_location=torch.device('cpu'))
    edgeconv.load_state_dict(edgeconv_state_dict)
    return edgeconv


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


def load_meta_model(path):
    with open(f"{path}/settings.json", "rb") as f:
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

    edgeconv_meta = Edgeconvmodel(
        node_in_features=1,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=True,
        **edgeconv_params
    )
    edgeconv_meta_state_dict = torch.load(f"{path}/model.pth", map_location=torch.device('cpu'))
    edgeconv_meta.load_state_dict(edgeconv_meta_state_dict)
    return edgeconv_meta


def load_transfer_model(path, data_type=None):
    with open(f"{path}/settings.json", "rb") as f:
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

    edgeconv_transfer = Edgeconvmodel(
        node_in_features=1,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=True,
        **edgeconv_params
    )
    if data_type is None:
        edgeconv_transfer_state_dict = torch.load(f"{path}/vanilla_model.pth", map_location=torch.device('cpu'))
    else:
        edgeconv_transfer_state_dict = torch.load(f"{path}/finetuned_{data_type}_vanilla_model.pth", map_location=torch.device('cpu'))
    edgeconv_transfer.load_state_dict(edgeconv_transfer_state_dict)
    optimizer = torch.optim.SGD(
        edgeconv_transfer.parameters(), 
        lr=0.05
    )

    return edgeconv_transfer, optimizer


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

ADAPTATION_STEPS=5
ADAPT_LR=0.05
META_LR=0.001


def eval_models(
    metamodel, 
    untrained_edgeconv, 
    optimizer_untrained, 
    trained_edgeconv,
    finetuned_edgeconv,
    transferlearn_edgeconv,
    finetuned_transfer,
    optimizer_transfer,
    datapath, 
    k_shots=5, 
    iterations=30, 
    adaptation_steps=10, 
    adapt_lr=0.05,
    verbose=False
):
    open_file = open(datapath, "rb")
    dataset = dill.load(open_file)
    train_dataset, test_dataset = Dataset.train_test_split(dataset, ratio=0.9, num_history=12, shuffle=True)
    torch.manual_seed(0)
    test_data_list = []
    for i in range(len(test_dataset)):
        test_data_list.append(test_dataset[i])
    
    train_data_list = []
    for i in range(len(train_dataset)):
        train_data_list.append(train_dataset[i])

    if k_shots > 0:
        train_loader_meta = DataLoader(train_data_list, batch_size=k_shots, shuffle=True)
    
    maml = l2l.algorithms.MAML(metamodel, lr=adapt_lr, first_order=False, allow_unused=True).to(DEVICE)
    lossfn = nn.MSELoss(reduction='mean')

    test_loader_eval = DataLoader(test_data_list, batch_size=20, shuffle=True)
    # query_data = next(iter(test_loader_eval)).to(DEVICE)


    query_loss_trained = 0
    query_loss_finetuned = 0
    query_loss_finetuned_transfer = 0
    query_loss_baseline = 0
    trained_edgeconv.eval()
    finetuned_edgeconv.eval()
    finetuned_transfer.eval()
    num_batch_test = 0
    with torch.no_grad():
        for batch in test_loader_eval:
            query_preds_trained = trained_edgeconv(batch.to(DEVICE))
            query_preds_finetuned = finetuned_edgeconv(batch.to(DEVICE))
            query_preds_finetuned_transfer = finetuned_transfer(batch.to(DEVICE))
            query_preds_baseline = batch.x[:, -1, :]

            query_loss_trained += lossfn(batch.y, query_preds_trained.view(batch.num_graphs, -1)).item()
            query_loss_finetuned += lossfn(batch.y, query_preds_finetuned.view(batch.num_graphs, -1)).item()
            query_loss_finetuned_transfer += lossfn(batch.y, query_preds_finetuned_transfer.view(batch.num_graphs, -1)).item()
            query_loss_baseline += lossfn(batch.y, query_preds_baseline.view(batch.num_graphs, -1)).item()

            num_batch_test += 1
    
    query_loss_trained = query_loss_trained / num_batch_test
    query_loss_finetuned = query_loss_finetuned / num_batch_test
    query_loss_finetuned_transfer = query_loss_finetuned_transfer / num_batch_test
    query_loss_baseline = query_loss_baseline / num_batch_test

    meta_loss = []
    trained_loss = []
    finetuned_loss = []
    untrained_loss = []
    transfer_loss = []
    finetuned_transfer_loss = []
    baseline_loss = []

    for epoch in range(iterations):
        learner = maml.clone()

        if k_shots > 0:
            support_data = next(iter(train_loader_meta)).to(DEVICE)

        if k_shots > 0:
            learner.train()
            untrained_edgeconv.train()
            transferlearn_edgeconv.train()
            for _ in range(adaptation_steps): # adaptation_steps
                # Adapt Metalearning model
                support_preds = learner(support_data)
                support_loss = lossfn(support_data.y, support_preds.view(support_data.num_graphs, -1))
                learner.adapt(support_loss)
                
                # Optimize untrained edgeconv model
                optimizer_untrained.zero_grad(set_to_none=True)
                out = untrained_edgeconv(support_data)
                loss = lossfn(support_data.y, out.view(support_data.num_graphs, -1))
                loss.backward()
                optimizer_untrained.step()

                optimizer_transfer.zero_grad(set_to_none=True)
                out = transferlearn_edgeconv(support_data)
                loss = lossfn(support_data.y, out.view(support_data.num_graphs, -1))
                loss.backward()
                optimizer_transfer.step()
        
        # PREDICT WITH ALL MODELS
        learner.eval()
        untrained_edgeconv.eval()
        transferlearn_edgeconv.eval()
        query_loss_meta = 0
        query_loss_untrained = 0
        query_loss_transfer = 0
        num_batch_test = 0
        with torch.no_grad():
            for batch in test_loader_eval:
                query_preds_meta = learner(batch.to(DEVICE))
                query_preds_untrained = untrained_edgeconv(batch.to(DEVICE))
                query_preds_transfer = transferlearn_edgeconv(batch.to(DEVICE))

                query_loss_meta += lossfn(batch.y, query_preds_meta.view(batch.num_graphs, -1)).item()
                query_loss_untrained += lossfn(batch.y, query_preds_untrained.view(batch.num_graphs, -1)).item()
                query_loss_transfer += lossfn(batch.y, query_preds_transfer.view(batch.num_graphs, -1)).item()

                num_batch_test += 1
            
            query_loss_meta = query_loss_meta / num_batch_test
            meta_loss.append(query_loss_meta)

            query_loss_untrained = query_loss_untrained / num_batch_test
            untrained_loss.append(query_loss_untrained)

            query_loss_transfer = query_loss_transfer / num_batch_test
            transfer_loss.append(query_loss_transfer)

            """learner.eval()
            query_preds_meta = learner(query_data)
            query_loss_meta = lossfn(query_data.y, query_preds_meta.view(query_data.num_graphs, -1)).cpu()
            meta_loss.append(query_loss_meta)

            untrained_edgeconv.eval()
            query_preds_untrained = untrained_edgeconv(query_data)
            query_loss_untrained = lossfn(query_data.y, query_preds_untrained.view(query_data.num_graphs, -1)).cpu()
            untrained_loss.append(query_loss_untrained)"""
            
            #query_loss_trained = lossfn(query_data.y, query_preds_trained.view(query_data.num_graphs, -1)).cpu()
            trained_loss.append(query_loss_trained)

            #query_loss_finetuned = lossfn(query_data.y, query_preds_finetuned.view(query_data.num_graphs, -1)).cpu()
            finetuned_loss.append(query_loss_finetuned)

            finetuned_transfer_loss.append(query_loss_finetuned_transfer)

            #query_loss_baseline = lossfn(query_data.y, query_preds_baseline.view(query_data.num_graphs, -1)).cpu()
            baseline_loss.append(query_loss_baseline)
            

        # Reset parameters of untrained model
        untrained_edgeconv.apply(weight_reset)

        if verbose:
            print(f"Epoch: {epoch+1}")
            logger.info(f"Meta Train Loss: {query_loss_meta}")
            logger.info(f"Finetuned loss: {query_loss_finetuned}")
            print(f"Trained Edgeconv loss: {query_loss_trained}")
            print(f"Untrained Edgeconv loss: {query_loss_untrained}")
            print(f"Transfer model loss: {query_loss_transfer}")
            print(f"Finetuned transfer model loss: {query_loss_finetuned_transfer}")
            print(f"Baseline loss: {query_loss_baseline}")
            print(8 * "#")
        
    return meta_loss, finetuned_loss, trained_loss, untrained_loss, transfer_loss, finetuned_transfer_loss, baseline_loss


EVAL_DICT = {}
for dataset in os.listdir(DATASET_FOLDER):
    #if dataset == "citibike-tripdata-HOUR1-REGION.pkl":
    if "citibike" not in dataset:
        continue
    if "GRID" in dataset:
        TYPE = "GRID"
    else:
        TYPE = "REGION"
    EVAL_DICT[dataset] = {}
    dataset_abs_path = os.path.abspath(os.path.join(DATASET_FOLDER, dataset))

    for metamodel_path in os.listdir(METAMODEL_FOLDER):
        metamodel_abs_path = os.path.abspath(os.path.join(METAMODEL_FOLDER, metamodel_path))
        # with open(f"{metamodel_abs_path}/settings.json") as settings_json:
        #     metamodel_exclude = json.load(settings_json)["exclude"]

        
        # if dataset.startswith(metamodel_exclude):
        if dataset.startswith(metamodel_path):
            logger.info("hello")
            logger.info(dataset)
            logger.info(metamodel_path)
            metamodel = load_meta_model(metamodel_abs_path)
            transferlearn_model, _ = load_transfer_model(metamodel_abs_path)
            finetuned_transfer, optimizer_transfer = load_transfer_model(metamodel_abs_path, data_type=TYPE)
            finetuned_model = load_finetuned_model(metamodel_abs_path, data_type=TYPE)

            break
    if finetuned_model is None:
        continue

    for edgeconv_path in os.listdir(EDGECONV_FOLDER):
        edgeconv_abs_path = os.path.abspath(os.path.join(EDGECONV_FOLDER, edgeconv_path))
        with open(f"{edgeconv_abs_path}/settings.json") as settings_json:
            edgeconv_settings = json.load(settings_json)
        
        if edgeconv_settings["data"].split("/")[-1] == dataset:
            edgeconv_trained_model = load_edgeconv_model(edgeconv_abs_path, trained=True)
            edgeconv_untrained_model, optimizer_untrained = load_edgeconv_model(edgeconv_abs_path, trained=False)

    logger.info(dataset_abs_path)
    for k in [0, 1, 5]:
        EVAL_DICT[dataset][str(k)] = {}
        losses = eval_models(
            metamodel=metamodel.to(DEVICE),
            untrained_edgeconv=edgeconv_untrained_model.to(DEVICE),
            optimizer_untrained=optimizer_untrained,
            trained_edgeconv=edgeconv_trained_model.to(DEVICE),
            finetuned_edgeconv=finetuned_model.to(DEVICE),
            transferlearn_edgeconv=transferlearn_model.to(DEVICE),
            finetuned_transfer=finetuned_transfer.to(DEVICE),
            optimizer_transfer=optimizer_transfer,
            datapath=dataset_abs_path,
            k_shots=k,
            iterations=30,
            adaptation_steps=5,
            adapt_lr=0.05,
            verbose=True
        )
        meta_loss, finetuned_loss, trained_loss, untrained_loss, transfer_loss, finetuned_transfer_loss, baseline_loss = losses
        
        EVAL_DICT[dataset][str(k)]["meta_loss"] = np.array(meta_loss)
        EVAL_DICT[dataset][str(k)]["finetuned_loss"] = np.array(finetuned_loss)
        EVAL_DICT[dataset][str(k)]["trained_loss"] = np.array(trained_loss)
        EVAL_DICT[dataset][str(k)]["untrained_loss"] = np.array(untrained_loss)
        EVAL_DICT[dataset][str(k)]["transfer_loss"] = np.array(transfer_loss)
        EVAL_DICT[dataset][str(k)]["finetuned_transfer_loss"] = np.array(finetuned_transfer_loss)
        EVAL_DICT[dataset][str(k)]["baseline_loss"] = np.array(baseline_loss)

logger.info(f"Saving results to {os.getcwd()}/meta_compare.pkl")

with open("NON-AUGMENTED-meta_compare.pkl", "wb") as outfile:
    dill.dump(EVAL_DICT, outfile)


