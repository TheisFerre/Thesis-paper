# -*- coding: utf-8 -*-
from data_processing.process_dataset import (
    load_csv_dataset,
    create_grid,
    create_grid_ids,
    neighbourhood_adjacency_matrix,
    correlation_adjacency_matrix,
    features_targets_and_externals,
    distance_adjacency_matrix,
    Dataset,
)
from data_processing.encode_externals import Weather_container, time_encoder
import numpy as np
import json
import os
import dill
import logging
from pathlib import Path
import sys


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (data/raw) into
    cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading .csv files from directory {input_filepath}")
    logger.info("making final data set from raw data")

    for infile in os.listdir(input_filepath):
        if infile[-3:] == "csv":
            logger.info(f"Processing file {input_filepath}/{infile}")
            infile_root = infile[:-4]
            if os.path.isfile(f"{input_filepath}/{infile_root}.json"):
                file_dict = json.load(open(f"{input_filepath}/{infile_root}.json"))
            else:
                continue

            for HOUR_INTERVAL in file_dict["HOUR_INTERVAL"]:
                logger.info(f"Aggregating over {HOUR_INTERVAL} hours")

                # we dont neccesarily need stations if we makes grids.
                if "STATION_COL" in file_dict:
                    station_column = file_dict["STATION_COL"]
                else:
                    station_column = None

                df = load_csv_dataset(
                    path=f"{input_filepath}/{infile}",
                    time_column=file_dict["TIME_COL"],
                    location_columns=[file_dict["LNG_COL"], file_dict["LAT_COL"]],
                    station_column=station_column,
                    time_intervals=HOUR_INTERVAL + "h",
                )

                for GRID_SPLIT in file_dict["GRID_SPLITS"]:
                    # ALWAYS CREATE GRID
                    GRID_SPLIT = int(GRID_SPLIT)
                    logger.info(f"Splitting into {GRID_SPLIT} grids")
                    df = create_grid(df, lng_col=file_dict["LNG_COL"], lat_col=file_dict["LAT_COL"], splits=GRID_SPLIT)
                    df["grid_id"] = create_grid_ids(
                        df, longitude_col=file_dict["LNG_COL"] + "_binned", lattitude_col=file_dict["LAT_COL"] + "_binned"
                    )
                    # THIS ORDERING HAS TO BE THE EXACT SAME ALL THE TIME!!!
                    region_ordering = df["grid_id"].unique()
                    if file_dict["GRAPH"]:
                        adj_mat = neighbourhood_adjacency_matrix(region_ordering=region_ordering)
                    else:
                        adj_mat = neighbourhood_adjacency_matrix(region_ordering=region_ordering)
                        # If we do not care about the graph (only for the edgeconv)
                        # adj_mat = np.eye(len(region_ordering))

                    # encode time & weather
                    mean_lon = df[file_dict["LNG_COL"]].mean()
                    mean_lat = df[file_dict["LAT_COL"]].mean()
                    weather = Weather_container(
                        longitude=mean_lon, latitude=mean_lat, time_interval=HOUR_INTERVAL + "H"
                    )

                    time_enc = time_encoder()

                    (
                        X,
                        lat_vals,
                        lng_vals,
                        targets,
                        time_encoding,
                        weather_array,
                        feature_scaler,
                        target_scaler,
                    ) = features_targets_and_externals(
                        df=df,
                        region_ordering=region_ordering,
                        id_col="grid_id",
                        time_col=file_dict["TIME_COL"],
                        time_encoder=time_enc,
                        weather=weather,
                        time_interval=HOUR_INTERVAL + "H",
                        latitude=file_dict["LAT_COL"],
                        longitude=file_dict["LNG_COL"],
                    )

                    dat = Dataset(
                        adjacency_matrix=adj_mat,
                        targets=targets,
                        X=X,
                        weather_information=weather_array,
                        time_encoding=time_encoding,
                        feature_scaler=feature_scaler,
                        target_scaler=target_scaler,
                        latitude=lat_vals,
                        longitude=lng_vals,
                    )
                    clean_data_dict = {
                        "features": X,
                        "targets": targets,
                    }
                    logger.info(f"SAVING PROCESSED DATA TO {output_filepath}/hour{HOUR_INTERVAL}/{infile_root}-HOUR{HOUR_INTERVAL}-GRID{GRID_SPLIT}.pkl")
                    outfile = open(f"{output_filepath}/hour{HOUR_INTERVAL}/{infile_root}-HOUR{HOUR_INTERVAL}-GRID{GRID_SPLIT}.pkl", "wb")
                    dill.dump(dat, outfile)
                    outfile.close()

                # outfile = open(f"{output_filepath}/{infile_root}_nodes.pkl", "wb")
                # dill.dump(clean_data_dict, outfile)
                # outfile.close()

                # CREATE GRAPH FROM STATIONS
                if station_column is not None:
                    df["grid_id"] = df[station_column]
                
                    # THIS ORDERING HAS TO BE THE EXACT SAME ALL THE TIME!!!
                    region_ordering = df["grid_id"].unique()
                    if file_dict["GRAPH"]:
                        adj_mat = correlation_adjacency_matrix(
                            rides_df=df, region_ordering=region_ordering, id_col="grid_id", time_col=file_dict["TIME_COL"]
                        )
                    else:
                        # creates adjacency based on distance
                        logger.info("Distance adjacency")
                        adj_mat = distance_adjacency_matrix(
                            rides_df=df,
                            region_ordering=region_ordering,
                            id_col="grid_id",
                            time_col=file_dict["TIME_COL"],
                            neighbours=10,
                            lat_col=file_dict["LAT_COL"],
                            lng_col=file_dict["LNG_COL"],
                        )
                        # If we do not care about the graph (only for the edgeconv)
                        # adj_mat = np.eye(len(region_ordering))

                    # encode time & weather
                    mean_lon = df[file_dict["LNG_COL"]].mean()
                    mean_lat = df[file_dict["LAT_COL"]].mean()
                    weather = Weather_container(
                        longitude=mean_lon, latitude=mean_lat, time_interval=HOUR_INTERVAL + "H"
                    )

                    time_enc = time_encoder()

                    (
                        X,
                        lat_vals,
                        lng_vals,
                        targets,
                        time_encoding,
                        weather_array,
                        feature_scaler,
                        target_scaler,
                    ) = features_targets_and_externals(
                        df=df,
                        region_ordering=region_ordering,
                        id_col="grid_id",
                        time_col=file_dict["TIME_COL"],
                        time_encoder=time_enc,
                        weather=weather,
                        time_interval=HOUR_INTERVAL + "H",
                        latitude=file_dict["LAT_COL"],
                        longitude=file_dict["LNG_COL"],
                    )

                    dat = Dataset(
                        adjacency_matrix=adj_mat,
                        targets=targets,
                        X=X,
                        weather_information=weather_array,
                        time_encoding=time_encoding,
                        feature_scaler=feature_scaler,
                        target_scaler=target_scaler,
                        latitude=lat_vals,
                        longitude=lng_vals,
                    )
                    clean_data_dict = {
                        "features": X,
                        "targets": targets,
                    }
                    logger.info(f"SAVING PROCESSED DATA TO {output_filepath}/hour{HOUR_INTERVAL}/{infile_root}-HOUR{HOUR_INTERVAL}-REGION.pkl")
                    outfile = open(f"{output_filepath}/hour{HOUR_INTERVAL}/{infile_root}-HOUR{HOUR_INTERVAL}-REGION.pkl", "wb")
                    dill.dump(dat, outfile)
                    outfile.close()

                    #outfile = open(f"{output_filepath}/{infile_root}_nodes.pkl", "wb")
                    #dill.dump(clean_data_dict, outfile)
                    #outfile.close()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    main(input_filepath, output_filepath)
