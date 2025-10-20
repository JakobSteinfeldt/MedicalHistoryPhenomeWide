#!/usr/bin/env python
# coding: utf-8

import os
import math
import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import clear_output

import warnings
import lifelines
from lifelines.utils import CensoringType
from lifelines.utils import concordance_index

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from formulaic.errors import FactorEvaluationError
import zstandard
import pickle
import yaml
import ray
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="A script to calculate cindex for endpoints")
    parser.add_argument('--iteration', type=int, required=True)
    args=parser.parse_args()
    return args

def save_pickle(data, data_path):
    with open(data_path, "wb") as fh:
        cctx = zstandard.ZstdCompressor()
        with cctx.stream_writer(fh) as compressor:
            compressor.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            
def load_pickle(fp):
    with open(fp, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as decompressor:
            data = pickle.loads(decompressor.read())
    return data

def load_data():
    base_path = "/sc-projects/sc-proj-ukb-cvd"
    print(base_path)

    project_label = "22_medical_records"
    project_path = f"{base_path}/results/projects/{project_label}"
    figure_path = f"{project_path}/figures"
    output_path = f"{project_path}/data"

    experiment = 230425
    experiment_path = f"{output_path}/{experiment}"
    pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    
    in_path = f"{experiment_path}/coxph/predictions"
    out_path = f"{experiment_path}/benchmarks"
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    prediction_paths = pd.read_feather(f"{experiment_path}/prediction_paths.feather")

    endpoints_md = pd.read_csv(f"{experiment_path}/endpoints.csv")
    endpoints = sorted(endpoints_md.endpoint.to_list())
    scores = ['ASCVD', 'ASCVD+MedicalHistory', 'Age+Sex',
       'Age+Sex+Comorbidities', 'Age+Sex+Comorbidities+MedicalHistory',
       'Age+Sex+Comorbidities+MedicalHistoryLM', 'Age+Sex+MedicalHistory',
       'Age+Sex+MedicalHistoryLM', 'Comorbidities', 'MedicalHistory',
       'MedicalHistoryLM', 'QRISK3', 'QRISK3+MedicalHistory', 'SCORE2',
       'SCORE2+MedicalHistory']
    
    eligable_eids = pd.read_feather(f"{output_path}/eligable_eids_220627.feather")
    eids_dict = eligable_eids.set_index("endpoint")["eid_list"].to_dict()

    return output_path, experiment_path, in_path, out_path, endpoints, scores, prediction_paths, eids_dict

def read_partitions(in_path, prediction_paths, endpoint, score, time):
    paths = prediction_paths.query("endpoint==@endpoint").query("score==@score").path.to_list()
    data_preds = pd.concat([pd.read_feather(f"{in_path}/{path}", columns=["eid", f"Ft_{time}"]) 
                      for path in paths], axis=0).set_index("eid").sort_index()
    data_preds.columns = ["Ft"]
    return data_preds

def prepare_data(in_path, prediction_paths, endpoint, score, t_eval, output_path):
    temp_preds = read_partitions(in_path, prediction_paths, endpoint, score, t_eval)
    temp_tte = pd.read_feather(f"{output_path}/baseline_outcomes_220627.feather", 
        columns= ["eid", f"{endpoint}_event", f"{endpoint}_time"]).set_index("eid")
    temp_tte.columns = ["event", "time"]
    temp_data = temp_preds.merge(temp_tte, left_index=True, right_index=True, how="left")
    
    condition = (temp_data['event'] == 0) | (temp_data['time'] > t_eval)
    
    temp_data["event"] = (np.where(condition, 0, 1))
    
    temp_data["time"] = (np.where(condition, t_eval, temp_data['time']))

    return temp_data

from lifelines.utils import concordance_index

def calculate_cindex(in_path, prediction_paths, endpoint, score, time, iteration, eids_i, output_path):  
    temp_data = prepare_data(in_path, prediction_paths, endpoint, score, time, output_path)
    temp_data = temp_data[temp_data.index.isin(eids_i)]
    
    del eids_i
    
    try:
        cindex = 1-concordance_index(temp_data["time"], temp_data["Ft"], temp_data["event"])
    except ZeroDivisionError: 
        cindex=np.nan
    
    del temp_data
    
    return {"endpoint":endpoint, "score": score, "iteration": iteration, "time":time, "cindex":cindex}

@ray.remote
def calculate_iteration(in_path, prediction_paths, endpoint, scores, time, iteration, eids_i, output_path):  
    dicts = []
    for score in scores:
        dicts.append(calculate_cindex(in_path, prediction_paths, endpoint, score, time, iteration, eids_i, output_path))
    return dicts
 
def main(args):

    # prepare env variables and initiate ray
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

    ray.init(address="auto")

    # read iteration and set seed
    iteration=args.iteration
    np.random.seed(iteration)

    # prepare setup
    date = 230425
    t_eval = 10
    name = f"benchmark_cindex_{date}_{iteration}"

    # load data
    output_path, experiment_path, in_path, out_path, endpoints, scores, prediction_paths, eids_dict = load_data()

    rows_ray = []
    for endpoint in tqdm(endpoints):
        eids_e = eids_dict[endpoint]
        eids_i = np.random.choice(eids_e, size=len(eids_e))
        ds = calculate_iteration.remote(in_path, prediction_paths, endpoint, scores, t_eval, iteration, eids_i, output_path)
        rows_ray.append(ds)

        del eids_e

    rows = [ray.get(r) for r in tqdm(rows_ray)]
    rows_finished = [item for sublist in rows for item in sublist]
    benchmark_endpoints = pd.DataFrame({}).append(rows_finished, ignore_index=True)
    
    benchmark_endpoints.to_feather(f"{experiment_path}/benchmarks/{name}.feather")
    
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    main(args)