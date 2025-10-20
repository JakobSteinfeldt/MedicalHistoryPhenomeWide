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

def get_score_defs():

    with open(r'/home/USER/code/MedicalHistoryPhenomeWide/2_downstream_processing/score_definitions.yaml') as file:
        score_defs = yaml.full_load(file)
    
    return score_defs

def get_features(endpoint, score_defs):
    features = {
        'Identity(Records)+MLP': {
            "MedicalHistory": [endpoint],
            "Age+Sex": score_defs["AgeSex"],
            "Comorbidities": score_defs["Comorbidities"],
            "SCORE2": score_defs["SCORE2"],
            "ASCVD": score_defs["ASCVD"],
            "QRISK3": score_defs["QRISK3"],
            "Age+Sex+Comorbidities": score_defs["AgeSex"] + score_defs["Comorbidities"],
            "Age+Sex+MedicalHistory": score_defs["AgeSex"] + [endpoint],
            "SCORE2+MedicalHistory": score_defs["SCORE2"] + [endpoint],
            "ASCVD+MedicalHistory": score_defs["ASCVD"] + [endpoint],
            "QRISK3+MedicalHistory": score_defs["QRISK3"] + [endpoint],
            "Age+Sex+Comorbidities+MedicalHistory": score_defs["AgeSex"] + score_defs["Comorbidities"] + [endpoint],
            },
        'Identity(Records)+Linear': {
            "MedicalHistoryLM": [endpoint],
            "Age+Sex+MedicalHistoryLM": score_defs["AgeSex"] + [endpoint],
            "Age+Sex+Comorbidities+MedicalHistoryLM": score_defs["AgeSex"] + score_defs["Comorbidities"] + [endpoint],
            }
    }
    return features

def get_train_data(in_path, partition, models, data_outcomes):
    train_data = {
        model: pd.read_feather(f"{in_path}/{model}/{partition}/train.feather")\
        .set_index("eid").merge(data_outcomes, left_index=True, right_index=True, how="left")
        for model in models}
        
    return train_data

def fit_cox(data_fit, feature_set, covariates, endpoint, penalizer, step_size=1):   
    # drop columns with only one value -> doesnt make sense as predictor
    data_fit = data_fit.select_dtypes(include=['int', 'float']).loc[:, data_fit.select_dtypes(include=['int', 'float']).nunique() > 1]
    
    # initialize and fit model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data_fit, f"{endpoint}_time", f"{endpoint}_event", fit_options={"step_size": step_size})
    return cph

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

def clean_covariates(endpoint, covariates):
    if endpoint=="phecode_181": # Autoimmune disease
        covariates = [c for c in covariates if c!="systemic_lupus_erythematosus"]
    if endpoint=="phecode_202": # Diabetes
        covariates = [c for c in covariates if c not in ['diabetes1', 'diabetes2', 'diabetes']]
    if endpoint=="phecode_202-1": # Diabetes 1
        covariates = [c for c in covariates if c!="diabetes1"]
    if endpoint=="phecode_202-2": # Diabetes 1
        covariates = [c for c in covariates if c!="diabetes2"]
    if endpoint=="phecode_286": # Mood [affective] disorders
        covariates = [c for c in covariates if c not in ['bipolar_disorder', 'major_depressive_disorder']]
    if endpoint=="phecode_286-1": # Bipolar disorder
        covariates = [c for c in covariates if c not in ['bipolar_disorder']]
    if endpoint=="phecode_286-2": # Major depressive disorder
        covariates = [c for c in covariates if c not in ['major_depressive_disorder']]
    if endpoint=="phecode_287": # psychotic disorders
        covariates = [c for c in covariates if c not in ['schizophrenia']]
    if endpoint=="phecode_287-1": # schizophrenia
        covariates = [c for c in covariates if c not in ['schizophrenia']]
    if endpoint=="phecode_331": # headache
        covariates = [c for c in covariates if c!="migraine"]
    if endpoint=="phecode_331-6": # headache
        covariates = [c for c in covariates if c!="migraine"]
    if endpoint=="phecode_416": # atrial fibrillation
        covariates = [c for c in covariates if c not in ['atrial_fibrillation']]
    if endpoint=="phecode_416-2": # atrial fibrillation and flutter
        covariates = [c for c in covariates if c not in ['atrial_fibrillation']]
    if endpoint=="phecode_416-21": # atrial fibrillation
        covariates = [c for c in covariates if c not in ['atrial_fibrillation']]
    if endpoint=="phecode_584": # Renal failure
        covariates = [c for c in covariates if c not in ['renal_failure']]
    if endpoint=="phecode_605": # Male sexual dysfuction
        covariates = [c for c in covariates if c not in ['sex_Male', 'male_erectile_dysfunction']]
    if endpoint=="phecode_605-1": # Male sexual dysfuction
        covariates = [c for c in covariates if c not in ['sex_Male', 'male_erectile_dysfunction']]
    if endpoint=="phecode_700": # Diffuse diseases of connective tissue
        covariates = [c for c in covariates if c not in ['systemic_lupus_erythematosus']]
    if endpoint=="phecode_700-1": # Lupus
        covariates = [c for c in covariates if c not in ['systemic_lupus_erythematosus']]
    if endpoint=="phecode_700-11": # Systemic lupus erythematosus [SLE]	
        covariates = [c for c in covariates if c not in ['systemic_lupus_erythematosus']]
    if endpoint=="phecode_705": # Rheumatoid arthritis and other inflammatory
        covariates = [c for c in covariates if c not in ['rheumatoid_arthritis']]
    if endpoint=="phecode_705-1": # Rheumatoid arthritis and other inflammatory
        covariates = [c for c in covariates if c not in ['rheumatoid_arthritis']]
    return covariates

def load_data(partition):
    base_path = "/sc-projects/sc-proj-ukb-cvd"
    print(base_path)

    project_label = "22_medical_records"
    project_path = f"{base_path}/results/projects/{project_label}"
    figure_path = f"{project_path}/figures"
    output_path = f"{project_path}/data"

    experiment = 230425
    experiment_path = f"{output_path}/{experiment}"
    pathlib.Path(experiment_path).mkdir(parents=True, exist_ok=True)
    
    in_path = pathlib.Path(f"{experiment_path}/coxph/input")
    in_path.mkdir(parents=True, exist_ok=True)

    model_path = f"{experiment_path}/coxph/models"
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    data_outcomes = pd.read_feather(f"{output_path}/baseline_outcomes_220627.feather").set_index("eid")
    
    endpoints_md = pd.read_csv(f"{experiment_path}/endpoints.csv")
    endpoints = sorted(endpoints_md.endpoint.to_list())
    
    endpoint_defs = pd.read_feather(f"{output_path}/phecode_defs_220306.feather").query("endpoint==@endpoints").sort_values("endpoint").set_index("endpoint")
    
    eligable_eids = pd.read_feather(f"{output_path}/eligable_eids_220627.feather")
    eids_dict = eligable_eids.set_index("endpoint")["eid_list"].to_dict()

    models = ['Identity(Records)+MLP', 'Identity(Records)+Linear']
    score_defs = get_score_defs()

    data_partition = get_train_data(in_path, partition, models, data_outcomes)

    return eids_dict, score_defs, endpoint_defs, endpoints, models, model_path, experiment_path, data_partition

@ray.remote(num_cpus=1)
def fit_endpoint(data_partition, eids_dict, score_defs, endpoint_defs, endpoint, partition, models, model_path, experiment_path):
    eids_incl = eids_dict[endpoint].tolist()
    features = get_features(endpoint, score_defs)
    eligibility = endpoint_defs.loc[endpoint]["sex"]
    for model in models:
        data_model = data_partition[model]
        for feature_set, covariates in features[model].items():
            cph_path = f"{model_path}/{endpoint}_{feature_set}_{partition}.p"
            if os.path.isfile(cph_path):
                try:
                    cph = load_pickle(cph_path)
                    success = True
                except:
                    success = False
                    pass
            if not os.path.isfile(cph_path) or success==False:
                if (eligibility != "Both") and ("sex_Male" in covariates): 
                    covariates = [c for c in covariates if c!="sex_Male"]
                
                # make sure cox models can fit ("LinAlgError: Matrix is singular")
                covariates = clean_covariates(endpoint, covariates)

                data_endpoint = data_model[covariates + [f"{endpoint}_event", f"{endpoint}_time"]].astype(np.float32)
                data_endpoint = data_endpoint[data_endpoint.index.isin(eids_incl)]
                cph = None
                for sz in [1, 0.5, 0.1, 0.01]:
                    if cph is not None:
                        break
                    try:
                        cph = fit_cox(data_endpoint, feature_set, covariates, endpoint, penalizer=0.0, step_size=sz)
                        save_pickle(cph, cph_path)
                        if sz<1: 
                            print("ConvergenceError", model, endpoint, feature_set, partition, f"trying with reduced step size ... {sz} successfull")
                    except (ValueError, ConvergenceError, KeyError, FactorEvaluationError) as e:
                        print("ConvergenceError", model, endpoint, feature_set, partition, f"trying with reduced step size ... {sz} failed")
                        if sz==0.01: 
                            save_pickle(data_endpoint, f"{experiment_path}/coxph/errordata_{endpoint}_{feature_set}_{partition}.p")
                        pass
                del cph
                        
def main(args):

    # prepare env variables for ray
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

    # prepare data
    partition = int(args[1])
    eids_dict, score_defs, endpoint_defs, endpoints, models, model_path, experiment_path, data_partition = load_data(partition)

    # setup ray and put files in plasma storage
    #ray.init(num_cpus=24) # crashes if num_cpus > 16, why not more possible?
    ray.init(address="auto")
    ray_eids = ray.put(eids_dict)
    ray_score_defs = ray.put(score_defs)
    ray_endpoint_defs = ray.put(endpoint_defs)
    ray_partition = ray.put(data_partition)
    
    # fit cox models via ray
    progress = []
    for endpoint in endpoints:
        progress.append(fit_endpoint.remote(ray_partition, ray_eids, ray_score_defs, ray_endpoint_defs, endpoint, partition, models, model_path, experiment_path))
    [ray.get(s) for s in tqdm(progress)]
    
    ray.shutdown()

if __name__ == "__main__":
    main(sys.argv)