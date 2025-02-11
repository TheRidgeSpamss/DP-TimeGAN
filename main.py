# 1. Training model
from train_and_test import train, dp_train, test
# 2. Data loading
from data.torch_dataloading import real_data_loading, sine_data_generation
# 3. Utilities
import torch_utils as tu
import os
import pandas as pd
import time
import yaml
import numpy as np
import torch
import random

def main(parameters, checkpoint_filename):
    start_time = time.time()

    file_dir = os.path.dirname(__file__)
    data_dir = os.path.join(file_dir, "..", "data")

    # Data loading - run torch_dataloading.py individually with the correct parameters to save the values as necessary
    ori_data, labels = None, None
    if parameters["data_name"] in ["ckd"]:
        data_file_name = "ckd_sequences.npy"
        label_file_name = "ckd_labels_hypertension.npy"

        data_file_path = os.path.join(data_dir, data_file_name)
        label_file_path = os.path.join(data_dir, label_file_name)

        ori_data = np.load(data_file_path)
        labels = np.load(label_file_path)
    elif parameters["data_name"] == "sines":
        ori_data = sine_data_generation(parameters["sine_no"], parameters["seq_len"], parameters["sine_dim"])
        labels = np.empty(3)

    print(parameters["data_name"] + ' dataset is ready.')

    # Training or Testing
    if parameters["test_only"]:
        test(ori_data, labels, parameters, checkpoint_filename)
    else:
        if parameters["use_dp"]:
            privacy_params = dp_train(ori_data, parameters, checkpoint_filename, delta=1e-5)
            test(ori_data, labels, parameters, checkpoint_filename, privacy_params)
        else:
            train(ori_data, parameters, checkpoint_filename)
            test(ori_data, labels, parameters, checkpoint_filename)


if __name__ == '__main__':

    with open("parameters.yaml", "r") as file:
        parameters = yaml.safe_load(file)
    print("Parameters loaded")

    # Call main function
    if parameters["checkpoint"] == "":
        no = f'_no{parameters["sine_no"]}' if parameters["data_name"] == 'sines' else ''
        dp = 'DP_' if parameters["use_dp"] else ''
        filename_additions = parameters["filename_additions"] + "_" if not parameters["filename_additions"] == "" else ""
        checkpoint_filename = f'{filename_additions}{dp}{parameters["data_name"]}_e{parameters["iterations"]}{no}_l{parameters["num_layers"]}_noise{parameters["noise_sd"]}'
    
    # Reproducibility code - seed setting and CUDA determinism   
    random_seeds = np.arange(1, 2) * 8

    for seed in random_seeds:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        parameters["seed"] = seed
        checkpoint_filename = f"seed{seed}_{checkpoint_filename}"
        print(f"Random seed: {seed}")

        main(parameters, checkpoint_filename)
    
    print("All random seeds done")

