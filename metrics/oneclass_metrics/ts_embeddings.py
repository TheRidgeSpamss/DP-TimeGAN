'''
Adapted from the original codebase:
A. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar, "How Faithful is your Synthetic Data? //
Sample-level Metrics for Evaluating and Auditing Generative Models," 
International Conference on Machine Learning (ICML), 2022.
'''

"""Time series embedding.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""
import os
import sys

import numpy as np

import torch
import torch.optim as optim


from metrics.oneclass_metrics.networks.seq2seq_autoencoder import Encoder, Decoder, Seq2Seq, train_seq2seq_autoencoder, iterate_eval_set
from metrics.oneclass_metrics.networks.seq2seq_autoencoder import utils as s2s_utils
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(".."))
from torch_utils import extract_time


# ----------------------------------------------------------------------------------------------------------------------
# Set experiment settings here:

# Options for `run_experiment`: 
#   - Learn embeddings:
#     "learn:dummy" 
#     "learn:amsterdam:combined_downsampled_subset" 
#     "learn:amsterdam:test_subset"
#     "learn:amsterdam:hns_subset"
#     "learn:googlestock"
#   - Apply existing embeddings:
#     "apply:amsterdam:hns_competition_data"
run_experiment = "apply:ckd"

models_dir = "./models/"
embeddings_dir = "./embeddings/"
experiment_settings = dict()

experiment_settings["learn:sines"] = {
    "train_frac": 0.7,
    "split_order": ["train", "val", "test"],
    "n_features": 5,
    # --------------------
    "include_time": False,
    "max_timesteps": 30,  # Calculated automatically.
    "pad_val": +7., #might need to tinker with this too - changes within main function now
    "eos_val": +777.,  #might need to tinker with this
    "seed": 42,
    # --------------------
    "n_epochs": 5000,
    "batch_size": 128, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    # "data_path": "/home/ankit/synthetic_data_generation_research/data/ckd_sequences.npy",
    # "model_name": "s2s_ae_ckd.pt",
    # "embeddings_name": "ckd_embeddings.npy"
}

experiment_settings["learn:ckd"] = {
    "train_frac": 0.7,
    "split_order": ["train", "val", "test"],
    "n_features": 7,
    # --------------------
    "include_time": False,
    "max_timesteps": 7,  # Calculated automatically.
    "pad_val": +7., #might need to tinker with this too - changes within main function now
    "eos_val": +777.,  #might need to tinker with this
    "seed": 42,
    # --------------------
    "n_epochs": 5000,
    "batch_size": 128, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    # "data_path": "/home/ankit/synthetic_data_generation_research/data/ckd_sequences.npy",
    # "model_name": "s2s_ae_ckd.pt",
    # "embeddings_name": "ckd_embeddings.npy"
}

experiment_settings["learn:eicu"] = {
    "train_frac": 0.7,
    "split_order": ["train", "val", "test"],
    "n_features": 5,
    # --------------------
    "include_time": False,
    "max_timesteps": 30,  # Calculated automatically.
    "pad_val": +30., #might need to tinker with this too - changes within main function now
    "eos_val": +777.,  #might need to tinker with this
    "seed": 42,
    # --------------------
    "n_epochs": 5000,
    "batch_size": 128, 
    "hidden_size": 70,
    "num_rnn_layers": 2,
    "lr": 0.01,
    # --------------------
    # "data_path": "/home/ankit/synthetic_data_generation_research/data/eicuVitalPeriodic_resample15min_seqlen30.npy",
    # "model_name": "s2s_ae_eicu.pt",
    # "embeddings_name": "eicu_embeddings.npy"
}

experiment_settings["apply:sines"] = {
    # "gen_data_path": "/home/ankit/synthetic_data_generation_research/data/ckd_e10000_l3_noise0.2.npy",
    "pad_val": experiment_settings["learn:sines"]["pad_val"], #overwritten
    "eos_val": experiment_settings["learn:sines"]["eos_val"],
    "max_timesteps": experiment_settings["learn:sines"]["max_timesteps"], #overwritten
    # --------------------
    # "model_path": "./models/s2s_ae_ckd.pt",
    "batch_size": experiment_settings["learn:sines"]["batch_size"], 
    "n_features": experiment_settings["learn:sines"]["n_features"], #overwritten
    "hidden_size": experiment_settings["learn:sines"]["hidden_size"],
    "num_rnn_layers": experiment_settings["learn:sines"]["num_rnn_layers"],
    # --------------------
    # "embeddings_path": "./embeddings/synth_ckd_embeddings.npy"
    # --------------------
}

experiment_settings["apply:ckd"] = {
    # "gen_data_path": "/home/ankit/synthetic_data_generation_research/data/ckd_e10000_l3_noise0.2.npy",
    "pad_val": experiment_settings["learn:ckd"]["pad_val"], #overwritten
    "eos_val": experiment_settings["learn:ckd"]["eos_val"],
    "max_timesteps": experiment_settings["learn:ckd"]["max_timesteps"], #overwritten
    # --------------------
    # "model_path": "./models/s2s_ae_ckd.pt",
    "batch_size": experiment_settings["learn:ckd"]["batch_size"], 
    "n_features": experiment_settings["learn:ckd"]["n_features"], #overwritten
    "hidden_size": experiment_settings["learn:ckd"]["hidden_size"],
    "num_rnn_layers": experiment_settings["learn:ckd"]["num_rnn_layers"],
    # --------------------
    # "embeddings_path": "./embeddings/synth_ckd_embeddings.npy"
    # --------------------
}

experiment_settings["apply:eicu"] = {
    # "gen_data_path": "/home/ankit/synthetic_data_generation_research/data/ckd_e10000_l3_noise0.2.npy",
    "pad_val": experiment_settings["learn:eicu"]["pad_val"], #overwritten
    "eos_val": experiment_settings["learn:eicu"]["eos_val"],
    "max_timesteps": experiment_settings["learn:eicu"]["max_timesteps"], #overwritten
    # --------------------
    # "model_path": "./models/s2s_ae_ckd.pt",
    "batch_size": experiment_settings["learn:eicu"]["batch_size"], 
    "n_features": experiment_settings["learn:eicu"]["n_features"], #overwritten
    "hidden_size": experiment_settings["learn:eicu"]["hidden_size"],
    "num_rnn_layers": experiment_settings["learn:eicu"]["num_rnn_layers"],
    # --------------------
    # "embeddings_path": "./embeddings/synth_ckd_embeddings.npy"
    # --------------------
}

# ----------------------------------------------------------------------------------------------------------------------
# Utilities.

# General utilities.
def make_all_dataloaders(data_dict, batch_size):
    dataloaders_dict = dict()
    for dataset_name, data_tensors in data_dict.items():
        dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=data_tensors, batch_size=batch_size, shuffle=False)
        dataloaders_dict[dataset_name] = dataloader
    return dataloaders_dict


def prep_datasets_for_s2s_ae_training(x_xlen_dict, device, pad_val, eos_val):
    data_dict = dict()
    for key in x_xlen_dict.keys():
        x, x_len = x_xlen_dict[key]
        x_rev, x_rev_shifted = s2s_utils.rearrange_data(x, x_len, pad_val, eos_val)
        data_dict[key] = s2s_utils.data_to_tensors(
            x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)
    return data_dict


# Dummy data utilities.

def generate_all_dummy_data(device):
    dummy_exp_stgs = experiment_settings["learn:dummy"]
    data_dict = dict()
    for key in ("train", "val", "test"):
        data_dict[key] = s2s_utils.generate_dummy_data(
            n_samples=dummy_exp_stgs[f"n_samples_{key}"], 
            min_timesteps=dummy_exp_stgs["min_timesteps"], 
            max_timesteps=dummy_exp_stgs["max_timesteps"], 
            n_features=dummy_exp_stgs["n_features"], 
            pad_val=dummy_exp_stgs["pad_val"], 
            eos_val=dummy_exp_stgs["eos_val"], 
            seed=dummy_exp_stgs["data_gen_seed"], 
            to_tensors=True,
            float_type=torch.float32, 
            device=device,
        )
    return data_dict


def get_hns_dataloader(x, x_len, device, exp_settings):
    x_rev, x_rev_shifted = s2s_utils.rearrange_data(
            x, x_len, exp_settings["pad_val"], exp_settings["eos_val"])

    X, X_len, X_rev, X_rev_shifted = s2s_utils.data_to_tensors(
        x, x_len, x_rev, x_rev_shifted, float_type=torch.float32, device=device)

    dataset, dataloader = s2s_utils.make_dataloader(
            data_tensors=(X, X_len, X_rev, X_rev_shifted), 
            batch_size=exp_settings["batch_size"], 
            shuffle=False)

    return dataset, dataloader


def seq2seq_orig_data_preproc_and_train(orig_data, parameters):
    selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_type = f'learn:{parameters["data_name"]}'
    exp_settings = experiment_settings[exp_type]
    exp_settings["seed"] = parameters["seed"]

    print("Start Dataloader creation")
    data_full = orig_data
    data_train, data_val = train_test_split(data_full, train_size=exp_settings["train_frac"], random_state=exp_settings["seed"])
    data_val, data_test = train_test_split(data_val, test_size=0.5, random_state=exp_settings["seed"])

    _, exp_settings["max_timesteps"], exp_settings["n_features"] = data_full.shape
    exp_settings["pad_val"] = exp_settings["max_timesteps"]
    data_list = [data_train, data_val, data_test, data_full]
    time_list = []
    for data in data_list:
        time, _ = extract_time(data)
        time_list.append(time)
    
    data_time_dict = {
        "train": (data_list[0], time_list[0]),
        "val": (data_list[1], time_list[1]),
        "test": (data_list[2], time_list[2]),
        "full": (data_list[3], time_list[3])
    }

    data_dict = prep_datasets_for_s2s_ae_training(
        x_xlen_dict=data_time_dict,
        device=selected_device,
        pad_val=exp_settings["pad_val"],
        eos_val=exp_settings["eos_val"]
    )
    dataloaders_dict = make_all_dataloaders(
        data_dict=data_dict,
        batch_size=exp_settings["batch_size"]
    )
    print("End Dataloader creation")
    encoder = Encoder(
        input_size=exp_settings["n_features"], 
        hidden_size=exp_settings["hidden_size"], 
        num_rnn_layers=exp_settings["num_rnn_layers"]
    )
    decoder = Decoder(
        input_size=exp_settings["n_features"], 
        hidden_size=exp_settings["hidden_size"], 
        num_rnn_layers=exp_settings["num_rnn_layers"]
    )
    s2s = Seq2Seq(encoder=encoder, decoder=decoder)
    s2s.to(selected_device)

    opt = optim.Adam(s2s.parameters(), lr=exp_settings["lr"])
    
    print("Start Seq2Seq model training")

    train_seq2seq_autoencoder(
        seq2seq=s2s, 
        optimizer=opt,
        train_dataloader=dataloaders_dict["train"],
        val_dataloader=dataloaders_dict["val"], 
        n_epochs=exp_settings["n_epochs"], 
        batch_size=exp_settings["batch_size"],
        padding_value=exp_settings["pad_val"],
        max_seq_len=exp_settings["max_timesteps"],
    )
    eval_loss = iterate_eval_set(
        seq2seq=s2s, 
        dataloader=dataloaders_dict["test"],
        padding_value=exp_settings["pad_val"],
        max_seq_len=exp_settings["max_timesteps"]
    )
    print(f"Ev.Ls.={eval_loss:.3f}")
    print("End Seq2Seq model training")

    embedding_dataloaders = (dataloaders_dict["train"], dataloaders_dict["val"], dataloaders_dict["test"])
    max_seq_len = exp_settings["max_timesteps"]
    embeddings = s2s_utils.get_embeddings(
        seq2seq=s2s, 
        dataloaders=embedding_dataloaders,
        padding_value=exp_settings["pad_val"],
        max_seq_len=max_seq_len
    )
    
    return s2s, embeddings

def seq2seq_synth_inference(s2s, synth_data, parameters):
    print("Start Seq2Seq inference")
    selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_type = f'apply:{parameters["data_name"]}'
    exp_settings = experiment_settings[exp_type]
    exp_settings["seed"] = parameters["seed"]

    print("Start Dataloader creation")
    _, exp_settings["max_timesteps"], exp_settings["n_features"] = synth_data.shape
    exp_settings["pad_val"] = exp_settings["max_timesteps"]
    seq_lens, _ = extract_time(synth_data)
    
    s2s.to(selected_device)
    dataset, dataloader = get_hns_dataloader(
                    x=synth_data, 
                    x_len=seq_lens, 
                    device=selected_device, 
                    exp_settings=exp_settings
                    )
    
    autoencoder_loss = iterate_eval_set(
                seq2seq=s2s, 
                dataloader=dataloader,
                padding_value=exp_settings["pad_val"],
                max_seq_len=exp_settings["max_timesteps"]
            )
    
    embeddings = s2s_utils.get_embeddings(
                seq2seq=s2s, 
                dataloaders=(dataloader,),
                padding_value=exp_settings["pad_val"],
                max_seq_len=exp_settings["max_timesteps"]
            )
    
    n_nan = np.isnan(embeddings).astype(int).sum()
    assert n_nan == 0
    print(f"AE Synth Loss = {autoencoder_loss:.3f}")
    print("End Seq2Seq inference")
    return embeddings