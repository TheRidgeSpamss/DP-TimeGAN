#running only from test file so import should change
import torch
import metrics.oneclass_metrics.ts_embeddings as s2s
import metrics.oneclass_metrics.compute_metrics as oc
import gc

def evaluate_oneclass_metrics(original_data, synthetic_data, parameters):
    '''Evaluate the OneClass metrics
    -------------Inputs-------------
    Original data (tensor): shape = (batch_size, seq_len, num_features)'
    Synthetic data (tensor): shape = (batch_size, seq_len, num_features)'
    parameters (dict): original parameters from DP-TimeGAN training
    ------------Returns-------------
    dict: statistical metrics based on the latent representation
    '''

    seq2seq, real_seq2seq_embeddings = s2s.seq2seq_orig_data_preproc_and_train(original_data, parameters)
    synthetic_seq2_seq_embeddings = s2s.seq2seq_synth_inference(seq2seq, synthetic_data, parameters)
    _, _, _, oneclass_metrics = oc.run_compute_alpha_precision(real_seq2seq_embeddings, synthetic_seq2_seq_embeddings, parameters)

    print(oneclass_metrics)
    del seq2seq, real_seq2seq_embeddings, synthetic_seq2_seq_embeddings
    gc.collect()
    return oneclass_metrics

