'''
Extracted directly from the original codebase:
A. Alaa, B. van Breugel, E. Saveliev, M. van der Schaar, "How Faithful is your Synthetic Data? //
Sample-level Metrics for Evaluating and Auditing Generative Models," 
International Conference on Machine Learning (ICML), 2022.
'''

"""Timeseries encoding to a fixed size vector representation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""

from .seq2seq_autoencoder import Encoder, Decoder, Seq2Seq, init_hidden, compute_loss
from .training import train_seq2seq_autoencoder, iterate_eval_set