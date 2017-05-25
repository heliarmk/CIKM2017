import tensorflow as tf
import datumio as dtd
import tqdm
import joblib
import fire
import numpy as np
import os
from conv_lstm_model import conv_lstm

def regression_loss(reg_preds, reg_labels):
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reg_labels, reg_preds)))
    tf.add_to_collection('losses', rmse)
    return rmse

def combind_loss(logits, labels, reg_preds, reg_labels):
    alpha = 1
    beta = 0.025
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cem = tf.reduce_mean(cross_entropy, name='cross_entropy')
    w_cem = cem * alpha
    tf.add_to_collection("losses", w_cem)
    reg_labels = tf.reshape(reg_labels, (-1, 1))
    # rmse = tf.sqrt(tf.losses.mean_squared_error(reg_labels, reg_preds, loss_collection=None))
    rmse = regression_loss(reg_preds, reg_labels)
    w_rmse = rmse * beta
    tf.add_to_collection("losses", w_rmse)

    return tf.add_n(tf.get_collection("losses"), name='combinded_loss'), cem, rmse

def train():

    #===============configuration parameters=============
    batch_size = 16
    display_step = 1
    snap_step = 1000
    start_lr = 0.0001
    lr_decay_step = 2000
    lr_decay_rate = 0.95
    n_class = 22
    dtype = tf.float32
    debug = False
    output = False
    finetuning = False

    dirpath = ""
    checkpoint_dir = os.path.join(dirpath, "ckeckpoints")
    log_dir = os.path.join(dirpath, "logs")

    is_training = tf.placeholder_with_default(True, None)
    keepprob = tf.placeholder(dtype=dtype, shape=None)
