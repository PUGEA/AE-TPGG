from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K


def train(adata, network, output_dir=None, optimizer='rmsprop', learning_rate=0.001,
          epochs=300, reduce_lr=10, output_subset=None, use_raw_as_output=True,
          early_stop=15, batch_size=32, clip_grad=5., save_weights=False,
          validation_split=0.1, tensorboard=False, verbose=True, threads=None,
          **kwds):

    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=threads, inter_op_parallelism_threads=threads)))
    model = network.model
    loss = network.loss
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)

    model.compile(loss=loss, optimizer=optimizer)

    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose: model.summary()

    inputs = {'semi-continuous-data': adata.X, 'size_factors': adata.obs.size_factors}

    if output_subset:
        gene_idx = [np.where(adata.raw.var_names == x)[0][0] for x in output_subset]
        output = adata.raw.X[:, gene_idx] if use_raw_as_output else adata.X[:, gene_idx]
    else:
        output = adata.raw.X if use_raw_as_output else adata.X

    loss = model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose,
                     **kwds)

    return loss

