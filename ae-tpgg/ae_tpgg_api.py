import os, tempfile, shutil, random
import anndata
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('ae_tpgg requires tensorflow.')


from .ae_tpgg_io import read_dataset, normalize
from .ae_tpgg_train import train
from .ae_tpgg_network import AE_types


def ae_tpgg(adata,
        mode='inference',
        ae_type='tpgg-conddisp',
        normalize_per_cell=False,
        scale=False,
        log1p=False,
        hidden_size=(64, 32, 64),
        hidden_dropout=0.,
        batchnorm=True,
        activation='relu',
        init='glorot_normal',
        network_kwds={},
        epochs=300,
        reduce_lr=20,
        early_stop=30,
        batch_size=32,
        optimizer='rmsprop',
        random_state=12,
        threads=None,
        verbose=True,
        training_kwds={},
        return_model=False,
        return_info=False,
        copy=False
        ):
    print('\n')
    print('\n')

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('inference', 'latent'), '%s is not a valid mode.' % mode

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.set_random_seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=copy)


    adata = normalize(adata,
                      filter_min_counts=False, # no filtering, keep cell and gene idxs same
                      size_factors=normalize_per_cell,
                      normalize_input=scale,
                      logtrans_input=log1p)

    network_kwds = {**network_kwds,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init
    }

    input_size = output_size = adata.n_vars
    net = AE_types[ae_type](input_size=input_size,
                            output_size=output_size,
                            **network_kwds)
    net.save()
    net.build()

    training_kwds = {**training_kwds,
        'epochs': epochs,
        'reduce_lr': reduce_lr,
        'early_stop': early_stop,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'verbose': verbose,
        'threads': threads
    }

    hist = train(adata[adata.obs.ae_tpgg_split == 'train'], net, **training_kwds)
    res = net.predict(adata, mode, return_info, copy)
    adata = res if copy else adata

    if return_info:
        adata.uns['ae_tpgg_loss_history'] = hist.history

    if return_model:
        return (adata, net) if copy else net
    else:
        return adata if copy else None
