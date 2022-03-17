import os
import pickle
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.regularizers import l1_l2
from keras.objectives import mean_squared_error

import tensorflow as tf

from .ae_tpgg_loss import TPGG
from .ae_tpgg_layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer



Act = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e6)

advanced_activations = ('PReLU', 'LeakyReLU')


re_par_l1 = 0.0000
re_par_l2 = 0.00001
class Autoencoder():
    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=(64, 32, 64),
                 l2_coef=re_par_l2,
                 l1_coef=re_par_l1,
                 l2_enc_coef=re_par_l2,
                 l1_enc_coef=re_par_l1,
                 ridge=0.0000,
                 hidden_dropout=0.,
                 input_dropout=0.,
                 batchnorm=True,
                 activation='relu',
                 init='glorot_uniform',
                 file_path=None,
                 debug=False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.loss = None
        self.file_path = file_path
        self.extra_models = {}
        self.model = None
        self.encoder = None
        self.decoder = None
        self.input_layer = None
        self.sf_layer = None
        self.debug = debug

        if self.output_size is None:
            self.output_size = input_size

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name='semi-continuous-data')
        self.sf_layer = Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer



        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'

            if self.l1_enc_coef != 0. and stage in ('center', 'encoder'):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0. and stage in ('center', 'encoder'):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            last_hidden = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                kernel_regularizer=l1_l2(l1, l2),
                                name=layer_name)(last_hidden)
            if self.batchnorm:
                last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)

            if self.activation in advanced_activations:
                last_hidden = keras.layers.__dict__[self.activation](name='%s_act'%layer_name)(last_hidden)
            else:
                last_hidden = Activation(self.activation, name='%s_act'%layer_name)(last_hidden)

            if hid_drop > 0.0:
                last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.decoder_output = last_hidden
        self.build_output()

    def build_output(self):

        self.loss = mean_squared_error
        mean = Dense(self.output_size, kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])

        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            with open(os.path.join(self.file_path, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)

    def load_weights(self, filename):
        self.model.load_weights(filename)
        self.encoder = self.get_encoder()
        self.decoder = None

    def get_decoder(self):
        i = 0
        for l in self.model.layers:
            if l.name == 'center_drop':
                break
            i += 1

        return Model(inputs=self.model.get_layer(index=i+1).input,
                     outputs=self.model.output)

    def get_encoder(self, activation=False):
        if activation:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center_act').output)
        else:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center').output)
        return ret

    def predict(self, adata, mode='inference', return_info=False, copy=False):

        assert mode in ('inference', 'latent', 'full'), 'Unknown mode'

        adata = adata.copy() if copy else adata

        if mode in ('inference', 'full'):
            print('AE-TPGG: Inferring alpha parameter...')

            adata.X = self.model.predict({'semi-continuous-data': adata.X,
                                          'size_factors': adata.obs.size_factors})

            adata.uns['ae_tpgg_loss'] = self.model.test_on_batch({'semi-continuous-data': adata.X,
                                                              'size_factors': adata.obs.size_factors},
                                                             adata.raw.X)
        if mode in ('latent', 'full'):
            print('AE-TPGG: Obtaining low dimensional representations...')

            adata.obsm['X_ae_tpgg'] = self.encoder.predict({'semi-continuous-data': adata.X,
                                                        'size_factors': adata.obs.size_factors})
        if mode == 'latent':
            adata.X = adata.raw.X.copy() #recover normalized expression values

        return adata if copy else None

    def write(self, adata, file_path, mode='inference', colnames=None):

        print('AE_TPGG: Saving output(s)...')
        os.makedirs(file_path, exist_ok=True)




class TPGGAutoencoder(Autoencoder):

    def build_output(self):
        pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.decoder_output)
        beta = Dense(self.output_size, activation=Act,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef,
                               self.l2_coef),
                           name='beta')(self.decoder_output)

        gamma = Dense(self.output_size, activation=Act,
                      kernel_initializer=self.init,
                      kernel_regularizer=l1_l2(self.l1_coef,
                                               self.l2_coef),
                      name='gamma')(self.decoder_output)

        alpha = Dense(self.output_size, activation=Act, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='alpha')(self.decoder_output)
        output = ColwiseMultLayer([alpha, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, beta, gamma, pi])




        tpgg = TPGG(pi, beta_hat=beta, gamma_hat=gamma, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = tpgg.loss

        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['alpha'] = Model(inputs=self.input_layer, outputs=alpha)
        self.extra_models['beta'] = Model(inputs=self.input_layer, outputs=beta)
        self.extra_models['gamma'] = Model(inputs=self.input_layer, outputs=gamma)

        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='inference', return_info=False, copy=False, colnames=None):

        adata = adata.copy() if copy else adata

        if return_info:
            print('Saving parameter inference results.................')
            adata.obsm['X_tpgg_pi'] = self.extra_models['pi'].predict(adata.X)
            adata.obsm['X_tpgg_beta'] = self.extra_models['beta'].predict(adata.X)
            adata.obsm['X_tpgg_gamma'] = self.extra_models['gamma'].predict(adata.X)


        super().predict(adata, mode, return_info, copy=False)
        return adata if copy else None

    def write(self, adata, file_path, mode='inference', colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames

        super().write(adata, file_path, mode, colnames=colnames)

AE_types = {'normal': Autoencoder,  'tpgg-conddisp': TPGGAutoencoder}

