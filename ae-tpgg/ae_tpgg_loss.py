import numpy as np
import tensorflow as tf

def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = tf.square(y_pred - y_true)

    return _reduce_mean(ret)


class GG(object):
    def __init__(self, beta_hat=None, gamma_hat=None, masking=False, scope='gg_loss/',
                 scale_factor=1.0, debug=True):

        # for numerical stability
        self.eps = 1e-6
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.beta_hat = beta_hat
        self.gamma_hat = gamma_hat

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            beta_hat = tf.minimum(self.beta_hat, 1e6)
            gamma_hat = tf.minimum(self.gamma_hat, 1e6)


            part1 = tf.log(y_pred+eps) + tf.lgamma(beta_hat+eps) - tf.log(gamma_hat+eps) - (beta_hat*gamma_hat-1)*(tf.log(y_true+eps)-tf.log(y_pred+eps))
            part2 = (y_true/y_pred + eps)**gamma_hat



            if self.debug:
                assert_ops = [
                        tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                        tf.verify_tensor_all_finite(part1, 'part1 has inf/nans'),
                        tf.verify_tensor_all_finite(part2, 'part2 has inf/nans')]

                tf.summary.histogram('part1', part1)
                tf.summary.histogram('part2', part2)

                with tf.control_dependencies(assert_ops):
                    final = part1 + part2

            else:
                final = part1 + part2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)


        return final

class TPGG(GG):
    def __init__(self, pi_hat, ridge_lambda=0.0, scope='tpgg_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi_hat = pi_hat
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            gg_case = super().loss(y_true, y_pred, mean=False) - tf.log(1.0-self.pi_hat+eps)

            y_true = tf.cast(y_true, tf.float32)

            zero_case = -tf.log(self.pi_hat + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, gg_case)
            ridge = self.ridge_lambda*tf.square(self.pi_hat)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

            if self.debug:
                tf.summary.histogram('gg_case', gg_case)
                tf.summary.histogram('zero_case', zero_case)
                tf.summary.histogram('ridge', ridge)

        return result

