from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import Conv2D


class ConvolutionBlock:

    def build_conv_layer(self, x, num_kernels, kernel_size=(4, 4), strides=2, activation=True):
        gen_img = Conv2D(num_kernels, kernel_size=kernel_size, padding='same', strides=strides, use_bias=False)(x)
        gen_img = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(gen_img)
        if activation:
            gen_img = LeakyReLU(alpha=0.2)(x)
        return gen_img
