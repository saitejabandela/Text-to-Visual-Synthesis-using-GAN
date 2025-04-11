from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

assert tf.__version__.startswith('2')

from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D


class Upsampler:

    def get_up_sampling_block(self, upscale, num_kernels):

        upscale = UpSampling2D(size=(2, 2))(upscale)
        upscale = Conv2D(num_kernels, kernel_size=(3, 3), padding='same', strides=1, use_bias=False)(upscale)
        upscale = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(upscale)
        upscale = ReLU()(upscale)
        return upscale
