from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ConditioningArgumentNetwork import ConditioningArgumentNetwork
from Upsampler import Upsampler
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from Utility import Utility

assert tf.__version__.startswith('2')

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator:
    def __init__(self):
        self.conditionArgumentNetwork = ConditioningArgumentNetwork()
        self.upsampler = Upsampler()
    def stage1_generator(self):
        input1 = Input(shape=(1024,))
        cond = Dense(256)(input1)
        cond = LeakyReLU(alpha=0.2)(cond)

        # Obtain the conditioned text
        cond_variable = Lambda(self.conditionArgumentNetwork.evaluate_conditional_variable)(cond)

        input_layer2 = Input(shape=(100,))
        concat = Concatenate(axis=1)([cond_variable, input_layer2])

        inp = Dense(16384, use_bias=False)(concat)
        inp = ReLU()(inp)
        inp = Reshape((4, 4, 1024), input_shape=(16384,))(inp)

        inp = self.upsampler.get_up_sampling_block(inp, 512)
        inp = self.upsampler.get_up_sampling_block(inp, 256)
        inp = self.upsampler.get_up_sampling_block(inp, 128)
        inp = self.upsampler.get_up_sampling_block(inp, 64)  # upsampled our image to 64*64*3

        inp = Conv2D(3, kernel_size=3, padding='same', strides=1, use_bias=False)(inp)
        inp = Activation('tanh')(inp)

        stage1_gen = Model(inputs=[input1, input_layer2], outputs=[inp, cond])
        return stage1_gen

    def build_stage2_generator(self):

        input_layer1 = Input(shape=(1024,))
        input_images = Input(shape=(64, 64, 3))

        # Conditioning Augmentation
        ca = Dense(256)(input_layer1)
        mls = LeakyReLU(alpha=0.2)(ca)
        c = Lambda(self.conditionArgumentNetwork.evaluate_conditional_variable)(mls)

        # Downsampling block
        x = ZeroPadding2D(padding=(1, 1))(input_images)
        x = Conv2D(128, kernel_size=(3, 3), strides=1, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        x = ReLU()(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(256, kernel_size=(4, 4), strides=2, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
        x = ReLU()(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(512, kernel_size=(4, 4), strides=2, use_bias=False,
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
        x = ReLU()(x)

        # Concatenate text conditioning block with the encoded image
        concat = Utility.concat_along_dims([c, x])

        # Residual Blocks
        x = ZeroPadding2D(padding=(1, 1))(concat)
        x = Conv2D(512, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_uniform')(x)
        x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
        x = ReLU()(x)

        x = Utility.residual_block(x)
        x = Utility.residual_block(x)
        x = Utility.residual_block(x)

        # Upsampling Blocks
        x = self.upsampler.get_up_sampling_block(x, 512)
        x = self.upsampler.get_up_sampling_block(x, 256)
        x = self.upsampler.get_up_sampling_block(x, 128)
        x = self.upsampler.get_up_sampling_block(x, 64)

        x = Conv2D(3, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_uniform')(x)
        x = Activation('tanh')(x)

        stage2_gen = Model(inputs=[input_layer1, input_images], outputs=[x, mls])
        return stage2_gen

