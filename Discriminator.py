from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ConvolutionBlock import ConvolutionBlock

assert tf.__version__.startswith('2')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add


class Discriminator:
    def __init__(self):
        self.convolutionBlock = ConvolutionBlock()

    # the discriminator is fed with two inputs, the feature from Generator and the text embedding
    def stage1_discriminator(self):

        gen_image = Input(shape=(64, 64, 3))

        gen_image = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', use_bias=False)(gen_image)
        gen_image = LeakyReLU(alpha=0.2)(gen_image)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 128)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 256)
        gen_image= self.convolutionBlock.build_conv_layer(gen_image, 512)

        # text embedding
        input2 = Input(shape=(4, 4, 128))
        concat = concatenate([gen_image, input2])

        total_input = Conv2D(512, kernel_size=(1, 1), padding='same', strides=1, use_bias=False,)(concat)
        total_input = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(gen_image)
        total_input = LeakyReLU(alpha=0.2)(gen_image)
        total_input = Flatten()(total_input)
        total_input = Dense(1)(total_input)
        total_input = Activation('sigmoid')(total_input)

        stage1_dis = Model(inputs=[gen_image, input2], outputs=[total_input])
        return stage1_dis

    def stage2_discriminator(self):

        input1 = Input(shape=(256, 256, 3))

        gen_image = Conv2D(64, kernel_size=(4, 4), padding='same', strides=2, use_bias=False)(input1)
        gen_image = LeakyReLU(alpha=0.2)(gen_image)

        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 128)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 512)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 1024)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 2048)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 1024, (1, 1), 1)
        gen_image = self.convolutionBlock.build_conv_layer(gen_image, 512, (1, 1), 1, False)
        gen_image1 = self.convolutionBlock.build_conv_layer(gen_image, 128, (1, 1), 1)
        gen_image1 = self.convolutionBlock.build_conv_layer(gen_image1, 128, (3, 3), 1)
        gen_image1 = self.convolutionBlock.build_conv_layer(gen_image1, 512, (3, 3), 1, False)
        gen_image2 = add([gen_image, gen_image1])
        gen_image2 = LeakyReLU(alpha=0.2)(gen_image2)

        # Concatenate compressed and spatially replicated embedding
        input2 = Input(shape=(4, 4, 128))
        concat = concatenate([gen_image2, input2])

        x3 = Conv2D(512, kernel_size=(1, 1), strides=1, padding='same')(concat)
        x3 = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x3)
        x3 = LeakyReLU(alpha=0.2)(x3)

        # Flatten and add a FC layer
        x3 = Flatten()(x3)
        x3 = Dense(1)(x3)
        x3 = Activation('sigmoid')(x3)

        stage2_dis = Model(inputs=[input1, input2], outputs=[x3])
        return stage2_dis
