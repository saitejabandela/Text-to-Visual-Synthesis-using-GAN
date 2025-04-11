from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

assert tf.__version__.startswith('2')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense


class EmbeddingCompressor:

    def build_embedding_compressor(self):

        input1 = Input(shape=(1024,))
        compress_input = Dense(128)(input1)
        compress_input = ReLU()(compress_input)

        model = Model(inputs=[input1], outputs=[compress_input])
        return model
