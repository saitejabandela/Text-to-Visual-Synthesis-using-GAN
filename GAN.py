from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.backend as K
assert tf.__version__.startswith('2')
from tensorflow.keras import Input, Model


class GAN:
    # Building GAN with Generator and Discriminator

    def stage1_adversarial(self, generator_model, discriminator_model):

        input1 = Input(shape=(1024,))
        input2 = Input(shape=(100,))
        input3 = Input(shape=(4, 4, 128))
        text, cond = generator_model([input1, input2])  # text,noise
        discriminator_model.trainable = False
        probabilities = discriminator_model([text, input3])
        adversarial_model = Model(inputs=[input1, input2, input3], outputs=[probabilities, cond])
        return adversarial_model


    def stage2_adversarial_network(self, stage2_disc, stage2_gen, stage1_gen):

        conditioned_embedding = Input(shape=(1024,))
        noise = Input(shape=(100,))
        compressed_replicated = Input(shape=(4, 4, 128))
        input_images, ca = stage1_gen([conditioned_embedding, noise])
        stage2_disc.trainable = False
        stage1_gen.trainable = False

        images, ca2 = stage2_gen([conditioned_embedding, input_images])
        probability = stage2_disc([images, compressed_replicated])

        return Model(inputs=[conditioned_embedding, noise, compressed_replicated],
                     outputs=[probability, ca2])
