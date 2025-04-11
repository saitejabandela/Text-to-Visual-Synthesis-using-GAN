from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ConditioningArgumentNetwork import ConditioningArgumentNetwork
from EmbeddingCompressor import EmbeddingCompressor
from GAN import GAN
from Generator import Generator
from Discriminator import Discriminator
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
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class StackGanStage2(object):

    def __init__(self, epochs=200, z_dim=100, batch_size=64, enable_function=True, stage2_generator_lr=0.001,stage2_discriminator_lr=0.001):
        generator = Generator()
        self.epochs = epochs
        self.z_dim = z_dim
        self.enable_function = enable_function
        self.stage1_generator_lr = stage2_generator_lr
        self.stage1_discriminator_lr = stage2_discriminator_lr
        self.low_image_size = 64
        self.high_image_size = 256
        self.conditioning_dim = 128
        self.batch_size = batch_size
        self.stage2_generator_optimizer = Adam(lr=stage2_generator_lr, beta_1=0.5, beta_2=0.999)
        self.stage2_discriminator_optimizer = Adam(lr=stage2_discriminator_lr, beta_1=0.5, beta_2=0.999)
        self.stage1_generator = generator.stage1_generator()
        self.stage1_generator.compile(loss='binary_crossentropy', optimizer=self.stage2_generator_optimizer)
        self.stage1_generator.load_weights('/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_gen.h5')
        self.stage2_generator = generator.build_stage2_generator()
        self.stage2_generator.compile(loss='binary_crossentropy', optimizer=self.stage2_generator_optimizer)
        self.stage2_discriminator = Discriminator().stage2_discriminator()
        self.stage2_discriminator.compile(loss='binary_crossentropy', optimizer=self.stage2_discriminator_optimizer)
        self.ca_network = ConditioningArgumentNetwork().build_ca_network()
        self.ca_network.compile(loss='binary_crossentropy', optimizer='Adam')

        self.embedding_compressor = EmbeddingCompressor().build_embedding_compressor()
        self.embedding_compressor.compile(loss='binary_crossentropy', optimizer='Adam')

        self.stage2_adversarial = GAN().stage2_adversarial_network(self.stage2_discriminator, self.stage2_generator,self.stage1_generator)
        self.stage2_adversarial.compile(loss='binary_crossentropy',optimizer=self.stage2_generator_optimizer)



    def train_stage2(self):
        data_dir = "/Users/tejasree/Downloads/birds_implementation-3/birds"
        train_dir = data_dir + "/train"
        test_dir = data_dir + "/test"
        embeddings_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
        embeddings_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"
        filename_path_train = train_dir + "/filenames.pickle"
        filename_path_test = test_dir + "/filenames.pickle"
        class_id_path_train = train_dir + "/class_info.pickle"
        class_id_path_test = test_dir + "/class_info.pickle"
        dataset_path = "/Users/tejasree/Downloads/birds_implementation-3/CUB_200_2011/CUB_200_2011"
        #Trains Stage 2 StackGAN.
        x_high_train, y_high_train, high_train_embeds = Utility.load_data(filename_path=filename_path_train,
                                                                  class_id_path=class_id_path_train,
                                                                  dataset_path=dataset_path,
                                                                  embeddings_path=embeddings_path_train,
                                                                  size=(256, 256))

        x_high_test, y_high_test, high_test_embeds = Utility.load_data(filename_path=filename_path_test,
                                                               class_id_path=class_id_path_test,
                                                               dataset_path=dataset_path,
                                                               embeddings_path=embeddings_path_test, size=(256, 256))

        x_low_train, y_low_train, low_train_embeds = Utility.load_data(filename_path=filename_path_train,
                                                               class_id_path=class_id_path_train,
                                                               dataset_path=dataset_path,
                                                               embeddings_path=embeddings_path_train, size=(64, 64))

        x_low_test, y_low_test, low_test_embeds = Utility.load_data(filename_path=filename_path_test,
                                                            class_id_path=class_id_path_test,
                                                            dataset_path=dataset_path,
                                                            embeddings_path=embeddings_path_test, size=(64, 64))

        real = np.ones((self.batch_size, 1), dtype='float') * 0.9
        fake = np.zeros((self.batch_size, 1), dtype='float') * 0.1

        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')

            gen_loss = []
            disc_loss = []

            num_batches = int(x_high_train.shape[0] / self.batch_size)

            for i in range(num_batches):

                latent_space = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                embedding_text = high_train_embeds[i * self.batch_size:(i + 1) * self.batch_size]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, self.conditioning_dim))
                compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

                image_batch = x_high_train[i * self.batch_size:(i + 1) * self.batch_size]
                image_batch = (image_batch - 127.5) / 127.5

                low_res_fakes, _ = self.stage1_generator.predict([embedding_text, latent_space], verbose=3)
                high_res_fakes, _ = self.stage2_generator.predict([embedding_text, low_res_fakes], verbose=3)

                discriminator_loss = self.stage2_discriminator.train_on_batch([image_batch, compressed_embedding],
                                                                              np.reshape(real, (self.batch_size, 1)))

                discriminator_loss_gen = self.stage2_discriminator.train_on_batch(
                    [high_res_fakes, compressed_embedding],
                    np.reshape(fake, (self.batch_size, 1)))

                discriminator_loss_fake = self.stage2_discriminator.train_on_batch(
                    [image_batch[:(self.batch_size - 1)], compressed_embedding[1:]],
                    np.reshape(fake[1:], (self.batch_size - 1, 1)))

                d_loss = 0.5 * np.add(discriminator_loss, 0.5 * np.add(discriminator_loss_gen, discriminator_loss_fake))
                disc_loss.append(d_loss)

                print(f'Discriminator Loss: {d_loss}')

                g_loss_stage2 = self.stage2_adversarial.train_on_batch([embedding_text, latent_space, compressed_embedding],
                                                                [K.ones((self.batch_size, 1)) * 0.9,
                                                                 K.ones((self.batch_size, 256)) * 0.9])
                total_generator_loss_stage2 = g_loss_stage2[0]
                generator_image_loss_stage2 = g_loss_stage2[1]
                generator_conditioning_loss_stage2 = g_loss_stage2[2]
                print(f'Total Generator Loss: {total_generator_loss_stage2}')
                print(f'Generator Image Loss: {generator_image_loss_stage2}')
                print(f'Generator Conditioning Loss: {generator_conditioning_loss_stage2}')
                g_loss_stage2.append(total_generator_loss_stage2)
                if epoch % 10 == 0:
                    latent_space = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                    embedding_batch = high_test_embeds[0: self.batch_size]

                    low_fake_images, _ = self.stage1_generator.predict([embedding_batch, latent_space], verbose=3)
                    high_fake_images, _ = self.stage2_generator.predict([embedding_batch, low_fake_images], verbose=3)

                    for i, image in enumerate(high_fake_images[:10]):
                        Utility.save_image(image,
                                   f'/Users/tejasree/Downloads/birds_implementation-3/results_stage2/2nd_{self.batch_size}_{epoch}_{i}.png')

                if epoch % 10 == 0:
                    self.stage2_generator.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_gen.h5')
                    self.stage2_discriminator.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_disc.h5')
                    self.ca_network.save_weights('/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_ca.h5')
                    self.embedding_compressor.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_embco.h5')
                    self.stage2_adversarial.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_adv.h5')
        try:
            plt.figure(figsize=(10,5))
            plt.plot(g_loss_stage2,self.epochs, label="Generator Loss",color='blue')
            plt.title("Generator Loss over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        except:
            pass
        self.stage2_generator.save_weights('/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_gen.h5')
        self.stage2_discriminator.save_weights("/Users/tejasree/Downloads/birds_implementation-3/weights/stage2_disc.h5")
