from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Utility import Utility
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from ConditioningArgumentNetwork import ConditioningArgumentNetwork
from Discriminator import Discriminator
from EmbeddingCompressor import EmbeddingCompressor
from GAN import GAN
from Generator import Generator

assert tf.__version__.startswith('2')

import PIL
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class StackGanStage1(object):
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

   
    def __init__(self, epochs=100, z_dim=100, batch_size=64, enable_function=True, stage1_generator_lr=0.001,
                 stage1_discriminator_lr=0.001):
        self.epochs = epochs
        self.z_dim = z_dim
        self.enable_function = enable_function
        self.stage1_generator_lr = stage1_generator_lr
        self.stage1_discriminator_lr = stage1_discriminator_lr
        self.image_size = 64
        self.conditioning_dim = 128
        self.batch_size = batch_size
        self.stage1_generator_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)
        self.stage1_discriminator_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
        self.stage1_generator = Generator().stage1_generator()
        self.stage1_generator.compile(loss='mse', optimizer=self.stage1_generator_optimizer)
        self.stage1_discriminator = Discriminator().stage1_discriminator()
        self.stage1_discriminator.compile(loss='binary_crossentropy', optimizer=self.stage1_discriminator_optimizer)
        self.ca_network = ConditioningArgumentNetwork().build_ca_network()
        self.ca_network.compile(loss='binary_crossentropy', optimizer='Adam')
        self.embedding_compressor = EmbeddingCompressor().build_embedding_compressor()
        self.embedding_compressor.compile(loss='binary_crossentropy', optimizer='Adam')
        self.stage1_adversarial = GAN().stage1_adversarial(self.stage1_generator, self.stage1_discriminator)
        self.stage1_adversarial.compile(loss= 'binary_crossentropy',optimizer=self.stage1_generator_optimizer)



    def train_stage1(self):

        x_train, y_train, train_embeds = Utility.load_data(filename_path=self.filename_path_train,
                                                           class_id_path=self.class_id_path_train,
                                                           dataset_path=self.dataset_path,
                                                           embeddings_path=self.embeddings_path_train, size=(64, 64))
        x_test, y_test, test_embeds = Utility.load_data(filename_path=self.filename_path_test,
                                                class_id_path=self.class_id_path_test, dataset_path=self.dataset_path,
                                                embeddings_path=self.embeddings_path_test, size=(64, 64))
        real = np.ones((self.batch_size, 1), dtype='float') * 0.9
        fake = np.zeros((self.batch_size, 1), dtype='float') * 0.1
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')
            gen_loss = []
            dis_loss = []
            num_batches = int(x_train.shape[0] / self.batch_size)

            for i in range(num_batches):
                latent_space = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                embedding_text = train_embeds[i * self.batch_size:(i + 1) * self.batch_size]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, 128))
                compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))
                image_batch = x_train[i * self.batch_size:(i + 1) * self.batch_size]
                image_batch = (image_batch - 127.5) / 127.5
                gen_images, _ = self.stage1_generator.predict([embedding_text, latent_space])

                discriminator_loss = self.stage1_discriminator.train_on_batch([image_batch, compressed_embedding],
                                                                              np.reshape(real, (self.batch_size, 1)))

                discriminator_loss_gen = self.stage1_discriminator.train_on_batch([gen_images, compressed_embedding],
                                                                                  np.reshape(fake,
                                                                                             (self.batch_size, 1)))

                discriminator_loss_wrong = self.stage1_discriminator.train_on_batch(
                    [gen_images[: self.batch_size - 1], compressed_embedding[1:]],
                    np.reshape(fake[1:], (self.batch_size - 1, 1)))
                # Discriminator loss
                d_loss = 0.5 * np.add(discriminator_loss,
                                      0.5 * np.add(discriminator_loss_gen, discriminator_loss_wrong))
                dis_loss.append(d_loss)
                print(f'Discriminator Loss: {d_loss}')

                # Generator loss
                g_loss = self.stage1_adversarial.train_on_batch([embedding_text, latent_space, compressed_embedding],
                                                                [K.ones((self.batch_size, 1)) * 0.9,
                                                                 K.ones((self.batch_size, 256)) * 0.9])
                total_generator_loss = g_loss[0]
                generator_image_loss = g_loss[1]
                generator_conditioning_loss = g_loss[2]
                print(f'Total Generator Loss: {total_generator_loss}')
                print(f'Generator Image Loss: {generator_image_loss}')
                print(f'Generator Text Embedding Loss: {generator_conditioning_loss}')
                gen_loss.append(g_loss)

                if epoch % 10 == 0:
                    latent_space = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                    embedding_batch = test_embeds[0: self.batch_size]
                    gen_images, _ = self.stage1_generator.predict_on_batch([embedding_batch, latent_space])
                    try:
                        for i, image in enumerate(gen_images[:10]):
                            Utility.save_image(image, f'/Users/tejasree/Downloads/birds_implementation-3/birds/test/gen_1_{epoch}_{i}')
                    except:
                        pass
                if epoch % 10 == 0:
                    self.stage1_generator.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_gen.h5')
                    self.stage1_discriminator.save_weights(
                        "/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_disc.h5")
                    self.ca_network.save_weights('/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_ca.h5')
                    self.embedding_compressor.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_embco.h5')
                    self.stage1_adversarial.save_weights(
                        '/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_adv.h5')
                if epoch>= 100 and epoch%100 ==0:
                    load_epoch = epoch-(epoch % 100)
                    self.stage1_generator.load_weights(f'/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_gen_epoch_{load_epoch}.h5')
                    self.stage1_discriminator.load_weights(f"/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_disc_epoch_{load_epoch}.h5")
                    self.ca_network.load_weights(f'/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_ca_epoch_{load_epoch}.h5')
                    self.embedding_compressor.load_weights(f'/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_embco_epoch_{load_epoch}.h5')
                    self.stage1_adversarial.load_weights(f'/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_adv_epoch_{load_epoch}.h5')
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(gen_loss, self.epochs, label="Generator Loss", color='blue')
            plt.title("Generator Loss Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.plot(dis_loss, self.epochs,label="Discriminator Loss", color='blue')
            plt.title("Discriminator Loss Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        except:
            pass
        self.stage1_generator.save_weights('/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_gen.h5')
        self.stage1_discriminator.save_weights("/Users/tejasree/Downloads/birds_implementation-3/weights/stage1_disc.h5")
