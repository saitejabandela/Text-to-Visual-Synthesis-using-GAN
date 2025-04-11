from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

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


class Utility:

    @staticmethod
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @staticmethod
    def load_class_ids_filenames(class_id_path, filename_path):
        with open(class_id_path, 'rb') as file:
            class_id = pickle.load(file, encoding='latin1')

        with open(filename_path, 'rb') as file:
            filename = pickle.load(file, encoding='latin1')

        return class_id, filename

    @staticmethod
    def load_text_embeddings(text_embeddings):
        with open(text_embeddings, 'rb') as file:
            embeds = pickle.load(file, encoding='latin1')
            embeds = np.array(embeds)

        return embeds

    @staticmethod
    def load_bbox(data_path):
        bbox_path = data_path + '/bounding_boxes.txt'
        image_path = data_path + '/images.txt'
        bbox_df = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        filename_df = pd.read_csv(image_path, delim_whitespace=True, header=None)

        filenames = filename_df[1].tolist()
        bbox_dict = {i[:-4]: [] for i in filenames[:2]}

        for i in range(0, len(filenames)):
            bbox = bbox_df.iloc[i][1:].tolist()
            dict_key = filenames[i][:-4]
            bbox_dict[dict_key] = bbox

        return bbox_dict

    @staticmethod
    def load_images(image_path, bounding_box, size):
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        if bounding_box is not None:
            r = int(np.maximum(bounding_box[2], bounding_box[3]) * 0.75)
            c_x = int((bounding_box[0] + bounding_box[2]) / 2)
            c_y = int((bounding_box[1] + bounding_box[3]) / 2)
            y1 = np.maximum(0, c_y - r)
            y2 = np.minimum(h, c_y + r)
            x1 = np.maximum(0, c_x - r)
            x2 = np.minimum(w, c_x + r)
            image = image.crop([x1, y1, x2, y2])

        image = image.resize(size, PIL.Image.BILINEAR)
        return image

    @staticmethod
    def load_data(filename_path, class_id_path, dataset_path, embeddings_path, size):

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
        class_id, filenames = Utility.load_class_ids_filenames(class_id_path, filename_path)
        embeddings = Utility.load_text_embeddings(embeddings_path)
        bbox_dict = Utility.load_bbox(dataset_path)

        x, y, embeds = [], [], []

        for i, filename in enumerate(filenames):
            bbox = bbox_dict[filename]
            try:
                image_path = f'{dataset_path}/images/{filename}.jpg'
                image = Utility.load_images(image_path, bbox, size)
                e = embeddings[i, :, :]
                embed_index = np.random.randint(0, e.shape[0] - 1)
                embed = e[embed_index, :]

                x.append(np.array(image))
                y.append(class_id[i])
                embeds.append(embed)
            except Exception as e:
                print(f'{e}')

        x = np.array(x)
        y = np.array(y)
        embeds = np.array(embeds)

        return x, y, embeds

    @staticmethod
    def save_image(file, save_path):
        """Saves the image at the specified file path.
        """
        image = plt.figure()
        ax = image.add_subplot(1, 1, 1)
        ax.imshow(file)
        ax.axis("off")
        plt.savefig(save_path)

    @staticmethod
    def concat_along_dims(inputs):
        """Joins the conditioned text with the encoded image along the dimensions.

        Args:
            inputs: consisting of conditioned text and encoded images as [c,x].

        Returns:
            Joint block along the dimensions.
        """
        c = inputs[0]
        x = inputs[1]

        c = K.expand_dims(c, axis=1)
        c = K.expand_dims(c, axis=1)
        c = K.tile(c, [1, 16, 16, 1])
        return K.concatenate([c, x], axis=3)

    @staticmethod
    def residual_block(input):
        """Residual block with plain identity connections.

        Args:
            inputs: input layer or an encoded layer

        Returns:
            Layer with computed identity mapping.
        """
        inp = Conv2D(512, kernel_size=(3, 3), padding='same', use_bias=False)(input)
        inp = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(inp)
        inp = ReLU()(inp)

        inp = Conv2D(512, kernel_size=(3, 3), padding='same', use_bias=False)(inp)
        inp = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(inp)

        inp = add([inp, input])
        inp = ReLU()(inp)
        return inp
