from __future__ import print_function, division
import scipy

# from keras_contrib.layers.normalization import InstanceNormalization
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from scipy.misc import imsave, imread, imresize


def data_loader():
    print("data")


# GNA class for generating stuff
class GAN:
    def __init__(self, width=128, height=128, channels=1):

        # set images dims
        self.width = width
        self.height = height
        self.channels = channels
        self.shape = (self.width, self.height, self.channels)
        self.latent = 100

        # optimizer for learning
        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        # setup discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # setup generator layer
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.combined = Model()
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):

        model = Sequential()
        model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(Reshape(self.shape))

        model.summary()

        return model

    def train(self, X, epochs=2000, batch_size=1, save_interval=100):
        for epochs in range(epochs):
            print(batch_size)

            if epochs % save_interval == 0:
                print("test")


if __name__ == "__main__":
    # train loader
    (X_train, _), (_, _) = mnist.load_data()

    # init gna
    gan = GAN()
    gan.train(X_train, epochs=2000, batch_size=1, save_interval=100)
