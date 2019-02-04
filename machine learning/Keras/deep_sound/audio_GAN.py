from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import data_loader as dl


# GAN for audio generation
# raw audio

class GAN:
    def __init__(self, dim, channels=2):

        # set dims
        self.dim = dim
        self.channels = channels
        self.shape = (self.dim, self.channels)
        self.latent = 100

        # optimizer for learning
        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        # setup discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # setup generator layer
        self.generator = self.build_generator()

        # create sound noise z to generate
        z = Input(shape=(self.latent,))
        sound = self.generator(z)

        # we won't train the discriminator for the combined model
        self.discriminator.trainable = False

        # determine validity of generated image
        valid = self.discriminator(sound)

        # combined models train the generator to fool discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer)

    def build_generator(self):

        model = Sequential()
        model.summary()

        noise = Input(shape=(self.latent,))
        sound = model(noise)

        return Model(noise, sound)

    def build_discriminator(self):

        model = Sequential()
        model.summary()

        sound = Input(shape=self.shape)
        validity = model(sound)

        return Model(sound, validity)

    def train(self, X, epochs=2000, batch_size=1, save_interval=50):

        # ground truth
        for epoch in range(epochs):
            print(batch_size)

            # save on interval
            if epoch % save_interval == 0:
                # save generated music
                self.save_sound(epoch)

    def save_sound(self, epoch):
        noise = np.random.normal()
        gen_sound = self.generator.predict(noise)
        # dl.matrix_to_wav()


if __name__ == "__main__":
    # fetch data wav file
    fs, data = dl.wav_to_matrix("levels")

    # get sound dim
    data_dim = data.shape[0]
    data_channel = data.shape[1]

    # create GAN
    # gan = GAN(data_dim, data_channel)

    # run train loop
    # gan.train(data)
