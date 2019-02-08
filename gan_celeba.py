"""
Training Generative Adversarial Network on CelebA dataset (people's faces).

The Generator network will learn to create faces images of not existing people.
"""

import os
import numpy as np
from keras.layers import Conv2D, Dropout, Conv2DTranspose
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing import image
from timeit import default_timer as timer


class GAN:

    def __init__(self, model_file):
        self.model_file = model_file
        self.img_rows = 218
        self.img_cols = 178
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128
        self.resume_step_number = 0
        self.resume_batch_index = 1

        discrimnator_optimizer = Adam(0.0002, 0.5)
        combined_optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=discrimnator_optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=combined_optimizer, metrics=['accuracy'])
        print('\n--- GAN ---')
        self.combined.summary()

    def resume(self, resume_step_number, resume_batch_index):
        self.resume_step_number = resume_step_number
        self.resume_batch_index = resume_batch_index

        print('\nDiscriminator weights [0][0] before load_weights():\n')
        print(self.discriminator.get_weights()[0][0])

        print('\nCombined weights [0][0] before load_weights():\n')
        print(self.combined.get_weights()[0][0])

        self.combined.load_weights(self.model_file)

        print('\n-------------------------------------------------------------------')
        print('\nDiscriminator weights [0][0] after load_weights():\n')
        print(self.discriminator.get_weights()[0][0])

        print('\nCombined weights [0][0] after load_weights():\n')
        print(self.combined.get_weights()[0][0])

        # TODO: set learning rates of optimizer?
        print('\nWeights loaded.\n')

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512 * 28 * 23, input_dim=self.latent_dim))
        model.add(LeakyReLU())
        model.add(Reshape((28, 23, 512)))

        model.add(Conv2D(filters=512, kernel_size=5, padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2D(filters=64, kernel_size=7))
        model.add(LeakyReLU())

        model.add(Conv2D(self.channels, kernel_size=7, activation='tanh', padding='same'))

        print('\n--- Generator ---')
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=3, input_shape=self.img_shape))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=128, kernel_size=4, strides=2))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=256, kernel_size=4, strides=2))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=512, kernel_size=4, strides=2))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        print('\n--- Discriminator ---')
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_celeba_data(self, directory, num_samples=5000):
        x_train = np.empty((num_samples, self.img_rows, self.img_cols, self.channels), dtype='uint8')

        onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        for i, imagefile in enumerate(onlyfiles):
            if i >= num_samples:
                break
            img = image.load_img(directory + "/" + imagefile)
            x_train[i] = image.img_to_array(img, data_format='channels_last')

        return x_train

    def load_celeba_batch(self, directory, start_index, stop_index):
        # start_index must be >= 1, since the first file in CelebA is 000001.jpg
        assert 1 <= start_index <= 250000
        assert 1 <= stop_index <= 250000
        assert start_index < stop_index

        num_samples = stop_index - start_index

        x_train = np.empty((num_samples, self.img_rows, self.img_cols, self.channels), dtype='uint8')

        for i in range(start_index, stop_index):
            imagefile = "{0}/{1:06d}.jpg".format(directory, i)
            img = image.load_img(imagefile)
            x_train[i - start_index] = image.img_to_array(img, data_format='channels_last')

        return x_train

    def train(self, max_steps, batch_size=16, sample_interval=10, dataset_dir='', save_dir=''):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        total_size = 50000  # dataset contains 50 000 images
        epoch_size = 2000   # loading 2000 images on each epoch
        epoch_batches = epoch_size // batch_size  # number of selected random batches on each epoch

        resume_index_used = False

        step = self.resume_step_number

        while True:

            start_batch_index = 1  # starting image index = first batch [1, 2000]

            # starting from resume_batch_index, but only once, next cycle from 1
            if not resume_index_used:
                start_batch_index = self.resume_batch_index
                resume_index_used = True

            for start_index in range(start_batch_index, total_size, epoch_size):

                x_train = self.load_celeba_batch(dataset_dir,
                                                 start_index=start_index,
                                                 stop_index=start_index + epoch_size)

                # Normalizes data
                x_train = (x_train.reshape(
                    (x_train.shape[0],) + (self.img_rows, self.img_cols, self.channels)
                ).astype('float32') - 127.5) / 127.5

                for batch in range(epoch_batches):

                    start = timer()

                    if step >= max_steps:
                        self.combined.save_weights(model_file)
                        print('Successfully finished.')
                        return

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx = np.random.randint(0, x_train.shape[0], batch_size)
                    imgs = x_train[idx]

                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                    # Generate a batch of new images
                    gen_imgs = self.generator.predict(noise)

                    # Trick with little random noise to the labels
                    valid_with_noise = valid + 0.05 * np.random.random(valid.shape)
                    fake_with_noise = fake + 0.05 * np.random.random(fake.shape)

                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(imgs, valid_with_noise)
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_with_noise)
                    d_loss = 0.5 * np.add(np.abs(d_loss_real), np.abs(d_loss_fake))

                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                    # Train the generator (to have the discriminator label samples as valid)
                    g_loss = self.combined.train_on_batch(noise, valid)

                    # Plot the progress
                    if step % 1 == 0:
                        end = timer()
                        elapsed = end - start
                        print(
                            "step {}  (images [{}, {}], batch {}):   D_Loss: {:.4f}, D_Acc: {:.2f}%  |  G_Loss: {:.4f}, G_Acc: {:.2f}%   Time: {:.0f} sec.".format(
                                step, start_index, start_index + epoch_size - 1, batch, d_loss[0], 100 * d_loss[1],
                                g_loss[0], 100 * g_loss[1], elapsed
                            ))

                    # If at save interval => save generated image samples
                    if step % sample_interval == 0:
                        img = image.array_to_img(gen_imgs[batch_size // 2] * 127.5 + 127.5, scale=False)
                        img.save(os.path.join(save_dir, 'generated_' + str(step) + '.png'))

                    # Save model
                    if (step + 1) % 100 == 0:
                        self.combined.save_weights(self.model_file)

                    step += 1


if __name__ == '__main__':
    gan = GAN(model_file='/home/asd/Desktop/keras/gan-celeba.h5')
    gan.resume(resume_step_number=301, resume_batch_index=4001)
    gan.train(max_steps=30000,
              batch_size=16,
              sample_interval=2,
              dataset_dir='/home/asd/Desktop/keras/celeba_dataset',
              save_dir='/home/asd/Desktop/keras/GAN')
