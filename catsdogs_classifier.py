import os
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


"""
Training a model for image classification (cats/dogs)

Using:
    - pretrained VGG16 model
    - custom classifier on top of VGG16
    - data augmentation
    - fine-tuning
"""


class CatsDogsClassifier:

    def __init__(self, base_dir, model_checkpoint_file, model_train_file, model_finetune_file, conv_base_type, batch_size):
        self.base_dir = base_dir
        self.conv_base_type = conv_base_type

        self.train_dir = os.path.join(base_dir, 'train')
        self.validation_dir = os.path.join(base_dir, 'validation')
        self.test_dir = os.path.join(base_dir, 'test')

        self.model_checkpoint_file = model_checkpoint_file
        self.model_train_file = model_train_file
        self.model_finetune_file = model_finetune_file

        self.batch_size = batch_size
        self.rows = 150
        self.cols = 150

        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

        self.checkpoint_callback = None

        self.conv_base = None
        self.model = None

    def make_convolutional_base(self):
        if self.conv_base_type.lower() == 'vgg16':
            self.conv_base = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.rows, self.cols, 3))
        else:
            raise ValueError('Not implemented conv base type: ' + self.conv_base_type)

    def build_model(self):
        self.make_convolutional_base()
        print(self.conv_base.summary())

        self.model = models.Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        self.build_generators()

    def build_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            self.train_dir,
            # All images will be resized to 150x150
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            class_mode='binary')

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.rows, self.cols),
            batch_size=self.batch_size,
            class_mode='binary')

    def get_checkpoint_callback(self):
        if not self.checkpoint_callback:
            self.checkpoint_callback = ModelCheckpoint(
                self.model_checkpoint_file,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)
        return self.checkpoint_callback

    @staticmethod
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def train(self, learning_rate, epochs, steps_per_epoch, steps_per_validation):
        print('The number of trainable weights before freezing the conv base:', len(self.model.trainable_weights))
        self.conv_base.trainable = False
        print('The number of trainable weights after freezing the conv base:', len(self.model.trainable_weights))

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['acc'])

        print('Starting training the model..')

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=steps_per_validation,
            callbacks=[self.get_checkpoint_callback()],
            verbose=1)

        self.model.save(self.model_train_file)

    def finetune(self, learning_rate, epochs, steps_per_epoch, steps_per_validation):
        print('Starting fine-tuning...')

        # set layers after 'block5_conv1' trainable
        self.conv_base.trainable = True

        set_trainable = False
        for layer in self.conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['acc'])

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[self.get_checkpoint_callback()],
            validation_steps=steps_per_validation)

        self.model.save(self.model_finetune_file)

    def test(self, steps):
        print('Starting testing...')
        test_loss, test_acc = self.model.evaluate_generator(self.test_generator, steps=steps)
        print('Test acc:', test_acc)

    def plot_history(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, CatsDogsClassifier.smooth_curve(acc), 'bo', label='Smoothed training acc')
        plt.plot(epochs, CatsDogsClassifier.smooth_curve(val_acc), 'b', label='Smoothed validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, CatsDogsClassifier.smooth_curve(loss), 'bo', label='Smoothed training loss')
        plt.plot(epochs, CatsDogsClassifier.smooth_curve(val_loss), 'b', label='Smoothed validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    """
    Images file structure:
        <base_dir>/
            |-train/
                |-cats/
                |-dogs/
            |-vaildation/
                |-cats/
                |-dogs/        
            |-test/
                |-cats/
                |-dogs/        
    """

    classifier = CatsDogsClassifier(base_dir='/home/asd/Desktop/keras/cats_and_dogs',
                                    model_checkpoint_file='/home/asd/Desktop/keras/vgg-model-chkpnt.h5',
                                    model_train_file='/home/asd/Desktop/keras/vgg-model-train.h5',
                                    model_finetune_file='/home/asd/Desktop/keras/vgg-model-finetune.h5',
                                    conv_base_type='vgg16',
                                    batch_size=50)
    classifier.build_model()

    classifier.train(learning_rate=0.00002, epochs=30, steps_per_epoch=300, steps_per_validation=100)
    classifier.plot_history()

    classifier.finetune(learning_rate=0.00001, epochs=100, steps_per_epoch=300, steps_per_validation=100)
    classifier.plot_history()

    classifier.test(steps=100)
