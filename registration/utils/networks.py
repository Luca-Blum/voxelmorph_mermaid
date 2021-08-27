from os import listdir
from os.path import join, isdir
import pathlib
from pathlib import Path
import re
import os
import voxelmorph as vxm
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

# turn off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Networks:
    """
    Class for training the voxelmorph and mermaid networks
    So far only voxelmorph is supported
    """

    def __init__(self, datahandler, losses=None, loss_weights=None, train_size=0.8):
        """
        :param datahandler: Datahandler object that takes care of the input and output of the files
        :param losses: loss functions. Default is NCC and L2
        :param loss_weights: weights of the corresponding loss functions
        :param train_size: train test split [0,1]
        """

        self.dh = datahandler
        self.name = self.dh.name

        if losses is None:
            self.losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
        else:
            self.losses = losses

        if loss_weights is None:
            self.loss_weights = [1, 0.01]
        else:
            self.loss_weights = loss_weights

        self.train_size = 0
        self.test_size = 0

        self.vol_shape = np.array([0, 0, 0])

        self.padding = ((0, 0), (5, 5), (3, 3), (5, 5))

        self.train, self.test = self._create_train_test_split(train_size)

        self.vxm_model = None

    def _create_train_test_split(self, train_size):
        """
        Splits data into training and testing data
        :param train_size: size of the training set [0,1]
        :return: Tuple of lists where a list contains the file names
        """

        files = self.dh.get_processed_file_names()

        train_files, test_files = train_test_split(files, train_size=train_size, random_state=13)

        self.train_size = len(train_files)
        self.test_size = len(test_files)

        self.vol_shape = nib.load(train_files[0]).get_fdata().shape
        self.vol_shape = self.vol_shape + np.array([self.padding[1][0] + self.padding[1][1],
                                                    self.padding[2][0] + self.padding[2][1],
                                                    self.padding[3][0] + self.padding[3][1]])

        return np.array(train_files), np.array(test_files)

    def vxm_data_generator(self, x_data, batch_size=32):
        """
        Adapted from voxelmorph tutorial :
        https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=dJcCEElPPXN2
        Generator that takes in array of file names, and yields data for
        our custom vxm model.

        inputs:  moving [bs, H, W, D], fixed image [bs, H, W, D]
        outputs: moved image [bs, H, W, D], zero-gradient [bs, H, W, 3]

        It is guaranteed that a pair of moving and target images are not the same

        :param x_data: data set used to generate the samples
        :param batch_size: size of batch
        :return: generator that yields input and output to networks
        """

        ndims = len(self.vol_shape)

        # prepare a zero array the size of the deformation
        zero_phi = np.zeros([batch_size, *self.vol_shape, ndims])

        while True:
            # prepare inputs:
            r = np.arange(len(x_data))
            idx1 = np.random.choice(r, size=batch_size)

            moving_images = x_data[idx1]

            moving_images = np.array([nib.load(f).get_fdata() for f in moving_images])

            moving_images = np.pad(moving_images, self.padding, 'constant')

            r = np.delete(r, idx1)
            idx2 = np.random.choice(r, size=batch_size)

            fixed_images = x_data[idx2]

            fixed_images = np.array([nib.load(f).get_fdata() for f in fixed_images])

            fixed_images = np.pad(fixed_images, self.padding, 'constant')

            inputs = [moving_images, fixed_images]

            # prepare outputs (the 'true' moved image):
            # of course, we don't have this, but we know we want to compare
            # the resulting moved image with the fixed image.
            # we also wish to penalize the deformation field.
            outputs = [fixed_images, zero_phi]

            yield inputs, outputs

    def build_model_vxm(self):
        """
        createds a default voxelmorph model
        """

        nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]

        self.vxm_model = vxm.networks.VxmDense(self.vol_shape, nb_features, int_steps=0)

        print("Losses")
        print(self.losses)

        print(self.loss_weights)

        self.vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=self.losses,
                               loss_weights=self.loss_weights)

    def train_vxm(self):
        """
        trains the voxelmorph model and creates a plot of the loss evolution
        :return: None
        """

        print("train voxelmorph for " + self.name)

        self.build_model_vxm()

        model_path = self.dh.get_processed_folder()

        model_path = join(model_path, "vxm_model" + str(int(time.time())))

        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

        model_path = join(model_path, "cp-{epoch:04d}.ckpt")

        # Create a callback that saves the model's weights
        callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                      save_weights_only=True,
                                                      verbose=1)

        print("store initial weights to: " + str(model_path))

        self.vxm_model.save_weights(model_path.format(epoch=0))

        batch_size = 1

        train_generator = self.vxm_data_generator(self.train, batch_size=batch_size)

        # use this if you want to train with every pair in each epoch
        # nb_pairs = self.train_size * (self.train_size - 1)/2.

        nb_pairs = 60
        epochs = 100

        steps_per_epoch = np.ceil(nb_pairs / batch_size)

        print("steps per epoch: " + str(steps_per_epoch))
        print("start training")

        # train model
        hist = self.vxm_model.fit(train_generator, callbacks=[callback], epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=2)

        plt.figure()
        plt.plot(hist.epoch, hist.history['loss'], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

        plt.savefig('hist_vxm.png')

        return None

    def train_from_weights_vxm(self, model_path=None):
        """
        trains a voxelmorph model from given weights and creates a plot of the loss evolution
        :param model_path: path to the folder that contains the weights for the model
        :return: None
        """

        self.load_vxm(model_path)

        model_path = self.dh.get_processed_folder()

        model_path = join(model_path, "vxm_model" + str(int(time.time())))

        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

        model_path = join(model_path, "cp-{epoch:04d}.ckpt")

        # Create a callback that saves the model's weights
        callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                      save_weights_only=True,
                                                      verbose=1)

        print("store initial weights to: " + str(model_path))

        self.vxm_model.save_weights(model_path.format(epoch=0))

        batch_size = 1

        train_generator = self.vxm_data_generator(self.train, batch_size=batch_size)

        # use this if you want to train with every pair in each epoch
        # nb_pairs = self.train_size * (self.train_size - 1)/2.

        nb_pairs = 60
        epochs = 100

        steps_per_epoch = np.ceil(nb_pairs / batch_size)

        print("steps per epoch: " + str(steps_per_epoch))
        print("start training")

        # train model
        hist = self.vxm_model.fit(train_generator, callbacks=[callback], epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=2)

        plt.figure()
        plt.plot(hist.epoch, hist.history['loss'], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

        plt.savefig('hist_vxm.png')

        return None

    def predict_vxm(self, nb_pairs=10):
        """
        predicts nb_pairs samples from the test set
        :param nb_pairs: number of predictions
        :return: Tuple of lists where the first element contains the warped image and the second the displacement fields
        """

        return self.predict_data_vxm(self.test, nb_pairs)

    def predict_data_vxm(self, data, nb_pairs=10):
        """
        predicts nb_pairs samples from the given data set
        :param data: data used for the prediction
        :param nb_pairs: number of predictions
        :return:Tuple of lists where the first element contains the warped image and the second the displacement fields
        """

        if self.vxm_model is None:
            self.load_vxm()

        val_generator = self.vxm_data_generator(data, batch_size=1)

        val_inputs = []
        val_preds = []
        for pair in range(nb_pairs):
            val_input, _ = next(val_generator)
            val_pred = self.vxm_model.predict(val_input)

            val_inputs.append(val_input)
            val_preds.append(val_pred)

        return val_inputs, val_preds

    def predict_one_pair_vxm(self, data):
        """
        predicts one pair given by data. Useful for non-random prediction/ not using the data generator
        :param data: List of TWO filenames
        :return: input and output of network
        """

        if self.vxm_model is None:
            self.load_vxm()

        ndims = len(self.vol_shape)

        zero_phi = np.zeros([1, *self.vol_shape, ndims])

        idx1 = np.array([0])

        moving_image = data[idx1]

        moving_image = np.array([nib.load(f).get_fdata() for f in moving_image])

        moving_image = np.pad(moving_image, self.padding, 'constant')

        idx2 = np.array([1])

        fixed_image = data[idx2]

        fixed_image = np.array([nib.load(f).get_fdata() for f in fixed_image])

        fixed_image = np.pad(fixed_image, self.padding, 'constant')

        inputs = [moving_image, fixed_image]

        outputs = [fixed_image, zero_phi]

        val_input, _ = (inputs, outputs)
        val_pred = self.vxm_model.predict(val_input)

        return [val_input], [val_pred]

    def load_vxm(self, model_path=''):
        """
        load a stored voxelmorph model
        :param model_path: path to the folder that contains the weights. If not given, the method searches for a model
        and uses the latest checkpoint
        :return: None
        """

        self.build_model_vxm()

        if model_path != '' and not isdir(model_path):
            raise NotADirectoryError("path to model does not exists")

        if model_path == '':

            model_path = self.dh.get_processed_folder()

            all_models = [Path(join(model_path, f)).name for f in listdir(model_path) if
                          isdir(join(model_path, f)) and 'vxm_model' in f]

            max_model = 0

            for name in all_models:
                nb = int(re.findall(r'\d+', name)[0])

                if max_model <= nb:
                    max_model = nb

            model_path = join(model_path, "vxm_model" + str(max_model))

        latest = tf.train.latest_checkpoint(model_path)

        if model_path != '':
            print("path to model given: " + str(model_path))

        print("loaded weights from: " + str(latest))

        self.vxm_model.load_weights(latest)

    def train_mermaid(self):
        """
        train a mermaid network. Unfortunately this could not achieved.
        :return: None
        """

        libs_path = self.dh.get_data_path("libs")

        prep_data_path = join(libs_path, "easyreg/scripts/prepare_data.py")
        train_path = join(libs_path, "easyreg/scripts/train_reg.py")
        eval_path = join(libs_path, "easyreg/scripts/eval_reg.py")

        # TODO: training
        print("train mermaid for " + self.name)

        # ???

        # TODO: store model
        model_name = "mermaid.h5"
        print("store mermaid for " + self.name + " " + model_name)

        return None

    def train_both(self):
        """
        train voxelmorph and mermaid one after another
        :return: None
        """

        self.train_vxm()
        self.train_mermaid()

    def get_training_data(self):
        """
        :return: List of file names for training data
        """

        return self.train

    def get_testing_data(self):
        """
        :return: List of file names for testing data
        """

        return self.test

    def get_vol_shape(self):
        """
        :return: shape of image volumes
        """
        return self.vol_shape
