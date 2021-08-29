from utils import normxcorr2
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path
import re
import neurite as ne
import matplotlib.pyplot as plt
import numpy as np


class Evaluation:
    def __init__(self, network):
        self.network = network

    def evaluate_axes_vxm(self, test_input, test_output, postfix=''):
        """
        Creates a plot for the given input and the corresponding predicted output.
        The plot shows the mid-slice from the sagital, coronal and traverse axis.
        :param test_input: sample that was given to the network to create a prediction
        :param test_output: prediction for the given sample
        :param postfix: added to the filename
        :return: None
        """

        vol_shape = test_input[0][0].shape[1:]

        print("number of pairs: " + str(len(test_input)))

        titles = ['fixed sagital', 'fixed coronal', 'fixed traverse', 'warped sagital', 'warped coronal',
                  'warped traverse']

        for i in range(len(test_input)):
            fixed = test_input[i][1].squeeze()
            warped = test_output[i][0].squeeze()

            mid_slices_fixed = [np.take(fixed, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
            mid_slices_fixed[0] = np.rot90(mid_slices_fixed[0], 1)
            mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

            mid_slices_pred = [np.take(warped, vol_shape[d] // 2, axis=d) for d in range(3)]

            mid_slices_pred[0] = np.rot90(mid_slices_pred[0], 1)
            mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
            mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
            fig, axes = ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True,
                                       grid=[2, 3], titles=titles)

            name = 'axes' + str(i) + '_' + postfix + '.png'
            fig_path = self.network.dh.get_plot_dir()
            fig_path = join(fig_path, name)

            fig.savefig(fig_path)

    def evaluate_displ_vxm(self, test_input, test_output, postfix=''):
        """
        Creates a plot for the given input and the corresponding predicted output.
        The plot shows the mid-slice from the coronal view and shows the displacement field
        :param test_input: sample that was given to the network to create a prediction
        :param test_output: prediction for the given sample
        :param postfix: added to the filename
        :return: None
        """

        vol_shape = test_input[0][0].shape[1:]

        titles = ['moving', 'fixed', 'warped', 'displ']

        print("number of pairs: " + str(len(test_input)))

        for i in range(len(test_input)):
            moving = test_input[i][0].squeeze()
            fixed = test_input[i][1].squeeze()
            warped = test_output[i][0].squeeze()
            field = test_output[i][1].squeeze()

            mid_slices_moving = [np.take(moving, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)

            mid_slices_fixed = [np.take(fixed, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)

            mid_slices_pred = [np.take(warped, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)

            mid_slices_flow = [np.take(field, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_flow[1] = np.rot90(mid_slices_flow[1], 1)

            images = [mid_slices_moving[1], mid_slices_fixed[1], mid_slices_pred[1], mid_slices_flow[1]]

            fig, axes = ne.plot.slices(images, cmaps=['gray'], grid=[1, 4], titles=titles)
            fig.tight_layout()

            name = 'displ' + str(i) + '_' + postfix + '.png'
            fig_path = self.network.dh.get_plot_dir()
            fig_path = join(fig_path, name)

            fig.savefig(fig_path)

    def evaluate_losses_vxm(self, test_input, test_output, loss='mse', postfix=''):
        """
        Creates a plot for the given input and the corresponding predicted output.
        The plot shows the difference of the warped moving image to the target image
        :param test_input: sample that was given to the network to create a prediction
        :param test_output: prediction for the given sample
        :param loss: metric to use to calculate the difference {'mse', ncc}
        :param postfix: added to the filename
        :return: None
        """

        vol_shape = test_input[0][0].shape[1:]

        titles = ['moving', 'fixed', 'warped', loss]

        print("number of pairs: " + str(len(test_input)))

        for i in range(len(test_input)):

            moving = test_input[i][0].squeeze()
            fixed = test_input[i][1].squeeze()
            warped = test_output[i][0].squeeze()

            mid_slices_moving = [np.take(moving, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)

            mid_slices_fixed = [np.take(fixed, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)

            mid_slices_pred = [np.take(warped, vol_shape[d] // 2, axis=d) for d in range(3)]
            mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)

            if loss == 'mse':
                mid_slices_loss = np.square(np.subtract(mid_slices_pred[1], mid_slices_fixed[1]))
            else:
                mid_slices_loss = normxcorr2(mid_slices_fixed[1], mid_slices_pred[1])

            images = [mid_slices_moving[1], mid_slices_fixed[1], mid_slices_pred[1], mid_slices_loss]

            fig, axes = ne.plot.slices(images, cmaps=['gray'], grid=[1, 4], titles=titles)
            fig.tight_layout()

            name = 'pred_loss' + str(i) + '_' + postfix + '.png'
            fig_path = self.network.dh.get_plot_dir()
            fig_path = join(fig_path, name)

            fig.savefig(fig_path)

    def evaluate_loss_history_model(self, model_path=None, metric='eval', plot=True, skip_first=False):
        """
        Creates plot of the loss history for the training and testing loss. The loss is approximated by 4 samples for
        each case.
        :param model_path: path to the model to evaluate
        :param metric: metric for evaluation. Can be 'eval' to use tf evaluate method, 'mse' or 'ncc'
        :param plot: can be disabled for loss calculation over multiple model
        :param skip_first: skip first epoch
        :return: train and test loss
        """

        self.network.build_model_vxm()

        if model_path is None:

            model_path = self.network.dh.get_processed_folder()

            all_models = [Path(join(model_path, f)).name for f in listdir(model_path) if
                          isdir(join(model_path, f)) and 'results' not in f]

            max_model = 0

            for name in all_models:
                nb = int(re.findall(r'\d+', name)[0])

                if max_model <= nb:
                    max_model = nb

            model_path = join(model_path, "vxm_model" + str(max_model))

        models = [join(model_path, f)[:-6] for f in listdir(model_path) if
                  isfile(join(model_path, f)) and "ckpt.index" in f]

        models = sorted(models)

        print(models)

        nb_train_pairs = 8
        nb_test_pairs = 8

        train_generator = self.network.vxm_data_generator(self.network.get_training_data(), batch_size=1)
        test_generator = self.network.vxm_data_generator(self.network.get_testing_data(), batch_size=1)

        train_losses = []
        test_losses = []
        epochs = []

        for idx, model in enumerate(models):
            if idx == 0 and skip_first:
                continue

            number = int(model[-9:-5])
            print(number)
            epochs.append(number)

            self.network.vxm_model.load_weights(model)

            train_loss_image = np.zeros(self.network.get_vol_shape())

            if metric == 'eval':
                train_losses.append(np.mean(self.network.vxm_model.evaluate(train_generator, steps=nb_train_pairs)))
                test_losses.append(np.mean(self.network.vxm_model.evaluate(test_generator, steps=nb_test_pairs)))

            else:
                for pair in range(nb_train_pairs):

                    train_input, _ = next(train_generator)
                    train_output = self.network.vxm_model.predict(train_input)

                    fixed = train_input[1].squeeze()
                    warped = train_output[0].squeeze()

                    if metric == 'mse':
                        train_loss_image = np.add(train_loss_image, np.square(np.subtract(fixed, warped)))
                    else:
                        train_loss_image = np.add(train_loss_image, normxcorr2(fixed, warped, mode="same"))

                train_losses.append(train_loss_image.mean())

                test_loss_image = np.zeros(self.network.get_vol_shape())

                for pair in range(nb_test_pairs):

                    test_input, _ = next(test_generator)
                    test_output = self.network.vxm_model.predict(test_input)

                    fixed = test_input[1].squeeze()
                    warped = test_output[0].squeeze()

                    if metric == 'mse':
                        test_loss_image = np.add(test_loss_image, np.square(np.subtract(fixed, warped)))
                    else:
                        metric = 'mncc'
                        test_loss_image = np.add(test_loss_image, normxcorr2(fixed, warped, mode="same"))

                test_losses.append(test_loss_image.mean())

        if plot:
            plt.figure()

            plt.plot(epochs, train_losses, '.-', label="training loss")
            plt.plot(epochs, test_losses, '.-', label="testing loss")
            plt.ylabel(metric + ' loss')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

            name = 'loss_epoch_' + metric + '.png'
            fig_path = self.network.dh.get_plot_dir()
            fig_path = join(fig_path, name)

            plt.savefig(fig_path)

        return train_losses, test_losses, epochs

    def evaluate_loss_history(self, metric='eval'):
        """
        Creates plot of the loss history for the training and testing loss over subsequently trained models
        :param metric: metric for evaluation. Can be 'eval' to use tf evaluate method, 'mse' or 'ncc'
        :return: train and test loss
        """

        self.network.build_model_vxm()

        model_path = self.network.dh.get_processed_folder()

        all_models = [join(model_path, f) for f in listdir(model_path) if
                      isdir(join(model_path, f)) and 'results' not in f]

        all_models = sorted(all_models)
        print(all_models)

        train_losses = []
        test_losses = []
        epochs = []
        max_epoch = -1

        for idx, model in enumerate(all_models):
            if idx == 0:
                train_loss, test_loss, epoch = self.evaluate_loss_history_model(model_path=model, metric=metric,
                                                                                plot=False, skip_first=False)
                max_epoch = epoch[-1]
            else:
                train_loss, test_loss, epoch = self.evaluate_loss_history_model(model_path=model, metric=metric,
                                                                                plot=False, skip_first=True)
                epoch = [e + max_epoch for e in epoch]

                max_epoch = epoch[-1]

            train_losses.extend(train_loss)
            test_losses.extend(test_loss)
            epochs.extend(epoch)

        plt.figure()

        plt.plot(epochs, train_losses, '.-', label="training loss")
        plt.plot(epochs, test_losses, '.-', label="testing loss")
        plt.ylabel(metric + ' loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

        name = 'loss_epoch_overall_' + metric + '.png'
        fig_path = self.network.dh.get_plot_dir()
        fig_path = join(fig_path, name)

        plt.savefig(fig_path)

    def evaluate_models(self, nb_predictions=2, model_path=''):
        """
        evaluates the latest model and creates a loss history for subsequently trained models. If a model path is given
        then the loss history is only for the specified model
        :param nb_predictions: number of predictions used to create various plots
        :param model_path: path to model to evaluate. If empty the latest model will be used
        :return: None
        """
        if model_path != '':
            self.network.load_vxm(model_path)

        test_input_temp, test_output_temp = self.network.predict_vxm(nb_predictions)

        self.evaluate_axes_vxm(test_input_temp, test_output_temp)

        self.evaluate_displ_vxm(test_input_temp, test_output_temp)

        self.evaluate_losses_vxm(test_input_temp, test_output_temp)
        
        if model_path != '':
            self.evaluate_loss_history_model(model_path=model_path, metric='eval', plot=True, skip_first=False)
        else:
            self.evaluate_loss_history(metric='eval')

    def evaluate_pair_evolution(self, indices=None):
        """
        evaluates two pairs for subsequently trained models and a pair for T1w-T1w and T2w-T2w for inter-modal case
        :return: None
        """

        if indices is None:
            indices = [0, 1, 2, 3, 0, 7, 5, 3]

        model_path = self.network.dh.get_processed_folder()

        all_models = [join(model_path, f) for f in listdir(model_path) if
                      isdir(join(model_path, f)) and 'results' not in f]

        all_models = sorted(all_models)
        print(all_models)

        data_temp1 = np.array([self.network.get_testing_data()[indices[0]],
                               self.network.get_testing_data()[indices[1]]])

        print(data_temp1)

        data_temp2 = np.array([self.network.get_testing_data()[indices[2]],
                               self.network.get_testing_data()[indices[3]]])

        print(data_temp2)

        for index, model in enumerate(all_models):

            self.network.load_vxm(model)

            test_input_temp, test_output_temp = self.network.predict_one_pair_vxm(data_temp1)

            self.evaluate_axes_vxm(test_input_temp, test_output_temp, postfix='p1_' + str(index))

            self.evaluate_displ_vxm(test_input_temp, test_output_temp, postfix='p1_' + str(index))

            self.evaluate_losses_vxm(test_input_temp, test_output_temp, postfix='p1_' + str(index))

            test_input_temp, test_output_temp = self.network.predict_one_pair_vxm(data_temp2)

            self.evaluate_axes_vxm(test_input_temp, test_output_temp, postfix='p2_' + str(index))

            self.evaluate_displ_vxm(test_input_temp, test_output_temp, postfix='p1_' + str(index))

            self.evaluate_losses_vxm(test_input_temp, test_output_temp, postfix='p1_' + str(index))

            # evaluate T1w-T1w and T2w-T2w registration
            if self.network.dh.get_name() == 'inter_modal_t1t2':
                data_temp_t1 = np.array([self.network.get_testing_data()[indices[4]],
                                         self.network.get_testing_data()[indices[5]]])

                test_input_temp, test_output_temp = self.network.predict_one_pair_vxm(data_temp_t1)

                self.evaluate_axes_vxm(test_input_temp, test_output_temp, postfix='t1_p1_' + str(index))

                self.evaluate_displ_vxm(test_input_temp, test_output_temp, postfix='t1_p1_' + str(index))

                self.evaluate_losses_vxm(test_input_temp, test_output_temp, postfix='t1_p1_' + str(index))

                data_temp_t2 = np.array([self.network.get_testing_data()[indices[6]],
                                         self.network.get_testing_data()[indices[7]]])

                test_input_temp, test_output_temp = self.network.predict_one_pair_vxm(data_temp_t2)

                self.evaluate_axes_vxm(test_input_temp, test_output_temp, postfix='t2_p1_' + str(index))

                self.evaluate_displ_vxm(test_input_temp, test_output_temp, postfix='t2_p1_' + str(index))

                self.evaluate_losses_vxm(test_input_temp, test_output_temp, postfix='t2_p1_' + str(index))
