import os
import random
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import bisect
from itertools import chain, combinations

from data_utils import Dataset_ASVspoof2021_eval, genSpoof_list
from model import RawNet


def find_threshold():
    """
        FIND OPTIMAL THRESHOLD
        1. For each class
        2. Average the LLR
        3. Find the separator with the max. dist. from both scores
    """

    scores = pd.read_csv('score.txt', delimiter='\s+')
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')

    scores_list = scores['score'].values.tolist()
    labels_list = metadata['key'].values.tolist()

    bonafide_scores = []
    spoof_scores = []

    for i, label in enumerate(labels_list):
        if label == 'bonafide':
            bonafide_scores.append(scores_list[i])
        else:
            spoof_scores.append(scores_list[i])

    avg_bonafide = sum(bonafide_scores) / len(bonafide_scores)
    avg_spoof = sum(spoof_scores) / len(spoof_scores)
    print('Avg. bonafide score:', avg_bonafide)
    print('Avg. spoof score:', avg_spoof)

    return 'Midpoint: {0}'.format((avg_bonafide + avg_spoof) / 2)


def get_rand_idx(data_size):
    return random.randint(0, data_size - 1)


def get_rand_subset_size(feature_indices):
    return random.randint(1, len(feature_indices) - 1)


def get_rand_subset(features, rand_subset_size, window):
    features_minus_window = [f for f in features if f != window]
    return sorted(random.sample(features_minus_window, rand_subset_size))


def replace(n, to_tensor, rand_inst):
    if n == 0:
        to_tensor[:12920] = rand_inst[:12920]
    if n == 1:
        to_tensor[12920:25840] = rand_inst[12920:25840]
    if n == 2:
        to_tensor[25840:38760] = rand_inst[25840:38760]
    if n == 3:
        to_tensor[38760:51680] = rand_inst[38760:51680]
    if n == 4:
        to_tensor[51680:64600] = rand_inst[51680:64600]

    return to_tensor


def confidence_level(model_output):
    """
        PROBABILITY OF BELONGING TO POSITIVE CLASS
        Classifier outputs a LLR score
        LLR = log(Pos. Likelihood Ratio - Neg. Likelihood ratio)
    """

    return 1 / (1 + math.exp(-model_output))


def evaluate_threshold(threshold):
    """
        @threshold : int, threshold for LLR score
        @scores : list, scores from the classifier
        @labels : list, ground truth labels

        PRECISION-RECALL CURVE FOR THE GIVEN THRESHOLD
        Classifier outputs a soft classification (LLR)
    """

    scores = pd.read_csv('score.txt', delimiter='\s+')
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')

    predictions = []
    scores_list = scores['score'].values.tolist()
    labels_list = metadata['key'].values.tolist()

    for score in scores_list:
        if score > threshold:
            predictions.append('bonafide')
        else:
            predictions.append('spoof')

    print('confusion matrix:', confusion_matrix(labels_list, predictions))
    print('f1 score for threshold {0} is: {1}'.format(threshold,
                                                      f1_score(labels_list, predictions, pos_label='spoof')))


def horizontal_plot(windows, values):
    """
        @windows : list, of windows (0..5)
        @values : list, shapley values for given windows

        @return : graph, horizontal bar chart (red = neg, blue = pos)
    """

    fig, ax = plt.subplots()
    bar_list = ax.barh(windows, values, height=0.3)
    for index, val in enumerate(values):
        if val < 0:
            bar_list[index].set_color('r')

    plt.ylabel('Window (seconds)')
    plt.xlabel('Shapley value')
    plt.title('Horizontal plot')
    plt.show()


def plot_waveform(audio, values):
    """
        @audio : torch.Tensor, tensor representation of the audio
        @values : list, shapley values for given windows

        @return : graph, waveform diagram with heatmap for the most important section
    """

    audio = audio.numpy()
    time = 64600 / 16000  # 16Hz sampling rate - Each audio clip has 64600 data points
    x = np.linspace(0, time, num=len(audio))

    significant_val = max(values)
    index = values.index(significant_val)
    end_slice = int((index + 1) * (64600 / 5))
    begin_slice = end_slice - 12920

    fig, ax = plt.subplots()
    plt.plot(x[begin_slice:end_slice], audio[begin_slice:end_slice], color='r')
    plt.plot(x[:begin_slice], audio[:begin_slice])
    plt.plot(x[end_slice:], audio[end_slice:])
    plt.xlabel('time')
    plt.show()


class Explainer:
    """
    @model : nn.Module, pre-trained pytorch model
    @data_set : torch.utils.Dataset, the test set

    Provides post-hoc explanation for a pytorch model

    Splits the audio clip into 5 time windows
    Evaluating each window's impact on the classifier's decision
    """

    def __init__(self, model, data_set):
        pass
        # self.model = model.eval()
        # self.tensors = data_set

    def get_size(self):
        data_set_size = self.tensors.numpy()
        data_set_size = data_set_size.shape
        data_set_size = data_set_size[0]

        return data_set_size

    # Replaces feature n with values from random instance

    def shap_values(self, no_of_iterations, window, data_point):
        """
            @no_of_iterations : int, recommended to be between 100 - 1000
            @window : int, no. between 0 and 4 (audio clip is divided into 5 windows)
            @data_point : torch.Tensor, the data point in question

            @return : float, Shapley value for given window

            MONTE-CARLO APPROXIMATION OF SHAPLEY VALUES
            1. Pick random instance from the data set
            2. Pick random subset of windows
            3. Construct 2 new data instances
                3.1 The random subset of features are replaced EXCEPT window n
                3.2 A random number of features are replaced WITH window n
            4. Calculate the marginal contribution
                4.1 model(with window) - model(without window)
            5. Repeat n times
            6. Return the mean marginal contribution
        """

        print('************************')
        print('Iterate {0} times Window {1}'.format(no_of_iterations, window))
        feature_indices = [0, 1, 2, 3, 4]
        data_size = self.get_size()
        marginal_contributions = []

        for _ in range(no_of_iterations):
            rand_idx = get_rand_idx(data_size)
            rand_instance = self.tensors[rand_idx]

            rand_subset_size = get_rand_subset_size(feature_indices)
            x_idx = get_rand_subset(feature_indices, rand_subset_size, window)

            x_with_window = deepcopy(data_point)
            x_with_window = x_with_window.numpy()
            x_without_window = deepcopy(data_point)
            x_without_window = x_without_window.numpy()

            for x in x_idx:
                x_with_window = replace(x, x_with_window, rand_instance)
                x_without_window = replace(x, x_without_window, rand_instance)
            x_without_window = replace(window, x_without_window, rand_instance)

            x_with_window = np.array([x_with_window])
            x_without_window = np.array([x_without_window])
            x_with_window = torch.from_numpy(x_with_window)
            x_without_window = torch.from_numpy(x_without_window)

            pred_1 = self.model(x_with_window)[0][1].item()  # snd dim of softmax is used as LLR
            pred_2 = self.model(x_without_window)[0][1].item()

            marginal_contribution = pred_1 - pred_2
            marginal_contributions.append(marginal_contribution)

        shap_val = sum(marginal_contributions) / len(marginal_contributions)

        return 'Approx. Shapley value for window {0} is {1}'.format(window, shap_val)

    def shap_values_exact(self, window, data_point):
        """
            @window : int, feature in question
            @data_point : torch.Tensor, data point in question

            @return : float, exact shapley value

            More computationally expensive hence not used in practice - O(2^n)
            Used for testing purposes to evaluate the accuracy of approx.
        """

        print('************************')
        print('Calculating exact Shapley value for Window {0}'.format(window))
        data_point.numpy()
        windows = [0, 1, 2, 3, 4]
        windows.remove(window)
        powerset = list(chain.from_iterable(combinations(windows, r) for r in range(len(windows) + 1)))
        N = 5
        out = []

        for set in powerset:
            set = list(set)
            # print('Current set', set)
            S = len(set)
            avg_factor = (math.factorial(S) * math.factorial(N - S - 1)) / math.factorial(N)
            # print('Averaging factor', avg_factor)

            without_i = np.zeros(64600)  # zeros is eqv. to not having the feature
            for win in set:
                replace(win, without_i, data_point)

            with_i = np.zeros(64600)  # zeros is eqv. to not having the feature
            bisect.insort(set, window)
            # print('Set with window', set)
            for win in set:
                replace(win, with_i, data_point)

            # print('Are equal', np.array_equal(with_i, without_i))
            with_i = np.array([with_i])
            without_i = np.array([without_i])
            with_i = torch.from_numpy(with_i).float()
            without_i = torch.from_numpy(without_i).float()

            val1 = self.model(with_i)[0][1].item()
            val2 = self.model(without_i)[0][1].item()
            # print('Value with_i is {0} and without_i is {1}'.format(val1, val2))

            out.append(avg_factor * (val1 - val2))

        return 'Exact Shapley value for window {0} is {1}'.format(window, sum(out))

    def test_efficiency(self, data_point, data_set):
        """
        @data_point : torch.Tensor
        @data_set : torch.Tensor

        Shapley values for a given data point should sum to
        the difference between the:
            1. Model prediction for said data point
            2. Average model prediction for the data set

        This is testing the EFFICIENT property of shapley values
        """
        data_point = data_point.numpy()
        data_point = np.array([data_point])
        data_point = torch.from_numpy(data_point)
        point = self.model(data_point)[0, 1]
        batch = self.model(data_set)[:, 1]
        print('Prediction for point is:', point)
        print('Prediction for batch is:', sum(batch))
        avg = sum(batch) / len(batch)

        return (point - avg).item()

    def library_shap_values(self, test_data, test_point):
        """
        @test_data : np.array, a subset of the test set

        @return : array, Shapley values
        """

        f = lambda x: self.model(torch.from_numpy(x)).detach().numpy()[:, 1]
        # explainer = shap.KernelExplainer(f, test_data)
        # vals = explainer.shap_values(test_point)
        # print(vals)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Missing the trained model...')

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # model
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    trained_model = sys.argv[1]
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
    print('Model loaded : {}'.format(trained_model))

    # dataset
    file_eval = genSpoof_list(dir_meta='./ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt', is_train=False,
                              is_eval=True)
    print('No. of eval trials:', len(file_eval))
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='./ASVspoof2021_DF_eval/')

    data_loader = DataLoader(eval_set, batch_size=128, shuffle=False, drop_last=False)
    for batch_x, utt_id in data_loader:
        example_clip = 0
        example_point = batch_x[example_clip]
        clip_id = utt_id[example_clip]
        print('Clip ID:', clip_id)

    # instantiate explainer
    shap_explainer = Explainer(model, batch_x)
