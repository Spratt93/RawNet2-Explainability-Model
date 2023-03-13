import logging
import math
import random
import sys
from copy import deepcopy
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from setup_model import load_model

test_file = datetime.now().strftime('test_%H:%M.log')
logging.basicConfig(filename=test_file, level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s",
                    datefmt="%I:%M:%S %p")


def get_rand_idx(data_size):
    return random.randint(0, data_size - 1)


def get_rand_subset_size(feature_indices):
    return random.randint(1, len(feature_indices) - 1)


def get_rand_subset(features, rand_subset_size, window):
    features_minus_window = [f for f in features if f != window]
    return sorted(random.sample(features_minus_window, rand_subset_size))


def get_audio_length(file):
    return librosa.get_duration(filename=file)


def replace(n, to_tensor, rand_inst):
    """
        @param n : int, the time window to replace
        @param to_tensor : torch.Tensor, the tensor used in the Shapley value equation
        @param rand_inst : torch.Tensor, the random data point used in the Monte Carlo Approx.

        @return : torch.Tensor, the 
    """
    end_slice = int((n + 1) * 12920)  # audio is normalised to arr of len 64600
    begin_slice = int(end_slice - 12920)
    to_tensor[begin_slice:end_slice] = rand_inst[begin_slice:end_slice]

    return to_tensor


def confidence_level(model_output):
    """
        PROBABILITY OF BELONGING TO POSITIVE CLASS
        Classifier outputs a LLR score
        LLR = log(Pos. Likelihood Ratio - Neg. Likelihood ratio)
    """

    return 1 / (1 + math.exp(-model_output))


def find_threshold():
    """
        FIND MIDPOINT THRESHOLD
        1. For each class
        2. Average the LLR
        3. Find the separator with the max. dist. from both scores
    """

    scores = pd.read_csv('score.txt', delimiter='\s+')
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')
    spoof_scores = []
    bonafide_scores = []

    with open('test_set.txt', 'r') as test_set:
        for clip in test_set:
            clip = clip.rstrip()  # strip trailing \n
            score = scores.loc[scores['trialID'] == clip]['score'].item()
            label = metadata.loc[scores['trialID'] == clip]['key'].item()

            if label == 'bonafide':
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)

    avg_bonafide = sum(bonafide_scores) / len(bonafide_scores)
    avg_spoof = sum(spoof_scores) / len(spoof_scores)
    logging.info('Avg. bonafide score: {}'.format(avg_bonafide))
    logging.info('Avg. spoof score: {}'.format(avg_spoof))

    return 'Midpoint: {0}'.format((avg_bonafide + avg_spoof) / 2)


def evaluate_threshold(threshold):
    """
        @param threshold : int, threshold for LLR score
        @param scores : list, scores from the classifier
        @param labels : list, ground truth labels

        PRECISION-RECALL CURVE FOR THE GIVEN THRESHOLD
        Classifier outputs a soft classification (LLR)
        Note: Best f1 score ~ -1.5 - even tho mid ~ -6.5
    """

    scores = pd.read_csv('score.txt', delimiter='\s+')
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')
    scores_list = []
    labels_list = []
    predictions = []

    with open('test_set.txt', 'r') as test_set:
        for clip in test_set:
            clip = clip.rstrip()  # strip trailing \n
            score = scores.loc[scores['trialID'] == clip]['score'].item()
            label = metadata.loc[scores['trialID'] == clip]['key'].item()
            scores_list.append(score)
            labels_list.append(label)

    for score in scores_list:
        if score > threshold:
            predictions.append('bonafide')
        else:
            predictions.append('spoof')

    logging.info('confusion matrix: {}'.format(
        confusion_matrix(labels_list, predictions)))

    return 'f1 score for threshold {0} is: {1}'.format(threshold,
                                                       f1_score(labels_list, predictions, pos_label='spoof'))


def create_test_set(vocoder_type, n):
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')
    id_list = metadata['trialID'].values.tolist()

    with open('{}.txt'.format(vocoder_type), 'w') as audio_file:
        count = 0
        for i, utt_id in enumerate(id_list):
            if count == n:
                break

            if metadata.iloc[i, 2] == 'low_mp3' and metadata.iloc[i, 8] == '{}'.format(vocoder_type):
                loaded_file = librosa.load(
                    './ASVspoof2021_DF_eval/flac/{}.flac'.format(utt_id), sr=16000)
                (audio, _) = loaded_file
                if 3.85 <= librosa.get_duration(y=audio, sr=16000) <= 4.15:
                    logging.info('Adding Clip: {}'.format(utt_id))
                    audio_file.write(utt_id + '\n')
                    count += 1


def model_prediction_avg(scores, test_set):
    """
        @param scores : pd.DataFrame, list of scores

        @return : the avg. model prediction for all data points in the batch

        Helper function for evaluating the EFFICIENT property of Shapley values
        Only for TEST SET
    """

    merged = pd.merge(scores, test_set, on=['trialID'])

    return merged['score'].mean()


def model_prediction(scores, clip):
    """
        @param scores : pd.DataFrame, list of scores
        @param clip : string, ID of audio clip

        @return : float, model prediction for said clip

        Helper function for testing
    """

    return scores.loc[scores['trialID'] == clip]['score'].item()


def plot_horizontal(name, windows, values):
    """
        @param name : string, name of audio clip
        @param windows : list, of windows (0..5)
        @param values : list, shapley values for given windows

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
    plt.savefig('{}.png'.format(name))


def plot_waveform(name, audio, values):
    """
        @param name : string, name of audio clip
        @param audio : np.array, float array representation of the audio
        @param values : list, Shapley values for given windows

        @return : graph, waveform diagram with most significant window highlighted

        SOLELY CONCERNED WITH THE LARGEST SHAPLEY VALUE
    """

    duration = len(audio)
    time = round(duration / 16000)  # 16KHz sampling rate
    x = np.linspace(0, time, num=duration)

    significant_val = max(values)
    index = values.index(significant_val)
    end_slice = round((index + 1) * (duration / 5))
    begin_slice = round(end_slice - (duration / 5))

    plt.plot(x[begin_slice:end_slice], audio[begin_slice:end_slice], color='r')
    plt.plot(x[:begin_slice], audio[:begin_slice])
    plt.plot(x[end_slice:], audio[end_slice:])
    plt.xlabel('time')
    plt.savefig('{}.png'.format(name))


# TODO
def plot_verbal(name, values):
    """
        @param name : string, name of audio clip
        @param values: list, Shapley values for the given clip
    """
    pass


class Explainer:
    """
        @param model : torch.nn.Module, pre-trained pytorch model
        @param data_set : torch.utils.Dataset, the test set

        Provides post-hoc explanation for a pytorch model

        Splits the audio clip into 5 time windows
        Evaluating each window's impact on the classifier's decision
    """

    def __init__(self, model, data_set):
        self.model = model.eval()  # sets the model to the 'eval' mode
        self.data_set = data_set

    def get_size(self):
        data_set_size = self.data_set.numpy()
        data_set_size = data_set_size.shape
        data_set_size = data_set_size[0]

        return data_set_size

    def shap_values(self, no_of_iterations, window, data_point, device):
        """
            @param no_of_iterations : int, recommended to be between 100 - 1000
            @param window : int, no. between 0 and 4 (audio clip is divided into 5 windows)
            @param data_point : torch.Tensor, the data point in question
            @param device : torch.device, if cuda gpu available it improves speed of the model

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

        # logging.info('Iterate {0} times Window {1}'.format(no_of_iterations, window))
        feature_indices = [0, 1, 2, 3, 4]
        data_size = self.get_size()
        marginal_contributions = []

        for _ in range(no_of_iterations):
            rand_idx = get_rand_idx(data_size)
            rand_instance = self.data_set[rand_idx]

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
            x_with_window = x_with_window.to(device)
            x_without_window = x_without_window.to(device)

            # snd dim of softmax is used as LLR
            pred_1 = self.model(x_with_window)[0][1].item()
            pred_2 = self.model(x_without_window)[0][1].item()

            marginal_contribution = pred_1 - pred_2
            marginal_contributions.append(marginal_contribution)

        shap_val = sum(marginal_contributions) / len(marginal_contributions)
        # logging.info('Shapley value: {}'.format(shap_val))

        return shap_val

    def average_percent_error(self, n, labels, scores, test_set, device):
        """
            @param n : int, no. of iterations for Monte Carlo approx.
            @param labels : list, names of the audio clips in data set
            @param scores : pd.DataFrame, list of scores
            @param test_set : pd.DataFrame, list of test set points
            @param device : torch.Device, to optimise model predictions

            @return : float, Average percentage error for the efficient property of Shapley values
        """
        avg = model_prediction_avg(scores, test_set)
        sums = []
        preds = []

        for i, x in enumerate(self.data_set):
            clip = labels[i]
            pred = model_prediction(scores, clip)
            preds.append(pred)
            vals = [self.shap_values(n, i, x, device) for i in range(5)]
            sums.append(sum(vals))

        diffs = []
        for i, s in enumerate(sums):
            e = (preds[i] - avg) # expected value
            logging.info('Approx. value for Clip {0} is {1}'.format(labels[i], s))
            logging.info('Expected value for Clip {0} is {1}'.format(labels[i], e))
            d = abs(s - e) # difference
            diff = (d / abs(e)) * 100 # percentage error
            diffs.append(diff)
            logging.info('Percentage error for Clip {0} is {1}%'.format(labels[i], diff))

        avg_err = sum(diffs) / len(diffs)
        logging.info('Average percentage error for {0} iterations: {1}%'.format(n, avg_err))

        return avg_err


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise TypeError('Trained model not provided...')

    model, batch, labels, device = load_model(sys.argv[1])

    # explainer
    shap_explainer = Explainer(model, batch)
    scores = pd.read_csv('score.txt', delimiter='\s+')
    test_set = pd.read_csv('test_set.txt', delimiter='\s+')

    # shap_explainer.average_percent_error(1000, labels, scores, test_set, device)
    shap_explainer.average_percent_error(10, labels, scores, test_set, device)
