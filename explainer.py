import logging
import math
import random
import sys
from copy import deepcopy

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2021_eval, genSpoof_list
from model import RawNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s",
                    datefmt="%I:%M:%S %p", )


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
        @threshold : int, threshold for LLR score
        @scores : list, scores from the classifier
        @labels : list, ground truth labels

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

    logging.info('confusion matrix: {}'.format(confusion_matrix(labels_list, predictions)))

    return 'f1 score for threshold {0} is: {1}'.format(threshold,
                                                             f1_score(labels_list, predictions, pos_label='spoof'))


def create_test_set(vocoder_type, n):
    metadata = pd.read_csv('trial_metadata.txt', delimiter='\s+')
    id_list = metadata['trialID'].values.tolist()

    file = open('{}.txt'.format(vocoder_type), 'w')
    count = 0
    for i, utt_id in enumerate(id_list):
        if count == n:
            break

        if metadata.iloc[i, 2] == 'low_mp3' and metadata.iloc[i, 8] == '{}'.format(vocoder_type):
            loaded_file = librosa.load('./ASVspoof2021_DF_eval/flac/{}.flac'.format(utt_id), sr=16000)
            (audio, _) = loaded_file
            if 3.85 <= librosa.get_duration(y=audio, sr=16000) <= 4.15:
                logging.info('Adding Clip: {}'.format(utt_id))
                file.write(utt_id + '\n')
                count += 1

    file.close()


def model_prediction_avg():
    """
        @model : torch.nn.Module, Instance of Rawnet model
        @batch : torch.utils.data.Dataset, Batch of size n

        @return : the avg. model prediction for all data points in the batch
    
        Helper function for evaluating the EFFICIENT property of Shapley values
    """
    
    scores = pd.read_csv('score.txt', delimiter='\s+')

    return scores['score'].mean()


def model_prediction(clip):
    """
        @clip : string, ID of audio clip

        @return : float, model prediction for said clip
    """

    scores = pd.read_csv('score.txt', delimiter='\s+')

    return scores.loc[scores['trialID'] == clip]['score'].item()


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
        @audio : np.array, float array representation of the audio
        @values : list, shapley values for given windows

        @return : graph, waveform diagram with most significant window highlighted
    """

    duration = len(audio)
    time = round(duration / 16000)  # 16KHz sampling rate
    x = np.linspace(0, time, num=duration)

    significant_val = max(values)
    index = values.index(significant_val)
    end_slice = round((index + 1) * (duration / 5))
    begin_slice = round(end_slice - (duration / 5))

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
        self.model = model.eval()  # sets the model to the 'eval' mode
        self.tensors = data_set

    def get_size(self):
        data_set_size = self.tensors.numpy()
        data_set_size = data_set_size.shape
        data_set_size = data_set_size[0]

        return data_set_size

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

        logging.info('Iterate {0} times Window {1}'.format(no_of_iterations, window))
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
        logging.info('Shapley value: {}'.format(shap_val))

        return shap_val


    def efficient(self, values, data_point, average):
        """
        @values : [float], shapley values for data_point
        @data_point : torch.Tensor
        @data_set : torch.Tensor

        @return : pair, should be equal to each other

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

        return sum(values), (point - average).item()

    def test_symmetry(self):
        pass

    def test_dummy(self):
        pass

    def test_additivity(self):
        pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Missing the trained model as an argument...')

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: {}'.format(device))

    # model
    dir_yaml = './model_config_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    trained_model = sys.argv[1]
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
    logging.info('Model loaded : {}'.format(trained_model))

    # dataset
    file_eval = genSpoof_list(dir_meta='./ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt', is_train=False,
                              is_eval=True)
    logging.info('No. of eval trials: {}'.format(len(file_eval)))
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='./ASVspoof2021_DF_eval/')

    data_loader = DataLoader(eval_set, batch_size=400, shuffle=False, drop_last=False)
    for batch_x, utt_id in data_loader:
        pass

    # explainer
    shap_explainer = Explainer(model, batch_x)
