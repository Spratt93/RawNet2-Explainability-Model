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

test_file = datetime.now().strftime('test-%H:%M.log')
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


def confidence_level(clip_id, model_output):
    """
        PROBABILITY OF BELONGING TO POSITIVE CLASS
        Classifier outputs a LLR score
        LLR = log(Pos. Likelihood Ratio - Neg. Likelihood ratio)
    """

    p = 1 / (1 + math.exp(-model_output))
    logging.info('Probability of being bona fide for Clip {0} is: {1}'.format(clip_id, p))

    return p


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

        EVALUATES THE OPTIMAL DECISION BOUNDARY FOR HARD CLASSIFICATION
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
            if clip != 'trialID\n':
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

    conf_mat = confusion_matrix(labels_list, predictions) 
    f1 = f1_score(labels_list, predictions, pos_label='spoof')
    logging.info('Confusion matrix for threshold {0}: {1}'.format(threshold, conf_mat))
    logging.info('F1 score for threshold {0}: {1}'.format(threshold, f1))

    return f1


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
    avg = merged['score'].mean()
    logging.info('Average score for test set is: {}'.format(avg))

    return avg


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
    plt.close()  # prevent graph glitching


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
    logging.info('Shapley values: {}'.format(values))
    logging.info('Largest Shapley value: {}'.format(significant_val))
    index = values.index(significant_val)
    end_slice = round((index + 1) * (duration / 5))
    begin_slice = round(end_slice - (duration / 5))
    logging.info('Begin slice: {0} End Slice: {1}'.format(begin_slice, end_slice))
    begin_slice = begin_slice * (1/16000)
    end_slice = end_slice * (1/16000)

    plt.plot(x, audio, color='b')
    plt.axvspan(begin_slice, end_slice, color='r', alpha=0.5)
    plt.xlabel('time')
    plt.savefig('./plots/{}.png'.format(name))
    plt.close()  # prevent graph glitching


# TODO
def plot_verbal(name, values):
    """
        @param name : string, name of audio clip
        @param values: list, Shapley values for the given clip
    """
    pass

# TODO
def plot_3D():
    pass


class Explainer:
    """
        @param model : torch.nn.Module, pre-trained pytorch model
        @param data_set : torch.utils.Dataset, the test set
        @param labels : string, the data point ID

        Provides post-hoc explanation for a pytorch model

        Splits the audio clip into 5 time windows
        Evaluating each window's impact on the classifier's decision
    """

    def __init__(self, model, data_set, labels):
        self.model = model.eval()  # sets the model to the 'eval' mode
        self.data_set = data_set
        self.labels = labels

    def get_size(self):
        data_set_size = self.data_set.numpy()
        data_set_size = data_set_size.shape
        data_set_size = data_set_size[0]

        return data_set_size

    def shap_values(self, n, window, data_point, device):
        """
            @param n : int, no. of iterations - recommended to be between 100 - 1000
            @param window : int, no. between 0 and 4 (audio clip is divided into 5 windows)
            @param data_point : torch.Tensor, the data point in question
            @param device : torch.device, if cuda gpu available it improves speed of the model

            @return : float, Shapley value for given window

            Monte-Carlo approximation of Shapley values
        """

        feature_indices = [0, 1, 2, 3, 4]
        data_size = self.get_size()
        marginal_contributions = []

        for _ in range(n):
            rand_idx = get_rand_idx(data_size)
            rand_instance = self.data_set[rand_idx]
            rand_subset_size = get_rand_subset_size(feature_indices)
            x_idx = get_rand_subset(feature_indices, rand_subset_size, [window])
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

        return shap_val


    def create_prediction_set(self):
        """
            Return a text file with the hard classification for each clip
            Compatible with pandas
            The threshold I am using is -3.5
        """

        batch_in = self.data_set.to(device)
        batch_out = self.model(batch_in)
        scores = batch_out[:, 1].cpu().detach().numpy()

        with open('prediction_set.txt', 'w') as f:
            f.write('trialID key\n')
            for i, s in enumerate(scores):
                if s > -3.5:
                    f.write('{0} {1}\n'.format(self.labels[i], 'bonafide'))
                else:
                    f.write('{0} {1}\n'.format(self.labels[i], 'spoof'))


    def efficiency_error(self, n, device):
        """
            @param n : int, no. of iterations for Monte Carlo approx.
            @param device : torch.Device, Supports CUDA optimisation

            @return : float, Average percentage error for the efficient property of Shapley values

            The efficient property simply means that the prediction is divided
            amongst the features
        """
        batch_in = self.data_set.to(device)
        batch_out = self.model(batch_in)  # returns a 60 dimensional tensor
        scores = batch_out[:, 1].cpu().detach().numpy()  # isolate the LLRs

        avg = scores.mean()
        logging.info('Average LLR is: {}'.format(avg))
        sums = []
        preds = []

        for i, s in enumerate(scores):
            preds.append(s)
            vals = [abs(self.shap_values(n, w, self.data_set[i], device)) for w in range(5)]
            sums.append(sum(vals))

        diffs = []
        for i, s in enumerate(sums):
            l = self.labels[i]
            p = preds[i]
            e = abs(p - avg)  # expected value
            d = s - e
            diff = abs(d / e) * 100  # percentage error
            diffs.append(diff)
            logging.info('Approx. value for Clip {0} is {1}'.format(l, s))
            logging.info('Model prediction for Clip {0} is {1}'.format(l, p))
            logging.info('Expected value for Clip {0} is {1}'.format(l, e))
            logging.info('Percentage error for Clip {0} is {1}%'.format(l, diff))

        avg_err = sum(diffs) / len(diffs)
        logging.info('Average percentage error for {0} iterations: {1}%'.format(n, avg_err))

        return avg_err


    def symmetry_error(self, n, data_point, vals, device):
        """
            @param n : int, no of iterations for approximation
            @param data_point : torch.Tensor, point in question
            @param vals : list, Shapley values for given data point
            @param device : for CUDA optimisation

            @return : float, percentage error for symmetry property of Shapley values

            Reverse engineering the symmetry property
            If a given data point has 2 Shapley values that are similar ( +-0.1 )
            Then the expected contribution of those 2 windows should be similar
        """
        
        window = None
        window1 = None
        for i, v in enumerate(vals):
            for i1, v1 in enumerate(vals):
                if abs(v - v1) < 0.1 and i != i1:
                    window = i
                    window1 = i1
        if not window:
            logging.info('There were no Shapley values similar enough to test symmetry...')
            return 

        errors = []
        windows = [0,1,2,3,4]
        windows.remove(window)
        windows.remove(window1)
        powerset = list(chain.from_iterable(combinations(windows, r) for r in range(len(windows) + 1)))
        for s in powerset:
            # model pred. with j
            # model pred. with k

            # insert j into s
            p1 = self.model(x)[0][1].item()
            p2 = self.model(y)[0][1].item()

            return p1 - p2
            
        
    def dummy_test(self, vals):
        """
            @param 

            Reverse engineer the dummy property
            Look for Shapley values that are zero
            If so should have a marginal contribution of 0
        """

        zero_val = None
        for v in vals:
            if abs(v) < 0.1:
                zero_val = v

        if not zero_val:
            logging.info('No Shapley value that is close enough to zero...')
            return

        # do the with and without j computation
        feature_indices = [0, 1, 2, 3, 4]
        data_size = self.get_size()
        marginal_contributions = []

        for _ in range(no_of_iterations):
            rand_idx = get_rand_idx(data_size)
            rand_instance = self.data_set[rand_idx]

            rand_subset_size = get_rand_subset_size(feature_indices)
            x_idx = get_rand_subset(feature_indices, rand_subset_size, [window])

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

        return shap_val

        


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise TypeError('Trained model not provided...')

    model, batch, labels, device = load_model(sys.argv[1])

    shap_explainer = Explainer(model, batch, labels)

    shap_explainer.create_prediction_set()

    # shap_explainer.efficiency_error(1000, labels, device)
