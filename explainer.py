import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from copy import deepcopy

'''
Provides post-hoc explanation for a pytorch model

Splits the audio clip into 5 time windows
Evaluating each window's impact on the classifier's decision

@model : nn.Module, pre-trained pytorch model
@data_set : torch.utils.Dataset, the test set
'''
class Explainer:
    def __init__(self, model, data_set):
        self.model = model
        self.data_set = data_set
        data_loader = DataLoader(data_set, batch_size=128, shuffle=False, drop_last=False)
        for tensors,_ in data_loader:
            self.tensors = tensors

    def get_size(self):
        data_set_size = self.tensors.numpy()
        data_set_size = data_set_size.shape
        data_set_size = data_set_size[0]

        return data_set_size

    def get_rand_idx(self, data_size):
        return random.randint(0, data_size-1)

    def get_rand_subset_size(self, feature_indices):
        return random.randint(1, len(feature_indices)-1)

    def get_rand_subset(self, features, rand_subset_size, window):
        features_minus_window = [f for f in features if f != window]
        return sorted(random.sample(features_minus_window, rand_subset_size))

    # Replaces feature n with values from random instance
    def replace(self, n, to_tensor, rand_inst):
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

    '''
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
    '''
    def shap_values(self, no_of_iterations, window, data_point):
        print('************************')
        print('Iterate {0} times Window {1}'.format(no_of_iterations, window))
        feature_indices = [0,1,2,3,4]
        data_size = self.get_size()
        marginal_contributions = []

        for _ in range(no_of_iterations):
            rand_idx = self.get_rand_idx(data_size)
            rand_instance = self.tensors[rand_idx]

            rand_subset_size = self.get_rand_subset_size(feature_indices)
            x_idx = self.get_rand_subset(feature_indices, rand_subset_size, window)

            x_with_window = deepcopy(data_point)
            x_with_window = x_with_window.numpy()
            x_without_window = deepcopy(data_point)
            x_without_window = x_without_window.numpy()

            for x in x_idx:
                x_with_window = self.replace(x, x_with_window, rand_instance)
                x_without_window = self.replace(x, x_without_window, rand_instance)
            x_without_window = self.replace(window, x_without_window, rand_instance)

            x_with_window = np.array([x_with_window])
            x_without_window = np.array([x_without_window])
            x_with_window = torch.from_numpy(x_with_window)
            x_without_window = torch.from_numpy(x_without_window)

            print(self.model(x_with_window))

            pred1 = self.model(x_with_window)[0][1].item() # snd dim of softmax is used as LLR
            pred2 = self.model(x_without_window)[0][1].item()

            marginal_contribution = pred1 - pred2
            marginal_contributions.append(marginal_contribution)
        
        shap_val = sum(marginal_contributions) / len(marginal_contributions)

        return 'Approx. Shapley value for window {0} is {1}'.format(window, shap_val)

    '''
    EXACT SHAPLEY VALUE
    More computationally expensive hence not used in practice
    '''
    def shap_values_exact(self, data_point):
        pass

    '''
    CONFIDENCE LEVEL BASED ON CONFIDENCE INTERVALS
    The classifier outputs a LLR score
    '''
    def confidence_level(self):
        pass

    '''
    @windows : list, of windows (0..5)
    @values : list, shapley values for given windows

    @return : graph, horizontal bar chart (red = neg, blue = pos)
    '''
    def horizontal_plot(self, windows, values):
        fig, ax = plt.subplots()
        bar_list = ax.barh(windows, values, height=0.3)
        for index, val in enumerate(values):
            if val < 0:
                bar_list[index].set_color('r')
                
        plt.ylabel('Window (seconds)')
        plt.xlabel('Shapley value')
        plt.title('Horizontal plot')
        plt.show()

    '''
    @audio : torch.Tensor, tensor representation of the audio
    @values : list, shapley values for given windows

    @return : graph, waveform diagram with heatmap for the most important section
    '''
    def plot_waveform(self, audio, values):
        audio = audio.numpy()
        time = 64600 / 16000 # 16Hz sampling rate - Each audio clip has 64600 data points
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