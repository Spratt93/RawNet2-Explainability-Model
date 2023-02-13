import random
import numpy as np
import torch
from torch.utils.data import DataLoader

'''
Provides post-hoc explanation for a pytorch model

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
        return random.randint(1, len(feature_indices))

    def get_rand_subset(self, features, rand_subset_size, window):
        if window in features:
            features.remove(window)
        return sorted(random.sample(features, rand_subset_size))

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

    MONTE-CARLO APPROXIMATION OF SHAPLEY VALUES 
    1. Pick random instance from the data set
    2. Pick random subset of windows
    3. Construct 2 new data instances
        3.1 A random number of features are replaced EXCEPT window n
        3.2 A random number of features are replaced WITH window n
    4. Calculate the marginal contribution
        4.1 model(with window) - model(without window)
    5. Repeat n times
    '''
    def shap_values(self, no_of_iterations, window, data_point):
        print('************************')
        print('Iterate {0} times Window {1}'.format(no_of_iterations, window))
        feature_indices = [0,1,2,3,4]
        marginal_contributions = []

        for _ in range(no_of_iterations):
            data_size = self.get_size()
            rand_idx = self.get_rand_idx(data_size)
            rand_instance = self.tensors[rand_idx]
            print('Random instance index', rand_idx)

            rand_subset_size = self.get_rand_subset_size(feature_indices)
            x_idx = self.get_rand_subset(feature_indices, rand_subset_size, window)
            print('The x indices', x_idx)

            x_with_j = data_point.numpy()
            x_without_j = data_point.numpy()
            for x in x_idx:
                x_with_j = self.replace(x, x_with_j, rand_instance)
                x_without_j = self.replace(x, x_without_j, rand_instance)
            x_without_j = self.replace(window, x_without_j, rand_instance)

            print('x with j', x_with_j)
            print('x without j', x_without_j)

            x_with_j = np.array([x_with_j])
            x_without_j = np.array([x_without_j])
            x_with_j = torch.from_numpy(x_with_j)
            x_without_j = torch.from_numpy(x_without_j)

            pred1 = self.model(x_with_j)[0][1].item() # second dim of softmax is used as LLR
            pred2 = self.model(x_without_j)[0][1].item()
            print('LLR 1:', pred1) 
            print('LLR 2:', pred2)

            marginal_contribution = pred1 - pred2
            marginal_contributions.append(marginal_contribution)
        
        shap_val = sum(marginal_contributions) / len(marginal_contributions)
        print('************************')

        return 'Approximated Shapley value for window {0} is {1}'.format(window, shap_val)