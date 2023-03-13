import logging

import yaml
import torch
from torch.utils.data import DataLoader

from data_utils import Dataset_ASVspoof2021_eval, genSpoof_list
from model import RawNet

def load_model(model_path):
    """
        Returns an instance of the model and the loaded dataset
        @model : torch.nn.Module, Instance of Rawnet model

        @return : triple, batch and labels
    """
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: {}'.format(device))

    # model
    dir_yaml = './model_config_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    model = RawNet(parser1['model'], device)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info('Model loaded : {}'.format(model_path))

    # dataset
    file_eval = genSpoof_list(dir_meta='./ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt', is_train=False,
                              is_eval=True)
    logging.info('No. of eval trials: {}'.format(len(file_eval)))
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='./ASVspoof2021_DF_eval/')

    data_loader = DataLoader(eval_set, batch_size=32, shuffle=False, drop_last=False)
    batch_x, utt_id = next(iter(data_loader))

    return model, batch_x, utt_id, device