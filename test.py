import os
import sys
import unittest
import logging

import yaml
import torch
from torch.utils.data import DataLoader

from data_utils import genSpoof_list, Dataset_ASVspoof2021_eval
from explainer import Explainer, model_prediction_avg, model_prediction
from model import RawNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s : %(message)s",
                    datefmt="%I:%M:%S %p", )


class TestShapley(unittest.TestCase):

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: {}'.format(device))

    # model
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
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
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir='./ASVspoof2021_DF_eval/')

    data_loader = DataLoader(eval_set, batch_size=400, shuffle=False, drop_last=False)

    for batch_x, utt_id in data_loader:
        pass

    # explainer
    shap_explainer = Explainer(model, batch_x)


    def test_efficiency(self):
        """
            Test EFFICIENT property of my Shapley value approx.
            For each data point of subset:
                1. Calculate model prediction
                2. Calculate approx. of Shapley values
                3. Verify sum(shap_vals) ~= f(x) - E[f(x)]
            Using 1000 iterations for accuracy
            Margin of error is 0.15
        """

        avg = model_prediction_avg()
        sums = []
        preds = []
        logging.info('Average score: {}'.format(avg))

        for batch_x, utt_id in self.data_loader:
            for i, x in enumerate(batch_x):
                clip = utt_id[i]
                logging.info('Calculating for Clip: {}'.format(clip))
                pred = model_prediction(clip)
                preds.append(pred)
                vals = [self.shap_explainer.shap_values(100, i, x) for i in range(5)]
                sums.append(sum(vals))
                logging.info('Model prediction: {}'.format(pred))

        for i, s in enumerate(sums):
            diff = abs(s - (preds[i] - avg))
            logging.info('Difference is: {}'.format(diff))
            if diff < 0.15:
                is_efficient = True
            else:
                is_efficient = False

            self.assertTrue(is_efficient, 'The difference is too significant: {}'.format(diff))
            

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
