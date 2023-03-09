import sys
import unittest
import logging

from explainer import Explainer, model_prediction_avg, model_prediction
from setup_model import load_model


class TestShapley(unittest.TestCase):

    model, batch, labels = load_model(sys.argv[1])
    shap_explainer = Explainer(model, batch)


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

        for i, x in enumerate(self.batch):
            clip = self.labels[i]
            logging.info('Calculating for Clip: {}'.format(clip))
            pred = model_prediction(clip)
            logging.info('Model prediction: {}'.format(pred))
            preds.append(pred)
            vals = [self.shap_explainer.shap_values(
                1000, i, x) for i in range(5)]
            sums.append(sum(vals))

        for i, s in enumerate(sums):
            diff = abs(s - (preds[i] - avg))
            logging.info('Difference is: {}'.format(diff))
            if diff < 0.15:
                is_efficient = True
            else:
                is_efficient = False

            self.assertTrue(
                is_efficient, 'The difference is too significant: {}'.format(diff))


if __name__ == '__main__':
    with open('testing.log', "w") as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner, argv=[
                      'first-arg-is-ignored'], exit=False)
