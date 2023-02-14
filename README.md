# Explainability model for Rawnet

Designed to work with the [Rawnet baseline system](https://github.com/asvspoof-challenge/2021)
from the ASVspoof challenge 2021. It is a black-box model that detects whether
audio clips are real or fake - where *fake* refers to a 'deepfake' audio clip.

This is an application specific solution, splitting the audio clips into
5 *windows*. Each time window's Shapley value is evaluated, determining which
segment of the audio has the biggest effect on the classifiers decision.

To use you have to have a trained Rawnet model e.g *trained_model.pth*

To run:
```
python main.py --track=DF --loss=CCE --is_eval --eval --model_path='pre_trained_DF_RawNet2.pth' --protocols_path='./' --database_path='./' --eval_output='pre_trained_eval_CM_scores.txt'
```
