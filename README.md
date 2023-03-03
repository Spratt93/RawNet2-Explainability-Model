# Explainability model for Rawnet

Designed to work with the [Rawnet baseline system](https://github.com/asvspoof-challenge/2021)
from the ASVspoof challenge 2021. It is a black-box model that detects whether
audio clips are real or fake - where *fake* refers to a 'deepfake' audio clip.

This is an application specific solution, splitting the audio clips into
5 *windows*. Each time window's Shapley value is evaluated, determining which
segment of the audio has the biggest effect on the classifiers decision.

To use you must have a trained Rawnet model - *trained_model.pth*

To run:
```
$ python3.6 -m venv .explainer
$ source .explainer/bin/activate
$ pip install -r requirements.txt
$ python explainer.py pre_trained_DF_RawNet2.pth
```

*This uses code from the Rawnet Baseline system - please don't sue me lol*
