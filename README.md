# Explainability model for Rawnet2 :mag_right:

Designed to work with the [Rawnet2 baseline system](https://github.com/asvspoof-challenge/2021/tree/main/DF/Baseline-RawNet2)
from the ASVspoof challenge 2021. It is a black-box model that classifies audio 
clips as real or fake - where *fake* refers to a deepfake.

This is an application specific solution, splitting the audio clips into
*5 time windows*. Each window's Shapley value is evaluated, and my solution
highlights the segment of audio on the waveform plot with the largest positive
Shapley value. Therefore it displays to the user the segment of the audio that has the
largest effect on the classifier's decision. Note that this solution only works for
audio clips of ~ 4 secs. Further work would include zero-padding the inputs to
allow the solution to work with all clips in the data set.

To use this model you must have a pre-trained Rawnet2 model *trained_model.pth*

The *ASVspoof2021_eval.zip* is the test set - which is an even split of 
bona fide and spoof data points. To find out more, read *trial_metadata.txt*

To run:
```
$ python3.6 -m venv .explainer
$ source .explainer/bin/activate
$ pip install -r requirements.txt
$ python explainer.py trained_model.pth
```
