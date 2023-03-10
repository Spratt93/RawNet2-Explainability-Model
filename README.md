# Explainability model for Rawnet2

Designed to work with the [Rawnet2 baseline system](https://github.com/asvspoof-challenge/2021/tree/main/DF/Baseline-RawNet2)
from the ASVspoof challenge 2021. It is a black-box model that classifies audio 
clips as real or fake - where *fake* refers to a deepfake.

This is an application specific solution, splitting the audio clips into
*5 time windows*. Each window's Shapley value is evaluated, and my solution
highlights the segment of audio on the waveform plot with the largest positive
Shapley value. Therefore it displays to the user the segment of the audio has the
biggest effect on the classifiers decision. Note that this solution only works for
audio clips of ~ 4 secs. Further work would include zero-padding the inputs to
allow the solution to work with all clips in the data set.

To use this model you must have the following files in the same directory:
- *trained_model.pth*
- From part00 of the [dataset](https://zenodo.org/record/4835108#.ZAs4mC-l3S5)
*ASVspoof2021_DF_eval* and *ASVspoof_DF_cm_protocols*
- From the [eval\_package](https://github.com/asvspoof-challenge/2021/tree/main/eval-package)
*score.txt* and *trial_metadata.txt*

To run:
```
$ python3.6 -m venv .explainer
$ source .explainer/bin/activate
$ pip install -r requirements.txt
$ python explainer.py pre_trained_DF_RawNet2.pth
```