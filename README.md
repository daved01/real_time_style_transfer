# Real Time Style Transfer
![GitHub](https://img.shields.io/github/license/daved01/real_time_style_transfer)

A program to train models for real-time style transfer, based on the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) by Johnson et al.


## How to use
While there are many files, only three are really important. In the `configuration.yaml` file you can select the paths for the content images, style image, and where the generated models are saved. Make sure these folders exist. Images have to be `.jpg` images and of the same size. The default size is `256x256`, but this can be changed. Additonally, you can configure from which layers features are extracted by the loss function. Make sure you also provide the correct dimensions here.

To train models, run the `train.py` file with the arguments. This will save models to the folder `models`.

Once you have trained models saved you can generate images using the `run.py` file.
