# Fourier Ptychography

## Getting Started

This is the Pytorch implementation of [Perceptually Driven Conditional GAN for Fourier Ptychography](https://ieeexplore.ieee.org/document/9049029). We propose a deep learning algorithm to perform the synthesized of high resolution image from low resolution image under low spectral overlap between samples, and show a significant improvement in phase reconstruction over existing DL algorithms. This repository also make uses of earlier version of [MS-SSIM](https://github.com/jorge-pessoa/pytorch-msssim). 

## Requirements
- torch
- Numpy 
- cv2
- pickle
- skimage
- progressbar

## File Structure
- **fpm** Folder contains implementation of simulation part of Fourier Ptychographic Microscopy (FPM), more details about the folder are mention in README file inside the folder.
- **pytorch_msssim** Folder contains implementation of MS-SSIM, which I borrowed it from pytorch implementation of [MS-SSIM](https://github.com/jorge-pessoa/pytorch-msssim)
- ``began.py`` Basic BEGAN implementation.
- ``began_siamese.py`` BEGAN which uses Siamese network.
- ``densenet_began_siam.py`` BEGAN which uses densenet and Siamese network.
- ``models.py`` Pytorch model architecture classes.
- ``utils.py`` utility functions.
- ``test_models.py`` and ``test_script.py`` are testing script for different models.
- ``wgan_mse`` WGAN which uses MSE as loss function.
- ``wgan_ssim`` WGAN which uses MSE+SSIM as loss function.

