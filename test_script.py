import numpy as np

import pickle
from torch import nn
import torch

import skimage.measure

import progressbar

from utils import evaluate_model

print 'U-Net results'
gen_model = evaluate_model('../model/u_net_0811_0050.pkl','../data/25_stacks/','full')

print 'U-Net WGAN results:'
gen_model = evaluate_model('../model/wgan_07_ssim_03_l2_0811_1509.pkl','../data/25_stacks/','full')

print 'U-Net BEGAN results:'
gen_model = evaluate_model('../model/began_siam_1911_1730.pkl','../data/25_natural_stacks/','full')

print 'U-Net FPGAN results:'
gen_model = evaluate_model('../model/began_densenet_siam_2011_0705.pkl','../data/25_stacks/','full')