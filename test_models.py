from torch import nn
import torch

from models import UNet, UDenseNet, USkipNet

class WGANGenerator(UNet):

	'''
	Defines a generator for WGAN which produces output in the range [0,1]. Derived from Generator_ with output activation changed
	'''

	def __init__(self, model):
		super(WGANGenerator,self).__init__(filter_size)

	def forward(self, x):
		x_ = 2*x - 1
		y_ = super(WGANGenerator,self).forward(x_)
		y_ = nn.Tanh()(y_)
		y = (y_ + 1)*0.5

		return y

class GANGenerator(UNet):

	'''
	Defines a generator for WGAN which produces output in the range [0,1]. Derived from Generator_ with output activation changed
	'''

	def __init__(self, filter_size = 3):
		super(GANGenerator,self).__init__(filter_size)

	def forward(self, x):
		y_ = super(GANGenerator,self).forward(x)
		y_ = nn.ReLU()(y_)
		y = torch.clamp(y_,0,1)
		return y

class BeGANGenerator(UNet):

	def __init__(self, filter_size = 3,input_channel=25):
		super(BeGANGenerator,self).__init__(filter_size,input_channel)

	def forward(self, x):
		x_ = 2*x - 1
		y_ = super(BeGANGenerator,self).forward(x_)
		y_ = nn.Tanh()(y_)
		y_ = (y_+1)*0.5
		return y_

class DenseBeGANGenerator(UDenseNet):

	def __init__(self, filter_size = 3):
		super(DenseBeGANGenerator,self).__init__(filter_size,25)

	def forward(self, x):
		x_ = 2*x - 1
		y_ = super(DenseBeGANGenerator,self).forward(x_)
		y_ = nn.Tanh()(y_)
		y = (y_ + 1)*0.5
		return y

class SkipBeGANGenerator(USkipNet):

	def __init__(self, filter_size = 3):
		super(SkipBeGANGenerator,self).__init__(filter_size,25)

	def forward(self, x):
		x_ = 2*x - 1
		y_ = super(SkipBeGANGenerator,self).forward(x_)
		y_ = nn.Tanh()(y_)
		y = (y_ + 1)*0.5
		return y