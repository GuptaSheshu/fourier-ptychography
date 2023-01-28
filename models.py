from torch import nn
import torch

class DenseBlock(nn.Module):
	'''
		Defines a fully connected block of the DenseNet architectture
		Inputs to constructor:
			Number of input filters
			Growth rate
			Number of layers
			Filter size
	'''
	def __init__(self, input_filters, growth_rate, num_layers = 3, kernel_size = 3, ConvLayer = nn.Conv2d):
		super(DenseBlock,self).__init__()
		self.num_layers = num_layers
		self.ConvLayer = ConvLayer
		self.conv_layers = nn.ModuleList()
		self.batchnorm_layers = nn.ModuleList()
		filters = input_filters
		for i in range(num_layers):
			self.conv_layers.append(self.ConvLayer(filters, growth_rate, (kernel_size, kernel_size), padding = kernel_size/2))
			self.batchnorm_layers.append(nn.BatchNorm2d(growth_rate))
			filters += growth_rate

	def forward(self,x):
		int_x = []
		int_x = x
		for i in range(self.num_layers):
			h = self.conv_layers[i](int_x)
			h = self.batchnorm_layers[i](h)
			int_x = torch.cat([int_x,h],1)
		y = int_x
		return y

class SkipBlock(nn.Module):
	'''
		Defines a fully connected block with a skip connection from input to output
		Inputs to constructor:
			Number of input filters
			Number of layers
			Filter size
	'''
	def __init__(self, input_filters, num_layers = 3, kernel_size = 3, ConvLayer = nn.Conv2d):
		super(SkipBlock,self).__init__()
		self.num_layers = num_layers
		self.ConvLayer = ConvLayer
		self.conv_layers = nn.ModuleList()
		self.batchnorm_layers = nn.ModuleList()
		for i in range(num_layers):
			self.conv_layers.append(self.ConvLayer(input_filters, input_filters, (kernel_size, kernel_size), padding = kernel_size/2))
			self.batchnorm_layers.append(nn.BatchNorm2d(input_filters))

	def forward(self,x):
		int_x = x
		for i in range(self.num_layers):
			h = self.conv_layers[i](int_x)
			h = self.batchnorm_layers[i](h)
			int_x = h
		y = torch.cat([int_x,x],1)
		return y

class ResidualBlock(nn.Module):
	'''
		Defines a fully connected block with a skip connection from input to output
		Inputs to constructor:
			Number of input filters
			Number of layers
			Filter size
	'''
	def __init__(self, input_filters, num_layers = 3, kernel_size = 3, ConvLayer = nn.Conv2d):
		super(SkipBlock,self).__init__()
		self.num_layers = num_layers
		self.ConvLayer = ConvLayer
		self.conv_layers = nn.ModuleList()
		self.batchnorm_layers = nn.ModuleList()
		for i in range(num_layers):
			self.conv_layers.append(self.ConvLayer(input_filters, input_filters, (kernel_size, kernel_size), padding = kernel_size/2))
			self.batchnorm_layers.append(nn.BatchNorm2d(input_filters))

	def forward(self,x):
		int_x = x
		for i in range(self.num_layers):
			h = self.conv_layers[i](int_x)
			h = self.batchnorm_layers[i](h)
			int_x = h
		y = x + int_x
		return y

class UNet(nn.Module):
	'''
		Defines a Generator which takes a stack as input and generates the objects magnitude and phase distributions.
		Input shape: (N,25,256,256)
		Output shape: (N,2,256,256)

		Note: Output activation is not applied to allow reusing the same class.
	'''
	def __init__(self,kernel_size = 3,input_channels=25):
		super(UNet,self).__init__()
		self.enc_batch1 = nn.BatchNorm2d(input_channels)
		self.conv1 = nn.Conv2d(input_channels, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch2 = nn.BatchNorm2d(32)
		self.pool1 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv2 = nn.Conv2d(32, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch3 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv3 = nn.Conv2d(64, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch4 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv4 = nn.Conv2d(128, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch5 = nn.BatchNorm2d(256)
		self.pool4 = nn.AvgPool2d((2,2), stride = (2,2))

		self.convtrans1 = nn.ConvTranspose2d(256, 512, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch1 = nn.BatchNorm2d(512)
		self.convtrans2 = nn.ConvTranspose2d(512 + 256, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch2 = nn.BatchNorm2d(256)
		self.convtrans3 = nn.ConvTranspose2d(256 + 128, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch3 = nn.BatchNorm2d(128)
		self.convtrans4 = nn.ConvTranspose2d(128 + 64, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch4 = nn.BatchNorm2d(64)
		self.convtrans5 = nn.ConvTranspose2d(64 + 32, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch5 = nn.BatchNorm2d(32)

		self.convtrans6 = nn.ConvTranspose2d(32, 2, (kernel_size,kernel_size), padding = kernel_size/2)

	def forward(self,x):
		h = self.enc_batch1(x)
		c1 = nn.ReLU()(self.conv1(h))
		c1 = self.enc_batch2(c1)
		h = self.pool1(c1)
		c2 = nn.ReLU()(self.conv2(h))
		c2 = self.enc_batch3(c2)
		h = self.pool2(c2)
		c3 = nn.ReLU()(self.conv3(h))
		c3 = self.enc_batch4(c3)
		h = self.pool3(c3)
		c4 = nn.ReLU()(self.conv4(h))
		c4 = self.enc_batch5(c4)
		h = self.pool4(c4)

		z = nn.ReLU()(self.convtrans1(h))
		z = self.dec_batch1(z)
		h = nn.functional.interpolate(z,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c4,h],1)
		h = nn.ReLU()(self.convtrans2(h))
		h = self.dec_batch2(h)
		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c3,h],1)
		h = nn.ReLU()(self.convtrans3(h))
		h = self.dec_batch3(h)
		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c2,h],1)
		h = nn.ReLU()(self.convtrans4(h))
		h = self.dec_batch4(h)
		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c1,h],1)
		h = nn.ReLU()(self.convtrans5(h))
		h = self.dec_batch5(h)

		y = self.convtrans6(h)

		return y

class UDenseNet(nn.Module):
	'''
		Defines a UNet where the UNet blocks are replaced by fully connected DenseNet blocks.
	'''
	def __init__(self, kernel_size = 3, input_channels = 25):
		super(UDenseNet,self).__init__()
		self.input_batch = nn.BatchNorm2d(input_channels)
		
		self.enc_dense_blocks = nn.ModuleList([])
		enc_dense1 = DenseBlock(input_channels, growth_rate = input_channels, ConvLayer = nn.Conv2d)
		self.conv1 = nn.Conv2d(input_channels*4, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch1 = nn.BatchNorm2d(32)
		self.pool1 = nn.AvgPool2d((2,2), stride = (2,2))

		enc_dense2 = DenseBlock(32, growth_rate = 32, ConvLayer = nn.Conv2d)
		self.conv2 = nn.Conv2d(32*4, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch2 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense3 = DenseBlock(64, growth_rate = 64, ConvLayer = nn.Conv2d)
		self.conv3 = nn.Conv2d(64*4, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch3 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense4 = DenseBlock(128, growth_rate = 128, ConvLayer = nn.Conv2d)
		self.conv4 = nn.Conv2d(128*4, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch4 = nn.BatchNorm2d(256)
		self.pool4 = nn.AvgPool2d((2,2), stride = (2,2))
		
		self.enc_dense_blocks = nn.ModuleList([enc_dense1, enc_dense2, enc_dense3, enc_dense4])

		self.convtrans1 = nn.ConvTranspose2d(256, 512, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch1 = nn.BatchNorm2d(512)

		self.convtrans2 = nn.ConvTranspose2d(512 + 256, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch2 = nn.BatchNorm2d(256)
		dec_dense1 = DenseBlock(256, growth_rate = 256, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans3 = nn.ConvTranspose2d(256*4 + 128, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch3 = nn.BatchNorm2d(128)
		dec_dense2 = DenseBlock(128, growth_rate = 128, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans4 = nn.ConvTranspose2d(128*4 + 64, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch4 = nn.BatchNorm2d(64)
		dec_dense3 = DenseBlock(64, growth_rate = 64, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans5 = nn.ConvTranspose2d(64*4 + 32, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch5 = nn.BatchNorm2d(32)
		dec_dense4 = DenseBlock(32, growth_rate = 32, ConvLayer = nn.ConvTranspose2d)
		
		self.dec_dense_blocks = nn.ModuleList([dec_dense1, dec_dense2, dec_dense3, dec_dense4])
		self.convtrans6 = nn.ConvTranspose2d(32*4, 2, (kernel_size,kernel_size), padding = kernel_size/2)

	def forward(self, x):

		_x = self.input_batch(x)

		h = self.enc_dense_blocks[0](_x)
		c1 = nn.ReLU()(self.conv1(h))
		c1 = self.enc_batch1(c1)
		h = self.pool1(c1)

		h = self.enc_dense_blocks[1](h)
		c2 = nn.ReLU()(self.conv2(h))
		c2 = self.enc_batch2(c2)
		h = self.pool2(c2)

		h = self.enc_dense_blocks[2](h)
		c3 = nn.ReLU()(self.conv3(h))
		c3 = self.enc_batch3(c3)
		h = self.pool3(c3)

		h = self.enc_dense_blocks[3](h)
		c4 = nn.ReLU()(self.conv4(h))
		c4 = self.enc_batch4(c4)
		h = self.pool4(c4)

		z = nn.ReLU()(self.convtrans1(h))
		z = self.dec_batch1(z)

		h = nn.functional.interpolate(z,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c4,h],1)
		h = nn.ReLU()(self.convtrans2(h))
		h = self.dec_batch2(h)
		h = self.dec_dense_blocks[0](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c3,h],1)
		h = nn.ReLU()(self.convtrans3(h))
		h = self.dec_batch3(h)
		h = self.dec_dense_blocks[1](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c2,h],1)
		h = nn.ReLU()(self.convtrans4(h))
		h = self.dec_batch4(h)
		h = self.dec_dense_blocks[2](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c1,h],1)
		h = nn.ReLU()(self.convtrans5(h))
		h = self.dec_batch5(h)
		h = self.dec_dense_blocks[3](h)

		y = self.convtrans6(h)

		return y

class USkipNet(nn.Module):
	'''
		Defines a UNet where the UNet blocks are replaced by fully connected DenseNet blocks.
	'''
	def __init__(self, kernel_size = 3, input_channels = 25):
		super(USkipNet,self).__init__()
		self.input_batch = nn.BatchNorm2d(input_channels)
		
		self.enc_dense_blocks = nn.ModuleList([])
		enc_dense1 = SkipBlock(input_channels, num_layers = 2, ConvLayer = nn.Conv2d)
		self.conv1 = nn.Conv2d(input_channels*2, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch1 = nn.BatchNorm2d(32)
		self.pool1 = nn.AvgPool2d((2,2), stride = (2,2))

		enc_dense2 = SkipBlock(32, num_layers = 2, ConvLayer = nn.Conv2d)
		self.conv2 = nn.Conv2d(32*2, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch2 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense3 = SkipBlock(64, num_layers = 2, ConvLayer = nn.Conv2d)
		self.conv3 = nn.Conv2d(64*2, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch3 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense4 = SkipBlock(128, num_layers = 2, ConvLayer = nn.Conv2d)
		self.conv4 = nn.Conv2d(128*2, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch4 = nn.BatchNorm2d(256)
		self.pool4 = nn.AvgPool2d((2,2), stride = (2,2))
		
		self.enc_dense_blocks = nn.ModuleList([enc_dense1, enc_dense2, enc_dense3, enc_dense4])

		self.convtrans1 = nn.ConvTranspose2d(256, 512, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch1 = nn.BatchNorm2d(512)

		self.convtrans2 = nn.ConvTranspose2d(512 + 256, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch2 = nn.BatchNorm2d(256)
		dec_dense1 = SkipBlock(256, num_layers = 2, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans3 = nn.ConvTranspose2d(256*2 + 128, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch3 = nn.BatchNorm2d(128)
		dec_dense2 = SkipBlock(128, num_layers = 2, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans4 = nn.ConvTranspose2d(128*2 + 64, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch4 = nn.BatchNorm2d(64)
		dec_dense3 = SkipBlock(64, num_layers = 2, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans5 = nn.ConvTranspose2d(64*2 + 32, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch5 = nn.BatchNorm2d(32)
		dec_dense4 = SkipBlock(32, num_layers = 2, ConvLayer = nn.ConvTranspose2d)
		
		self.dec_dense_blocks = nn.ModuleList([dec_dense1, dec_dense2, dec_dense3, dec_dense4])
		self.convtrans6 = nn.ConvTranspose2d(32*2, 2, (kernel_size,kernel_size), padding = kernel_size/2)

	def forward(self, x):

		_x = self.input_batch(x)

		h = self.enc_dense_blocks[0](_x)
		c1 = nn.ReLU()(self.conv1(h))
		c1 = self.enc_batch1(c1)
		h = self.pool1(c1)

		h = self.enc_dense_blocks[1](h)
		c2 = nn.ReLU()(self.conv2(h))
		c2 = self.enc_batch2(c2)
		h = self.pool2(c2)

		h = self.enc_dense_blocks[2](h)
		c3 = nn.ReLU()(self.conv3(h))
		c3 = self.enc_batch3(c3)
		h = self.pool3(c3)

		h = self.enc_dense_blocks[3](h)
		c4 = nn.ReLU()(self.conv4(h))
		c4 = self.enc_batch4(c4)
		h = self.pool4(c4)

		z = nn.ReLU()(self.convtrans1(h))
		z = self.dec_batch1(z)

		h = nn.functional.interpolate(z,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c4,h],1)
		h = nn.ReLU()(self.convtrans2(h))
		h = self.dec_batch2(h)
		h = self.dec_dense_blocks[0](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c3,h],1)
		h = nn.ReLU()(self.convtrans3(h))
		h = self.dec_batch3(h)
		h = self.dec_dense_blocks[1](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c2,h],1)
		h = nn.ReLU()(self.convtrans4(h))
		h = self.dec_batch4(h)
		h = self.dec_dense_blocks[2](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c1,h],1)
		h = nn.ReLU()(self.convtrans5(h))
		h = self.dec_batch5(h)
		h = self.dec_dense_blocks[3](h)

		y = self.convtrans6(h)

		return y

class UResNet(nn.Module):
	'''
		Defines a UNet where the UNet blocks are replaced by fully connected DenseNet blocks.
	'''
	def __init__(self, kernel_size = 3, input_channels = 25):
		super(USkipNet,self).__init__()
		self.input_batch = nn.BatchNorm2d(input_channels)
		
		self.enc_dense_blocks = nn.ModuleList([])
		enc_dense1 = ResidualBlock(input_channels, ConvLayer = nn.Conv2d)
		self.conv1 = nn.Conv2d(input_channels, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch1 = nn.BatchNorm2d(32)
		self.pool1 = nn.AvgPool2d((2,2), stride = (2,2))

		enc_dense2 = ResidualBlock(32, ConvLayer = nn.Conv2d)
		self.conv2 = nn.Conv2d(32, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch2 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense3 = ResidualBlock(64, ConvLayer = nn.Conv2d)
		self.conv3 = nn.Conv2d(64, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch3 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d((2,2), stride = (2,2))
		
		enc_dense4 = ResidualBlock(128, ConvLayer = nn.Conv2d)
		self.conv4 = nn.Conv2d(128, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.enc_batch4 = nn.BatchNorm2d(256)
		self.pool4 = nn.AvgPool2d((2,2), stride = (2,2))
		
		self.enc_dense_blocks = nn.ModuleList([enc_dense1, enc_dense2, enc_dense3, enc_dense4])

		self.convtrans1 = nn.ConvTranspose2d(256, 512, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch1 = nn.BatchNorm2d(512)

		self.convtrans2 = nn.ConvTranspose2d(512 + 256, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch2 = nn.BatchNorm2d(256)
		dec_dense1 = ResidualBlock(256, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans3 = nn.ConvTranspose2d(256 + 128, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch3 = nn.BatchNorm2d(128)
		dec_dense2 = ResidualBlock(128, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans4 = nn.ConvTranspose2d(128 + 64, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch4 = nn.BatchNorm2d(64)
		dec_dense3 = ResidualBlock(64, ConvLayer = nn.ConvTranspose2d)
		
		self.convtrans5 = nn.ConvTranspose2d(64 + 32, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.dec_batch5 = nn.BatchNorm2d(32)
		dec_dense4 = ResidualBlock(32, ConvLayer = nn.ConvTranspose2d)
		
		self.dec_dense_blocks = nn.ModuleList([dec_dense1, dec_dense2, dec_dense3, dec_dense4])
		self.convtrans6 = nn.ConvTranspose2d(32, 2, (kernel_size,kernel_size), padding = kernel_size/2)

	def forward(self, x):

		_x = self.input_batch(x)

		h = self.enc_dense_blocks[0](_x)
		c1 = nn.ReLU()(self.conv1(h))
		c1 = self.enc_batch1(c1)
		h = self.pool1(c1)

		h = self.enc_dense_blocks[1](h)
		c2 = nn.ReLU()(self.conv2(h))
		c2 = self.enc_batch2(c2)
		h = self.pool2(c2)

		h = self.enc_dense_blocks[2](h)
		c3 = nn.ReLU()(self.conv3(h))
		c3 = self.enc_batch3(c3)
		h = self.pool3(c3)

		h = self.enc_dense_blocks[3](h)
		c4 = nn.ReLU()(self.conv4(h))
		c4 = self.enc_batch4(c4)
		h = self.pool4(c4)

		z = nn.ReLU()(self.convtrans1(h))
		z = self.dec_batch1(z)

		h = nn.functional.interpolate(z,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c4,h],1)
		h = nn.ReLU()(self.convtrans2(h))
		h = self.dec_batch2(h)
		h = self.dec_dense_blocks[0](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c3,h],1)
		h = nn.ReLU()(self.convtrans3(h))
		h = self.dec_batch3(h)
		h = self.dec_dense_blocks[1](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c2,h],1)
		h = nn.ReLU()(self.convtrans4(h))
		h = self.dec_batch4(h)
		h = self.dec_dense_blocks[2](h)

		h = nn.functional.interpolate(h,scale_factor = 2, mode = 'bilinear', align_corners = True)
		h = torch.cat([c1,h],1)
		h = nn.ReLU()(self.convtrans5(h))
		h = self.dec_batch5(h)
		h = self.dec_dense_blocks[3](h)

		y = self.convtrans6(h)

		return y

class ConvNet(nn.Module):
	'''
		Defines a Discriminator/Critic, which takes a magnitude and phase distribution and outputs a scalar value function.
		When used with sigmoid activation, this can be interpreted as the probability of the input being real.
		Input shape: (N,2,256,256)
		Output shape: (N,1)

		Note: Output activation is not applied to allow reusing the same class.
	'''

	def __init__(self,kernel_size = 3, output_size = 1):
		super(ConvNet,self).__init__()
		self.conv1 = nn.Conv2d(2, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch1 = nn.BatchNorm2d(32)
		self.pool1 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv2 = nn.Conv2d(32, 32, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch2 = nn.BatchNorm2d(32)
		self.pool2 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv3 = nn.Conv2d(32, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch3 = nn.BatchNorm2d(64)
		self.pool3 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv4 = nn.Conv2d(64, 64, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch4 = nn.BatchNorm2d(64)
		self.pool4 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv5 = nn.Conv2d(64, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch5 = nn.BatchNorm2d(128)
		self.pool5 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv6 = nn.Conv2d(128, 128, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch6 = nn.BatchNorm2d(128)
		self.pool6 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv7 = nn.Conv2d(128, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch7 = nn.BatchNorm2d(256)
		self.pool7 = nn.AvgPool2d((2,2), stride = (2,2))
		self.conv8 = nn.Conv2d(256, 256, (kernel_size,kernel_size), padding = kernel_size/2)
		self.batch8 = nn.BatchNorm2d(256)

		self.dense1 = nn.Linear(1024,output_size)

	def forward(self,input):
		h = nn.ReLU()(self.conv1(input))
		h = self.batch1(h)
		h = self.pool1(h)
		h = nn.ReLU()(self.conv2(h))
		h = self.batch2(h)
		h = self.pool2(h)
		h = nn.ReLU()(self.conv3(h))
		h = self.batch3(h)
		h = self.pool3(h)
		h = nn.ReLU()(self.conv4(h))
		h = self.batch4(h)
		h = self.pool4(h)
		h = nn.ReLU()(self.conv5(h))
		h = self.batch5(h)
		h = self.pool5(h)
		h = nn.ReLU()(self.conv6(h))
		h = self.batch6(h)
		h = self.pool6(h)
		h = nn.ReLU()(self.conv7(h))
		h = self.batch7(h)
		h = self.pool7(h)
		h = nn.ReLU()(self.conv8(h))
		h = self.batch8(h)
		h = h.view(h.size(0),-1)

		y = self.dense1(h)

		return y

class SiameseNetwork(nn.Module):
	'''
		Defines a Siamese Network.
	'''
	def __init__(self, filter_size = 3):
		super(SiameseNetwork,self).__init__()
		self.shared_block = ConvNet(output_size = 128)
		self.add_module('shared_block', self.shared_block)
		self.dense1 = nn.Linear(128,64)
		self.dense2 = nn.Linear(64,16)
		self.dense3 = nn.Linear(16,1)
		self.dense3.weight.data.fill_(0.0)
		self.dense3.bias.data.fill_(0.0)

	def forward(self, x1, x2):
		z1 = nn.ReLU()(self.shared_block(x1))
		z2 = nn.ReLU()(self.shared_block(x2))
		z = (z1 - z2)**2
		h = nn.ReLU()(self.dense1(z))
		h = nn.ReLU()(self.dense2(h))
		y = self.dense3(h)
		return y

class SingleHeadNetwork(nn.Module):
	'''
		Defines a Siamese Network.
	'''
	def __init__(self, filter_size = 3):
		super(SingleHeadNetwork,self).__init__()
		self.shared_block = ConvNet(output_size = 128)
		self.add_module('shared_block', self.shared_block)
		self.dense1 = nn.Linear(128,64)
		self.dense2 = nn.Linear(64,16)
		self.dense3 = nn.Linear(16,1)
		self.dense3.weight.data.fill_(0.0)
		self.dense3.bias.data.fill_(0.0)

	def forward(self, x1):
		z1 = nn.ReLU()(self.shared_block(x1))
		z = z1
		h = nn.ReLU()(self.dense1(z))
		h = nn.ReLU()(self.dense2(h))
		y = self.dense3(h)
		return y

class WGANGenerator(UNet):

	'''
	Defines a generator for WGAN which produces output in the range [-1,1]. Derived from UNet with output activation changed
	'''

	def __init__(self, filter_size = 3):
		super(WGANGenerator,self).__init__(filter_size)

	def forward(self, x):
		y = super(WGANGenerator,self).forward(x)
		return nn.Tanh()(y)

class WGANCritic(ConvNet):

	'''
	Defines a critic for WGAN. Derived from ConvNet with no change.
	'''

	def __init__(self, filter_size = 3):
		super(WGANCritic,self).__init__(filter_size)

	def forward(self, x):
		y = super(WGANCritic,self).forward(x)
		return y

class GANGenerator(UNet):

	'''
	Defines a generator for WGAN which produces output in the range [-1,1]. Derived from UNet with output activation changed
	'''

	def __init__(self, filter_size = 3):
		super(GANGenerator,self).__init__(filter_size)

	def forward(self, x):
		y = super(GANGenerator,self).forward(x)
		return nn.ReLU()(y)

class GANCritic(ConvNet):

	'''
	Defines a critic for WGAN. Derived from ConvNet with no change.
	'''

	def __init__(self, filter_size = 3):
		super(GANCritic,self).__init__(filter_size)

	def forward(self, x):
		y = super(GANCritic,self).forward(x)
		return nn.Sigmoid()(y)

class BeGANGenerator(UNet):

	def __init__(self, filter_size = 3,input_channels=25):
		super(BeGANGenerator,self).__init__(filter_size,input_channels)

	def forward(self, x):
		y = super(BeGANGenerator,self).forward(x)
		return nn.Tanh()(y)

class BeGANDescriminator(UNet):

	def __init__(self,filter_size = 3,input_channels=2):
		super(BeGANDescriminator,self).__init__(filter_size,input_channels)

	def forward(self,x):
		y = super(BeGANDescriminator,self).forward(x)
		return nn.Tanh()(y)

class BeGANSiameseCritic(SiameseNetwork):

	def __init__(self,filter_size = 3):
		super(BeGANSiameseCritic,self).__init__(filter_size)

	def forward(self,x1, x2):
		y = super(BeGANSiameseCritic,self).forward(x1, x2)
		return nn.Sigmoid()(y)

class BeGANCritic(SingleHeadNetwork):

	def __init__(self,filter_size = 3):
		super(BeGANCritic,self).__init__(filter_size)

	def forward(self,x1):
		y = super(BeGANCritic,self).forward(x1)
		return nn.Sigmoid()(y)

class DenseBeGANGenerator(UDenseNet):

	def __init__(self, filter_size = 3):
		super(DenseBeGANGenerator,self).__init__(filter_size,25)

	def forward(self, x):
		y = super(DenseBeGANGenerator,self).forward(x)
		return nn.Tanh()(y)

class DenseBeGANDescriminator(UDenseNet):

	def __init__(self,filter_size = 3,input_channels=2):
		super(DenseBeGANDescriminator,self).__init__(filter_size,input_channels)

	def forward(self,x):
		y = super(DenseBeGANDescriminator,self).forward(x)
		return nn.Tanh()(y)

class SkipBeGANGenerator(USkipNet):

	def __init__(self, filter_size = 3):
		super(SkipBeGANGenerator,self).__init__(filter_size,25)

	def forward(self, x):
		y = super(SkipBeGANGenerator,self).forward(x)
		return nn.Tanh()(y)

class SkipBeGANDescriminator(USkipNet):

	def __init__(self,filter_size = 3,input_channels=2):
		super(SkipBeGANDescriminator,self).__init__(filter_size,2)

	def forward(self,x):
		y = super(SkipBeGANDescriminator,self).forward(x)
		return nn.Tanh()(y)

class ResBeGANGenerator(USkipNet):

	def __init__(self, filter_size = 3):
		super(ResBeGANGenerator,self).__init__(filter_size,25)

	def forward(self, x):
		y = super(ResBeGANGenerator,self).forward(x)
		return nn.Tanh()(y)

class ResBeGANDescriminator(USkipNet):

	def __init__(self,filter_size = 3,input_channels=2):
		super(ResBeGANDescriminator,self).__init__(filter_size,2)

	def forward(self,x):
		y = super(ResBeGANDescriminator,self).forward(x)
		return nn.Tanh()(y)
