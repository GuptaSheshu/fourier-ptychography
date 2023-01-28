from torch import nn
import torch
from torch import autograd
from torch.optim import RMSprop
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytorch_msssim

plt.ion()

import progressbar
from models import BeGANDescriminator,BeGANGenerator
from utils import data_generator, plot_pair

torch.manual_seed(0)


generator = BeGANGenerator().to("cuda:0")
descriminator = BeGANDescriminator().to("cuda:0")

batch_size = 8
steps = 2384/batch_size
epochs = 100

train_descriminator_gen = data_generator('../data/25_stacks/', mode = 'train', batch_size = batch_size)
train_generator_gen = data_generator('../data/25_stacks/', mode = 'train', batch_size = batch_size)

generator_opt = RMSprop(generator.parameters(), lr = 5e-3, alpha = 0.9)
desciminator_opt = RMSprop(descriminator.parameters(), lr = 5e-3, alpha = 0.9)

widgets=[
    progressbar.ETA(),
    progressbar.Bar(),
    ' ',progressbar.DynamicMessage('DesLoss'),
    ' ',progressbar.DynamicMessage('MSE'),
    ' ',progressbar.DynamicMessage('SSIM'),
    ' ',progressbar.DynamicMessage('boundary_loss'),
    ' ',progressbar.DynamicMessage('k'),
    ' ',progressbar.DynamicMessage('PerformanceMeasure')
]

def compute_disc_loss(outputs_d_x, data, outputs_d_z, gen_z):
    real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
    fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
    return real_loss_d, fake_loss_d
        
def compute_gen_loss(outputs_g_z, gen_z):
    return torch.mean(torch.abs(outputs_g_z - gen_z))


z = 0.0
k = 0.5
lamda = 0.01
gamma = 0.5
alpha = 0.7
measure_history=[]

ssim_funct = pytorch_msssim.SSIM()

for epoch in range(epochs):
	c_loss = 0
	ssim = 0
	mse = 0
	beg = 0
	print "Epoch: {}/{}".format(epoch + 1,epochs)
	with progressbar.ProgressBar(max_value = steps, widgets = widgets) as bar:
		for i in range(steps):
			
			[x,y] = train_descriminator_gen.next()

			x = 2*x - 1
			y = 2*y - 1

			gx = generator.forward(x)
			fgx = descriminator.forward(gx)
			fy = descriminator.forward(y)

			desciminator_opt.zero_grad()

			real_loss_d,fake_loss_d = compute_disc_loss(fy,y,fgx,gx)
			loss = real_loss_d - k * fake_loss_d
			
			loss.backward(retain_graph = True)
			desciminator_opt.step()

			c_loss += loss.item()

			generator_opt.zero_grad()
			
			ssim_value = ssim_funct.forward((y+1)/2.0, (gx+1)/2.0)
			ssim_loss = 1.0 - ssim_value

			mse_loss = ((y - gx)**2).mean(0).mean(1).mean(1)

			gen_loss = alpha*ssim_loss + (1 - alpha)*mse_loss
			
			beg_loss = fake_loss_d
			loss = 1*beg_loss + 1000*gen_loss


			loss.backward()
			generator_opt.step()

			balance = gamma - fake_loss_d.detach()/real_loss_d.detach()
			k += lamda*(balance)
			k =  min(max(0,k),1)
			convg_measure = real_loss_d.item()+torch.abs(balance)
			measure_history.append(convg_measure)

			mse += torch.mean((y - gx)**2).item()
			ssim += ssim_value.item()
			beg += fake_loss_d.item()/real_loss_d.item()
			bar.update(i,DesLoss = c_loss/(i+1), MSE = mse/(i+1), SSIM = ssim/(i+1),boundary_loss=beg/(i+1),k=k,PerformanceMeasure = convg_measure)

x,y = train_generator_gen.next()
x = 2*x - 1
y = 2*y - 1
y = y.cpu().numpy()
pred = generator.forward(x)
pred = pred.cpu().detach().numpy()

plot_pair((y[0,:,:,:] + 1)/2.0, (pred[0,:,:,:] + 1)/2.0)