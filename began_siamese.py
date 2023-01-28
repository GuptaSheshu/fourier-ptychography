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
from models import BeGANSiameseCritic,BeGANGenerator
from utils import data_generator, plot_pair

torch.manual_seed(0)
np.random.seed(0)

generator = BeGANGenerator().to("cuda:0")
descriminator = BeGANSiameseCritic().to("cuda:0")

batch_size = 8
steps = 2384/batch_size
epochs = 150

train_descriminator_gen = data_generator('../data/25_natural_stacks/', mode = 'train', batch_size = batch_size)
train_generator_gen = data_generator('../data/25_natural_stacks/', mode = 'train', batch_size = batch_size)

generator_opt = RMSprop(generator.parameters(), lr = 1e-3, alpha = 0.9)
desciminator_opt = RMSprop(descriminator.parameters(), lr = 1e-3, alpha = 0.9)

widgets=[
    progressbar.ETA(),
    progressbar.Bar(),
    ' ',progressbar.DynamicMessage('MSE'),
    ' ',progressbar.DynamicMessage('SSIM'),
    ' ',progressbar.DynamicMessage('RealLoss'),
    ' ',progressbar.DynamicMessage('FakeLoss'),
    ' ',progressbar.DynamicMessage('k')
]

def compute_disc_loss(outputs_d_x, data, outputs_d_z, gen_z):
    real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
    fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
    return real_loss_d, fake_loss_d
        
def compute_gen_loss(outputs_g_z, gen_z):
    return torch.mean(torch.abs(outputs_g_z - gen_z))


z=1.0
z_rec = 0.0
k = 0.0
k_rec = 0.0
lamda = 0.01
lambd2 = 0.0
gamma = 0.5
alpha = 0.5
measure_history=[]

ssim_funct = pytorch_msssim.SSIM()
bce = nn.BCELoss()
ones = torch.ones(batch_size,1).to("cuda:0")
for epoch in range(epochs):
	c_loss = 0
	ssim = 0
	mse = 0
	ratio = 0.0
	ratio_rec = 0.0
	cc=1	
	real_var1=1e-8
	real_var2=1e-8
	fake_var1=1e-8
	fake_var2=1e-8
	phase_loss2 = 1e-8
	mag_loss2 = 1e-8
	balance1 = 0
	balance1_rec = 0

	print "Epoch: {}/{}".format(epoch + 1,epochs)
	with progressbar.ProgressBar(max_value = steps, widgets = widgets) as bar:
		for i in range(steps):

			[x,y] = train_descriminator_gen.next()

			x = 2*x - 1
			y = 2*y - 1

			gx = generator.forward(x)
			fgx = descriminator.forward(gx,y)
			fy = descriminator.forward(y,y)

			desciminator_opt.zero_grad()
			real_loss_d = fy.mean()
			fake_loss_d = fgx.mean()

			loss = real_loss_d - k*fake_loss_d

			loss.backward(retain_graph = True)
			desciminator_opt.step()

			c_loss += loss.item()


			generator_opt.zero_grad()

			ssim_value = ssim_funct.forward((y+1)/2.0, (gx+1)/2.0)
			ssim_loss = 1.0 - ssim_value

			mse_loss = ((y - gx)**2).mean(0).mean(1).mean(1)

			recon_loss = alpha*ssim_loss + (1.0 - alpha)*mse_loss
			

			gen_loss = 0.5*(recon_loss[0] + recon_loss[1])
			
			beg_loss = fake_loss_d
			loss = 100*beg_loss + 10*gen_loss


			loss.backward()
			generator_opt.step()

			balance = gamma - (fake_loss_d.detach() - real_loss_d.detach())
			
			k += lamda*balance
			
			k = torch.clamp(k,0,1.0)
			balance_rec = 1.0 - recon_loss[1].detach()/recon_loss[0].detach()

			mse += torch.mean((y - gx)**2).detach().item()
			ssim += msssim_value.mean().item()
			ratio += real_loss_d.detach()
			ratio_rec += fake_loss_d.detach()
			bar.update(i,MSE = mse/(i+1), SSIM = ssim/(i+1),RealLoss=ratio/(i+1),FakeLoss=ratio_rec/(i+1),k=k)
			cc+=1

x,y = train_generator_gen.next()
x = 2*x - 1
y = 2*y - 1
y = y.cpu().numpy()
pred = generator.forward(x)
pred = pred.cpu().detach().numpy()

plot_pair((y[0,:,:,:] + 1)/2.0, (pred[0,:,:,:] + 1)/2.0)