import torch

from torch.optim import RMSprop
import pytorch_msssim

import progressbar

from models import WGANGenerator, WGANCritic
from utils import data_generator, plot_pair

torch.manual_seed(0)

generator = WGANGenerator().to("cuda:0")
critic = WGANCritic().to("cuda:0")

n_critic = 1
batch_size = 4
steps = 2384/batch_size
epochs = 100

train_critic_gen = data_generator('../data/25_stacks/', mode = 'train', batch_size = batch_size)
train_generator_gen = data_generator('../data/25_stacks/', mode = 'train', batch_size = batch_size)

generator_opt = RMSprop(generator.parameters(), lr = 1e-3, alpha = 0.9)
critic_opt = RMSprop(critic.parameters(), lr = 1e-3, alpha = 0.9)

widgets=[
    progressbar.ETA(),
    progressbar.Bar(),
    ' ',progressbar.DynamicMessage('CriticLoss'),
    ' ',progressbar.DynamicMessage('MeanSSIM')
]

ssim_funct = pytorch_msssim.SSIM()

for epoch in range(epochs):
	c_loss = 0
	ssim = 0

	print "Epoch: {}/{}".format(epoch + 1,epochs)
	with progressbar.ProgressBar(max_value = steps, widgets = widgets) as bar:
		for i in range(steps):
			for param in generator.parameters():
				param.requires_grad = False

			for j in range(n_critic):
				[x,y] = train_critic_gen.next()

				x = 2*x - 1
				y = 2*y - 1

				gx = generator.forward(x)
				fgx = critic.forward(gx)
				fy = critic.forward(y)

				critic_opt.zero_grad()

				loss = torch.mean(fgx - fy)
				loss.backward()
				critic_opt.step()

				for param in critic.parameters():
					param.data.clamp_(-0.01,0.01)

				c_loss += loss.item()

			for param in generator.parameters():
				param.requires_grad = True

			for param in critic.parameters():
				param.requires_grad = False


			[x,y] = train_generator_gen.next()

			x = 2*x - 1
			y = 2*y - 1

			gx = generator.forward(x)
			fgx = critic.forward(gx)

			generator_opt.zero_grad()

			ssim_loss = 1.0 - torch.mean(ssim_funct.forward((y+1)/2.0,(gx+1)/2.0))
			gen_loss = 0.7*ssim_loss + 0.3*torch.mean(torch.abs(y - gx))

			critic_loss = - torch.mean(fgx)
			loss = 1000*gen_loss + critic_loss

			loss.backward()
			generator_opt.step()

			for param in critic.parameters():
				param.requires_grad = True

			ssim += 1 - ssim_loss.item()

			bar.update(i,CriticLoss = c_loss/(n_critic*(i+1)), MeanSSIM = ssim/(i+1))

x,y = train_generator_gen.next()
x = 2*x - 1
y = 2*y - 1
y = y.cpu().numpy()
pred = generator.forward(x)
pred = pred.cpu().detach().numpy()

plot_pair((y[0,:,:,:] + 1)/2.0, (pred[0,:,:,:] + 1)/2.0)