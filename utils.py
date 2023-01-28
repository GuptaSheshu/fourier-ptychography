import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pickle
import progressbar

import skimage

import test_models

class TestUnpickler(pickle.Unpickler,object):
	def find_class(self, module, name):
		'''
		Redirects class names from models to test_models to have the desired testing time behaviour.
		'''
		if module == 'models' and 'Generator' in name:
			return getattr(test_models,name)
		else:
			return super(TestUnpickler,self).find_class(module,name)

def data_generator(file_path, mode, batch_size = 50, shuffle = True):

	'''
		Defines a data generator.

		Arguments:

		mode: 'train', 'test', 'full' to generate training data, testing data and all data respectively.
			  Training data must be organized into x_train.npy and y_train.npy
			  Testing data must be organized into x_test.npy and y_test.npy
			  Combined data must be organized into stacks.npy and targets.npy

		batch_size (Default 50):  Number of samples per batch.

		shuffle (Default True): Whether data must be shuffled.

		Returns a tuple of (batch_size,25,256,256) and (batch_size,2,256,256) tenors
	'''

	if mode != 'train' and mode != 'test' and mode != 'full':
		print 'Invalid mode'
		return

	if mode == 'train':
		x = np.load(file_path + 'x_train.npy')
		y = np.load(file_path + 'y_train.npy')
	elif mode == 'test':
		x = np.load(file_path + 'x_test.npy')
		y = np.load(file_path + 'y_test.npy')
	else:
		x = np.load(file_path + 'stacks.npy')
		y = np.load(file_path + 'targets.npy')

	x_batch = np.zeros((batch_size,256,256,25))

	if shuffle:
		while True:
			p_inds = np.random.permutation(x.shape[0])

			for ind in range(0,len(p_inds) - batch_size,batch_size):
				for p_ind in range(ind,ind + batch_size):
					x_batch[p_ind - ind] = cv2.resize(x[p_inds[p_ind]],(256,256))
				y_batch = y[p_inds[ind:ind+batch_size]]
				yield(torch.from_numpy(np.transpose(x_batch,(0,3,1,2))).to("cuda:0").float(),torch.from_numpy(np.transpose(y_batch,(0,3,1,2))).to("cuda:0").float())
	else:
		while True:
			for ind in range(0,x.shape[0] - batch_size,batch_size):
				for p_ind in range(ind,ind + batch_size):
					x_batch[p_ind - ind] = cv2.resize(x[p_ind],(256,256))
				y_batch = y[ind:ind+batch_size]
				yield(torch.from_numpy(np.transpose(x_batch,(0,3,1,2))).to("cuda:0").float(),torch.from_numpy(np.transpose(y_batch,(0,3,1,2))).to("cuda:0").float())

def plot_pair(true_image, gen_image, psnr = None, ssim = None):

	'''
	Plots a pair of true and predicted magnitude and phase distributions in the format

	True Magnitude      | True Phase
	--------------------------------------
	Predicted Magnitude | Predicted Phase

	Title contains the PSNRs and SSIMs between the true and predicted magnitude and phase distributions.

	Arguments:

	true_image: True distribution of shape (256,256,2)

	gen_image: Predicted distribution of shape (256,256,2)

	pnsr: (Default None) Tuple of the form (Magnitude PSNR, Phase PSNR)

	ssim: (Default None) Tuple of the form (Magnitude SSIM, Phase SSIM)
	'''

	plt.ion()

	plt.figure()
	plt.subplot(2,2,1)
	plt.imshow(true_image[0,:,:],cmap='gray')
	plt.subplot(2,2,2)
	plt.imshow(true_image[1,:,:],cmap='gray')
	plt.subplot(2,2,3)
	plt.imshow(gen_image[0,:,:],cmap='gray')
	plt.subplot(2,2,4)
	plt.imshow(gen_image[1,:,:],cmap='gray')

	if psnr is not None and ssim is not None:
		plt.suptitle("PSNRs - Magnitude: {}, Phase: {}, SSIMs - Magnitude: {}, Phase: {}".format(psnr[0],psnr[1],ssim[0],ssim[1]))
	else:
		plt.suptitle('True and predicted Magnitude and Phase distributions.')

def evaluate_model(model_path, data_path, mode, return_model = False):
	
	'''
	Evaluates a model on data. Metrics reported are PSNR and SSIM at various overlaps

	Arguments:

	model_path: Path from which model is loaded
	
	data_path: Path from which data is loaded

	mode: 'plot' Displays 4 plots, one for each value of overlap
		  'full' Reports mean SSIM and PSNR for magnitude and phase distributions over the full dataset for each value of overlap
	
	return_model: (Default False) Returns loaded model if True
	'''

	if mode != 'plot' and mode != 'full':
		print 'Invalid mode'
		return

	mag_ssims = []
	mag_psnrs = []
	phase_ssims = []
	phase_psnrs = []
	mse = []

	unpickler = TestUnpickler(file(model_path,'rb'))
	gen_model = unpickler.load().to("cuda:0").eval()

	if mode == 'plot':

		stacks  = np.load(data_path + 'stacks.npy')
		targets = np.load(data_path + 'targets.npy')

		x_test = []
		y_test = []
		for i in range(4):
			x_test.append(cv2.resize(stacks[i*745 + 250],(256,256)))
			y_test.append(targets[i*745 + 250])

		x_test = np.array(x_test)
		y_test = np.array(y_test).astype('float32')

		y_pred = gen_model.forward(torch.from_numpy(np.transpose(x_test,(0,3,1,2))).to("cuda:0").float())
		y_pred = y_pred.cpu().detach().numpy()
		y_pred = np.transpose(y_pred,(0,2,3,1))

		for i in range(4):
			mag_psnrs.append(skimage.measure.compare_psnr((255*y_test[i,:,:,0]).astype('uint8'),(255*y_pred[i,:,:,0]).astype('uint8')))
			phase_psnrs.append(skimage.measure.compare_psnr((255*y_test[i,:,:,1]).astype('uint8'),(255*y_pred[i,:,:,1]).astype('uint8')))
			mag_ssims.append(skimage.measure.compare_ssim((255*y_test[i,:,:,0]).astype('uint8'),(255*y_pred[i,:,:,0]).astype('uint8')))
			phase_ssims.append(skimage.measure.compare_ssim((255*y_test[i,:,:,1]).astype('uint8'),(255*y_pred[i,:,:,1]).astype('uint8')))

		y_test = np.transpose(y_test,(0,3,1,2))
		y_pred = np.transpose(y_pred,(0,3,1,2))
		for i in range(4):
			plot_pair(y_test[i],y_pred[i],(mag_psnrs[i],phase_psnrs[i]),(mag_ssims[i],phase_ssims[i]))

	else:

		widgets=[
			    progressbar.ETA(),
			    progressbar.Bar()
		]

		data_gen = data_generator(data_path, mode = 'full', shuffle = False, batch_size = 1)

		with progressbar.ProgressBar(max_value = 2980, widgets = widgets) as bar:
			for i in range(2980):
				x,y = data_gen.next()
				y_pred = gen_model.forward(x)
				y_pred = y_pred.cpu().detach().numpy()
				y_pred = np.transpose(y_pred,(0,2,3,1))

				y_test = y.cpu().detach().numpy()
				y_test = np.transpose(y_test,(0,2,3,1))

				mag_psnrs.append(skimage.measure.compare_psnr((255*y_test[0,:,:,0]).astype('uint8'),(255*y_pred[0,:,:,0]).astype('uint8')))
				phase_psnrs.append(skimage.measure.compare_psnr((255*y_test[0,:,:,1]).astype('uint8'),(255*y_pred[0,:,:,1]).astype('uint8')))
				mag_ssims.append(skimage.measure.compare_ssim((255*y_test[0,:,:,0]).astype('uint8'),(255*y_pred[0,:,:,0]).astype('uint8')))
				phase_ssims.append(skimage.measure.compare_ssim((255*y_test[0,:,:,1]).astype('uint8'),(255*y_pred[0,:,:,1]).astype('uint8')))

				mse.append(np.mean((y_test - y_pred)**2))
				bar.update(i)

		for i in range(4):
			print "{}% overlap - Magnitude PSNR: {}, Magnitude SSIM: {}, Phase PSNR: {}, Phase SSIM: {}".format(5*i,np.mean(mag_psnrs[745*i:745*(i+1)]),np.mean(mag_ssims[745*i:745*(i+1)]),np.mean(phase_psnrs[745*i:745*(i+1)]),np.mean(phase_ssims[745*i:745*(i+1)]))
		print "MSE is {}".format(np.mean(mse))

	if return_model:
		return gen_model