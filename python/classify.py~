import numpy as np
import matplotlib.pyplot as plt
import caffe
import random
import time
import sys, os
from multiprocessing import Pool
from initial import *

# load 4 instances of the net for multiprocessing
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

def through(imagefolder,imagename):
	image = caffe.io.load_image(imagefolder+imagename)
	prediction = net.predict([image])
	return net.blobs

def train_and_save(imagefolder,imagelist,tofile,layer):
	classifyfile = open(tofile+'.txt','w')
	for imagename in imagelist:
		image = caffe.io.load_image(imagefolder+imagename)
		prediction = net.predict([image])
		classifyfile.write(imagename+' '+IMGCLASS[prediction[0].argmax()]+'\n')
		data = net.blobs[layer].data
		np.save('output/'+imagename+'.npy',data)

foldername = 'none'
ifoldername = 'none'

def train_and_save_si(imagename):
	image = caffe.io.load_image(ifoldername+imagename)
	prediction = net.predict([image])
	data = net.blobs['fc7'].data
	np.save(foldername+'_4096/'+imagename+'.npy',data)
	np.save(foldername+'_1000/'+imagename+'.npy',prediction[0])
	return imagename+' '+IMGCLASS[prediction[0].argmax()]+'\n'

def train_and_save_mul(imagelist,tofile,tofolder,imgfolder):
	global foldername,ifoldername
	ifoldername = imgfolder
	foldername = tofolder
	pool = Pool(4)
	names = pool.map(train_and_save_si, imagelist)
	classifyfile = open(tofile+'.txt','w')
	for name in names:
		classifyfile.write(name)

def mainA():
	stdlog('running data through net')
	t1 = time.time()
	train_and_save_mul(ALL_ADATA,'adata_test2','rawAdata',ADATA_FOLDER)
	stdlog('net run and saved in %.2f sec' % (time.time() - t1) )

def mainB():
	stdlog('running data through net')
	t1 = time.time()
	train_and_save_mul(ALL_BDATA,'bdata_test2','rawBdata',BDATA_FOLDER)
	stdlog('net run and saved in %.2f sec' % (time.time() - t1) )

if __name__ == '__main__':
	pass
