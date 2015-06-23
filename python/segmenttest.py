import numpy as np
import matplotlib.pyplot as plt
import caffe
import random
import time
import sys, os
from scikits.learn import svm, linear_model #Todo
from scikits.learn.externals import joblib
import datetime
import Image

rel = '''
import segmenttest
reload(segmenttest)
from segmenttest import *
'''

CAFFE_ROOT = '../caffe/'
ADATA_FOLDER = '../DATA_AWATER/'
BDATA_FOLDER = '../DATA_BWATER/'

MODEL_FILE = CAFFE_ROOT+'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = CAFFE_ROOT+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMGCLASS = [line[:-1] for line in open('imnetclasses.txt')]

CLFP = joblib.load('svmmodels/linearMixed/clfP.pkl')
CLFA = joblib.load('svmmodels/linearMixed/clfA.pkl')

caffe.set_mode_cpu()

if not 'NET' in locals():
	NET = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

def log(s):
	print s

def cnn_through(image):
	prediction = NET.predict([image])
	log('Imagenet thinks for %.2f it sees a %s'%(prediction.max(),IMGCLASS[prediction.argmax()] ) )
	data = NET.blobs['fc7'].data
	vec = data.mean(axis=0)
	return vec

def svm_through(vec):
	p = CLFP.predict([vec])[0]
	a = CLFA.predict([vec])[0]
	log('SVM sees plastic:%d, animals:%d'%(p,a) )
	return p,a

def through(image,deep):
	t = time.time()
	ifull = image#mul_im(image,deep)
	vec = cnn_through(ifull)
	p,a = svm_through(vec)
	log('classification took %f sec'%(time.time()-t) )
	inew = label(image,p,a)
	return inew

def mul_im(image,deep):
	row = np.vstack( tuple([image]*((2**deep))) )
	full = np.hstack( tuple([row]*((2**deep))) )
	return full

def stitch_im(i1,i2,i3,i4):
	i1i2 = np.vstack((i1,i2))
	i3i4 = np.vstack((i4,i3))
	image = np.hstack((i1i2,i3i4))
	return image

def split_im(image):
	w,h,d = image.shape
	w2,h2 = w/2,h/2
	i1 = image[:w2,:h2,:]
	i2 = image[w2:,:h2,:]
	i3 = image[w2:,h2:,:]
	i4 = image[:w2,h2:,:]
	return i1,i2,i3,i4

def label(image,p,a):
	w,h,d = image.shape
	inew = np.array(image)
	if p == 1:
		for i in range(0,w,23):
			for j in range(0,h):
				inew[(i+j)%w,j,:] = [1,0,0]
				inew[(i+j+1)%w,j,:] = [1,0,0]
		#inew[range(0,w,11),range(0,h,11),:] = np.array([0,1,0]).tile((47,88,1))
	if a == 1:
		for i in range(0,w,23):
			for j in range(0,h):
				inew[(i-j)%w,j,:] = [0,1,0]
				inew[(i-j+1)%w,j,:] = [0,1,0]
		#inew[range(0,w,7),range(0,h,7),:] = np.array([1,0,0])
	return inew

#def run(image):

from multiprocessing import Pool

def rec_run_mul((image,depth,deep)):
	if depth <= 0:
		log('maximum depth reached')
		inew = through(image,deep)
		return [inew]
	
	i1,i2,i3,i4 = split_im(image)
	
	pool = Pool(4)
	out = pool.map(rec_run,[(i1,depth-1,deep+1),(i2,depth-1,deep+1),(i3,depth-1,deep+1),(i4,depth-1,deep+1)])
	
	nlist = []
	for i in range(0,len(out[0])):
		tmp = stitch_im(out[0][i],out[1][i],out[2][i],out[3][i])
		nlist.append(tmp)
	
	log('at depth %d'%(depth) )
	inew = through(image,deep)
	nlist.append(inew)
	return nlist

def rec_run((image,depth,deep)):
	if depth <= 0:
		log('maximum depth reached')
		inew = through(image,deep)
		return [inew]
	
	i1,i2,i3,i4 = split_im(image)
	
	i1n = rec_run((i1,depth-1,deep+1))
	i2n = rec_run((i2,depth-1,deep+1))
	i3n = rec_run((i3,depth-1,deep+1))
	i4n = rec_run((i4,depth-1,deep+1))
	
	nlist = []
	for i in range(0,len(i1n)):
		tmp = stitch_im(i1n[i],i2n[i],i3n[i],i4n[i])
		nlist.append(tmp)
	
	log('at depth %d'%(depth) )
	inew = through(image,deep)
	nlist.append(inew)
	
	return nlist
	
def run_and_save(imagefolder,imagename,depth):
	image = caffe.io.load_image(imagefolder+imagename)
	ilist = rec_run_mul((image,depth,0))
	ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S__')
	log('saving...')
	for i,im in enumerate(ilist):
		plt.imshow(im)
		plt.savefig('segout/'+ts+imagename+'__'+str(i)+'__.png')

if __name__ == '__main__':
	if not len(sys.argv) == 4:
		log('us as followd:\n  python segmenttest.py [imagefolder] [imagename] [depth]')
	run_and_save(sys.argv[1],sys.argv[2],int(sys.argv[3]))

