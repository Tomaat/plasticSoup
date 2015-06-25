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

CAFFE_ROOT = '../caffe/'
ADATA_FOLDER = '../DATA_AWATER/'
BDATA_FOLDER = '../DATA_BWATER/'

MODEL_FILE = CAFFE_ROOT+'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = CAFFE_ROOT+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMGCLASS = [line[:-1] for line in open('imnetclasses.txt')]

CLFP = joblib.load('svmmodels/allprob/clfP.pkl')
CLFA = joblib.load('svmmodels/allprob/clfA.pkl')

caffe.set_mode_cpu()
NET = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

# use custom log for printing (possibly to file)
def log(s):
	print s

# Train the SVM with a train-set of size NUM
# saves the result
def train_and_save(NUM=200):
	tD = np.load('data_4096s/ALLDATAR.npy')
	cp = (tD[:NUM,2] == 1) | (tD[:NUM,2] == 3)
	ca = (tD[:NUM,2] == 2) | (tD[:NUM,2] == 3)
	D = tD[:NUM,3:]
	
	kwargs = {'kernel':'linear', 'C':1.0, 'probability':True}
	
	clfP = svm.SVC(**kwargs)
	clfA = svm.SVC(**kwargs)

	log('fitting plastic')
	t = time.time()
	clfP.fit(D,cp)
	ttime = time.time()-t
	log('in %f sec' % (time.time()-t) )
	log('fitting animals')
	t = time.time()
	clfA.fit(D,ca)
	ttime += (time.time()-t)
	log('in %f sec' % (time.time()-t) )
	joblib.dump(clfP, 'svmmodels/allprob/clfP.pkl')
	joblib.dump(clfA, 'svmmodels/allprob/clfA.pkl')

# Pull an image through the network and return the second-to-last layer
def cnn_through(image):
	prediction = NET.predict([image])
	#log('Imagenet thinks for %.2f it sees a %s'%(prediction.max(),IMGCLASS[prediction.argmax()] ) )
	data = NET.blobs['fc7'].data
	vec = data.mean(axis=0)
	return vec

# Pull a vector through the SVM and return the confidence of the classes
def svm_through(vec):
	p = CLFP.predict_proba([vec])[0,1]
	a = CLFA.predict_proba([vec])[0,1]
	#log('SVM sees plastic:%.2f, animals:%.2f'%(p,a) )
	return p,a

# Given a widht, heights and depths, return a list of bounds
# that describe the sub-images
def makeboxes(w,h,depth):
	ans = []
	for d in range(0,depth+1):
		nums = 2**d
		lw = [i*w/nums for i in range(0,nums)]
		uw = [(i+1)*w/nums for i in range(0,nums)]
		wb = zip(lw,uw)
		lh = [i*h/nums for i in range(0,nums)]
		uh = [(i+1)*h/nums for i in range(0,nums)]
		hb = zip(lh,uh)
		bs = [(x[0],x[1],y[0],y[1]) for x in wb for y in hb]
		ans += bs
	return ans

# Run using different threads
def run_mul((i,tot,image,(b0,b1,b2,b3))):
	t = time.time()
	vec = cnn_through(image[b0:b1,b2:b3,:])
	p,a = svm_through(vec)
	log('classification %d of %d took %f sec'%(i,tot,time.time()-t) )
	return (p,a)
	
from multiprocessing import Pool

# Run with given image and depth, possibly using multithreading
def run(image,depth,multi=True):
	w,h,d = image.shape
	iplastic = np.ones((w,h,d),'float32')
	ianimals = np.ones((w,h,d),'float32')
	boxes = makeboxes(w,h,depth)
	norm = depth+1
	tot = len(boxes)
	
	if multi:
		log('using multi process')
		allbox = [(i,tot,image,box) for i,box in enumerate(boxes)]
	
		pool = Pool(3)
		results = pool.map(run_mul, allbox)
		pool.close()
		for (p,a),(b0,b1,b2,b3) in zip(results,boxes):
			iplastic[b0:b1,b2:b3,:] += [0,-p/norm,-p/norm]
			ianimals[b0:b1,b2:b3,:] += [-a/norm,0,-a/norm]
	else:
		log('not using multi process')
		for i,(b0,b1,b2,b3) in enumerate(boxes):
			t = time.time()
			vec = cnn_through(image[b0:b1,b2:b3,:])
			p,a = svm_through(vec)
			log('classification %d of %d took %f sec'%(i,tot ,time.time()-t) )
			iplastic[b0:b1,b2:b3,:] += [0,-p/norm,-p/norm]
			ianimals[b0:b1,b2:b3,:] += [-a/norm,0,-a/norm]

	#show(image,iplastic,ianimals)
	return iplastic,ianimals

# show three images in one figure
def show(image,ia,ip):
	plt.subplot(1,3,1)
	plt.imshow(image)
	plt.subplot(1,3,2)
	plt.imshow(ia)
	plt.subplot(1,3,3)
	plt.imshow(ip)
	plt.show()

# Given a folder, filename and depth, run the pipeline
def run_and_save(imagefolder,imagename,depth):
	image = caffe.io.load_image(imagefolder+imagename)
	ip,ia = run(image,depth)#,False)
	ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S__')
	log('saving...')
	plt.imshow(ip)
	plt.savefig('segout/'+ts+imagename+'__plastic__.png')
	plt.imshow(ia)
	plt.savefig('segout/'+ts+imagename+'__animals__.png')
	plt.imshow(image)
	plt.savefig('segout/'+ts+imagename+'__image__.png')
	#show(image,ia,ip)

# Given a file with each line containg 'folder|||filename'
# run each of them with the given depth
def read_and_run(filename,depth):
	f = open(filename,'r')
	alf = [l[:-1].split('|||') for l in f if len(l) > 0 and not l[0] == '#' and '|||' in l]
	for fol,nam in alf:
		run_and_save(fol,nam,depth)


if __name__ == '__main__':
	if len(sys.argv) == 4:
		run_and_save(sys.argv[1],sys.argv[2],int(sys.argv[3]))
	if len(sys.argv) == 3:
		read_and_run(sys.argv[1],int(sys.argv[2]))
	else:
		log('us as followd:\n  python segmenttest2.py [imagefolder] [imagename] [depth]\n or\n  python segmenttest2.py [file-with-image-names] [depth]')