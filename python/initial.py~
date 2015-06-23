import numpy as np
import matplotlib.pyplot as plt
import caffe
import random
import time
import sys, os
import datetime

CAFFE_ROOT = '../caffe/'
ADATA_FOLDER = '../DATA_AWATER/'
BDATA_FOLDER = '../DATA_BWATER/'

MODEL_FILE = CAFFE_ROOT+'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = CAFFE_ROOT+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMGCLASS = [line[:-1] for line in open('imnetclasses.txt')]
# Initialise the dataset
TRAIN_SLICE = .7
VAL_SLICE = .1
TEST_SLICE = 1.0 - TRAIN_SLICE - VAL_SLICE

ALL_ADATA = [f for f in os.listdir(ADATA_FOLDER) if os.path.isfile(ADATA_FOLDER+f)]
ALL_BDATA = [f for f in os.listdir(BDATA_FOLDER) if os.path.isfile(BDATA_FOLDER+f)]

random.seed(42)
random.shuffle(ALL_ADATA)
random.shuffle(ALL_BDATA)

tmp = len(ALL_ADATA)
ADATA_TRAIN = ALL_ADATA[:int(TRAIN_SLICE*tmp)]
ADATA_VAL = ALL_ADATA[int(TRAIN_SLICE*tmp):int((TRAIN_SLICE+VAL_SLICE)*tmp)]
ADATA_TEST = ALL_ADATA[int((TRAIN_SLICE+VAL_SLICE)*tmp):]

tmp = len(ALL_BDATA)
BDATA_TRAIN = ALL_BDATA[:int(TRAIN_SLICE*tmp)]
BDATA_VAL = ALL_BDATA[int(TRAIN_SLICE*tmp):int((TRAIN_SLICE+VAL_SLICE)*tmp)]
BDATA_TEST = ALL_BDATA[int((TRAIN_SLICE+VAL_SLICE)*tmp):]

def stdlog(s):
	logfile = open('stdlog.log','a')
	ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S -- ')
	print ts+s
	logfile.write(ts+s)
	logfile.write('\n')
	logfile.close()

#btrain_outlayer = []
#btrain_outlayer_files = os.listdir('output')
#for f in btrain_outlayer_files:
#	btrain_outlayer.append(np.load('output/'+f))
#btrain_matrix = np.array(btrain_outlayer)
