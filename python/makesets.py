from initial import *

trainfile = open(CAFFE_ROOT+'data/plastic/train.txt','w')
valfile = open(CAFFE_ROOT+'data/plastic/val.txt','w')
testfile = open(CAFFE_ROOT+'data/plastic/test.txt','w')

for (thisfile,dataset) in [(trainfile,BDATA_TRAIN),(valfile,BDATA_VALI),(testfile,BDATA_TEST)]:
	for filename in dataset:
		[num,rest] = filename.split('_')
		t = int(rest[0:2],2)
		name = "alldata_4096/%s.npy %s\n" % (filename, t)
		thisfile.write(name)

