from initial import *

rel = """
import matrixify
reload(matrixify)
from matrixify import *
"""

def loadall2(data=BDATA_TRAIN,dataFolder='rawBdata_4096'):
	dt = np.load(dataFolder+data[0]+'.npy')
	[num,rest] = data[0].split('_')
	typeI = int(rest[0:2],2)
	dt = np.insert(dt,0,typeI,axis=1)
	dt = dt.mean(axis=0)
	
	q1 = len(data)/4
	q2 = q1+q1
	q3 = q1+q1+q1
	stdlog('devided in %d, %d, %d' % (q1,q2,q3) )
	for f in data[1:q1]:
		a = np.load(dataFolder+f+'.npy')
		[num,rest] = f.split('_')
		typeI = int(rest[0:2],2)
		b = np.insert(a,0,typeI,axis=1)
		b = b.mean(axis=0)
		dt = np.vstack((dt,b))
	stdlog('quarter done')
	for f in data[q1:q2]:
		a = np.load(dataFolder+f+'.npy')
		[num,rest] = f.split('_')
		typeI = int(rest[0:2],2)
		b = np.insert(a,0,typeI,axis=1)
		b = b.mean(axis=0)
		dt = np.vstack((dt,b))
	stdlog('halfway')
	for f in data[q2:q3]:
		a = np.load(dataFolder+f+'.npy')
		[num,rest] = f.split('_')
		typeI = int(rest[0:2],2)
		b = np.insert(a,0,typeI,axis=1)
		b = b.mean(axis=0)
		dt = np.vstack((dt,b))
	stdlog('3/4, almost there')
	for f in data[q3:]:
		a = np.load(dataFolder+f+'.npy')
		[num,rest] = f.split('_')
		typeI = int(rest[0:2],2)
		b = np.insert(a,0,typeI,axis=1)
		b = b.mean(axis=0)
		dt = np.vstack((dt,b))
	return dt

oldload = '''
def resave4ks(data=BDATA_TRAIN,dirt='train/'):
	for f in data:
		a = np.load(dataFolder+f+'.npy')
		[num,rest] = f.split('_')
		typeI = int(rest[0:2],2)
		b = np.insert(a,0,typeI,axis=1)
		for i in range(0,10):
			v = b[i,:]
			np.save('data_4096s/'+dirt+str(i)+'-'+f+'.npy',v)


def loadall():
	# train
	print 'loading train'
	fs = os.listdir('data_4096s/train')
	trd = np.load('data_4096s/train/'+fs[0])
	for f in fs:
		v = np.load('data_4096s/train/'+f)
		trd = np.vstack((trd,v))
	# val
	print 'loading val'
	fs = os.listdir('data_4096s/val')
	vld = np.load('data_4096s/val/'+fs[0])
	for f in fs:
		v = np.load('data_4096s/val/'+f)
		vld = np.vstack((vld,v))
	# test
	print 'loading test'
	fs = os.listdir('data_4096s/test')
	tsd = np.load('data_4096s/test/'+fs[0])
	for f in fs:
		v = np.load('data_4096s/test/'+f)
		ted = np.vstack((tsd,v))
	return trd,vld,tsd
'''	

def mainA():
	stdlog('using ADATA')
	stdlog('reload test')
	trd = loadall2(ADATA_TEST,'rawAdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/AtestM.npy',trd)
	stdlog('reload val')
	trd = loadall2(ADATA_VAL,'rawAdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/AvalM.npy',trd)
	stdlog('reload train')
	trd = loadall2(ADATA_TRAIN,'rawAdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/AtrainM.npy',trd)

def mainB():
	stdlog('using BDATA')
	stdlog('reload test')
	trd = loadall2(BDATA_TEST,'rawBdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/BtestM.npy',trd)
	stdlog('reload val')
	trd = loadall2(BDATA_VAL,'rawBdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/BvalM.npy',trd)
	stdlog('reload train')
	trd = loadall2(BDATA_TRAIN,'rawBdata_4096/')
	stdlog('save as array')
	np.save('data_4096s/BtrainM.npy',trd)

if __name__ == '__main__':
	pass
