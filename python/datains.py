from initial import *

rel = '''
import datains
import initial
reload(datains)
reload(initial)
from datains import *
from initial import *
'''

def load(num=0):
	d1f = os.listdir('alldata_1000')
	d1 = []
	if num==0 or num>len(d1f):
		a = d1f
		for f in d1f:
			d1.append(np.load('alldata_1000/'+f))
	else:
		a = d1f[0:num]
		for i in range(0,num):
			d1.append(np.load('alldata_1000/'+d1f[i]))
	
	return a,np.array(d1)

def loadTVT(num=0):
	tr,vl,te = [],[],[]
	if num==0 or num>len(BDATA_VAL):
		a,b,c = BDATA_TRAIN,BDATA_VAL,BDATA_TEST
		for f in BDATA_TRAIN:
			tr.append(np.load('alldata_1000/'+f+'.npy'))
		for f in BDATA_VAL:
			vl.append(np.load('alldata_1000/'+f+'.npy'))
		for f in BDATA_TEST:
			te.append(np.load('alldata_1000/'+f+'.npy'))
	else:
		a,b,c = BDATA_TRAIN[0:num],BDATA_VAL[0:num],BDATA_TEST[0:num]
		for f in BDATA_TRAIN[0:num]:
			tr.append(np.load('alldata_1000/'+f+'.npy'))
		for f in BDATA_VAL[0:num]:
			vl.append(np.load('alldata_1000/'+f+'.npy'))
		for f in BDATA_TEST[0:num]:
			te.append(np.load('alldata_1000/'+f+'.npy'))
	return (a,b,c),(np.array(tr),np.array(vl),np.array(te))

def split(files,M):
	ans = [[0,0]]*len(files)
	plastic,fauna,none,both = [],[],[],[]
	for i in range(0,len(files)):
		[num,rest] = files[i].split('_')
		ans[i] = [int(c) for c in rest[0:2]]
		if ans[i] == [0,1]:
			plastic.append(M[i,:])
		elif ans[i] == [1,0]:
			fauna.append(M[i,:])
		elif ans[i] == [1,1]:
			both.append(M[i,:])
		else:
			none.append(M[i,:])
	return np.array(plastic),np.array(fauna),np.array(none),np.array(both),ans

def pca(M):
	mu = M.mean(axis=0)
	X = M-mu
	cov = np.dot(np.transpose(X),X)
	g,d,s = np.linalg.svd(cov)
	return g,d,s

def bestMatch(D,E,v):
	vE = np.dot(E,v)
	dv = D - np.tile(vE,(D.shape[0],1))
	sumsq = (dv**2).sum(axis=0)
	return sumsq

def bestMatch(D,v):
	dv = D - np.tile(v,(D.shape[0],1))
	sumsq = (dv**2).sum(axis=0)
	return sumsq

def maxnames(a,num=1,**kwargs):
	x = np.array(a)
	ans = []
	for i in range(0,num):
		imax = x.argmax(**kwargs)
		vmax = x[imax]
		ans.append([imax,vmax])
		x = x[x != vmax]
	return ans

old = '''
def testData(inpt=loadTVT(),num=0):
	(ltr,lvl,lte),(tr,vl,te) = inpt
	if num == 0:
		num = te.shape[0]
		output = open('sumsqtest0.txt','w')
		for i in range(0,num):
			v = te[i]
			n = maxnames(bestMatch(tr,v),5)
			ans = '[%s] = 1:%s, 2:%s, 3:%s, 4:%s ,5:%s \n' % (lte[i],ltr[n[0][0]],ltr[n[1][0]],ltr[n[2][0]],ltr[n[3][0]],ltr[n[4][0]])
			output.write(ans)
	else:
		output = []
		for i in range(0,num):
			v = te[i]
			n = maxnames(bestMatch(tr,v),5)
			ans = '[%s] = 1:%s, 2:%s, 3:%s, 4:%s, 5:%s \n' % (lte[i],ltr[n[0][0]],ltr[n[1][0]],ltr[n[2][0]],ltr[n[3][0]],ltr[n[4][0]])
			output.append(ans)
		return output

if __name__ == '__main__':
	testData()
'''
