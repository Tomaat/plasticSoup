'''
from initial import *
#import classify as clfy
#import matrixify as mtfy
import svmtest

#clfy.mainA()
#mtfy.mainA()

clfP,clfA = svmtest.joblib.load('svmmodels/linearModel/clfP.pkl'), svmtest.joblib.load('svmmodels/linearModel/clfA.pkl')
D,p,a = svmtest.loadData('Atrain')

svmtest.test2(clfP,clfA,D,p,a)
'''

'''
from initial import *

alldatanames = ['Btrain','Btest','Bval','Atrain','Atest','Aval']

X = np.load('data_4096s/'+alldatanames[0]+'M.npy')

for name in alldatanames[1:]:
	print name
	x = np.load('data_4096s/'+name+'M.npy')
	X = np.vstack((X,x))

print X.shape
np.save('data_4096s/ALLDATA.npy',X)


'''
'''
from initial import *

ids = [[0,0]]*37165
i = 0
print 'split'
for filename in BDATA_TRAIN:
	[num,rest] = filename.split('_')
	ids[i] = [0,int(num)]
	i += 1
for filename in BDATA_TEST:
	[num,rest] = filename.split('_')
	ids[i] = [0,int(num)]
	i += 1
for filename in BDATA_VAL:
	[num,rest] = filename.split('_')
	ids[i] = [0,int(num)]
	i += 1
for filename in ADATA_TRAIN:
	[num,rest] = filename.split('_')
	ids[i] = [1,int(num)]
	i += 1
for filename in ADATA_TEST:
	[num,rest] = filename.split('_')
	ids[i] = [1,int(num)]
	i += 1
for filename in ADATA_VAL:
	[num,rest] = filename.split('_')
	ids[i] = [1,int(num)]
	i += 1

print 'transform',i
ID = np.array(ids)
del ids
print ID

print 'load'
X = np.load('data_4096s/ALLDATA.npy')
print 'stack'
Y = np.hstack((ID,X))
del ID, X
print Y.shape
np.random.shuffle(Y)
np.save('data_4096s/ALLDATA2.npy',Y)


'''
from initial import *
import svmtest

#X = np.load('data_4096s/ALLDATA.npy')
def getData():
	D,p,a,i = svmtest.loadData('all')

	sl = len(D)/2
	Da,Db = D[:sl,:],D[sl:,:]
	aa,ab = a[:sl],a[sl:]
	pa,pb = p[:sl],p[sl:]
	ia,ib = i[:sl,:],i[sl:,:]
	return (Da,pa,aa,ia),(Db,pb,ab,ib)

from multiprocessing import Pool
Dt,pt,ca,De,pe,ae = None,None,None,None,None,None
def multi(s):
	clfP,clfA,ttime = svmtest.fitting(Dt[:s,:],pt[:s],ca[:s])
	ans = svmtest.test2(clfP,clfA,De,pe,ae)
	return ('linear_1.0',ttime,ans,s)

def main(swi):
	global Dt,pt,ca,De,pe,ae
	stdlog('open data')
	#(Dt,pt,ca,it),(De,pe,ae,ie) = getData()
	if swi == 1:
		Dt,pt,ca = svmtest.loadData('Btrain')
		De,pe,ae = svmtest.loadData('Btest')
	if swi == 2:
		Dt,pt,ca = svmtest.loadData('Btrain')
		De,pe,ae = svmtest.loadData('Atest')
	if swi == 3:
		Dt,pt,ca = svmtest.loadData('Atrain')
		De,pe,ae = svmtest.loadData('Atest')
	if swi == 4:
		Dt,pt,ca = svmtest.loadData('Atrain')
		De,pe,ae = svmtest.loadData('Btest')
	
	stdlog('loop slices')
	slices = [18,180,1800,18000,9,90,900,9000]
	out = []

	pool = Pool(8)
	output = pool.map(multi, slices)
	stdlog(str(output))
	stdlog('write output')
	#svmtest.writeOutput('svmout/mixedlinear.out',output)
	svmtest.writeOutput('svmout/traintest'+str(swi)+'.out',output)

if __name__ == '__main__':
	svmtest.log('=====START=====')
	main(1)
	main(2)
	main(3)
	main(4)

'''


#'''
