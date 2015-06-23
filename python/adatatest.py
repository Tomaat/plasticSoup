from initial import *
import classify as cl
from scikits.learn import svm
from scikits.learn.externals import joblib
#import svmtest as st

rel = """
import adatatest
reload(adatatest)
from adatatest import *
"""

clfP = joblib.load('svmmodels/test/SVMl_clfP.pkl')
clfA = joblib.load('svmmodels/test/SVMl_clfA.pkl')

def through(imagefolder,imagename):
	b = cl.through(imagefolder,imagename)
	dm = b['fc7'].data.mean(axis=0)
	a = clfA.predict([dm])
	p = clfP.predict([dm])
	return int(a[0]),int(p[0])

def test(num=100):
	bg = pg = ag = bf = 0
	for image in ADATA_TRAIN[:num]:
		a,p = through(ADATA_FOLDER,image)
		[num,rest] = image.split('_')
		typeR = rest[0:2]
		at = int(typeR[0])
		pt = int(typeR[1])
		print a,at,p,pt
		if a == at and p == pt:
			bg += 1
		elif p == pt:
			pg += 1
		elif a == at:
			ag += 1
		else:
			bf += 1
	print bg,pg,ag,bf
	# with num=1000
	#  546 347 84 23, so ~90% of plastic data is correct!!
