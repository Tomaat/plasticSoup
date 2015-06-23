from initial import *

rel = '''
import training
reload(training)
from training import *
'''

class BData:
	TYPES = ['none','plastic','fauna','both']
	
	def __init__(self,bwdata,i):
		self.name = str(i)+'-'+bwdata.name
		self.typeR = bwdata.typeR
		self.typeI = bwdata.typeI
		self.type = bwdata.type
		self.d4ki = bwdata.d4k[i,:]
		self.out2 = np.array([0]*4)
		self.out2[self.typeI] = 1
		self.out = np.array([int(c) for c in self.typeR])
		
	
	def plot(self):
		plt.plot(self.d4ki)
		plt.show()

class BWData:
	TYPES = ['none','plastic','fauna','both']
	
	def __init__(self,filename):
		self.name = filename
		self.image = caffe.io.load_image(BDATA_FOLDER+filename)
		self.d4k = np.load('alldata_4096/'+filename+'.npy')
		[num,rest] = filename.split('_')
		self.typeR = rest[0:2]
		self.typeI = int(self.typeR,2)
		self.type = BWData.TYPES[self.typeI]
	
	def show(self):
		plt.imshow(self.image)
		plt.show()
	
	def plot(self):
		plt.plot(np.transpose(self.d4k))
		plt.show()
	
	def getBDs(self):
		ans = [None]*10
		for i in range(0,10):
			ans[i] = BData(self,i)
		return ans

class BDNetwork:
	 
	def __init__(self, nHidden=10):
		# learning rate
		self.alpha = 0.1
												 
		# number of neurons in each layer
		self.nIn = 4096
		self.nHidden = nHidden
		#self.nOut = 4
		self.nOut = 2
		 
		# initialize weights randomly (+1 for bias)
		self.hWeights = np.random.random((self.nHidden, self.nIn+1)) 
		self.oWeights = np.random.random((self.nOut, self.nHidden+1))
		 
		# activations of neurons (sum of inputs)
		self.hActivation = np.zeros((self.nHidden, 1), dtype=float)
		self.oActivation = np.zeros((self.nOut, 1), dtype=float)
		 
		# outputs of neurons (after sigmoid function)
		self.iOutput = np.zeros((self.nIn+1, 1), dtype=float)	  # +1 for bias
		self.hOutput = np.zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
		self.oOutput = np.zeros((self.nOut), dtype=float)
		 
		# deltas for hidden and output layer
		self.hDelta = np.zeros((self.nHidden), dtype=float)
		self.oDelta = np.zeros((self.nOut), dtype=float)   
	 
	def forward(self, input):
		if not isinstance(input,BData):
			print 'input not of type: BData'
			raise TypeError
		# set input as output of first layer (bias neuron = 1.0)
		self.iOutput[:-1, 0] = input.d4ki / 25
		self.iOutput[-1:, 0] = 1.0
		 
		# hidden layer
		self.hActivation = np.dot(self.hWeights, self.iOutput)
		self.hOutput[:-1, :] = np.tanh(self.hActivation)
		 
		# set bias neuron in hidden layer to 1.0
		self.hOutput[-1:, :] = 1.0
		 
		# output layer
		self.oActivation = np.dot(self.oWeights, self.hOutput)
		tmp = np.tanh(self.oActivation)
		self.oOutput = tmp[:,0]
			 
	def backward(self, teach):
		if not isinstance(teach,BData):
			print 'input not of type: BData'
			raise TypeError
		error = self.oOutput - teach.out
		
		# deltas of output neurons
		self.oDelta = (1 - np.tanh(self.oActivation)) * np.tanh(self.oActivation) * error
				 
		# deltas of hidden neurons
		self.hDelta = (1 - np.tanh(self.hActivation)) * np.tanh(self.hActivation) * np.dot(self.oWeights[:,:-1].transpose(), self.oDelta)
				 
		# apply weight changes
		self.hWeights = self.hWeights - self.alpha * np.dot(self.hDelta, np.tile(self.iOutput.transpose(),(self.nOut,1)))
		self.oWeights = self.oWeights - self.alpha * np.dot(self.oDelta, np.tile(self.hOutput.transpose(),(self.nOut,1)))
	 
	def getOutput(self):
		return self.oOutput

class BDPerceptron:	 
	def __init__(self):
		# learning rate
		self.alpha = 0.1
												 
		# number of neurons in each layer
		self.nIn = 4096
		self.nHidden = 2
				 
		# initialize weights randomly (+1 for bias)
		self.hWeights = np.random.random((self.nHidden, self.nIn+1)) 
		 
		# activations of neurons (sum of inputs)
		self.hActivation = np.zeros((self.nHidden, 1), dtype=float)
		 
		# outputs of neurons (after sigmoid function)
		self.iOutput = np.zeros((self.nIn+1, 1), dtype=float)	  # +1 for bias
		self.hOutput = np.zeros((self.nHidden, 1), dtype=float)  # +1 for bias
		 
		# deltas for hidden and output layer
		self.hDelta = np.zeros((self.nHidden), dtype=float)
	 
	def forward(self, input):
		if not isinstance(input,BData):
			print 'input not of type: BData'
			raise TypeError
		# set input as output of first layer (bias neuron = 1.0)
		self.iOutput[:-1, 0] = input.d4ki / 250
		self.iOutput[-1:, 0] = 1.0
		 
		# hidden layer
		self.hActivation = np.dot(self.hWeights, self.iOutput)
		tmp = np.tanh(self.hActivation)
		self.hOutput = tmp
			 
	def backward(self, teach):
		if not isinstance(teach,BData):
			print 'input not of type: BData'
			raise TypeError
		error = self.hOutput - teach.out
		
		# deltas of output neurons
		self.hDelta = (1 - np.tanh(self.hActivation)) * np.tanh(self.hActivation) * error
				 
		# apply weight changes
		self.hWeights = self.hWeights - self.alpha * np.dot(self.hDelta, np.tile(self.iOutput.transpose(),(self.nHidden,1)))
	 
	def getOutput(self):
		return self.hOutput

do = '''
#net = BDPerceptron()
di = BWData(BDATA_TRAIN[0])
dil = di.getBDs()
'''
import ffn
def train(num=len(BDATA_TRAIN)):
	#net = BDNetwork(16)
	#net = BDPerceptron()
	t = time.time()
	net = ffn.FeedForwardNetwork(4096,4096,2)
	for i in range(0,num):
		di = BWData(BDATA_TRAIN[i])
		dil = di.getBDs()
		for n in range(0,10):
			net.forward(dil[n].d4ki/25)
			net.backward(dil[n].out)
		#if i % (num / 10) == 0:
		#	print 'at',i,
	print 'training in',str(time.time()-t),'sec'	
	return net

def test(net, num = len(BDATA_TEST) ):
	#tp, fp, tn, fn = 0,0,0,0hWeights
	good = 0
	t = time.time()
	output = open('nettestout6.txt','w')
	for i in range(0,num):
		di = BWData(BDATA_TEST[i])
		dil = di.getBDs()
		for n in range(0,10):
			net.forward(dil[n].d4ki)
			out = net.getOutput()
			#output.write('o[%.2f,%.2f,%.2f,%.2f] - t[%d,%d,%d,%d] f\n' % (out[0], out[1], out[2], out[3],dil[n].out2[0],dil[n].out2[1],dil[n].out2[2],dil[n].out2[3] ) )
			output.write('o[%.4f,%.4f] - t[%d,%d] f\n' % (out[0], out[1],dil[n].out[0],dil[n].out[1] ) )
			w = (0 > 0.1) & (0 > 0.9)
			t = np.sum(w == dil[n].out) == 4
			if t:
				good += 1
	output.close()
	print 'testing in',str(time.time()-t),'sec'
	return good

if __name__ == '__main__':
	print 'training'
	net = train()
	print 'testing'
	outcome = test(net)
	print 'from %d data, %d was correct' % (len(BDATA_TEST)*10,outcome)
