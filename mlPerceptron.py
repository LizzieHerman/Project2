'''
created on Oct 2 2017
@author LizzieHerman
'''
import kClustering
import random

class node:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = []

class inputLayer:
    def __init__(self, i=0):
        self.nInputs = i
        self.inputs = []

class hiddenLayer:
    def __init__(self, i=0):
        self.nInputs = i
        self.inputs = []
        self.outputs = []
        self.weights = {}
        for a in range(i):
            self.weights.update({a:{}})
            for b in range(i):
                self.weights[a].update({b:0})

    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nInputs):
                self.weights[i][j] = random.uniform(-a,a)


class outputLayer:
    def __init__(self, targs, i=0, o=0):
        self.nInputs = i
        self.nOutputs = o
        self.inputs = []
        self.outputs = []
        self.targets = targs
        self.weights = {}
        for a in range(i):
            self.weights.update({a:{}})
            for b in range(o):
                self.weights[a].update({b:0})

    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                self.weights[i][j] = random.uniform(-a,a)

    def findOutput(self, i=0, j=0):
        return i * j

    def errorFunc(self, j=0):
        expec = self.targets[j]
        temp = 0
        for i in range(self.nInputs):
            temp += pow((expec - self.findOutput(i,j)),2)
        return (temp / 2)

    def updateWeights(self, eta=0.0):
        return eta


class MLP:
    def __init__(self, pnts, targs, l=0, i=0, o=0, lr=0.0):
        self.points = pnts
        self.nOutputs = o
        self.iLayer = inputLayer(i)
        self.hLayers = []
        for i in range(l):
            self.hLayers.append(hiddenLayer(i))
        self.oLayer = outputLayer(targs,i,o)
        self.eta = lr

    def generateWeights(self, a=0.0):
        self.oLayer.generateWeights(a)
        for hid in self.hLayers:
            hid.generateWeights(a)

    def updateWeights(self):
        return self.oLayer.updateWeights(self.eta)

    def backPropagation(self):
        '''
            \delta\(N),l = targ,l - out(N),l
            \delta\(n),l = (SIGMAoverk \delta\(n+1),k*w(n+1),lk)*f'(SIGMAoverj out(n-1),j W(n),jl)
            DELTAw(n),hl = eta SIGMAoverp(\delta\(n),l*out(n-1),h)
                #activation functions
                #f(x)  = math.tanh(x)
                #f'(x) = 1 - math.pow(math.tanh(x),2)
            '''
        total = 1.0
        return total

    def adalineRule(self):
        # w <- w + eta(targ - out)x
        # E = (targ - out)^2
        total = 1.0
        return total

'''
NOTES
SSE Cost function. linear output activations, sigmoid hidden activation
E(cost)SSE = (1/2){(SIGMA over p)[(SIGMA over j) (targetp,j - output(N)p,j)^2]}
DELTA w(m),kl = - eta ( partial E({w(n),ij})) / (partial w(m),kl)
(m/n) are the layer, for weights its the layer they are going to
w,ij weight from i to j
final layer outputs depend onm all earlier layers of weights not just final one, 
  algorithm should automatically adjust all these earlier weights and outputs
'''
'''
Training for multi-layer networks is similar to that for single layer networks:
5. Apply the weight update DELTAwjk(n)= −etapartialE(w(n),jk)/partialw(n),jk to each 
   weight w(n),jk for each training pattern p. One set of updates of all the weights for
   all the training patterns is called one epoch of training.
6. Repeat step 5 until the network error function is “small enough”.
To be practical, algebraic expressions need to be derived for the weight updates
'''