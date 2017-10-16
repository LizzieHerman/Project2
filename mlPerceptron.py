'''
created on Oct 13 2017
@author LizzieHerman
'''
import random
import numpy


class Layer:
    def __init__(self, o=0, d=0):
        self.nOutputs = o
        self.nDim = d
        self.outputs = numpy.zeros(shape=(o, d))

    def updateOutputs(self, inp=numpy.matrix):
        self.outputs = inp

    def getOutputs(self):
        return self.outputs

class hiddenLayer(Layer):
    def __init__(self, prev=Layer, i=0, o=0, d=0):
        Layer.__init__(self, o, d)
        self.prevLayer = prev
        self.nInputs = i
        self.weights = numpy.zeros(shape=(i, o))

    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                self.weights[i][j] = random.uniform(-a, a)

    def updateOutputs(self, inp=numpy.matrix):
        temp = (self.weights.T) * inp
        self.outputs = numpy.tanh(temp)
        return self.outputs

    def updateWeights(self, sigma=numpy.matrix, eta=0.0):
        # back prop partial(q) = output(q) * (1 - output(q)) * SIGMAoverr w(qr)* partial deriv(r)
        partial = self.outputs * (1 - self.outputs) * sigma
        # new weight(p>q) = old weight(p>q) + eta * output(p) * partial deriv from back prop(q)
        self.weights = self.weights * partial.T * eta
        # return SIGMAoverq w(pq)* partial deriv(q)
        return self.weights * partial


class outputLayer(Layer):
    def __init__(self, targs=numpy.matrix, prev=Layer, i=0, o=0, d=0):
        Layer.__init__(self, o, d)
        self.prevLayer = prev
        self.nInputs = i
        self.targets = targs # will be a (o x 1) matrix
        self.weights = numpy.zeros(shape=(i, o))
        self.outputs = numpy.zeros(shape=(o,1))

    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                self.weights[i][j] = random.uniform(-a, a)

    def updateOutputs(self, inp=numpy.matrix):
        temp = (self.weights.T) * inp
        self.outputs = temp * numpy.ones(shape=(self.nDim, 1))
        return self.outputs

    def updateWeights(self, eta=0.0):
        # partial(q) = (target(q) - output(q)) * output(q) * (1 - output(q))
        partial = (self.targets - self.outputs) * self.outputs * (1 - self.outputs)
        # new weight(p>q) = old weight(p>q) + eta * output(p) * partial deriv from back prop(q)
        self.weights = self.weights * partial.T * eta
        # return SIGMAoverq w(pq)* partial deriv(q)
        return self.weights * partial

    def errorFunc(self):
        temp = numpy.power((self.targets - self.outputs),2)
        return (0.5 * numpy.sum(temp))


class MLP:
    def __init__(self, inps=numpy.matrix, targs=numpy.matrix, i=0, h=0, o=0, d=0, l=0, lr=0.0):
        self.iLayer = Layer(i, d)
        self.nLayers = l
        self.nDim = d
        self.hLayers = []
        last = self.iLayer
        for a in range(l):
            numins = h
            if a == 0:
                numins = i
            last = hiddenLayer(last, numins, h, d)
            self.hLayers.append(last)
        self.oLayer = outputLayer(targs, last, h, o, d)
        self.eta = lr
        self.iLayer.updateOutputs(inps)

    def generateWeights(self, a=0.0):
        self.oLayer.generateWeights(a)
        for hid in self.hLayers:
            hid.generateWeights(a)

    def calcOutputs(self):
        inputs = self.iLayer.getOutputs()
        for i in range(self.nLayers):
            inputs = self.hLayers[i].updateOutputs(inputs)
        inputs = self.oLayer.updateOutputs(inputs)

    def epoch(self):
        deltas = self.oLayer.updateWeights(self.eta)
        for i in range((self.nLayers - 1), -1, -1):
            deltas = self.hLayers[i].updateWeights(deltas, self.eta)

    def train(self, a=0.0):
        epochCount = 0
        self.generateWeights(a)
        self.calcOutputs()
        error = self.oLayer.errorFunc()
        while(error > 0.5 and epochCount < 1000):
            self.epoch()
            epochCount += 1
            self.calcOutputs()
            error = self.oLayer.errorFunc()
        print "Error: " + error + " Epoch Count: " + epochCount

    def test(self, inp=numpy.matrix):
        self.iLayer.updateOutputs(inp)
        self.calcOutputs()
        return self.oLayer.getOutputs()
