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

    def setOutputs(self, inp=numpy.matrix):
        self.outputs = inp

    def getOutputs(self):
        return self.outputs

class hiddenLayer(Layer):
    def __init__(self, prev=Layer, i=0, o=0, d=0):
        Layer.__init__(self, o, d)
        self.prevLayer = prev
        self.nInputs = i
        self.weights = numpy.zeros(shape=(i, o))
        self.lastWeightChange = numpy.zeros(shape=(i, o))

    # random generation of the weights of the inputs to this layer
    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                self.weights[i][j] = random.uniform(-a, a)

    # calculates the output of this layer
    def updateOutputs(self, inp=numpy.matrix):
        temp = numpy.dot(self.weights.T, inp)
        self.outputs = numpy.tanh(temp)
        return self.outputs

    # updates the weights that go along with the inputs into this layer, using back propagation
    # is overloaded so it can include momentum
    def updateWeights(self, sigma=numpy.matrix, eta=0.0, alpha=None):
        # back prop partial(q) = output(q) * (1 - output(q)) * SIGMAoverr w(qr)* partial deriv(r)
        partial = numpy.dot(numpy.dot(self.outputs, (1 - self.outputs.T)), sigma)
        if alpha is None:
            # new weight(p>q) = old weight(p>q) + eta * output(p) * partial deriv from back prop(q)
            self.weights = self.weights + eta * numpy.dot(self.prevLayer.outputs, partial.T)
        else:
            # delta weight(p>q)(t) = (1 - alpha) * eta * output(p) * partial deriv from back prop(q) + alpha * delta weight(p>q)(t-1)
            self.lastWeightChange = (1 - alpha) * eta * numpy.dot(self.prevLayer.outputs, partial.T) + alpha * self.lastWeightChange
            # weight(p>q)(t) = weight(p>q)(t-1) + delta weight(p>q)(t)
            self.weights = self.weights + self.lastWeightChange
        # return SIGMAoverq w(pq)* partial deriv(q)
        return numpy.dot(self.weights, partial)

class outputLayer(Layer):
    def __init__(self, targs=numpy.matrix, prev=Layer, i=0, o=0, d=0):
        Layer.__init__(self, o, d)
        self.prevLayer = prev
        self.nInputs = i
        self.targets = targs # will be a (o x 1) matrix
        self.weights = numpy.zeros(shape=(i, o))
        self.outputs = numpy.zeros(shape=(o,1))
        self.lastWeightChange = numpy.zeros(shape=(i, o))

    # random generation of the weights of the inputs to this layer
    def generateWeights(self, a=0.0):
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                self.weights[i][j] = random.uniform(-a, a)

    # calculates the output of this network
    def updateOutputs(self, inp=numpy.matrix):
        temp = numpy.dot(self.weights.T, inp)
        self.outputs = numpy.tanh(temp)
        return self.outputs

    # updates the weights that go along with the inputs into this layer
    # is overloaded so it can include momentum
    def updateWeights(self, eta=0.0, alpha=None):
        # partial(q) = (target(q) - output(q)) * output(q) * (1 - output(q))
        partial = numpy.dot(numpy.dot((self.targets - self.outputs), self.outputs.T), (1 - self.outputs))
        if alpha is None:
            # new weight(p>q) = old weight(p>q) + eta * output(p) * partial deriv from back prop(q)
            self.weights = self.weights + numpy.dot(self.prevLayer.outputs, partial.T) * eta
        else:
            # delta weight(p>q)(t) =  (1 - alpha)(eta * output(p) * partial deriv from back prop(q) + alpha * delta weight(p>q)(t-1)
            self.lastWeightChange = (1 - alpha) * eta * numpy.dot(self.prevLayer.outputs, partial.T) + alpha * self.lastWeightChange
            # weight(p>q)(t) = weight(p>q)(t-1) + delta weight(p>q)(t)
            self.weights = self.weights + self.lastWeightChange
            # return SIGMAoverq w(pq)* partial deriv(q)
        return numpy.dot(self.weights, partial)

    # calculates the error of the given output
    def errorFunc(self):
        temp = numpy.power((self.targets - self.outputs),2)
        return (0.5 * numpy.sum(temp))


class MLP:
    def __init__(self, inps=numpy.matrix, targs=numpy.matrix, i=0, h=0, o=0, d=0, l=0):
        self.iLayer = Layer(i, d)
        self.nLayers = l
        self.numInputs = i
        self.nDim = d
        self.hLayers = []
        last = self.iLayer
        if l == 0:
            self.oLayer = outputLayer(targs, last, i, o, d)
        else:
            for a in range(l):
                numins = h
                if a == 0:
                    numins = i
                last = hiddenLayer(last, numins, h, d)
                self.hLayers.append(last)
            self.oLayer = outputLayer(targs, last, h, o, d)
        self.iLayer.setOutputs(inps)

    # random generation of all weights in network
    def generateWeights(self, a=0.0):
        self.oLayer.generateWeights(a)
        for hid in self.hLayers:
            hid.generateWeights(a)

    # calculates the output of the current network
    def calcOutputs(self):
        inputs = self.iLayer.getOutputs()
        for i in range(self.nLayers):
            inputs = self.hLayers[i].updateOutputs(inputs)
        inputs = self.oLayer.updateOutputs(inputs)

    # one epoch is an update to all the weights in the network
    # overloaded so that momentum can be included if desired
    def epoch(self, eta=0.0, alpha=None):
        if alpha is None:
            deltas = self.oLayer.updateWeights(eta)
            for i in range((self.nLayers - 1), -1, -1):
                deltas = self.hLayers[i].updateWeights(deltas, eta)
        else:
            deltas = self.oLayer.updateWeights(eta, alpha)
            for i in range((self.nLayers - 1), -1, -1):
                deltas = self.hLayers[i].updateWeights(deltas, eta, alpha)

    # trains the network/ learns the function
    # overloaded so that momentum can be included if desired
    def train(self, a=0.0, eta=0.0, alpha=None):
        epochCount = 0
        self.generateWeights(a)
        self.calcOutputs()
        error = self.oLayer.errorFunc()
        while(error > 0.5 and epochCount < 100000):
            if alpha is None:
                self.epoch(eta)
            else:
                self.epoch(eta, alpha)
            epochCount += 1
            self.calcOutputs()
            error = self.oLayer.errorFunc()
        print "\t\tError: ", error, " Epoch Count: ", epochCount

    # tests the given input vectors on the trained network
    def test(self, inp=numpy.matrix):
        orginp = self.iLayer.getOutputs()
        self.iLayer.updateOutputs(inp)
        self.calcOutputs()
        output = numpy.zeros(shape=(self.numInputs, self.nDim + 1))
        for i in range(self.numInputs):
            output[i][0] = self.oLayer.outputs[i][0]
            for j in range(self.nDim):
                output[i][j + 1] = orginp[i][j]
        return output


