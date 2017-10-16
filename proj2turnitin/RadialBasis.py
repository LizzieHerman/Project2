'''
created on Oct 13 2017
@author LizzieHerman
'''

import random
import math
import numpy

class Cluster:
    def __init__(self, k=0, l=0, h=0, d=0):
        self.k = k # number of clusters to make
        self.low = l # low end of the range points are in
        self.high = h # high end of the range points are in
        self.dim = d # dimension of the vectors
        self.centroids = numpy.zeros(shape=(k,d))
        self.clusters = {} # key is the int of the index of the centroid in centroids
        self.numIters = 0

    # create random k centroids
    def genRandVecs(self):
        for i in range(self.k):
            for j in range(self.dim):
                self.centroids[i][j] = random.randint(self.low, self.high)
            self.clusters.update({i:[]})

    # assign each input vector to the nearest centroid
    def assignPoints(self, points=numpy.matrix):
        psize = points.shape[0]
        for i in range(psize):
            temp = self.centroids - points[i]
            temp = numpy.power(temp,2)
            temp = numpy.dot(temp, numpy.ones(shape=(self.dim, 1)))
            cent = numpy.argmax(temp)
            self.clusters.get(cent).append(points[i])

    # update the centroids so that if a centroid has no input vectors associated with it (no input vector is close to it) a new one is randomly assigned
    # for the centroids that have close inputs find the center of those inputs make this the new centroid
    def updateCentroids(self, points):
        newCentroids = numpy.zeros(self.centroids.shape)
        for i in range(self.k):
            temp = numpy.ones(shape=(1,self.dim))
            # check to see if centroid has no points to be center of
            # reinitialize if this happens
            if self.clusters.get(i) is None:
                for j in range(self.dim):
                    newCentroids[i][j] = random.randint(self.low, self.high)
            else:
                a = 1
                for clust in self.clusters.get(i):
                    a += 1
                    temp = temp + clust
                temp = temp / a
                temp = temp.astype(int)
                newCentroids[i] = temp
        return newCentroids

    # check if the algorithm should stop
    # if the algorithm has gone through to many iters or has not changed since the last update
    def shouldStop(self, newCentroids):
        if self.numIters >= 100000:
            return True
        return numpy.array_equal(newCentroids, self.centroids)

    # runs the k=means clustering algorithm
    def run(self, points):
        self.numIters += 1
        self.genRandVecs()
        self.assignPoints(points)
        newCentroids = self.updateCentroids(points)
        while not self.shouldStop(newCentroids):
            self.numIters += 1
            self.centroids = newCentroids
            for val in self.clusters.values():
                del val[:]
            self.assignPoints(points)
            newCentroids = self.updateCentroids(points)
        print "\t\t", self.k, "-Clustering:   Iteration Count: ", self.numIters
        return newCentroids

class RBF:
    def __init__(self, inps=numpy.matrix, targ=numpy.matrix, i=0, k=0, o=0, d=0):
        self.targets = targ
        self.k = k  # number of clusters to make
        self.dim = d  # dimension of the vectors
        self.centroids = numpy.zeros(shape=(k,d))
        self.sigma = 0.0
        self.inputs =  inps
        self.numInputs = i
        self.weights = numpy.zeros(shape=(k, o))
        self.numOutputs = o
        self.acts = numpy.zeros(shape=(k,1))
        self.outputs = numpy.zeros(shape=(o,d))
        self.actFunc = 0

    # runs the k-means clustering algorithm
    def getCentroids(self):
        clust = Cluster(self.k, 0, 9, self.dim)
        return clust.run(self.inputs)

    # finds the max distance between our centroids
    def finddmax(self):
        maxDist = 0.0
        for i in range(self.k):
            temp = self.centroids - self.centroids[i]
            temp = numpy.power(temp, 2)
            temp = numpy.dot(temp, numpy.ones(shape=(self.dim, 1)))
            dist = numpy.max(temp)
            if(dist > maxDist):
                maxDist = dist
        return maxDist

    # calculates our static sigma = dmax/sqrt(2k)
    def calcSigma(self):
        return (self.finddmax() / math.sqrt(2*self.k))

    # sets random initial weights for all weights in the network
    def setWeights(self, a=0.0):
        for i in range(self.k):
            for j in range(self.numOutputs):
                self.weights[i][j] = random.uniform(-a,a)

    # linear activation function done on centroid j
    def linearActFunc(self, j=0):
        temp = self.inputs - self.centroids[j]
        temp = numpy.power(temp, 2)
        temp = numpy.dot(temp, numpy.ones(shape=(self.dim, 1))) * self.sigma
        return numpy.sum(temp)

    # gaussian activation function done on centroid j
    def gaussianActFunc(self, j=0):
        temp = self.inputs - self.centroids[j]
        temp = numpy.power(temp, 2)
        temp = numpy.dot(temp, numpy.ones(shape=(self.dim, 1)))
        temp = numpy.power(temp, 2)
        top = numpy.sum(temp)
        bottom = 2 * math.pow(self.sigma, 2)
        return math.exp(-(top / bottom))

    # hardys activation function done on centroid j
    def hardysActFunc(self, j=0):
        left = math.pow(self.sigma, 2)
        temp = self.inputs - self.centroids[j]
        temp = numpy.power(temp, 2)
        temp = numpy.dot(temp, numpy.ones(shape=(self.dim, 1)))
        temp = numpy.power(temp, 2)
        right = numpy.sum(temp)
        return math.sqrt(left + right)

    # calculates the output given by the current network
    def calcOutput(self):
        for j in range(self.k):
            if self.actFunc == 0:
                act = self.linearActFunc(j)
            elif self.actFunc == 1:
                act = self.gaussianActFunc(j)
            else:
                act = self.hardysActFunc(j)
            self.acts[j][0] += act
        self.outputs = numpy.dot(self.weights.T, self.acts)

    # calculates the error of the current output
    def errorFunc(self):
        err = self.outputs - self.targets
        return numpy.sum(err)

    # updates all the weights in the network
    def weightUpdate(self, eta=0.0):
        err = self.outputs - self.targets
        update = numpy.dot(self.acts, err.T) * eta
        self.weights = self.weights + update

        # updates all the sigmas in the network
    def sigmaUpdate(self, eta=0.0):
        err = self.outputs - self.targets
        temp = numpy.dot(err, self.acts.T)
        part = numpy.zeros(shape=(self.k,1))
        for i in range(self.k):
            norm = self.inputs - self.centroids[i]
            o = numpy.sum(norm) / self.numInputs
            part[i] = (math.pow(o,2) / math.pow(self.sigma, 3))
        update = numpy.sum(numpy.dot(temp, part)) * eta
        self.sigma = self.sigma + update

    # one epoch is an update of all the weights and sigmas in the network
    def epoch(self, weta=0.0, seta=0.0):
        self.weightUpdate(weta)
        self.sigmaUpdate(seta)

    # trains the neural network, learns the function
    def train(self, a=0.0, weta=0.0, seta=0.0, act=0):
        epochCount = 0
        self.actFunc = act
        self.centroids = self.getCentroids()
        self.sigma = self.calcSigma()
        self.setWeights(a)
        self.calcOutput()
        error = self.errorFunc()
        while(error > 0.5 and epochCount < 100000):
            self.epoch(weta, seta)
            epochCount += 1
            self.calcOutput()
            error = self.errorFunc()
            print "\t\tError: ", error, " Epoch Count: ", epochCount

    # tests the given input on the trained neural net
    def test(self, inp=numpy.matrix):
        orginp = self.inputs
        self.inputs = inp
        self.calcOutput()
        output = numpy.zeros(shape=(self.numInputs,self.dim+1))
        for i in range(self.numInputs):
            output[i][0] = self.outputs[i]
            for j in range(self.dim):
                output[i][j+1] = orginp[i][j]
        return output
