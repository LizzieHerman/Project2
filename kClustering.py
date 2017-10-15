'''
created on Oct 2 2017
@author LizzieHerman
'''
import random

class Vector:
    def __init__(self, pos, d):
        self.position = pos
        self.dim = d

    #d the number of dimensions of the vector
    def makeRandVec(self, d=0, low=0, high=0):
        temp = []
        for i in range(d):
            temp.append(random.uniform(low,high))
        return Vector(temp, d)

    def distance(self, other, d):
        temp = 0
        for i in range(d):
            temp += pow(abs(self.position[i] - other.position[i]),d)
        dist = pow(temp,1/d)
        return dist

    def __eq__(self, other):
        if self.dim != other.dim:
            return False
        for i in range(self.dim):
            if self.position[i] != other.position[i]:
                return False
        return True

    def getPosition(self, i=0):
        return self.position[i]

class Cluster:
    def __init__(self, k=0, l=0, h=0, d=0):
        self.k = k # number of clusters to make
        self.low = l # low end of the range points are in
        self.high = h # high end of the range points are in
        self.dim = d # dimension of the vectors
        self.centroids = []
        self.clusters = {} # key is the int of the index of the centroid in centroids
        self.numIters = 0

    def genRandVecs(self):
        for i in range(self.k):
            vec = Vector.makeRandVec( self.dim, self.low, self.high)
            self.centroids.append(vec)
            self.clusters.update({i:[]})

    def assignPoints(self, points):
        for point in points:
            lowest = point.distance(self.centroids[1])
            low = 1
            for i in range(self.k):
                temp = point.distance(self.centroids[i])
                if temp < lowest:
                    lowest = temp
                    low = i
            self.clusters.get(low).append(point)
            num = len(self.clusters.get(low))

    def updateCentroids(self, points):
        newCentroids = []
        for i in range(self.k):
            temp = []
            vec = None
            # check to see if centroid has no points to be center of
            # reinitialize if this happens
            if self.clusters.get(i) in None:
                vec = Vector.makeRandVec( self.dim, self.low, self.high)
            else:
                for j in range(self.dim):
                    temp.append(0)
                    for clust in self.clusters.get(i):
                        temp[j] += clust.getPosition(j)
                    temp[j] = (temp[j] / self.dim)
                vec = Vector(temp, self.dim)
            newCentroids.append(vec)
        return newCentroids

    def shouldStop(self, newCentroids):
        if self.numIters >= 1000:
            return True
        new = set(newCentroids)
        old = set(self.centroids)
        if new == old:
            return True
        return False

    def run(self, points):
        self.numIters += 1
        self.genRandVecs()
        self.assignPoints(points)
        newCentroids = self.updateCentroids(points)
        while not self.shouldStop(newCentroids):
            self.numIters += 1
            self.centroids[:] = []
            self.centroids = newCentroids[:]
            for val in self.clusters.values():
                del val[:]
            self.assignPoints(points)

            newCentroids = self.updateCentroids(points)
        return newCentroids
            