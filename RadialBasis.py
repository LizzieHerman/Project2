'''
created on Oct 2 2017
@author LizzieHerman
'''

import kClustering
import random
import math

class RBF:
    def __init__(self, point, k=0, d=0, l=0, h=0):
        self.points = point
        self.k = k  # number of clusters to make
        self.dim = d  # dimension of the vectors
        self.low = l  # low end of the range points are in
        self.high = h  # high end of the range points are in
        self.centroids = []
        self.sigma = 0.0
        self.weights = []
        self.center = None

    def getCentroids(self):
        clust = kClustering.Cluster(self.k, self.low, self.high, self.dim)
        return clust.run(self.points)

    def finddmax(self):
        maxDist = 0.0
        for i in range(self.k):
            for j in range( (i+1), self.k):
                temp = self.centroids[i].distance(self.centroids[j])
                if temp > maxDist:
                    maxDist = temp
        return maxDist

    def calcSigma(self):
        return (self.finddmax() / math.sqrt(2*self.k))

    def setWeights(self, a=0.0):
        for i in range(self.k):
            self.weights.append(random.uniform(-a,a))

    def findCenter(self):
        temp = []
        for i in range(self.dim):
            temp.append(0)
            for j in range(self.k):
                temp[i] += self.centroids[j].getPosition(i)
            temp[i] = temp[i] / self.k
        vec = kClustering.Vector(temp, self.dim)
        return vec

    def linearActFunc(self, j=0):
        return self.center.distance(self.centroids[j])

    def gaussianActFunc(self, j=0):
        top = math.pow(self.center.distance(self.centroids[j]), 2)
        bottom = 2 * math.pow(self.sigma, 2)
        return math.exp(-(top / bottom))

    def hardysActFunc(self, j=0):
        left = math.pow(self.sigma, 2) 
        right = math.pow(self.center.distance(self.centroids[j]), 2)
        return math.sqrt(left + right)

    def weightUpdate(self, j=0, alpha=0.0, eta=0.0, prevChange=0.0):
        err = 1.0
        left = ((1 - alpha) * eta * err) + (alpha * prevChange)
        #∆wj^t = (1-α)*η*∇wj*Err + α*∆wj^(t-1)

    def run(self, a=0.0):
        self.centroids = self.getCentroids()
        self.sigma = self.calcSigma()
        self.setWeights(a)
        self.center = self.findCenter()


'''
#1. Choose the number (m) and initial coordinates of the centres (R) of the RBF functions.
m = k R = centroids

#2. Choose the initial value of the spread parameter (σ) for each centre (R).
we're using fixed σ for all points σ = (dmax)/(sqrt(2k))
σ = sigma

#3. Initialise the weights/coefficients (w) to small random values [-1,1].
overloaded function can include a float parameter a where w will be in range [-a,a], dont include a parameter w [-1,1]
w = weights index matches with centroids'

4. For each epoch (e)

    5. For each input vector/pattern (x(p))

        6. Calculate the output (y) of each output node (o) using eq. 1.

        7. Update the network parameters (w, R, σ) using eqs. 6, 7, 8, 9.

    8. end for (p = n)

9. end for (e = total epochs)
Note: Steps 1 and 2 can be performed using a clustering algorithm such as Kohonen SOMs.
'''
'''
∇∆←αηϕσ√
equations
-spread σ
  σ = dmax/√(2*k)
  dmax is the greatest distance between the selected centroids
    σ is RBF.sigma, dmax is given by RBF.finddmax(), k is RBF.k
-weight update
  ∆wj^t = (1-α)*η*∇wj*Err + α*∆wj^(t-1)
  α and η are tunable parameters, t represents time, j denotes which centroid this is the weight of, Err is the error function
    α is @@@.alpha, η is @@@.eta, t is @@@.time, j is the index of weights, Err is @@@@@@@@
-Linear Activation Function
  ϕj(x) = ||x - xj||
    xj is the current centroid, x is the centroid of the centroids
      xj is RBF.centroids[j], x is RBF.center
-Gaussian Activation Function
  ϕj(x) = exp[-(||x - xj||^2)/(2*σ^2)] 
    xj is the current centroid, x is the centroid of the centroids, σ is the spread from the centroid
      xj is RBF.centroids[j], x is RBF.center, σ is RBF.sigma
-Hardy’s Multiquadric Activattion Function
  ϕj(x) = √(σ^2 + ||x - xj||^2)
    xj is the current centroid, x is the centroid of the centroids, σ is the spread from the centroid
      xj is RBF.centroids[j], x is RBF.center, σ is RBF.sigma
'''