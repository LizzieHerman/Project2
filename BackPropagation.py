'''
created on Oct 13 2017
@author LizzieHerman
'''
import random
import numpy

class Node:
    def __init__(self):
        self.prevNodes = [Node]
        self.connectsTo = [Node]
        self.outputs = numpy.matrix

    def getOutputs(self):
        return self.outputs

    def blah(self):
        print("blah")
