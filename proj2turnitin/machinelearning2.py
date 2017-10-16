'''
created on Oct 4 2017
@author RyanFreivalds, LizzieHerman
'''
#Alright, so this will give us our grid,
#but we need a way to pull values out of it for inputs.

#import numpy as np
import random
import numpy
import mlPerceptron
import RadialBasis

#need to make layers, not individual nodes. Still unsure if there's any way to "keep our inputs intact" but as of now it seems we have to split every simension into another class.

#neuralNet = [(node1, node2, node3), (node4, node5, node6, node7, node8, node9), (node10)]


#reference page: http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5
#the later pat of this article actually has the author implementing a neural network, using python, and it is beyond me. 

#The one method I made that I think could actually be useful
def rosenbrock(variables):
    value = 0
    #print (4**2)
    for x in range(0, len(variables)-1, 1):
        #print (variables[x])
        #print (variables[x+1])
        value = value + (1 - variables[x])**2 + 100*(variables[x+1]-variables[x]**2)**2 #the Equasion taken straight from the assignment sheet,
    #combinedInputs.append(variables, value) 
    return value

#Our FUNCTIONING Input generation methods:

#For each of our 100 indeces, our X counts from 0 to 9, and then our Y iterates by 1. The last value is the Rosenbrock of our X and Y.    
def rosen2D():       
     rosenOutput = [[y for x in range(2+1)] for y in range(100)]
     for x in range (100):
        rosenOutput[x][0] = x%10       #our "x" position counts from 0 to 9
        rosenOutput[x][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
        rosenOutput[x][2] = rosenbrock([x%10,int(x/10)])    #our last position is always the Rosenbrock of X and Y
        #print(rosenOutput[x])  #To display the output of this function, decomment this line.
     return rosenOutput

#For each of our 1000 indeces, our X counts from 0 to 9, and then our Y iterates by 1. When Y iterates past 9 we return it to 0 and iterate Z, The last value is the Rosenbrock of our X and Y.    
def rosen3D():       
     rosenOutput = [[y for x in range(3+1)] for y in range(1000)]
     trueIndex = 0
     for y in range (10):
        for x in range (100):
           rosenOutput[trueIndex][0] = x%10       #our "x" position counts from 0 to 9
           rosenOutput[trueIndex][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
           rosenOutput[trueIndex][2] = y          #our "z" position which counts how many 10s of Ys has occured
           rosenOutput[trueIndex][3] = rosenbrock([x%10,int(x/10),y])    #our last position is always the Rosenbrock of X and Y and Z
           #print (x*(y+1), trueIndex) #debugging line
           trueIndex+=1
           #print(rosenOutput[x*(y+1)])  #To display the output of this function, decomment this line.
     return rosenOutput      
        
#For each of our 10000 indeces, our X counts from 0 to 9, and then our Y iterates by 1. When Y iterates past 9 we return it to 0 and iterate Z, and when Z iterates past 9... The last value is the Rosenbrock of our X and Y.    
def rosen4D():       
     rosenOutput = [[y for x in range(4+1)] for y in range(10000)]
     zedCount = 0
     trueIndex = 0
     for y in range (100):
        for x in range (100):
           rosenOutput[trueIndex][0] = x%10       #our "x" position counts from 0 to 9
           rosenOutput[trueIndex][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
           rosenOutput[trueIndex][2] = y%10       #our "z" position which counts how many 10s of Ys has occured
           rosenOutput[trueIndex][3] = int(zedCount/10)       #our dimension four position, which continues the trend
           rosenOutput[trueIndex][4] = rosenbrock([x%10,int(x/10),y%10, int(zedCount/10)])    #our last position is always the Rosenbrock of X and Y and Z
           #print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line. 
           #print (x*(y+1), trueIndex) #debugging line
           trueIndex += 1
        zedCount +=1
     #print "occured1"
     return rosenOutput
#For each of our 100000 indeces, our X counts from 0 to 9, and then our Y iterates by 1. When Y iterates past 9 we return it to 0 and iterate Z, and when Z iterates past 9... The last value is the Rosenbrock of our X and Y.
def rosen5D():
     rosenOutput = [[y for x in range(5+1)] for y in range(100000)]
     zedCount = 0
     trueIndex = 0
     for y in range (100):
        for x in range (1000):
           rosenOutput[trueIndex][0] = x%10       #our "x" position counts from 0 to 9
           rosenOutput[trueIndex][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
           rosenOutput[trueIndex][2] = y%10       #our "z" position which counts how many 10s of Ys has occured
           rosenOutput[trueIndex][3] = int(zedCount / 10)  # our dimension four position, which continues the trend
           rosenOutput[trueIndex][4] = int(zedCount % 10)  # our dimension 5 position, which continues the trend
           rosenOutput[trueIndex][5] = rosenbrock([x % 10, int(x / 10), y % 10, int(zedCount / 10), int(zedCount%10)])  # our last position is always the Rosenbrock of X and Y and Z
           #print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line.
           #print (x*(y+1), trueIndex) #debugging line
           trueIndex += 1
        zedCount +=1
     #print "occured1"
     return rosenOutput
#For each of our 1000000 indeces, our X counts from 0 to 9, and then our Y iterates by 1. When Y iterates past 9 we return it to 0 and iterate Z, and when Z iterates past 9... The last value is the Rosenbrock of our X and Y.
def rosen6D():
     rosenOutput = [[y for x in range(6+1)] for y in range(1000000)]
     zedCount = 0
     trueIndex = 0
     for y in range (1000):
        for x in range (1000):
           rosenOutput[trueIndex][0] = x%10       #our "x" position counts from 0 to 9
           rosenOutput[trueIndex][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
           rosenOutput[trueIndex][2] = y%10       #our "z" position which counts how many 10s of Ys has occured
           rosenOutput[trueIndex][3] = int(zedCount/10)       #our dimension four position, which continues the trend
           rosenOutput[trueIndex][4] = int(zedCount % 10)  # our dimension five position, which continues the trend
           rosenOutput[trueIndex][5] = int(y / 10)  # our dimension six position, which continues the trend
           rosenOutput[trueIndex][6] = rosenbrock([x % 10, int(x / 10), y % 10, int(zedCount / 10), int(zedCount%10), int(y/10)])  # our last position is always the Rosenbrock of X and Y and Z
           #print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line.
           #print (x*(y+1), trueIndex) #debugging line
           trueIndex += 1
        zedCount +=1
     #print "occured1"
     return rosenOutput

#these "small" methods are functionally identical to the above, but make 3 by 3 planes rather than 10 by 10s.
def smallRosen4D():       
     rosenOutput = [[y for x in range(4+1)] for y in range(81)]
     trueIndex=0
     zedCount = 0
     for y in range (27):
        for x in range (3):
           rosenOutput[trueIndex][0] = x%3       #our "x" position counts from 0 to 3
           rosenOutput[trueIndex][1] = int(y/3)%3  #our "y" position counts how many 3 of Xes have occured
           rosenOutput[trueIndex][2] = y%3       #our "z" position which counts how many 3s of Ys has occured
           rosenOutput[trueIndex][3] = int(zedCount/3)%3       #our dimension four position, which continues the trend
           rosenOutput[trueIndex][4] = rosenbrock([x%3,int(y/3)%3,y%3, int(zedCount/3)%3 ])    #our last position is always the Rosenbrock of X and Y and Z
           print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line. 
           #print (x*(y+1), trueIndex) #debugging line
           trueIndex+=1
        zedCount +=.3
     return rosenOutput
        
def smallRosen3D():       
     rosenOutput = [[y for x in range(3+1)] for y in range(27)]
     trueIndex=0
     for y in range (9):
        for x in range (3):
           rosenOutput[trueIndex][0] = x%3       #our "x" position counts from 0 to 3
           rosenOutput[trueIndex][1] = int(y/3)%3  #our "y" position counts how many 3s of Xes have occured
           rosenOutput[trueIndex][2] = y%3       #our "z" position which counts how many 3s of Ys has occured
           rosenOutput[trueIndex][3] = rosenbrock([x%3,int(y/3)%3,y%3])    #our last position is always the Rosenbrock of X and Y and Z
           #print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line.
           trueIndex+=1
     return rosenOutput
          
def smallRosen2D():       
     rosenOutput = [[y for x in range(2+1)] for y in range(9)]
     trueIndex=0
     for y in range (3):
        for x in range (3):
           rosenOutput[trueIndex][0] = x%3       #our "x" position counts from 0 to 3
           rosenOutput[trueIndex][1] = y%3       #our "z" position which counts how many 3s of Ys has occured
           rosenOutput[trueIndex][2] = rosenbrock([x%3,y%3])    #our last position is always the Rosenbrock of X and Y and Z
           #print trueIndex Debugging
           #print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line.
           trueIndex +=1
     #print rosenOutput Debugging
     return rosenOutput

# generates a manageable sample from our compelete range of inputs for use in K-fold cross validation
# four our smallest, and only for our smallest sample spaces, it makes more sense just to use the entire 3 by 3 sample size.
# were using 5 X 2 cross validation
def extractSample(input, sampleSize):
    # print "occured"
    for a in range(5):
        sampleEntry = [0 for y in range(sampleSize)]
        sample = [[] for y in range(2)]
        for j in range(sampleSize):
            for i in range(2):
                sampleEntry[j] = input[random.randint(0, len(input) - 1)]
                sample[i].append(sampleEntry[j])
        train(sample[0], sample[1])
        train(sample[1], sample[0])


def assessOutput(output):
    tollerenceVal = 0
    accuracy = 0
    while (accuracy < float(0.49)): #our accuracy cutoff
        tollerenceVal+=1
        numCorrect = 0
        criticalIndex = len(output[0])-1    #where the rosenbrock output is stored
        #print criticalIndex
        vectorIndex = len(output[0])-1      #the range of indecies where vector coordubates are kept
        #print vectorIndex
        for x in range(0, len(output)):    #for every vector in our output matrix

            vector = []
            for y in range(0, vectorIndex): #for all vector coordinates
                vector.extend([output[x][y]])   #put them together so we can plug them back into the rosenbrock

            #print vector #debugging line
            actualValue = rosenbrock(vector)    #the actual value our function was trying to approximate, for this vector
            #print actualValue, output[x][criticalIndex] #debugging line

            if output[x][criticalIndex] is actualValue:
                numCorrect +=1
                #print"exact match" #debugging line
            elif output[x][criticalIndex] < actualValue:  #if we under-approximated
                if output[x][criticalIndex] * tollerenceVal >= actualValue: #check if the tollerance value brings us in "range" of a correct answer
                    numCorrect +=1
                    #print "tollerance match" #debugging line
            elif output[x][criticalIndex] > actualValue:  #if we over-approximated, do the same
                if output[x][criticalIndex] / tollerenceVal <= actualValue:
                    numCorrect +=1
                    #print "second tollerance match" #debugging line
        accuracy = (numCorrect/float(len(output)))
        printableAcc = '{:1.3f}'.format(accuracy)
        print "Accuracy of ", printableAcc, "for tollerance of ", tollerenceVal
    print "Final tolerance of ", tollerenceVal, " achieved an accuracy of ", printableAcc

#finds average error of the outputs
def averageError(input=numpy.matrix):
    # input in form first val is output given by function rest is the input vector
    error = 0
    for i in range(len(input)):
        out = input[i][0]
        vector = []
        for j in range(1, len(input[i])):
            vector.append(input[i][j])
        actual = rosenbrock(vector)
        error += abs(actual-out)
    error = error / len(input)
    print "\tThe average error produced by this Neural Net was: ", error

#builds networks, trains and tests them
def train(input, test):
    dim = len(input[0])
    length = len(input)
    points = numpy.zeros(shape=(length, dim-1))
    targs = numpy.zeros(shape=(length,1))
    tpoints = numpy.zeros(shape=(length, dim-1))
    ttargs = numpy.zeros(shape=(length,1))
    output = numpy.zeros(shape=(length, dim))
    for i in range(length):
        targs[i] = input[i][dim - 1]
        ttargs[i] = test[i][dim-1]
        output[i][0] = test[i][dim-1]
        for j in range(dim-1):
            points[i][j] = input[i][j]
            tpoints[i][j] = test[i][j]
            output[i][j+1] = test[i][j]
    for a in range(3):
        print "\n\tMLP- Hidden Layers: ", a, " # inputs: ", length, " Nodes in Hidden Layers: 15 Learning Rate: 0.1 RandWeightBounds: [-0.5,0.5]"
        percept = mlPerceptron.MLP(points, targs, length, (length - 5), length, dim - 1, a)
        #percept.train(0.5, 0.1) # train without momentum
        print "\t\tTraining with momentum, alpha = 0.25"
        percept.train(0.5,0.1,0.25) # train with momentum
        #output = percept.test(tpoints)
        #assessOutput(output)
        averageError(output)
    for a in range(3):
        if a == 0:
            name = "Linear"
        elif a == 1:
            name = "Guassian"
        else:
            name = "Hardy's"
        print "\n\tRBF- Activation Function: ", name, " # inputs: ", length, " Clusters: ", length/3,  " Learning Rates: 0.1 RandWeightBounds: [-0.5,0.5]"
        radial = RadialBasis.RBF(points,targs,length,length/3,length,dim-1)
        radial.train(0.5,0.1,0.1,a)
        #output = radial.test(tpoints)
        #assessOutput(output)
        averageError(output)


print "2D Rosenbrock"
fullInputRange = rosen2D()
extractSample(fullInputRange,20)
print "3D Rosenbrock"
fullInputRange = rosen3D()
extractSample(fullInputRange,20)
print "4D Rosenbrock"
fullInputRange = rosen4D()
extractSample(fullInputRange,20)
print "5D Rosenbrock"
fullInputRange = rosen5D()
extractSample(fullInputRange,20)
print "6D Rosenbrock"
fullInputRange = rosen6D()
extractSample(fullInputRange,20)
