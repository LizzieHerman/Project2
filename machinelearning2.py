#Alright, so this will give us our grid,
#but we need a way to pull values out of it for inputs.

#import numpy as np
import random

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
        
        
#Makes a matrix to represent the edges that connect layers to facilitate the matric multiplication between an input matrix and a layer of nodes. 
#Then just need the activation function for the nodes to turn the result into another input.
def newLayer(rows, columns, weight):    #weight should start arbitrairily low, such as .1, and ramp up/refine with backprop
    for x in range(rows):    
        rowVal = [weight for i in range(columns)]
    weightMatrix.append(rowVal) #adding a new row to our list of rows.
#last issue is updating individual weights algorithmically -- or just save each individual layer of weights -- so we can continue to use those.
        
        
        
#using Rosenbrock to produce input matrixes:

#not quite working as a generic function, abandoned for making a method for a 2d rosen, 3d, 4d, etc...
#DISREGUARD -- not operational
def rosenInput(dimensions):                         #IF WE WANT A SMALLER TESTPLANE, REDUCE THE 10** to the size of the new square for each plane
    rosenOutput = [[0 for x in range(dimensions+1)] for y in range(10**dimensions)]  #dimensions for a number of rows matching the dimensionality of our Rosenbrock, +1 column for the output of Rosenbrock for those values in the row 
    
    #10^dimensions to hold all X, Y (Z, etc...) combinations for a 10 by 10 (by 10 by 10...) grid.
    #in this particular case, for a 2D Rosenbrock... we need a loop for each dimension, how do we do that, besides making a seperate method for each dimensionality we plan to use.
    offset = -1
    for d in range(dimensions-1):
        offset += 1     #so that our matrix gives combinations of vectors, not the same 10 vectors over and over and over again.
        for x in range(0,10**dimensions):    
            #print('x = ', x)
            if x%10 == 0 and x != 0:
                rosenOutput[x][d] = 10
            else:
                rosenOutput[x][d] = x%10+offset #our first column is our X values, our second our Y
                #now that our matrix is full our input vectors, we need to get the output of those vectors when plugged into the Rosenbrock for the last value in a given row.
            print(rosenOutput[x])
    for x in range(rosenOutput[y]):   
        vector = ''
        for y in range(dimensions-1):
            vector.extend(rosenOutput[x][y]) #getting all of our row's vectors in a single list
        rosenOutput[x][dimensions+1] = rosenbrock(vector)   #our last value in the row is output of this vector which comprises the row plugged into the rosenbrock function
            #we can only do this one way-- each column (or, alternatively, row) of our input matrix must be entirely X, Y, Z, or etc values, and the last column would be the output of the Rosenbrock
    print("rosenbrock finished")
    print(rosenOutput)
    
    
#Our FUNCTIONING Input generation methods:

#For each of our 100 indeces, our X counts from 0 to 9, and then our Y iterates by 1. The last value is the Rosenbrock of our X and Y.    
def rosen2D():       
     rosenOutput = [[y for x in range(2+1)] for y in range(100)]
     for x in range (100):
        rosenOutput[x][0] = x%10       #our "x" position counts from 0 to 9
        rosenOutput[x][1] = int(x/10)  #our "y" position counts how many 10s of Xes have occured
        rosenOutput[x][2] = rosenbrock([x%10,int(x/10)])    #our last position is always the Rosenbrock of X and Y
        #print(rosenOutput[x])  #To display the output of this function, decomment this line.

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
           print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line. 
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
           print(rosenOutput[trueIndex])  #To display the output of this function, decomment this line. 
           trueIndex +=1
     #print rosenOutput Debugging
     return rosenOutput

#a = [[1, 0], [0, 1]]
#b = [[4, 1], [2, 2]]
#result = np.dot(a, b)		#testing how np.dot() works
#rosenInput(3)


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

            if output[x][criticalIndex] == actualValue:
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
        print ("Accuracy of ", printableAcc, "for tollerance of ", tollerenceVal)
    print ("Final tolerance of ", tollerenceVal, " achieved an accuracy of ", printableAcc)
    
    #algortithm training logic to be implemented.
def train(input):
    pass

#generates a manageable sample from our compelete range of inputs for use in K-fold cross validation
#four our smallest, and only for our smallest sample spaces, it makes more sense just to use the entire 3 by 3 sample size.
def extractSample(input, sampleSize, numSamples):
    #print "occured"
    sampleEntry = [0 for y in range(sampleSize)]
    sample = [[] for y in range(numSamples)]
    for i in range(sampleSize): 
        for j in range(numSamples):
            sampleEntry[j] = input[random.randint(0,len(input))]
            sample[i].append(sampleEntry[j])
        print sample[i]
    #we have our samples, now we perform cross validation:
    for k in range(sampleSize):     #our K folds
        for i in range(sampleSize): 
            if k == i:
                pass
            else:
                train(sample[i]) #we train on all but one sample,
        assessOutput(sample[k]) #then validate our tests using that one excluded sampme, and repeat K times.
        

def selftest():
    print "Initial values:"
    testtest1 = smallRosen2D()
    print "Second Matrix:"
    testtest2 = smallRosen3D()
    
    print
    print
    print "Modified first matrix: initial Accuracy should be 33% and final Tolerance Value 2"

    testtest1[0][2] = 2
    print testtest1[0]
    testtest1[1][2] = 51
    print testtest1[1]
    testtest1[2][2] = 801
    print testtest1[2]
    testtest1[3][2] = 51
    print testtest1[3]
    testtest1[4][2] = 101
    print testtest1[4]
    testtest1[5][2] = 851
    print testtest1[5]
    print testtest1[6],  'unmodified'
    print testtest1[7],  'unmodified'
    print testtest1[8],  'unmodified'
    
    print
    print
    print "Modified first matrix: initial Accuracy should be 7% and final Tolerance Value 5"
    
    testtest2[0][3] = .4
    print testtest2[0]
    testtest2[1][3] = 20.2
    print testtest2[1]
    testtest2[2][3] = 321
    print testtest2[2]
    testtest2[3][3] = 510
    print testtest2[3]
    testtest2[4][3] = 1005
    print testtest2[4]
    testtest2[5][3] = 851
    print testtest2[5]
    testtest2[6][3] = 1
    print testtest2[6]
    testtest2[7][3] = 51
    print testtest2[7]
    testtest2[8][3] = 4001
    print testtest2[8]
    testtest2[9][3] = 1005
    print testtest2[9]
    testtest2[10][3] = 500
    print testtest2[10]
    testtest2[11][3] = 5005
    print testtest2[11]
    testtest2[12][3] = 505
    print testtest2[12]
    testtest2[13][3] = 851
    print testtest2[13]
    testtest2[14][3] = 181
    print testtest2[14]
    testtest2[15][3] = 1005
    print testtest2[15]
    testtest2[16][3] = 1000
    print testtest2[16]
    testtest2[17][3] = 5005
    print testtest2[17]
    testtest2[18][3] = 10010
    print testtest2[18]
    testtest2[19][3] = 341
    print testtest2[19]
    testtest2[20][3] = 10010
    print testtest2[20]
    testtest2[21][3] = 6510
    print testtest2[21]
    testtest2[22][3] = 5005
    print testtest2[22]
    testtest2[23][3] = 261
    print testtest2[23]
    testtest2[24][3] = 161
    print testtest2[24]
    print testtest2[25], 'unmodified'
    print testtest2[26], 'unmodified'
    
    #print len(testtest2) #returns 27, as expected
    
    assessOutput(testtest1)
    print
    assessOutput(testtest2)




#selftest()
fullInputRange = rosen4D()
extractSample(fullInputRange,10,10)


#input = an m by n matrix, where n is our number of examples and m is how many features each example has. 
#weights = a matrix where each row are the weights of an edge leading from the previous layer to the next.


##incomplete conseptualizations
#def feedfwd(inputs):
#	#for the first layer only
#	layer.activations = np.dot(layer, layer.weights)
#	for hiddenLayer in neuralNet:
#		hiddenLayer.activations = hiddenLayer.activations * np.dot(hiddenLayer, hiddenLayer.weights)
#
##Not even close to complete, probably not even sensical at this point.		
#def backprop():  #here's where the reference material completely loses me. 
#    for layer in neuralNet:
#        for node in layer:
#            node.weight = deriv * node.weight * NextLayerNode.weight	#as best as I can understand, this the function used to update weights, the actual multiplication is a matrix multiplication though.
#			#node.weight = np.dot(deriv, node.weight) * NextLayerNode.weight?