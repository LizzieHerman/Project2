#It turns out I kind of hate object based python, but that just might be because this is my first excusion into it.

input = [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
weights = 0.1


class Node(object):
    parent = ""
    child = 0
    

    # The class "constructor" - It's actually an initializer 
    def __init__(self, parent, child, weights):
        print ("node initialized")
        self.parent = parent
        self.age = child
        self.major = stuff

def newNode(parent, child, weights):
    node = Node(parent, child, weights)
    return node

def updateWeights(input):
    pass
    
#Not a true matrix multiplication, yet. I'm still trying to figure out if it could possibly be output as a different dimension
#The only sane way seems to be that our matricies will always be the same size square, so no change in output dimensionality, no new sized arrays and no checking for compatability.
def scaler(input):
    for x in range(len(input)):
        for y in range(len(input[x])):
            print(input[x][y])
            input[x][y] = input[x][y]*weights
            print(input[x][y])
            

#The below function multipies a matrix we know, such as our weights, and uses it as the first factor in a matrix multiplication where the second factor is an input matrix
#the logic was built assuming only square matrixies of the same size, but should work for any size matrix fitting this criteria.

#Visualization of a 3 by 3 matrix's coordinate grid:
#(0,0)    (1,0)   (2,0)
#(0,1)    (1,1)   (2,1)
#(0,2)    (1,2)   (2,2)

def matrixMult(input):  #this method looks far too simple for the nightmare it turned out to be.
    
    #"Matrix" * Input
    
    input = [[1,2,3], [3,2,1], [1,2,3]] #                                                     #***This Matrix is a LIST of COLUMNS***
    matrix = [[1,0,0], [3,2,0], [0,0,1]] # the identity matrix, or an aproximation, for testing  #***This matrix is a LIST of ROWS***

    outputMatrix = [[0,0,0],[0,0,0],[0,0,0]] #an empty matrix for saving outputs
    
    #CRITICAL: "matrix" IS OUR FIRST FACTOR, AND input IS OUR SECOND FACTOR IN THIS MULTIPLICATION
    
    xCoord = -1     #helps keep track of the index we're reading in more readable terms. (0,0) denotes the top left corner of our matrix, where X = 0 is the top and X = 2 would be the bottom, Y=0 the leftmost and Y=2 the rightmost of the 3 by 3 matrix.
    for column in input:    #for each column in our input matrix
        yCoord = -1
        xCoord += 1     #helps keep track of the index we're reading in more readable terms
        dotProduct = 0
        
        #outputX = len(outputMatrix)-(1+yCoord+1)     #short for "output X coordinate," to determine what cell the output of our multiplication will go to in the output matrix. It counts backwards in rows of the output matrix as we go further in rows of our first factor.
        outputY = xCoord                           #The Y position is simply the same "height" as the value we're pulling from our first factor.
        
        for columnVal in column: #for each value held in that column
            yCoord += 1
            outputX = yCoord
            
            rowNum = -1
            for row in matrix: #for each row in our first factor
                rowNum += 1
                print ("taking value " + str(column[rowNum]) + " times the value " + str(matrix[yCoord][rowNum]))    # simplifided debugging line
                dotProduct = dotProduct + (column[rowNum] * matrix[yCoord][rowNum])
            print ("our dot product for these opperations is " + str(dotProduct))
            print ("our output goes to our output's matrix coordinates " + str(outputY) + "," + str(outputX) + " in our output matrix")  #debugging line
            outputMatrix[outputY][outputX] = dotProduct # outputMatrix is CURRENTLY FORMATTED AS A LIST OF COLUMNS
        print (outputMatrix)
           
        

#processInput(input)
matrixMult(input)
            