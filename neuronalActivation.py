import numpy
import math
  
def sigmoidFunction(inputData):
        
    hOutput = numpy.zeros(len(inputData))
        
    for i in range(len(inputData)):
        hOutput[i] = 1 / (1 + math.e**(-inputData[i]))
        
    return hOutput