import numpy
import math
  
def sigmoidFunction(inputData):
        
    hOutput = numpy.zeros([len(inputData),1])
        
    for i, inputSample in enumerate(inputData):
        hOutput[i] = 1 / (1 + math.e**(-inputSample))
        
    return hOutput

def errorFunction(inputData, trainingData):
    
    error = numpy.zeros([len(inputData),1])
    
    for i, trainingSample in enumerate(trainingData):
        error[i] = (inputData[i] - trainingSample)**2 
        
    return error

def importMNIST():
    
    dataFile = open("./data/mnist_train.csv","r")
    dataList = dataFile.readlines()
    dataFile.close()
    return dataList