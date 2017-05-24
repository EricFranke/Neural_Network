import numpy as np
import math

# neural activation function  
def sigmoidFunction(inputData):
        
    hOutput = np.zeros([len(inputData),1])
        
    for i, inputSample in enumerate(inputData):
        hOutput[i] = 1 / (1 + math.e**(-inputSample))
        
    return hOutput

# error function
def errorFunction(outputData, trainingData):
    
    error = np.zeros((max(np.shape(trainingData)),1))
    
    for i, trainingSample in enumerate(trainingData):
        error[i] = (outputData[i] - trainingSample)**2 
        
    return error

# imports the mnist csv-file and transforms it according to the ANN architecture
# return values: 
#    - dataMatrix: every row represents a 28-by-28 image (784 elements)
#    - labelMatrix: every row represents the correct result for training purposes (10 elements - one for each number)
def importMNIST(file):
    
    # import data
    dataFile = open("./data/" + file,"r")
    fullData = dataFile.readlines()
    dataFile.close()
    
    # pre-allocate space
    dataMatrix = np.zeros((len(fullData), len(fullData[0].split(","))-1))
    labelMatrix = np.zeros((len(fullData), 10)) + 0.1
    
    for i, currentDataSet in enumerate(fullData):
        # transforms imported data into matrix containing integer values
        currentDataSet = list(map(int,currentDataSet.split(",")))
        
        labelMatrix[i,currentDataSet[0]] = 0.999
        dataMatrix[i] = currentDataSet[1:]
    
    # integer range from [0 255] -> [~0.1 ~0.999] to fit the range of the sigmoid function
    dataMatrix = ((dataMatrix / 255)  + 0.01) * 0.99
    
    return (dataMatrix, labelMatrix)