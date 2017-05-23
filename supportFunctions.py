import numpy as np
import math

# neural activation function  
def sigmoidFunction(inputData):
        
    hOutput = np.zeros([len(inputData),1])
        
    for i, inputSample in enumerate(inputData):
        hOutput[i] = 1 / (1 + math.e**(-inputSample))
        
    return hOutput

# error function
def errorFunction(inputData, trainingData):
    
    error = np.zeros([len(inputData),1])
    
    for i, trainingSample in enumerate(trainingData):
        error[i] = (inputData[i] - trainingSample)**2 
        
    return error

# imports the dataset from the .csv-file and transforms it according to the ANN architecture
def importMNIST(name):
    
    # import data
    dataFile = open("./data/" + name,"r")
    fullData = dataFile.readlines()
    dataFile.close()
    
    # pre-allocate space
    dataMatrix = np.zeros((len(fullData), len(fullData[0].split(","))))
    
    for i, currentDataSet in enumerate(fullData):
        # transforms imported data into matrix containing integer values
        dataMatrix[i] = list(map(int,currentDataSet.split(",")))
    
    # integer range from [0 255] -> [~0.1 ~0.999] to fit the intervall of the sigmoid function
    dataMatrix[:,1:] = ((dataMatrix[:,1:] / 255)  + 0.01) * 0.99
    
    # dataMatrix[i] returns the i-th dataset with the correct result as the first element
    return dataMatrix