import numpy as np
import supportFunctions

# generic framework to initialize and train a neural network
# TODO: variable amount of hidden layer
class neuralNetwork:
    
    # initializes the ANN
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # number of nodes in each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        # learning rate
        self.learningRate = learningRate
        
        # random initialization of the weight matrices, i = input, h = hidden, o = output
        self.wih = np.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.who = np.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
        pass
    
    # supervised learning over a specified amount of epochs
    # parameter:
    #    - inputMatrix: each line represents one data set
    #    - labelMatrix: each line represents one label
    #    - epochs: number of times the inputMatrix is used to train the network
    def learn(self, inputMatrix, labelMatrix, epochs):
        
        if labelMatrix.shape[1] != self.outputNodes:
            print("Length of output vector does not equal amount of nodes in the output layer.")
            pass
        
            
        if inputMatrix.shape[1] != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass   
        
        while(epochs > 0):
            
            # iterate over both inputMatrix and labelMatrix
            for inputData, labelData in zip(inputMatrix, labelMatrix):
                # transform to column vector for matrix multiplication
                currentInput = (np.reshape(inputData,(max(np.shape(inputData)),1)))
                currentLabel = (np.reshape(labelData,(max(np.shape(labelData)),1)))
        
                # calculate output for the hidden layer
                hInput = np.dot(self.wih, currentInput)
                hOutput = supportFunctions.sigmoidFunction(hInput)
        
                # calculate final output
                oInput = np.dot(self.who, hOutput)
                oOutput = supportFunctions.sigmoidFunction(oInput)
        
                # get the error
                outputError = currentLabel - oOutput
                hiddenError = np.dot(np.transpose(self.who), outputError)
        
                # change weights between hidden- and output-layer
                self.who += self.learningRate * np.dot(outputError * oOutput * (1.0 - oOutput), np.transpose(hOutput))
        
                # change weights between input- and hidden-layer
                self.wih += self.learningRate * np.dot(hiddenError * hOutput * (1.0 - hOutput), np.transpose(currentInput))
               
            # decrease epoch-counter
            epochs -= 1;
                           
        print(self.wih)
                                                
        pass
        
    # use the ANN to classify the inputData
    # parameter:
    #    - inputData: a vector containing one data set
    # output:
    #    - oOutput: 1-by-self.outputnodes vector containing the classification
    def run(self, inputData):
        
        if max(np.shape(inputData)) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass
        
        currentInput = (np.reshape(inputData,(max(np.shape(inputData)),1)))
        
        # calculate output for the hidden layer
        hInput = np.dot(self.wih, currentInput)
        hOutput = supportFunctions.sigmoidFunction(hInput)
        
        # calculate final output
        oInput = np.dot(self.who, hOutput)
        oOutput = supportFunctions.sigmoidFunction(oInput)
        
        return oOutput
    
    # store the configuration of the neural network on the hard drive
    def saveConfig(self):
        pass
    
    # load the configuration of a neural network from the hard drive
    def loadConfig(self, path):
        pass
    
    # end of class
    pass