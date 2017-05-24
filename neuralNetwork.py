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
    def learn(self, inputData, trainingData, epochs):
        
        if max(np.shape(trainingData)) != self.outputNodes:
            print("Length of output vector does not equal amount of nodes in the output layer.")
            pass
        
            
        if max(np.shape(inputData)) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass   
        
        while(epochs > 0):
            # transform to column vector for matrix multiplication
            self.input = (np.reshape(inputData,(max(np.shape(inputData)),1)))
            self.label = (np.reshape(trainingData,(max(np.shape(trainingData)),1)))
        
            # calculate output for the hidden layer
            self.hInput = np.dot(self.wih, self.input)
            self.hOutput = supportFunctions.sigmoidFunction(self.hInput)
        
            # calculate final output
            self.oInput = np.dot(self.who, self.hOutput)
            self.oOutput = supportFunctions.sigmoidFunction(self.oInput)
        
            # get the error
            self.outputError = supportFunctions.errorFunction(self.oOutput, self.label)
            self.hiddenError = np.dot(np.transpose(self.who), self.outputError)
        
            # change weights between hidden- and output-layer
            self.who += self.learningRate * np.dot(self.outputError * self.oOutput * (1 - self.oOutput), np.transpose(self.hOutput))
        
            # change weights between input- and hidden-layer
            self.wih += self.learningRate * np.dot(self.hiddenError * self.hOutput * (1 - self.hOutput), np.transpose(self.input))
               
            # decrease epoch-counter
            epochs -= 1;
                                                
        pass
        
    # use the ANN to classify the inputData
    def run(self, inputData):
        
        if max(np.shape(inputData)) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass
        
        self.input = (np.reshape(inputData,(max(np.shape(inputData)),1)))
        
        # calculate output for the hidden layer
        self.hInput = np.dot(self.wih, self.input)
        self.hOutput = supportFunctions.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = np.dot(self.who, self.hOutput)
        self.oOutput = supportFunctions.sigmoidFunction(self.oInput)
        
        return self.oOutput
    
    # store the configuration of the neural network on the hard drive
    def saveConfig(self):
        pass
    
    # load the configuration of a neural network from the hard drive
    def loadConfig(self, path):
        pass
    
    # end of class
    pass
