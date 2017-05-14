import numpy
import math
from numpy import ones

# generic framework to initialize and train a neural network
class neuralNetwork:
    
    # TODO: allow more hidden layer
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # number of nodes in each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        # learning rate
        self.learningRate = learningRate
        
        # weight matrices, i = input, h = hidden, o = output
        self.wih = numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.who = numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
        pass
    
    def learn(self, inputData, trainingData, epochs):
        if len(trainingData) != self.outputNodes:
            print("Length of output vector does not equal amount of nodes in the output layer.")
            pass
        
        if len(inputData) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass       
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, inputData)
        self.hOutput = self.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = self.sigmoidFunction(self.oInput)
        
        # calculate the error
        self.error = numpy.zeros(len(self.hOutput))
        
        for i in range(len(trainingData)):
            self.error[i] = (self.hOutput[i] - trainingData[i])**2 
        
        return self.error
    
    def run(self, inputData):
        
        if len(inputData) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, inputData)
        self.hOutput = self.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = self.sigmoidFunction(self.oInput)
        
        return self.oOutput
    
    @staticmethod
    def sigmoidFunction(inputData):
        
        hOutput = numpy.zeros(len(inputData))
        
        for i in range(len(inputData)):
            hOutput[i] = 1 / (1 + math.e**(-inputData[i]))
        
        return hOutput
    # end of class
    pass