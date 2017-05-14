import numpy
import neuronalFunctions

# generic framework to initialize and train a neural network
class neuralNetwork:
    
    # initializes the ANN
    # TODO: allow more hidden layer
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # number of nodes in each layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        # learning rate
        self.learningRate = learningRate
        
        # random initialization of the weight matrices, i = input, h = hidden, o = output
        self.wih = numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.who = numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
        pass
    
    # supervised learning over a specified amount of epochs
    # TODO: backpropagation, change weights, include epochs
    def learn(self, inputData, trainingData, epochs):
        if len(trainingData) != self.outputNodes:
            print("Length of output vector does not equal amount of nodes in the output layer.")
            pass
        
        if len(inputData) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass       
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, inputData)
        self.hOutput = neuronalFunctions.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = neuronalFunctions.sigmoidFunction(self.oInput)
        
        # get the error
        self.error = neuronalFunctions.errorFunction(self.oOutput, trainingData)
        
    
    # use the ANN to classify the inputData
    def run(self, inputData):
        
        if len(inputData) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, inputData)
        self.hOutput = neuronalFunctions.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = neuronalFunctions.sigmoidFunction(self.oInput)
        
        return self.oOutput
    
    # end of class
    pass