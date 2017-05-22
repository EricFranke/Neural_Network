import numpy
import supportFunctions

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
        
        self.input = numpy.array(inputData,ndmin=2)
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, self.input)
        self.hOutput = supportFunctions.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = supportFunctions.sigmoidFunction(self.oInput)
        
        # get the error
        self.OutputError = supportFunctions.errorFunction(self.oOutput, trainingData)
        
        # change weights between hidden- and output-layer
        self.who += self.learningRate * numpy.dot(self.OutputError * self.oOutput * (1 - self.oOutput), numpy.transpose(self.hOutput))
        
        # change weights between input- and hidden-layer
        # self.HiddenError
        
    # use the ANN to classify the inputData
    def run(self, inputData):
        
        if len(inputData) != self.inputNodes:
            print("Length of input vector does not equal the amount of nodes in the input layer.")
            pass
        
        # calculate output for the hidden layer
        self.hInput = numpy.dot(self.wih, inputData)
        self.hOutput = supportFunctions.sigmoidFunction(self.hInput)
        
        # calculate final output
        self.oInput = numpy.dot(self.who, self.hOutput)
        self.oOutput = supportFunctions.sigmoidFunction(self.oInput)
        
        return self.oOutput
    
    # end of class
    pass

myNetwork = neuralNetwork(3,5,3,0.2)
trainingData = numpy.random.rand(3,1)
inputData = numpy.random.rand(3,1)
myNetwork.learn(inputData, trainingData, 1)