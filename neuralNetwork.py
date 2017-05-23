import numpy
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
        self.wih = numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5
        self.who = numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5
        pass
    
    # supervised learning over a specified amount of epochs
    def learn(self, inputData, trainingData, epochs):
        
        print(self.who)
        
        while(epochs > 0):
            
            if len(trainingData) != self.outputNodes:
                print("Length of output vector does not equal amount of nodes in the output layer.")
                pass
        
            
            if len(inputData) != self.inputNodes:
                print("Length of input vector does not equal the amount of nodes in the input layer.")
                pass       
        
            self.input = numpy.array(inputData, ndmin=2)
        
            # calculate output for the hidden layer
            self.hInput = numpy.dot(self.wih, self.input)
            self.hOutput = supportFunctions.sigmoidFunction(self.hInput)
        
            # calculate final output
            self.oInput = numpy.dot(self.who, self.hOutput)
            self.oOutput = supportFunctions.sigmoidFunction(self.oInput)
        
            # get the error
            self.outputError = supportFunctions.errorFunction(self.oOutput, trainingData)
            self.hiddenError = numpy.dot(numpy.transpose(self.who), self.outputError)
        
            # change weights between hidden- and output-layer
            self.who += self.learningRate * numpy.dot(self.outputError * self.oOutput * (1 - self.oOutput), numpy.transpose(self.hOutput))
        
            # change weights between input- and hidden-layer
            self.wih += self.learningRate * numpy.dot(self.hiddenError * self.hOutput * (1 - self.hOutput), numpy.transpose(self.input))
               
            # decrease epoch-counter
            epochs -= 1;
            
        print(self.who)
                                                
        pass
        
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
    
    # store the configuration of the neural network on the hard drive
    def saveConfig(self):
        pass
    
    # load the configuration of a neural network from the hard drive
    def loadConfig(self, path):
        pass
    
    # end of class
    pass