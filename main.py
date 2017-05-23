import supportFunctions
import numpy
import matplotlib.pyplot as plt

dataList = supportFunctions.importMNIST();

allValues = dataList[0].split(",")
image = numpy.asfarray(allValues[1:]).reshape((28,28))

plt.imshow(image)