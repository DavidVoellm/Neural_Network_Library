from timeit import timeit 
from library import *
import nnfs
from nnfs.datasets import spiral_data, vertical_data
nnfs.init()
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values

myNN = Network(LossFunc=CategoricalCrossEntropy())
myNN.addLayer(Layer_Dense(2, 5, ReLU()))
myNN.addLayer(Layer_Dense(5, 3, ReLU()))
myNN.addLayer(Layer_Dense(3, 3, Softmax()))

print(myNN.run(np.array([3,1])))
print("pause")
print(myNN.run(np.array(X))[:5])
print("Loss:\t"+ str(myNN.calcLoss(y)))
print("Accuracy:\t"+ str(myNN.calcAccuracy(y)))