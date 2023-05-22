import numpy as np
import nnfs
import matplotlib.pyplot as plt
from library import *
from nnfs.datasets import spiral_data, vertical_data
nnfs.init()

# Create Data
X, y = vertical_data(samples = 100, classes = 3)
plt.style.use('dark_background')
plt.title("Data")
plt.scatter(X[:,0], X[:,1], c=y, s=1, cmap="brg")
plt.show()

# Prediction Visualisation
def predict(NN, name = "Neural Network", intensity = 130, area:list[tuple[int]] = [(np.amin(X[:,0])-0.05,np.amax(X[:,0])+0.05),(np.amin(X[:,1])-0.05,np.amax(X[:,1])+0.05)]):
    plt.title(name)
    plt.style.use('dark_background')
    x1, x2 = area[0]
    y1,y2 = area[1]
    predX = []
    n=intensity
    for i in range (int(n * (x2-x1))):
        for j in range (int(n * (y2-y1))):
            predX.append(np.array([i/n+x1,j/n+y1]))
    predX = np.array(predX)

    output = NN(predX)
    plt.scatter(predX[:,0], predX[:,1], c=output, s=80, cmap="brg", alpha=0.03)
    plt.scatter(X[:,0], X[:,1], c=y, s=1, cmap="brg")
    plt.show()

# Build Network
myNN = Network(Softmax_and_CategoricalCrossEntropy(), SGD())
myNN.addLayer(Layer_Dense(2, 64, ReLU()))
myNN.addLayer(Layer_Dense(64, 64, ReLU()))
myNN.addLayer(Layer_Dense(64, 3, None))

# show predicted output
output = myNN(X)
print("Loss:\t"+str(myNN.calcLoss(y)))
print("Accuracy:\t"+ str(myNN.calcAccuracy(y)))
predict(myNN, "Output")

# Backpropagate 1
myNN.run(X)
myNN.optimize(X,y,100, 1000,1)
myNN.run(X)
print("ACC:\t"+str(myNN.calcAccuracy(y)))
print(f"Loss:\t{myNN.calcLoss(y):.3f}")

predict(myNN, "Backpropagation")

# optimize Randomly
myNN.optimizeRandomly(1000, 0.2, X, y)
output = myNN(X)
print("ACC:\t"+str(myNN.calcAccuracy(y)))
print(myNN.calcLoss(y))

predict(myNN,"Random")

# Backpropagate 2
myNN.run(X)
myNN.backPropagation(myNN.output,y)
myNN.optimize(X,y,100, 1000,0.0001)
myNN.run(X)
print("ACC:\t"+str(myNN.calcAccuracy(y)))
print(f"Loss:\t{myNN.calcLoss(y):.3f}")

predict(myNN, "Backpropagation 2")