import numpy as np
class ActivationFunction:
    def __init__(self):
        pass
    def run(self, inputs):
        return inputs
    def __call__(self, inputs):
        return self.run(inputs)
class ReLU(ActivationFunction):
    def __init__(self):
        pass
    def run(self, inputs):
        self.output = np.maximum(0,inputs)
        return self.output
class Softmax(ActivationFunction):
    def __init__(self):
        pass
    def run(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = propabilities
        return self.output
class Sigmoid(ActivationFunction):
    def __init__(self):
        pass
    def run(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        return self.output
class Loss:
    def __init__(self):
        pass
    def calc(self, output, y):
        
        sampleLosses = self.run(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
class CategorialCrossEntropy(Loss):
    def __init__(self):
        pass
    def run(self, y_pred, y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
class MeanSquaredError(Loss):
    def __init__(self):
        pass
    def run(self, y_pred, y):
        loss = np.sum(np.square((y_pred - y) / y.shape[0]))
        return loss
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, ActivationFunction:ActivationFunction=None):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activationFunction = ActivationFunction
    def run(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activationFunction is not None:
            self.output = self.activationFunction.run(self.output)
        return self.output
    def __call__(self, inputs):
        return self.run(inputs)
class Network:
    def __init__(self, LossFunc:Loss):
        self.layerList:list(Layer_Dense) = []
        self.lossFunction:Loss = LossFunc
    def addLayer(self,layer:Layer_Dense):
        self.layerList.append(layer)
    def run(self, input, oneHot=False):
        for i in range(len(self.layerList)):
            input = self.layerList[i].run(input)
            #print("calc:\t"+str(i))
        self.output = input
        if oneHot:
            return np.argmax(self.output, axis=1)
        else:
            return self.output
    def __call__(self, input, oneHot=True):
        return self.run(input, oneHot)
    def calcLoss(self,y) -> float:
        return self.lossFunction.calc(self.output, y)
    def calcAccuracy(self, y) -> float:
        predictions = np.argmax(self.output, axis=1)
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy
    def backPropagation(self,X,y):
        self.run(X)
        lowestLoss = self.calcLoss(y)
        pass
    def optimizeRandomly(self, iterations, learningRate, X, y):
        self.run(X)
        lowestLoss = self.calcLoss(y)
        for i in range(iterations):
            weights = []
            biases = []
            for layer in self.layerList:
                weights.append(layer.weights.copy())
                biases.append(layer.biases.copy())
                shape = layer.weights.shape
                layer.weights += learningRate * np.random.randn(shape[0], shape[1])
                layer.biases += learningRate * np.random.randn(1,shape[1])
            self(X)
            loss = self.calcLoss(y)

            if loss < lowestLoss:
                lowestLoss = loss
                print("New optimal found: \t" + str(loss))
            else:
                for i in range(len(self.layerList)):
                    self.layerList[i].weights = weights[i]
                    self.layerList[i].biases = biases[i]