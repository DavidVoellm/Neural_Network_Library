import numpy as np
class ActivationFunction:
    def __init__(self):
        pass
    def run(self, inputs):
        return inputs
    def backward(self, dvalues):
        return dvalues
    def __call__(self, inputs):
        return self.run(inputs)
class ReLU(ActivationFunction):
    def __init__(self):
        pass
    def run(self, inputs):
        self.output = np.maximum(0,inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0
        return self.dinputs
class Softmax(ActivationFunction):
    def __init__(self):
        pass
    def run(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = propabilities
        return self.output
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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
class CategoricalCrossEntropy(Loss):
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
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y.shape) == 1:
            y = np.eye(labels)[y]
        self.dinputs = -y / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs
class Softmax_and_CategoricalCrossEntropy(Loss):
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
    def run(self, inputs, y):
        self.output = self.activation(inputs)
        return self.loss.run(self.output,y)
        
    def backward(self, dvalues, y):
        samples = len(dvalues)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y]-=1
        self.dinputs = self.dinputs / samples
        return self.dinputs
class MeanSquaredError(Loss):
    def __init__(self):
        pass
    def run(self, y_pred, y):
        loss = np.sum(np.square((y_pred - y) / y.shape[0]))
        return loss
    def backward(self, dvalues, y):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2*(y-dvalues) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, ActivationFunction:ActivationFunction=None):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activationFunction = ActivationFunction
    def run(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activationFunction is not None:
            self.output = self.activationFunction.run(self.output)
        return self.output
    def backward(self, dvalues):
        if self.activationFunction is not None:
            dvalues = self.activationFunction.backward(dvalues)

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
    def __call__(self, inputs):
        return self.run(inputs)
class Optimizer:
    def __init__(self, learningrate=1.0):
        self.learningrate = learningrate
    def update_Layer(self, layer):
        pass
    def setLearningrate(self, learningrate):
        self.learningrate = learningrate
class SGD(Optimizer):
    def update_Layer(self, layer):
        layer.weights -= self.learningrate * layer.dweights
        layer.biases -= self.learningrate * layer.biases

class Network:
    def __init__(self, LossFunc:Loss, optimizer:Optimizer=None):
        self.layerList:list(Layer_Dense) = []
        self.lossFunction:Loss = LossFunc
        self.optimizer = optimizer
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
    def backPropagation(self,y_pred,y):
        dinputs = self.lossFunction.backward(y_pred,y)
        for i in range(len(self.layerList)-1,-1,-1):
            dinputs = self.layerList[i].backward(dinputs)
       
    def optimizeRandomly(self, iterations, learningRate, X, y):
        if learningRate == None:
            learningRate = 1.0
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
    def optimize(self,X,y, iterations, feedback=100, learningrate=None):
        if self.optimizer == None:
            self.optimizeRandomly(iterations, learningrate, X, y)
        else:
            if not learningrate is None:
                self.optimizer.setLearningrate(learningrate)
            for i in range(iterations):
                y_pred = self.run(X)
                self.backPropagation(y_pred, y)
                for layer in self.layerList:
                    self.optimizer.update_Layer(layer)
                if i % feedback == 0:
                    self.run(X)
                    loss = self.calcLoss(y)
                    accuracy= self.calcAccuracy(y)
                    print(f'epoche: {i}, acc: {accuracy:.3f}, loss: {loss:.3f}')