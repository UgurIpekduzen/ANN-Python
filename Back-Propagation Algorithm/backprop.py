from activation import sigmoid
import numpy as np
from sympy import *


class NeuralNetwork:
    def __init__(self, intInputNumber, intOutputNumber, intHiddenNeuronNumber=0, intLayerNumber=2):
        self.network = []

        if intHiddenNeuronNumber is 0:
            intHiddenNeuronNumber = intInputNumber

        listHiddenLayerWeights = np.random.uniform(low=-1, high=1, size=(intInputNumber, intHiddenNeuronNumber))
        self.network.append(listHiddenLayerWeights)

        listOutputLayerWeights = np.random.uniform(low=-1, high=1, size=(intHiddenNeuronNumber, intOutputNumber))
        self.network.append(listOutputLayerWeights)

        self.bias = np.random.uniform(low=0, high=1, size=intLayerNumber)
        print("BIAS DEĞERLERİ: \n")
        print(self.bias)
        self.listTargets = np.random.randint(2, size=intOutputNumber)
        print("HEDEF ÇIKTILAR: \n")
        print(self.listTargets)

    def activate(self, weights, inputs):
        net = 0.0
        for i in range(len(weights)):
            net += np.multiply(weights[i], inputs[i])
        return net

    def forwardPropagation(self, row, count):
        listInputs = row
        for layer in self.network:
            listHidLayerInputs = []
            for neuron in layer:
                activation = self.activate(neuron, listInputs) + self.bias[count]
                output = sigmoid(activation)
                listHidLayerInputs.append(output)
            listInputs = listHidLayerInputs

        return listInputs

    def backwardPropagationError(self, outputs):
        intOutputNumber = len(outputs)
        listAllErrorsBackward = []
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            listLayerErrors = []
            for j in range(intOutputNumber):
                listNeuronErrors = []
                for k in range(layer.shape[1]):
                    delta_0 = (-2/intOutputNumber) * (self.listTargets[j] - outputs[j])
                    delta_1 = outputs[j] * (1 - outputs[j])
                    delta_2 = layer[j][k]
                    delta_weight = delta_0 * delta_1 * delta_2

                    listNeuronErrors.append(np.array(delta_weight))

                listLayerErrors.append(np.array(listNeuronErrors))

            listAllErrorsBackward.append(np.array(listLayerErrors))

        listAllErrorsBackward = np.array(listAllErrorsBackward)

        print("AĞIRLIK GERİ YAYILIM HATALARI: \n")
        print(np.flip(listAllErrorsBackward, 0))
        return np.flip(listAllErrorsBackward, 0)

    def updateWeights(self, errors, floatLearningRate):
        allNetworkWeights = []
        print("\nNETWORK'DEKİ MEVCUT AĞIRLIKLAR: \n")
        print(self.network)
        for i in reversed(range(len(self.network))):
            layerWeights = self.network[i]
            layerErrors = errors[i]
            allNetworkWeights.append(np.subtract(layerWeights, floatLearningRate * layerErrors))
        print("\nNETWORK'DEKİ GÜNCEL AĞIRLIKLAR: \n")
        print(allNetworkWeights)


    def train(self, listDataSet, floatLearningRate):
        intCount = 0 # Layer'lara ait bias değerlerini çağırır
        for row in listDataSet:
            outputs = self.forwardPropagation(row, intCount)
            print("\nÇIKTILAR: \n")
            print(outputs)
            errors = self.backwardPropagationError(outputs)
            self.updateWeights(errors, floatLearningRate)
            intCount += 1



