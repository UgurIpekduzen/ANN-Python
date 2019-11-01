import numpy as np

class Perceptron:

    def __init__(self, intInputCount, floatLearningRate, floatThreshold, floatBias):
        self.matrixWeights = np.random.uniform(low=-1, high=1, size=intInputCount)
        self.n = floatLearningRate
        self.sigma = floatThreshold
        self.b = floatBias

    def predict(self, inputs):
        net = np.sum(np.dot(self.matrixWeights, inputs)) + self.b
        return self.activation(net)

    def activation(self, net):
        return 1 if net > self.sigma else 0

    def train(self, matrixInputs, intDesired):
            # Nöron ateşleme
            intPredicted = self.predict(matrixInputs)
            # Ağırlık güncelleme
            self.matrixWeights = np.sum([self.matrixWeights, self.n * (intDesired - intPredicted) * matrixInputs], axis=0)