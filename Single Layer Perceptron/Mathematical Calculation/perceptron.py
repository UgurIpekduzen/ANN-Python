import numpy as np

class Perceptron:

    def __init__(self, intInputCount, floatThreshold, floatBias, floatLearningRate):
        # Ağırlık matrisi üretir
        self.matrixWeights = np.random.uniform(low=-1, high=1, size=intInputCount)
        # İstenen sonuç matrisi üretir
        self.matrixDesired = np.random.randint(2, size=intInputCount)

        self.n = floatLearningRate
        self.sigma = floatThreshold
        self.b = floatBias

        print("\nÜretilen Ağırlık Matrisi:")
        print(self.matrixWeights)
        print("\nÜretilen İstenilen Sonuç Matrisi:")
        print(self.matrixDesired)

    def predict(self, inputs):
        #Net hesaplar
        net = np.sum(np.dot(self.matrixWeights, inputs)) + self.b
        return self.activation(net)

    def activation(self, net):
        # Nöronun ateşlenmesine karar verir
        return 1 if net > self.sigma else 0

    def train(self, matrixInputs):
        print("\nEğitme işlemi:")
        for i in range(0, np.size(matrixInputs, 0)):

            intPredicted = self.predict(matrixInputs[i])
            # Tahmin edilen sonuç ile istenen sonuç arasındaki hata farkını hesaplar
            intError = self.matrixDesired[i] - intPredicted
            # Hata 1 ise ağırlık güncellemesi yapılır, 0 ise güncelleme yapmadan aynı hesabı gerçekleşir
            self.matrixWeights = np.sum([self.matrixWeights, self.n * intError * matrixInputs[i]], axis=0)

            print("Adım " + str(i + 1) + " , Ağırlıklar:")
            print(self.matrixWeights)


print("Örnek 1: x = [[1, 0], [0, 1]], sigma = -1, b = 0, n = 0.1")
x1 = np.array([[1, 0], [0, 1]])
p1 = Perceptron(np.size(x1, 1), -1.0, 0.0, 0.1)
p1.train(x1)

print("\nÖrnek 2: x = [0, 0, 1], sigma = 0, b = -0.7, n = 0.2")
x2 = np.array([0, 0, 1])
p2 = Perceptron(np.size(x2), 0.0, -0.7, 0.2)
p2.train(x2)

print("\nÖrnek 3: x = [[0, 1, 0], [0, 0, 1], [0, 1, 1]], sigma = 0, b = 0, n = 0.1")
x3 = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 1]])
p3 = Perceptron(np.size(x3, 1), 0.0, 0.0, 0.1)
p3.train(x3)

