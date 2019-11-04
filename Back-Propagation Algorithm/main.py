from backprop import NeuralNetwork
import numpy as np

if __name__ == "__main__":


    dataset = np.random.uniform(low=-1, high=1, size=(1, 2))
    print("GİRDİLER: \n")
    print(dataset)

    n = 0.1
    print("\nÖĞRENME ORANI: " + str(n) + "\n")

    nn = NeuralNetwork(2, 2, 2)
    nn.train(dataset, n)