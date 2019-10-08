import activation
import matplotlib.pyplot as plt
import numpy as np

def drawLinePlot(start, end, activationFunction):
    inputs = []
    outputs = []
    for x in np.arange(start, end, 0.5):
        inputs.append(x)
        outputs.append(activationFunction(x))
    return inputs, outputs

x, y = drawLinePlot(-3, 3, activation.linear)
plt.plot(x, y, label='Linear')
x, y = drawLinePlot(-3, 3, activation.sigmoid)
plt.plot(x, y, label='Sigmoid')
x, y = drawLinePlot(-3, 3, activation.leakyRelu)
plt.plot(x, y, label='Leaky ReLU')
x, y = drawLinePlot(-3, 3, activation.tanh)
plt.plot(x, y, label='Tanh')
x, y = drawLinePlot(-3, 3, activation.relu)
plt.plot(x, y, label='ReLU')
x, y = drawLinePlot(-3, 3, activation.swish)
plt.plot(x, y, label='Swish')
plt.plot([-3, -2, -1, 0, 1, 2, 3], activation.softmax([-3, -2, -1, 0, 1, 2, 3]), label='Softmax')
plt.axvline()
plt.xlabel("Inputs")
plt.ylabel("Outputs")
plt.title("Activation Functions")
plt.legend()
plt.show()
