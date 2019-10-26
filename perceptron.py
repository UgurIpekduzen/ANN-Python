import numpy as np
#
# floatBias = 0.0
# floatLearningRate = 0.1
# intLabel = 1
# matrixWeights = np.array([-1.0, 0.0, 1.5, 1.0])
# matrixInputs = np.array([[2.0, 1.0, -1.5, -1.0], [1.0, -1.0, -0.5, 1.5], [0.2, 1.0, -1.0, 0.0]])
#
# for row in matrixInputs:
#     intActivation = 1 if np.sum(np.dot(matrixWeights[np.newaxis], row)) + floatBias > 0 else 0
#     if intActivation is 0:
#         matrixWeights = np.sum([matrixWeights, floatLearningRate * (intLabel - intActivation) * row], axis=0)
#
#     print(matrixWeights)