import numpy as np
class Inputs:
	def __init__(self, x, y, ans):
		self.inputs = np.array([x, y])
		self.answer = ans