import numpy as np
from cmath import *
import nqs

class randomize:
    def __init__(self,nH,nV):
        # Create a randomized set of weights a biases
        self.nh = nH  # number of hidden spins (to be determined from file)
        self.nv = nV  # number of visible spins (to be determiend from file)
        self.W = np.zeros((nH, nV))  # neural network weights (matrix of W_ij)
        self.a = np.zeros(nV)  # neural network visible bias (vector)
        self.b = np.zeros(nH)  # neural network hidden bias (vector)
        magnitude = 1
        self.a = self.random_number(magnitude, self.nV)
        self.b = self.random_number(magnitude, self.nH)
        self.W = self.random_number(magnitude, self.nH, self.nV)


    def random_number(self,magnitude, number1, number2=False):
        if number2:
            return 2 * magnitude * (np.random.rand(number1, number2) - 0.5) + np.pi * 2j * np.random.rand(number1,
                                                                                                          number2)
        return 2 * magnitude * (np.random.rand(number1) - 0.5) + np.pi * 2j * np.random.rand(number1)


