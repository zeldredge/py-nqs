import numpy as np
from scipy.linalg import circulant

class NqsTI:
    def __init__(self, nv, density):
        self.alpha = density
        self.nv = nv

        self.W = np.empty((self.alpha, self.nv))
        self.a = 0
        self.b = np.empty(self.alpha)

        self.Lt = np.empty(self.alpha)

    def log_val(self, state):
        # The linear algebra in TI case uses a single weight vector and dots it against a circulant matrix made of the
        # state. Equivalent to applying W to every possible shift
        value = self.a * np.sum(state)
        value += np.sum(np.log(np.cosh(self.b + np.dot(circulant(state).T, self.W))))
        return value

    def log_pop(self, state, flips):
        # Log of Psi'/Psi when we start in state Psi and flip the spins identified by index in list flips
        if len(flips) == 0:  # No flips? We out
            return 0

        if len(flips) == 1 and flips == [None]:
            return 0

        if len(flips) == 2:
            if not np.any(flips - flips[0]):  # If it's this one that means no flips
                return 0

        logpop = 0 + 0j

        # First, we take into account the change due to the visible bias
        logpop += -np.sum(2 * state[flips] * self.a)

        # Changing a spin affects the hidden units interestingly, because of the translation invariance a flipped
        # spin "hits" every element of the weight vector. So we get a sum over W for each flip
        longflips = [np.mod(flips + [i] * len(flips), self.nv) for i in range(self.nv)]
        logpop += np.sum(np.log(np.cosh(self.Lt
                                        - 2 * np.sum(state[longflips]*self.W[longflips], axis=1))) - np.log(np.cosh(self.Lt)))

        return logpop

    def pop(self, state, flips):
        return np.exp(self.log_pop(state, flips))

    def init_lt(self, state):
        self.Lt = self.b + np.dot(circulant(state).T, self.W)

    def update_lt(self, state, flips):
        self.Lt -= 2 * np.sum(state[flips]) * np.sum(self.W, axis=1)

    def load_parameters(self,filename):
        temp_file = np.load(filename)
        self.a = temp_file['a']
        self.b = temp_file['b']
        self.W = temp_file['W']
        (self.alpha, self.nv) = self.W.shape

    def save_paramaters(self,filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W)