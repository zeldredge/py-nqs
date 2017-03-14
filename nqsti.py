import numpy as np
from scipy.linalg import circulant

class NqsTI:
    def __init__(self, nv, density):
        self.alpha = density
        self.nv = nv

        self.W = np.empty((self.alpha, self.nv),dtype=complex)

        # First we take W and, for each feature, produce a matrix that twists it so we get one "unsymmetrized"
        # weight matrix. Then we add concatenate the features to get the one big array we want
        self.Wfull = np.array([np.array([np.roll(self.W[a], -f) for f in range(self.nv)]) for a in range(density)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)

        self.a = 0
        self.b = np.empty(self.alpha,dtype=complex)
        # We use a similar scheme for b
        self.bfull = np.concatenate(np.array([self.b[a]*np.ones(nv) for a in range(density)]))

        # Note: I don't really need to establish these arrays here in the initialization per se
        # But it helps you see what they WILL BE when there's actually something there and not np.empty
        self.Lt = np.empty(self.alpha*self.nv,dtype=complex)

    def log_val(self, state):
        # Refers to the existing look-up tables to get a value
        value = self.a * np.sum(state)
        value += np.sum(np.log(np.cosh(self.Lt)))
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
        logpop += -2*self.a*np.sum(state[flips])

        # Changing a spin affects the hidden units interestingly, because of the translation invariance a flipped
        # spin "hits" every element of the weight vector. So we get a sum over W for each flip
        logpop += np.sum(np.log(np.cosh(self.Lt - 2 * np.dot(state[flips], self.Wfull[flips])))
                         - np.log(np.cosh(self.Lt)))

        return logpop

    def pop(self, state, flips):
        return np.exp(self.log_pop(state, flips))

    def init_lt(self, state):

        #Just as in init...
        self.Wfull = np.array([np.array([np.roll(self.W[a], -f) for f in range(self.nv)]) for a in range(self.alpha)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)
        self.bfull = np.concatenate(np.array([self.b[a] * np.ones(self.nv) for a in range(self.alpha)]))

        self.Lt = self.bfull + np.dot(state, self.Wfull)

    def update_lt(self, state, flips):
        self.Lt -= 2 * np.dot(state[flips], self.Wfull[flips])

    def load_parameters(self,filename):
        temp_file = np.load(filename)
        self.a = temp_file['a']
        self.b = temp_file['b']
        self.W = temp_file['W']
        self.Wfull = np.array([np.roll(self.W, -f) for f in range(self.nv)])
        (self.alpha, self.nv) = self.W.shape

    def save_parameters(self, filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W)