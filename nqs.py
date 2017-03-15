import numpy as np
from cmath import *
from scipy.linalg import circulant


class Nqs:
    def __init__(self, filename):
        # Initializing a bunch of variables. Not so necessary in python! but doing it anyway
        self.W = np.zeros((1, 1))  # neural network weights (matrix of W_ij)
        self.a = np.zeros(1)  # neural network visible bias (vector)
        self.b = np.zeros(1)  # neural network hidden bias (vector)
        self.nh = 0  # number of hidden spins (to be determined from file)
        self.nv = 0  # number of visible spins (to be determiend from file)

        self.Lt = np.zeros(1)  # look-up table for angles

        self.log2 = log(2)  # Apparently easier to precompute this?

        # Ok, now get all the parameters
        self.load_parameters(filename)

    def log_val(self, state):  # computes the logarithm of the wavefunction in a particular state
        # Just uses the formula in C1 with the logarithm used to take a sum
        # rbm = sum([self.a[v] * state[v] for v in range(self.nv)])  # Start with all visible biases
        rbm = np.dot(self.a, state)
        # The two sums below: inner sum is over all v (each hidden unit accounts for all of its visible connections)
        # outer sum is over all h (each cosh in the product)
        # rbm = rbm + sum([lncosh(sum([self.b[h] + self.W[v][h] * state[v] for v in range(self.nv)]))
        #                for h in range(self.nh)])
        rbm += np.sum(np.log(np.cosh((self.b + np.dot(state, self.W)))))

        return rbm

    # Next function is LogPoP, computes Log Psi'/Psi, where Psi' is the state with certain flipped spins
    # Look-up tables are used for speed; the vector flips tells us which are flipped

    def log_pop(self, state, flips):
        if len(flips) == 0:  # No flips? We out
            return 0

        if len(flips) == 1 and flips == [None]:
            return 0

        if len(flips) == 2:
            if not np.any(flips - flips[0]):  # If it's this one that means no flips
                return 0

        logpop = 0 + 0j  # Initialize the variable

        # This is the change due to visible biases
        # logpop = logpop - sum([self.a[flip] * 2.0 * state[flip] for flip in flips])
        logpop -= 2 * np.dot(self.a[flips], state[flips])
        # This is the change due to the interaction weights
        logpop += np.sum(np.log(np.cosh((self.Lt - 2 * np.dot(state[flips], self.W[flips]))))
                         - np.log(np.cosh(self.Lt)))

        return logpop

    def pop(self, state, flips):  # This does the obvious
        return np.exp(self.log_pop(state, flips))

    def init_lt(self, state):  # Initialize lookup tables
        self.Lt = np.zeros(self.nh)  # See eqn C7
        self.Lt = self.b + np.dot(state, self.W)
        # self.Lt = [self.b[h] + sum([self.W[v][h] * state[v] for v in range(self.nv)]) for h in range(self.nh)]
        return None

    def update_lt(self, state, flips):  # Update lookup tables after flips
        if len(flips) == 0:  # Again, if no flips, leave
            return None

        self.Lt -= 2 * np.dot(state[flips], self.W[flips])
        return None

    def load_parameters_c(self, filename):
        with open(filename, 'r') as f:
            self.nv = int(f.readline())
            self.nh = int(f.readline())

            self.a = np.array([ctopy_complex(f.readline()) for i in range(self.nv)])  # had to write a function to
            # parse the C++ complex output, which is (real, imaginary)
            self.b = np.array([ctopy_complex(f.readline()) for i in range(self.nh)])
            self.W = np.array([[ctopy_complex(f.readline()) for i in range(self.nh)] for j in range(self.nv)])

            print("NQS loaded from file: {}".format(filename))
            print("N_visbile = {0}      N_hidden = {1}".format(self.nv, self.nh))

    def load_parameters(self, filename):
        temp_file = np.load(filename)
        self.a = temp_file['a']
        self.b = temp_file['b']
        self.W = temp_file['W']
        self.nv = len(self.a)
        self.nh = len(self.b)

    def save_parameters(self, filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W)

    def nspins(self):  # This function exists for some reason, and I don't want to break anything
        return self.nv


class NqsTI:
    # Dedicated class for translation-invariant neural networks
    def __init__(self, nv, density):
        # Initialize by providing the number of physical variables (spins) and the hidden unit density
        self.alpha = density
        self.nv = nv

        self.W = np.zeros((self.alpha, self.nv),dtype=complex)
        # W is all the weights; for each feature there is a vector describing its weights

        # First we take W and, for each feature, produce a matrix that twists it so we get one "unsymmetrized"
        # weight matrix. Then we add concatenate the features to get the one big array we want
        self.Wfull = np.array([np.array([np.roll(self.W[a], -f) for f in range(self.nv)]) for a in range(density)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)

        self.a = 0 # There's only one visible bias in this case, because of TI
        self.b = np.empty(self.alpha,dtype=complex) # One bias per feature
        # We use a similar scheme for b
        self.bfull = np.concatenate(np.array([self.b[a]*np.ones(nv) for a in range(density)]))

        # Note: I don't really need to establish these arrays here in the initialization per se
        # But it helps you see what they WILL BE when there's actually something there and not np.zeros
        self.Lt = np.zeros(self.alpha*self.nv,dtype=complex)

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

        # Since have constructed Wfull, we can basically use same code as we did in the non-symmetric case
        logpop += np.sum(np.log(np.cosh(self.Lt - 2 * np.dot(state[flips], self.Wfull[flips])))
                         - np.log(np.cosh(self.Lt)))

        return logpop

    def pop(self, state, flips):
        return np.exp(self.log_pop(state, flips))

    def init_lt(self, state):

        # Just as in init...
        # We use roll to move the W vectors backwards somewhat, and then concatenate them on top of each other
        # The result is one big matrix we can use to get the weights
        self.Wfull = np.array([np.array([np.roll(self.W[a], -f) for f in range(self.nv)]) for a in range(self.alpha)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)

        # Same principle for bfull
        self.bfull = np.concatenate(np.array([self.b[a] * np.ones(self.nv) for a in range(self.alpha)]))

        # One Wfull and bfull are constructed, other operations can proceed without knowing about the symmetry
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


class NqsSymmetric:
    # Dedicated class for arbitrary-symmetric neural networks
    def __init__(self, nv, density, group):
        # Initialize by providing the number of physical variables (spins), the hidden unit density,
        # and the set of transformations you want the NN to be symmetric under
        self.alpha = density
        self.nv = nv
        self.t_group = group
        self.t_size = group.shape[0] #number of transformations

        self.W = np.zeros((self.alpha, self.nv), dtype=complex)
        # W is all the weights; for each feature there is a vector describing its weights

        # First we take W and, for each feature, produce a matrix that twists it so we get one "unsymmetrized"
        # weight matrix. Then we add concatenate the features to get the one big array we want
        self.Wfull = np.array([np.array([np.dot(t, self.W[a]) for t in self.t_group]) for a in range(self.alpha)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)

        self.a = np.zeros(nv//self.t_size)  # Every available symmetry cuts the number of visible neurons
        self.b = np.zeros(self.alpha, dtype=complex) # One bias per feature
        # We use a similar scheme for b
        self.bfull = np.concatenate(np.array([self.b[a]*np.ones(nv) for a in range(density)]))

        # Note: I don't really need to establish these arrays here in the initialization per se
        # But it helps you see what they WILL BE when there's actually something there and not np.zeros
        self.Lt = np.zeros(self.alpha*self.nv, dtype=complex)

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

        # Since have constructed Wfull, we can basically use same code as we did in the non-symmetric case
        logpop += np.sum(np.log(np.cosh(self.Lt - 2 * np.dot(state[flips], self.Wfull[flips])))
                         - np.log(np.cosh(self.Lt)))

        return logpop

    def pop(self, state, flips):
        return np.exp(self.log_pop(state, flips))

    def init_lt(self, state):

        # Just as in init...
        # We use the group to transform the W vectors, and then concatenate them on top of each other
        # The result is one big matrix we can use to get the weights
        self.Wfull = np.array([np.array([np.dot(t, self.W[a]) for t in self.t_group]) for a in range(self.alpha)])
        self.Wfull = np.concatenate(self.Wfull, axis=1)

        # Same principle for bfull
        self.bfull = np.concatenate(np.array([self.b[a] * np.ones(self.nv) for a in range(self.alpha)]))

        # One Wfull and bfull are constructed, other operations can proceed without knowing about the symmetry
        self.Lt = self.bfull + np.dot(state, self.Wfull)

    def update_lt(self, state, flips):
        self.Lt -= 2 * np.dot(state[flips], self.Wfull[flips])

    def load_parameters(self,filename):
        temp_file = np.load(filename)
        self.a = temp_file['a']
        self.b = temp_file['b']
        self.W = temp_file['W']
        (self.alpha, self.nv) = self.W.shape
        self.Wfull = np.array(np.array([np.dot(t, self.W[a]) for t in self.t_group]) for a in range(self.alpha))
        self.Wfull = np.concatenate(self.Wfull, axis=1)
        self.bfull = np.concatenate(np.array([self.b[a] * np.ones(self.nv) for a in range(self.alpha)]))

    def save_parameters(self, filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W)

def ctopy_complex(instring):
    coordinates = instring.translate({ord(c): None for c in '()\n'})  # strip out parentheses and newline
    coordinates = coordinates.split(",")  # split the coordinates into two strings at the comma
    outnum = float(coordinates[0]) + 1j * float(coordinates[1])
    return outnum


def lncosh(x):
    # I don't really understand why they write a more complicated function than this -- I think this should work though
    return np.log(np.cosh(x))
