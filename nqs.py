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

class NqsSymmetric:
    # WARNING: class not currently tested; do not use
    def __init__(self, nspins, t_group, feature_density):
        self.alpha = feature_density
        self.t_group = t_group
        self.nspins = nspins
        self.group_size = t_group.shape[0]

        self.W = np.empty((self.alpha, self.nspins))
        self.a = np.empty(self.alpha)
        self.b = np.empty(self.alpha)

        self.Lt = np.empty(self.alpha * self.group_size)

    def log_val(self, state):  # log amplitude in a particular state
        self.init_Lt(state)
        value = 0
        value += np.sum(self.a * np.dot(self.t_group, state))
        value += np.log(np.cosh(self.Lt))
        return value

    def init_Lt(self, state):
        b_expanded = np.tile(self.b, self.group_size)
        W_expanded = np.tile(self.W, self.group_size)
        self.Lt = b_expanded + np.dot(W_expanded, np.dot(self.t_group, state))

    def update_Lt(self, flips):
        if len(flips) == 0:  # Again, if no flips, leave
            return None

        W_expanded = np.tile(self.W, self.group_size)
        self.Lt -= 2 * np.dot(W_expanded, np.dot(self.t_group, flips))

    def pop(self, flips):
        return exp(self.log_pop(flips))

    def log_pop(self, flips):

        if np.all(flips == 0):
            return 0
        # For symmetrized version of the code, flips is a length-Nspins vector which will be 0 in most places and
        # +/- 2 if there is a flip in a spot, so that the new spin vector S' = S + flips
        # We use this because we aren't guaranteed the sparsity will survive symmetry operations

        logpop = 0 + 0j  # Initialize the variable

        # This is the change due to visible biases
        logpop -= np.sum(self.a * np.dot(self.t_group, flips))

        # This is the change due to weights
        w_expanded = np.tile(self.W, self.group_size)
        logpop -= np.log(np.cosh(self.Lt)) - np.log(np.cosh(self.Lt -
                                                            2 * np.dot(w_expanded, np.dot(self.t_group, flips))))
        return logpop

    def load_parameters(self, filename):
        temp_file = np.load(filename)
        self.a = temp_file['a']
        self.b = temp_file['b']
        self.W = temp_file['W']
        self.t_group = temp_file['t_group']
        self.nv = len(self.a)
        self.nh = len(self.b)

    def save_parameters(self, filename):
        np.savez(filename, a=self.a, b=self.b, W=self.W, t_group=self.t_group)


def ctopy_complex(instring):
    coordinates = instring.translate({ord(c): None for c in '()\n'})  # strip out parentheses and newline
    coordinates = coordinates.split(",")  # split the coordinates into two strings at the comma
    outnum = float(coordinates[0]) + 1j * float(coordinates[1])
    return outnum


def lncosh(x):
    # I don't really understand why they write a more complicated function than this -- I think this should work though
    return np.log(np.cosh(x))
