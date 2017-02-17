import numpy as np
from cmath import *


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
        #rbm = rbm + sum([lncosh(sum([self.b[h] + self.W[v][h] * state[v] for v in range(self.nv)]))
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
        #for h in range(self.nh):
            #self.Lt[h] -= sum([2 * state[flip] * self.W[flip][h] for flip in flips])
        self.Lt -= 2 * np.dot(state[flips], self.W[flips])
        return None

    def load_parameters(self, filename):
        with open(filename, 'r') as f:
            self.nv = int(f.readline())
            self.nh = int(f.readline())

            self.a = np.array([ctopy_complex(f.readline()) for i in range(self.nv)])  # had to write a function to
            # parse the C++ complex output, which is (real, imag)
            self.b = np.array([ctopy_complex(f.readline()) for i in range(self.nh)])
            self.W = np.array([[ctopy_complex(f.readline()) for i in range(self.nh)] for j in range(self.nv)])

            print("NQS loaded from file: {}".format(filename))
            print("N_visbile = {0}      N_hidden = {1}".format(self.nv, self.nh))

    def nspins(self):  # This function exists for some reason, and I don't want to break anything
        return self.nv


def ctopy_complex(instring):
    coordinates = instring.translate({ord(c): None for c in '()\n'})  # strip out parentheses and newline
    coordinates = coordinates.split(",")  # split the coordinates into two strings at the comma
    outnum = float(coordinates[0]) + 1j * float(coordinates[1])
    return outnum


def lncosh(x):
    # I don't really understand why they write a more complicated function than this -- I think this should work though
    return np.log(np.cosh(x))

