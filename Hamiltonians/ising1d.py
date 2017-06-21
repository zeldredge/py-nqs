import numpy as np


class Ising1d:
    def __init__(self, nspins: int , hfield: float, pbc: bool=True):
        # Input args are number of spins, external field, and whether periodic boundary conditions apply
        self.nspins = nspins
        self.hfield = hfield
        self.pbc = pbc
        self.minflips = 1
        print("# Using the 1d TF Ising model with h = {}".format(hfield))

    def find_conn(self, state):  # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1
        mel = -1*self.hfield*np.ones(self.nspins + 1)  # basically all matrix elements are h except the diagonal one
        mel[0] = 0

        # all the flipsh are going to be "[j-1]" except the diagonal one
        # which we will make NaN because nqs knows how to handle that
        flipsh  = np.reshape(np.arange(self.nspins+1, dtype = float), (self.nspins+1, 1)) - 1
        flipsh[0] = np.array([np.nan])

        # Now we do the ZZ interaction
        mel[0] -= sum([state[i]*state[i+1] for i in range(self.nspins - 1)])
        if self.pbc:
            mel[0] -= state[-1]*state[0]

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection
