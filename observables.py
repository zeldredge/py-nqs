import numpy as np

# This file stores observables to be fed to the Sampler
# They should look like the Hamiltonians, but they're not used that way, so I'll put them here

class Sigmax:
    def __init__(self, nspins: int, which_spin: int):
        # Input args are number of spins, external field, and whether periodic boundary conditions apply
        self.nspins = nspins
        self.which = which_spin  # Which spin the sigmax is on
        self.minflips = 1

    def find_conn(self, state):  # Given a state, find all states connected to it by this observable
        # State should be a vector of \pm 1

        # For sigma x, there is only one connected state -- it has matrix element 1 and flips a single spin
        mel = [1]

        flipsh = np.array([[self.which]])

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection
