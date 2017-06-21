import numpy as np


class Heisenberg1d:
    def __init__(self, nspins, jz, pbc=True):
        # Input args are number of spins, Jz coupling, and whether periodic boundary conditions apply
        self.nspins = nspins
        self.jz = jz
        self.pbc = pbc
        self.minflips = 2
        print("# Using the 1d Heisenberg model with J_z = {}".format(jz))

    def find_conn(self, state):  # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1
        mel = np.zeros(1)
        flipsh = np.array([[np.nan,np.nan]])  # No flips entry

        # First we do the ZZ interaction
        mel[0] = sum([state[i] * state[i + 1] for i in range(self.nspins - 1)])
        if self.pbc:  # also check the ends if pbc
            mel[0] += state[0] * state[-1]
        mel[0] *= self.jz  # Multiply by couplign constant

        # Look for possible spin flips
        for i in range(self.nspins - 1):
            if state[i] != state[i + 1]:
                mel = np.append(mel, [-2])
                flipsh = np.append(flipsh, [[i, i + 1]], axis=0)

        if self.pbc:
            if state[-1] != state[0]:
                mel = np.append(mel, [-2])
                flipsh = np.append(flipsh, [[-1, 0]], axis=0)

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection
