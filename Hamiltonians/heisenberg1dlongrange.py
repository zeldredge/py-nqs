import numpy as np


class Heisenberg1dLongRange:
    def __init__(self, nspins, jz, k, fstrength, pbc=True):
        # Input args are number of spins, Jz coupling, k is the range,
        # and whether periodic boundary conditions apply
        self.nspins = nspins
        self.jz = jz
        self.pbc = pbc
        self.minflips = 2
        self.k = k
        self.fstrength = fstrength
        print("# Using the 1d long-range model with J_z = {}, Range = {}".format(jz, k))

    def find_conn(self, state):  # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1
        mel = np.zeros(1)
        flipsh = np.array([[0, 0]])  # The "0,0" entry is a dummy, and is dealt with differently by nqs

        # loop over the neighbours (k-1 because if k=1, it's just nearest neighbour
        for nb in range(self.k):
            # First we do the ZZ interaction
            if self.pbc:  # if periodic loop over, if not terminate at the end
                mel[0] += sum([state[i] * state[(i + 1 + nb) % self.nspins] for i in range(self.nspins - 1)])
            else:
                mel[0] += sum([state[i] * state[i + 1 + nb] for i in range(self.nspins - 1 - nb)])
            mel[0] *= self.jz * self.fstrength(nb)  # Multiply by couplign constant

        # Look for possible spin flips
        if self.pbc:
            for nb in range(self.k):
                for i in range(self.nspins - 1):
                    if state[i] != state[(i + 1 + nb) % self.nspins]:
                        mel = np.append(mel, [-2 * self.fstrength(nb)])
                        flipsh = np.append(flipsh, [[i, i + 1 + nb]], axis=0)
        else:
            for nb in range(self.k):
                for i in range(self.nspins - 1 - nb):
                    if state[i] != state[i + 1 + nb]:
                        mel = np.append(mel, [-2 * self.fstrength(nb)])
                        flipsh = np.append(flipsh, [[i, i + 1 + nb]], axis=0)

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection
