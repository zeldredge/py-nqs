import numpy as np


class Heisenberg2d:

    def __init__(self, nspins: int, jz: float, length: int, pbc: bool = True):
        if length*length != nspins:
            raise ValueError("The number of spins given is not compatible with the dimension of the square lattice")

        #  Define the nearest neighbors
        self.nn = np.zeros((nspins, 2), dtype=int)
        self.nspins = nspins

        #  Spins are indexed like so: Spin 0 is at the bottom left (the "origin") and also at point (0,0).
        #  Coordinates increase to the right and up, with the x direction being the fast one
        #  So spin j has nearest neighbors j + 1,  j + length
        #  Since we only want every NN pair once, we only check up and right

        for j in range(nspins):
            self.nn[j] = [j + 1, j + length]  # if not on an edge, this is it

            if j % length == length - 1:  # if we are at the right end of a horizontal edge
                if pbc:
                    self.nn[j][0] = j - length + 1
                else:
                    del self.nn[j][0]

            if nspins - j <= length:  # if we are on the top row
                if pbc:
                    self.nn[j][1] = j % length  # same position in the bottom row
                else:
                    del self.nn[j][1]  # if not pbc, delete this entry

        print("Using the 2d Heisenberg model with Jz = {}".format(jz))
        self.minflips = 2

    def find_conn(self, state):
        # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1
        mel = np.zeros(1)
        flipsh = np.array([[np.nan, np.nan]])  # The "0,0" entry is a dummy, and is dealt with differently by nqs

        # First we do the ZZ interaction
        mel[0] = sum([sum([state[j]*state[k] for k in self.nn[j]]) for j in range(self.nspins)])

        # Look for possible spin flips
        for i in range(self.nspins):
            for j in self.nn[i]:
                if state[i] != state[j]:
                    mel = np.append(mel, [-2])
                    flipsh = np.append(flipsh, [[i, j]], axis=0)

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection






