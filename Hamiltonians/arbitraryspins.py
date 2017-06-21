import numpy as np


class ArbitrarySpinModel:
    def __init__(self, nspins, coupling_mat, field_list):
        """
        Class that realizes an arbitrary spin Hamiltonian of the form:
        H = \sum_(pairs) J_ij Sz_i Sz_j + \sum_(spin) B \dot S_i

        :param nspins: Number of spins in the model
        :param coupling_mat: the Jij matrix, see above
        :param field_list: An nspins x 3 ndarray, which such that field_list[i][j] is the jth component (Bx, By, Bz) of
        the magnetic field acting on spin i
        """
        self.nspins = nspins
        self.J = coupling_mat
        self.fields = field_list
        self.minflips = 1
        print("# Using an arbitrary spin model...")

    def find_conn(self, state):  # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1

        #Start with the Z terms, since these do not change the state
        mel = np.array([np.sum(state*self.fields[:][:-1])])
        flipsh = np.array([[np.nan,np.nan]])  # NQS will remove all NaNs

        for i in range(self.nspins): # First let's do the two-body terms. If there's a nonzero Jij, add that flip
            for j in range(i):
                if self.J[i,j] != 0:
                    print("Found coupling")
                    mel = np.append(mel, self.J[i,j])
                    flipsh = np.append(flipsh, np.array([[j,i]]), axis = 0)

        for i in range(self.nspins):  # Now we do one body terms, which have two types
            if sum(self.fields[i][:-1]) != 0: # First, one body terms that flip a spin (the Jx, Jy terms)
                print("Found one-body flip")
                mel = np.append(mel, sum(self.fields[i][:-1]))
                flipsh = np.append(flipsh, np.array([[i, np.nan]]), axis = 0)

        return mel, flipsh  # So what we have here is a vector of all matrix elements,
        # and then a vector which tells you which states to flip to get that connection