import numpy as np


class FermionHop: # Class implementing Hamiltonian for \sum_(i) c_i+1^dag c_i+ h.c., fermions

    def __init__(self, nspins, t, pbc=True):
        # Input args are number of sites, tunneling constant, and whether periodic boundary conditions apply
        self.nspins = nspins
        self.t = t
        self.pbc = pbc
        self.minflips = 1
        print("# Using the 1d free fermion model with t = {}".format(t))

    def find_conn(self, state):
        # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1 -- S, element s_i
        # That vector defines a state given by \prod (c^dag_i)^( 1 + s_i / 2) |vac>
        mel = [] # set of matrix elements
        flips = [] # set of flips
        for i in range(self.nspins - 1):
            #sign = (-1) ** len([x for x in state[:i] if x == 1])
            sign = 1
            if state[i] == 1 and state[i+1] == -1: # if we can make the hop (i.e., have a fermion and a space for it)
                flips.append(np.array([i,i+1]))

                mel.append(self.t*sign)

            elif state[i+1] == 1 and state[i] == -1: # or the other hop
                flips.append(np.array([i, i+ 1]))
                mel.append(self.t*sign)

        if self.pbc: # Periodic boundaries?
            sign = (-1) ** (len([x for x in state if x == 1]) - 1)
            if state[-1] == -1 and state[0] == 1:
                flips.append(np.array([0, -1]))
                mel.append(self.t*sign)

            elif state[-1] == 1 and state[0] == -1:
                flips.append(np.array([0, - 1]))
                mel.append(self.t*sign)  #

        return mel, flips
