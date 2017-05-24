import numpy as np


class FermionHop: # Class implementing Hamiltonian for \sum_(i) c_i+1^dag c_i+ h.c., fermions

    def __init__(self, nspins, t, pbc=True):
        # Input args are number of sites, Jz coupling, and whether periodic boundary conditions apply
        self.nspins = nspins
        self.t = t
        self.pbc = pbc
        self.minflips = 2
        print("# Using the 1d free fermion model with t = {}".format(t))

    def find_conn(self, state):
        # Given a state, find all states connected to it by this Hamiltonian
        # State should be a vector of \pm 1 -- S, element s_i
        # That vector defines a state given by \prod (c^dag_i)^( 1 + s_i / 2) |vac>
        mel = [] # set of matrix elements
        flips = [] # set of flips
        for i in range(self.nspins - 1):
            if state[i] == 1 and state[i+1] == -1: # if we can make the hop (i.e., have a fermion and a space for it)
                flips.append(np.array([i,i+1]))
                # Now we just need to determine the sign. Remember that if a site i is occupied (state[i] = +1) then
                # it has a c^dag_i on the ket. We will further assume that the first c^\dag_0 is applied and then
                # in order of increasing i. So the question is what sign accumulates when we commute c^\dag_i+1 c_i
                # past the string. Well, we need to move it past all c^\dag_k with k > i + 1. The annhilation operator
                # commutes without complication, the creation operator accumulates a -1 for every one it crosses
                sign = (-1)**len([x for x in state[:i] if x == 1])
                mel.append(self.t*sign)

            elif state[i+1] == 1 and state[i] == -1: # or the other hop
                flips.append(np.array([i, i+ 1]))
                sign = (-1) ** len([x for x in state[:i] if x == 1])
                mel.append(self.t * sign)

        if self.pbc: # Periodic boundaries?
            if state[-1] == 1 and state[0] == -1:
                flips.append(np.array([0, -1]))
                sign = (-1) ** len([x for x in state if x == 1])
                mel.append(self.t*sign) # In this case, c_0 commutes through and c^dag_N is already at the end -- no sign

            elif state[-1] == -1 and state[0] == 1:
                flips.append(np.array([0, - 1]))

                mel.append(self.t)  # In this case, c^\dag_0 commutes through and c_N is already at the end

        return mel, flips
