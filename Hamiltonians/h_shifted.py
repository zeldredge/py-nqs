import numpy as np

#  This class is used for first-excited state calcualations
#  We accomplish this by minimizing within the orthogonal subspace, adding a penalty term for overlaps with the
#  ground state

class HShifted:
    def __init__(self, h0, gswf, penalty):
        # Inputs:
        # h0: The original Hamiltonian we have already found the ground state of
        # gswf: The ground state wf, an instance of NQS that can return Psi(S)
        # Penalty: A positive number, the energy penalty suffered by overlap with the ground state
        self.h0 = h0
        self.gswf = gswf
        self.penalty = penalty

    def find_conn(self, state):
        # This is the same "find_conn" used by the other Hamiltonians
        # All we do is call that function
        mel, flipsh = self.h0.find_conn(state)  # Matrix element and flips to get connected states
        mel[0] += self.penalty*np.abs(np.exp(self.gswf.log_val(state)))

        return mel, flipsh