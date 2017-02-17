import numpy as np


class Trainer:
    def __init__(self, h, reg_list=(100, 0.9, 1e-4), mag0=True):
        self.h = h  # Hamiltonian to evaluate wf against
        self.nspins = h.nspins
        self.reg_list = reg_list  # Parameters for regularization
        self.step_count = 0

    def init_random_state(self, mag0=True):

        if not mag0:  # if we don't enforce magnetization = 0, it is easy
            state = np.random.choice([-1, 1], self.nspins)  # make a bunch of random -1, 1

        if mag0:  # if we do, need to be cleverer
            if self.nspins % 2 != 0:
                raise ValueError('Need even number of spins to have zero magnetization!')
            base_array = np.concatenate((np.ones(int(self.nspins / 2)),
                                         -1 * np.ones(int(self.nspins / 2))))  # make an array of half 1, half -1
            state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

        return state

    def update_vector(self, wf, state, batch_size, gamma):  # Get the vector of updates
        elocals = np.zeros(batch_size, dtype=float)  # Elocal resuls at each sample
        deriv_vectors = np.zeros((batch_size, self.nspins), dtype=complex)

        for sample in range(batch_size):
            state = self.get_next_state(state)
            elocals[i] = self.get_elocal(state, wf)
            deriv_vectors[i] = self.get_deriv_vector(state, wf)

        # Now that we have all the data from sampling let's run our statistics
        cov = get_covariance(deriv_vectors)
        forces = get_forces(elocals, deriv_vectors)

        # Now we calculate the updates as
        updates = gamma * np.dot(np.linalg.inv(cov), forces)

        return updates

    def get_elocals(self, state, wf):  # Function to calculate local energies; see equation A2 in Carleo and Troyer
        eloc = 0  # Start with 0
        mel, flips = self.h.find_conn(state)  # Get all S' that connect to S via H and their matrix elems
        for flip in range(len(flips)):
            eloc += mel[flip] * wf.pop(flips[flip])

        return eloc
