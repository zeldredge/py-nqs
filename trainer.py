import numpy as np
import sampler


class Trainer:
    def __init__(self, h, reg_list=(100, 0.9, 1e-4), mag0=True):
        self.h = h  # Hamiltonian to evaluate wf against
        self.nspins = h.nspins
        self.reg_list = reg_list  # Parameters for regularization
        self.step_count = 0

    def update_vector(self, wf, state, batch_size, gamma):  # Get the vector of updates
        samp = sampler.Sampler(wf, self.h)  # start a sampler
        samp.nflips = self.h.minflips
        samp.state = state

        elocals = np.zeros(batch_size, dtype=complex)  # Elocal results at each sample
        deriv_vectors = np.zeros((batch_size, wf.nh + wf.nv + wf.nh*wf.nv), dtype=complex)

        for sample in range(batch_size):
            samp.reset_av()  # reset whether we have accepted a move
            while samp.accepted == 0:  # until we find a good spin
                samp.move()
            elocals[sample] = self.get_elocal(samp.state, wf)
            deriv_vectors[sample] = self.get_deriv_vector(samp.state, wf)

        # Now that we have all the data from sampling let's run our statistics
        cov = self.get_covariance(deriv_vectors)
        forces = self.get_forces(elocals, deriv_vectors)

        # Now we calculate the updates as
        updates = gamma * np.dot(np.linalg.pinv(cov), forces)

        return updates

    def get_elocal(self, state, wf):  # Function to calculate local energies; see equation A2 in Carleo and Troyer
        eloc = 0j  # Start with 0
        mel, flips = self.h.find_conn(state)  # Get all S' that connect to S via H and their matrix elems
        for flip in range(len(flips)):
            eloc += mel[flip] * wf.pop(state, flips[flip])

        return eloc

    def get_deriv_vector(self, state, wf):
        # The derivative vector is a vector which contains the following elements in one column:
        # First: derivative of psi(S) w.r.t. visible unit biases (wf.Nv of them)
        # Second: the hidden unit biases (wf.Nh of them)
        # Third: The weights (wf.Nh * wf.Nv)
        # See Carleo C3-5 for formulas

        vector = np.zeros(wf.nv + wf.nh + wf.nv * wf.nh,dtype=complex)  # initialize

        for bias in range(wf.nv):  # visible unit biases
            vector[bias] = state[bias]

        for bias in range(wf.nv, wf.nh):  # hidden unit biases
            vector[bias] = np.tanh(wf.Lt[bias])

        for v in range(wf.nv):
            for h in range(wf.nh):
                vector[wf.nh + wf.nv + wf.nh * v + h] = state[v] * np.tanh(wf.Lt[h])

        return vector

    def get_covariance(self, deriv_vectors):
        # I'm writing this rather than using np.cov because I am not sure numpy handles complex values right

        Omean = np.mean(deriv_vectors, axis=0)  # First get the mean O vector
        outers = []  # empty list of outer product matrices
        for vec in deriv_vectors:
            outers.append(np.outer(np.conj(vec), vec))  # First term in A4
        mean_outer = np.mean(outers, axis=0)  # Get the mean outer product matrix
        smat = mean_outer - np.outer(np.conj(Omean), Omean)  # Eq A4
        return smat

    def get_forces(self, elocals, deriv_vectors):
        emean = np.mean(elocals)  # mean local energy
        omean = np.mean(deriv_vectors, axis=0)  # mean derivative vector
        correlator = np.mean([i[0] * np.conj(i[1]) for i in zip(elocals, deriv_vectors)])
        # pair the local energies with Ovecs and then calculate mean

        return correlator - emean * np.conj(omean)
