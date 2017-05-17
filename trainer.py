import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import sampler
from joblib import Parallel, delayed


class Trainer:
    def __init__(self, h, reg_list=(100, 0.9, 1e-4)):
        self.h = h  # Hamiltonian to evaluate wf against
        self.nspins = h.nspins
        self.reg_list = reg_list  # Parameters for regularization
        self.step_count = 0
        self.nvar = 0

    def train(self, wf, init_state, batch_size, num_steps, gamma_fun, print_freq=25, file='', out_freq=0):
        state = init_state
        elist = np.zeros(num_steps, dtype=complex)  # list of energies to evaluate
        for step in range(num_steps):
            # First call the update_vector function to get our set of updates and the new state (so process thermalizes)
            updates, state, elist[step] = self.update_vector(wf, state, batch_size, gamma_fun(step), step)
            # Now apply appropriate parts of the update vector to wavefunction parameters
            self.apply_update(updates, wf)

            if step % print_freq == 0:
                print("Completed training step {}".format(step))
                print("Current energy per spin: {}".format(elist[step]))

            if out_freq > 0 and step % out_freq == 0:
                wf.save_parameters(file + str(step))

        return wf, elist

    def update_vector(self, wf, init_state, batch_size, gamma, step, therm=False):  # Get the vector of updates
        self.nvar = self.get_nvar(wf)
        wf.init_lt(init_state)
        samp = sampler.Sampler(wf, self.h)  # start a sampler
        samp.nflips = self.h.minflips
        samp.state = np.copy(init_state)
        samp.reset_av()
        if therm == True:
            samp.thermalize(batch_size)
        results = []  # Results from all the samples

        for sample in range(batch_size):
            results.append(self.get_sample(samp))

        elocals = np.array([i[0] for i in results])
        deriv_vectors= np.array([i[1] for i in results])

        # Now that we have all the data from sampling let's run our statistics
        # cov = self.get_covariance(deriv_vectors)
        cov_operator = LinearOperator((self.nvar, self.nvar), dtype=complex,
                                      matvec=lambda v: self.cov_operator(v, deriv_vectors, step))

        forces = self.get_forces(elocals, deriv_vectors)

        # Now we calculate the updates as
        # updates = -gamma * np.dot(np.linalg.pinv(cov), forces)
        vec, info = cg(cov_operator, forces)
        # vec, info = cg(cov, forces)
        updates = -gamma * vec
        self.step_count += batch_size
        return updates, samp.state, np.mean(elocals) / self.nspins

    def get_sample(self,sampler):
        for i in range(sampler.nspins):
            sampler.move()
        return self.get_elocal(sampler.state, sampler.wf), self.get_deriv_vector(sampler.state, sampler.wf)

    def get_elocal(self, state, wf):  # Function to calculate local energies; see equation A2 in Carleo and Troyer

        if not all(state == wf.state): # make sure wavefunction lookup table is properly initialized
            wf.init_lt(state)

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

        vector = np.zeros(self.nvar, dtype=complex)  # initialize

        for bias in range(wf.nv):  # visible unit biases
            vector[bias] = state[bias]

        for bias in range(wf.nh):  # hidden unit biases
            vector[wf.nv + bias] = np.tanh(wf.Lt[bias])

        for v in range(wf.nv):
            for h in range(wf.nh):
                vector[wf.nh + wf.nv + wf.nh * v + h] = state[v] * np.tanh(wf.Lt[h])

        return vector

    def get_forces(self, elocals, deriv_vectors):
        emean = np.mean(elocals)  # mean local energy
        omean = np.mean(deriv_vectors, axis=0)  # mean derivative vector
        correlator = np.mean([i[0] * np.conj(i[1]) for i in zip(elocals, deriv_vectors)], axis=0)
        # pair the local energies with Ovecs and then calculate mean

        return correlator - emean * np.conj(omean)

    def cov_operator(self, vec, deriv_vectors, step):  # Callable function for evaluating S*v
        tvec = np.dot(deriv_vectors, vec)  # vector of t-values
        term1 = np.dot(deriv_vectors.T.conj(), tvec) / deriv_vectors.shape[0]
        term2 = np.mean(deriv_vectors.conj(), axis=0) * np.mean(tvec)
        reg = max(self.reg_list[0] * self.reg_list[1] ** step, self.reg_list[2]) * vec
        return term1 - term2 + reg

    def apply_update(self, updates, wf):
        wf.a += updates[0:wf.nv]
        wf.b += updates[wf.nv:wf.nh + wf.nv]
        wf.W += np.reshape(updates[wf.nv + wf.nh:], wf.W.shape)

    def get_nvar(self, wf):
        return wf.nh + wf.nv + wf.nh * wf.nv


class TrainerTI(Trainer):
    def __init__(self, h, reg_list=(100, 0.9, 1e-4)):
        Trainer.__init__(self, h, reg_list=reg_list)

    def apply_update(self, updates, wf):
        wf.a += updates[0]
        wf.b += updates[1:wf.alpha + 1]
        wf.W += updates[wf.alpha + 1:].reshape(wf.W.shape)

    def get_deriv_vector(self, state, wf):
        # The derivative vector is a vector which contains the following elements in one column:
        # First: derivative of psi(S) w.r.t. visible unit biases (1 of them)
        # Second: the hidden unit biases (wf.alpha of them)
        # Third: The weights (wf.Nh * wf.alpha)
        # See Carleo C3-5 for formulas

        vector = np.zeros(1 + wf.alpha + wf.nv * wf.alpha, dtype=complex)  # initialize

        vector[0] = np.sum(state)

        for a in range(wf.alpha):
            vector[1 + a] = np.sum(np.tanh(wf.Lt[a * wf.nv:(a + 1) * wf.nv]))

            # for j in range(wf.nv):
            # for a in range(wf.alpha):
            #    vector[wf.alpha + 1 + j*a + a] = np.sum(np.roll(state, -j)*np.tanh(wf.Lt))
        for j in range(wf.alpha * wf.nv):
            vector[wf.alpha + 1 + j] = np.sum(
                [state[(j - s) % wf.nv] * np.tanh(wf.Lt[s + wf.nv * (j // wf.nv)]) for s in range(wf.nv)])

        return vector

    def get_nvar(self, wf):
        return 1 + wf.alpha + wf.alpha * wf.nv


class TrainerSymmetric(Trainer):
    def __init__(self, h, reg_list=(100, 0.9, 1e-4)):
        Trainer.__init__(self, h, reg_list=reg_list)

    def apply_update(self, updates, wf):
        wf.a += updates[0:wf.a.size]
        wf.b += updates[wf.a.size:wf.a.size + wf.b.size]
        wf.W += updates[wf.a.size + wf.b.size:].reshape(wf.W.shape)

    def get_deriv_vector(self, state, wf):
        # The derivative vector is a vector which contains the following elements in one column:
        # First: derivative of psi(S) w.r.t. visible unit biases (1 of them)
        # Second: the hidden unit biases (wf.alpha of them)
        # Third: The weights (wf.Nh * wf.alpha)
        # See Carleo C3-5 for formulas

        vector = np.zeros(wf.a.size + wf.b.size + wf.W.size, dtype=complex)  # initialize

        for a in range(wf.a.size):
            vector[a] = np.sum(np.dot(wf.t_group, state))

        for a in range(wf.b.size):
            vector[wf.a.size + a] = np.sum(np.tanh(wf.Lt[a * wf.nv:(a + 1) * wf.nv]))

        offset = wf.a.size + wf.b.size

        for f in range(wf.alpha):  # for each feature
            vector[offset + f * wf.nv: offset + (f + 1) * wf.nv] = np.dot(np.dot(state, wf.t_group).T,
                                                                          np.tanh(wf.Lt[f * wf.nv:(f + 1) * wf.nv]))
            # for v in range(wf.nv):
            #     vector[offset + f*wf.nv + v] = np.sum( np.array([ np.dot(state,wf.t_group[u])[v] * np.tanh(wf.Lt[f*wf.nv + u]) for u in range(wf.t_size)] ))

        return vector

    def get_nvar(self, wf):
        return wf.a.size + wf.b.size + wf.W.size


class TrainerLocal(Trainer):
    def __init__(self, h, reg_list=(100, 0.9, 1e-4)):
        Trainer.__init__(self, h, reg_list=reg_list)

    def apply_update(self, updates, wf):
        wf.a += updates[:wf.a.size]
        wf.b += updates[wf.a.size:wf.a.size + wf.b.size].reshape(wf.b.shape)
        wf.W += updates[wf.a.size + wf.b.size:].reshape(wf.W.shape)

    def get_nvar(self, wf):
        return wf.a.size + wf.b.size + wf.W.size

    def get_deriv_vector(self, state, wf):
        # The derivative vector is a vector which contains the following elements in one column:
        # First: derivative of psi(S) w.r.t. visible unit biases (wf.Nv of them)
        # Second: the hidden unit biases (wf.Nh of them)
        # Third: The weights (wf.Nh * wf.Nv)
        # See Carleo C3-5 for formulas

        vector = np.zeros(wf.a.size + wf.b.size + wf.W.size, dtype=complex)  # initialize

        for bias in range(wf.nv):  # visible unit biases
            vector[bias] = state[bias]

        for v in range(wf.nv):  # hidden unit biases
            for b in range(wf.alpha):
                vector[wf.a.size + v * wf.alpha + b] = np.tanh(wf.Lt[v, b])

        for v in range(wf.nv):
            for b in range(wf.alpha):
                for l in wf.indices + wf.k:
                    vector[wf.a.size + wf.b.size + v * wf.alpha * (2 * wf.k + 1) + b * (2 * wf.k + 1) + l] = state[(
                                                                                                                       v + l - wf.k) % wf.nv] * np.tanh(
                        wf.Lt[v, b])

        return vector


class TrainerLocalTI(Trainer):
    def __init__(self, h, reg_list=(100, 0.9, 1e-4)):
        Trainer.__init__(self, h, reg_list=reg_list)

    def apply_update(self, updates, wf):
        wf.a += updates[0]
        wf.b += updates[1:wf.alpha + 1]
        wf.Wloc += updates[wf.alpha + 1:].reshape(wf.Wloc.shape)

    def get_deriv_vector(self, state, wf):

        locality_range = 2*wf.k+1

        vector = np.zeros(self.nvar, dtype = complex)

        vector[0] = np.sum(state)  # visible unit bias

        for a in range(wf.alpha):  # hidden unit biases
            vector[1 + a] = np.sum(np.tanh(wf.Lt[a * wf.nv:(a + 1) * wf.nv]))

        for a in range(wf.alpha):
            for j in range(-wf.k, wf.k + 1):
                vector[wf.alpha + 1 + a*locality_range + j + wf.k] = np.sum(
                    [state[(j - s) % wf.nv] * np.tanh(wf.Lt[s + wf.nv * a]) for s in range(wf.nv)])

        return vector

    def get_nvar(self, wf):
        return 1 + wf.alpha + wf.alpha*(2*wf.k + 1)

def build_trainer(wf, h, reg_list = (100, 0.9, 1e-4)):
    """
    Function to get the appropriate Trainer class depending on wavefunction symmetry
    
    :param wf: Wavefunction the trainer is being built for
    :param h: Hamiltonian to pass to the trainer
    :param reg_list: Regulator list
    :return: Trainer object of appropriate symmetry
    """

    s = wf.symmetry
    if wf.symmetry == "None":
        return Trainer(h, reg_list)
    if wf.symmetry == "Local":
        return TrainerLocal(h, reg_list)
    if wf.symmetry ==  "TI":
        return TrainerTI(h, reg_list)
    if wf.symmetry == "LocalTI":
        return TrainerLocalTI(h,reg_list)
    if wf.symmetry == "Symmetric":
        return TrainerSymmetric(h, reg_list)
    # If none of those work, print an error and return nothing
    raise ValueError("No trainer of appropriate symmetry found. Check Nqs.symmetric string.")