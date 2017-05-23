import sampler
import numpy as np
import trainer
from cmath import *


# Functions to calculate the Fubini-Study distances we need to use to evaluate error
# See Eqn B2 in Carleo + Troyer, also Fig 6b

def getZ(wf, nruns):
    # i = IdentityOp(wf.nv)
    # s = sampler.Sampler(wf, i)
    # s.run(nruns)
    # return s.estav
    return 1.0


def get_ddist(wf, H, delta, nruns):  # get the quantity D_0^2
    # see milanote for the breakdown into expectation values

    z = getZ(wf, nruns)  # <I>
    h2op = Hsq(H)  # build the H^2 sampler
    hsq_samp = sampler.Sampler(wf, h2op)
    hsq_samp.run(nruns)
    h2 = hsq_samp.estav * wf.nv  # calculate <H^2>

    h_samp = sampler.Sampler(wf, H)
    h_samp.run(nruns)  # calculate < H>
    e = h_samp.estav * wf.nv  # Sampler naturally returns a per-spin result

    frac = (z ** 2 + delta ** 2 * e ** 2) / (z * (z + delta ** 2 * h2))

    return acos(sqrt(frac)) ** 2


def get_rdist(wf, nruns, h):  # get the quantity R_0^2

    h2op = Hsq(h)  # build the H^2 sampler
    hsq_samp = sampler.Sampler(wf, h2op)
    hsq_samp.run(nruns)
    h2 = hsq_samp.estav * wf.nv  # calculate <H^2>
    s = sampler.Sampler(wf, IdentityOp)  # build a sampler with generic (identity) observable
    #s.state = np.random.permutation(np.concatenate(
     #   (np.ones(int(20)), -1 * np.ones(int(20)))))
    s.state = np.ones(wf.nv)
    s.thermalize(1000)  # thermalize
    state = s.state  # take the ended state
    t = trainer.build_trainer(wf, h, reg_list=(0, 0, 0))
    # Get -i*S^-1*F for our wavefunction
    u = t.update_vector(wf, state, 100, 1j, 1)[0]
    dtpsi2  = []  # this is the denominator
    states = []
    avg1 = 0
    for j in range(nruns):
        wf.init_lt(state)
        for i in range(s.nspins): # Do Monte Carlo sweeps
            s.move()  # make a move
        state = s.state  # make that move the new state, now we calculate (H*dt)_local
        states.append(state)

        d = t.get_deriv_vector(state, s.wf)  # get the vector of quantities (1/psi) dpsi/d(parameter)
        eloc = t.get_elocal(state, s.wf) # get the local energy
        avg1 += eloc*np.dot(d,u) #numerator of the rdist
        dtpsi2.append(abs(np.dot(d, u))**2 )# The <dtPsi>^2 part

    avg1 = abs(avg1/nruns)**2
    avg2 = np.mean(dtpsi2)
    rdist = acos(sqrt(avg1 / (avg2 * h2))) ** 2
    return rdist


class IdentityOp:  # because I need to get the partition function
    def __init__(self, nspins):
        self.minflips = 1
        self.nspins = nspins

    def find_conn(self, state):
        flipsh = [[None]]
        mel = [1]
        return mel, flipsh


class Hsq:  # automagically construct the Hamiltonian**2 given the original Hamiltonian
    def __init__(self, baseh):
        self.minflips = baseh.minflips
        self.nspins = baseh.nspins
        self.baseh = baseh

    def find_conn(self, state):
        # First, apply the Hamiltonian once -- get a list of connected states
        hmel, hflips = self.baseh.find_conn(state)
        mel = []
        flipsh = []
        # Now, for every connected state, we need to go through all of ITS connected states
        # The total matrix element for a given flip will include all these possible paths
        for f in range(len(hflips)):
            state2 = np.copy(state)
            if hflips[f] != [None]:
                state2[hflips[f]] *= -1
            h2mel, h2flips = self.baseh.find_conn(state2)

            for s in range(len(h2flips)):
                # Now we need to compose the flips, so that we can get a single object describing both applications of the
                # Hamiltonian. We do this by first creating an arary of all ones:
                test = np.ones(self.nspins)
                # Then we apply both sets of flips
                if hflips[f] != [None]:
                    test[hflips[f]] *= -1
                if h2flips[s] != [None]:
                    test[h2flips[s]] *= -1
                # The ones which need to be flipped are now the -1 that remain after both operations
                flips_comp = [x for x in np.arange(self.nspins) if test[x] < 0]
                # There's a chance flips_comp ends up empty, in which case we need to set it to none
                if not flips_comp:
                    flips_comp = [None]

                # Now we check whether the composite flip is in the total flips list (flipsh) yet

                if flips_comp in flipsh:  # If so, find its index and add the product of the matrix elements to the sum
                    ind = flipsh.index(flips_comp)
                    mel[ind] += h2mel[s] * hmel[f]
                else:  # Otherwise, append it
                    flipsh.append(flips_comp)
                    mel.append(h2mel[s] * hmel[f])
        return mel, flipsh
