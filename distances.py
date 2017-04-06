import sampler
import numpy as np
import trainer
from cmath import *


# Functions to calculate the Fubini-Study distances we need to use to evaluate error
# See Eqn B2 in Carleo + Troyer, also Fig 6b

def getZ(wf, nruns):
    i = IdentityOp()
    s = sampler.Sampler(wf, i)
    s.run(nruns)
    return s.estav


def get_ddist(wf, H, delta, nruns):  # get the quantity D_0^2
    # see milanote for the breakdown into expectation values

    z = getZ(wf, nruns)  # <I>
    h2op = Hsq(H)  # build the H^2 sampler
    hsq_samp = sampler.Sampler(wf, h2op)
    hsq_samp.run(nruns)
    h2 = hsq_samp.estav  # calculate <H^2>

    h_samp = sampler.Sampler(wf, H)
    h_samp.run(nruns)  # calculate < H>
    e = h_samp.estav

    frac = abs(z - delta * e) ** 2 / (z * (z - delta ** 2 * h2))
    return acos(sqrt(frac)) ** 2


def get_rdist(wf, nruns, h):  # get the quantity R_0^2

    h2op = Hsq(h)  # build the H^2 sampler
    hsq_samp = sampler.Sampler(wf, h2op)
    hsq_samp.run(nruns)
    h2 = hsq_samp.estav  # calculate <H^2>

    s = sampler.Sampler(wf, IdentityOp)
    s.state = np.ones(wf.nspins)
    s.thermalize(1000)
    state = s.state
    t = trainer.Trainer(h)
    u = t.update_vector(wf, state, nruns, .01j)
    avg1 = 0
    avg2 = 0
    for j in range(nruns):
        s.move()  # make a move
        state = s.state
        mel, flips = h.find_conn(state)
        d2 = t.get_deriv_vector(state2, wf)
        avg2 += np.abs(np.dot(d2, u)) ** 2 / nruns
        for i in range(flips):
            state2 = state
            state2[flips[i]] *= -1
            d = t.get_deriv_vector(state2, wf)
            lpop = wf.log_pop(flips[i])
            avg1 += lpop * mel[i] * np.dot(d, u)
    avg1 /= nruns
    avg1 **= 2
    return acos(sqrt(avg1 / (avg2 * h2))) ** 2


class IdentityOp:  # because I need to get the partition function
    def __init__(self):
        self.minflips = 1

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
            state2 = state
            state2[hflips[f]] *= -1
            h2mel, h2flips = self.baseh.find_conn(state2)
            for s in range(len(h2flips)):
                if h2flips[s] in flipsh:
                    ind = flipsh.index(h2flips[s])
                    mel[ind] += h2mel[s] * hmel[f]
                else:
                    flipsh.append(h2flips[s])
                    mel.append(h2mel[s] * hmel[f])
        return mel, flipsh
