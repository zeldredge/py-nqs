import sampler
import numpy as np
import trainer
from cmath import *


# Functions to calculate the Fubini-Study distances we need to use to evaluate error
# See Eqn B2 in Carleo + Troyer, also Fig 6b

def getZ(wf, nruns):
    #i = IdentityOp(wf.nv)
    #s = sampler.Sampler(wf, i)
    #s.run(nruns)
    #return s.estav
    return 1.0


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

    s = sampler.Sampler(wf, IdentityOp) # build a sampler with generic (identity) observable
    s.state = np.ones(wf.nv) # start in all ones
    s.thermalize(1000) # thermalize
    state = s.state # take the ended state
    t = trainer.build_trainer(wf,h)
    u = t.update_vector(wf, state, nruns, .01j,1)[0] # Get -i*S^-1*F for our wavefunction
    avg1 = 0 # this is the numerator of the Rdist quantity
    avg2 = 0 # this is the denominator
    for j in range(nruns):
        s.move()  # make a move
        state = s.state # make that move the new state, now we calculate (H*dt)_local
        mel, flips = h.find_conn(state) # find all states connected to S
        d2 = t.get_deriv_vector(state, wf) # get the vector of quantities (1/psi) dpsi/d(parameter)
        avg2 += np.dot(d2,u)*np.dot(d2.conj(),u.conj()) # The <dtPsi>^2 part
        for i in range(len(flips)): # Now we do the H*dt part by going through all connected states via Hamiltonian
            state2 = state # get the connected state
            state2[flips[i]] *= -1
            d = t.get_deriv_vector(state2, wf) # calculate the derivative vector in that state
            lpop = wf.log_pop(state,flips[i]) # get psi'/psi
            avg1 += lpop * mel[i] * np.dot(d, u) # keep running average
    avg1 /= nruns
    avg1 = abs(avg1)**2
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
