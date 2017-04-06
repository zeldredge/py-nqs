import heisenberg1d
import ising1d
import nqs
import trainer
import sampler
import numpy as np
import matplotlib.pyplot as plt

nruns = 1000
k = 2
gam = .025
h = heisenberg1d.Heisenberg1d(40,1)
#h = ising1d.Ising1d(40,1)

wf = nqs.NqsLocalTI(40, 1, k)  # A translation invariant NQS instance

wf.Wloc = 0.1*np.random.random(wf.Wloc.shape) + 0j # Fill in with starting values
wf.a = 0.1*np.random.uniform() + 0j
wf.b = 0.1*np.random.random(wf.b.shape) + 0j


base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)

s = sampler.Sampler(wf, h)
s.run(nruns)

state = s.state
wf.init_lt(state)

def gamma_fun(p):
    return gam

t = trainer.TrainerLocalTI(h)

wf, elist = t.train(wf,state,100,201,gamma_fun, file='../Outputs/Ising1d05', out_freq=0)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)
s.run(nruns)

plt.plot(elist)