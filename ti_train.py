import heisenberg1d
import nqs
import trainer
import sampler
import numpy as np

nruns = 1000

#n1 = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.npz")  # a full, normal nqs without translation invariance
wf = nqs.NqsTI(40, 2)  # A translation invariant NQS instance

wf.W = 0.1*np.random.random(wf.W.shape) + 0j # Fill in with starting values
wf.a = 0.1*np.random.uniform() + 0j
wf.b = 0.1*np.random.random(wf.b.shape) + 0j

h = heisenberg1d.Heisenberg1d(40, 1)
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)

s = sampler.Sampler(wf, h)
s.run(nruns)

state = s.state
wf.init_lt(state)

def gamma_fun(p):
    return .01

t = trainer.TrainerTI(h)

wf, elist = t.train(wf,state,100,101,gamma_fun, file='Outputs/test', out_freq=20)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)
s.run(nruns)