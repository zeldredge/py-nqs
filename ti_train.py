import heisenberg1d
import nqs
import nqsti
import sampler
import trainer_ti
import numpy as np

nruns = 1000

np.random.seed(5291992)

n1 = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.npz")  # a full, normal nqs without translation invariance
wf = nqsti.NqsTI(n1.nv, 1)  # A translation invariant NQS instance

wf.W = n1.W[0]  # Fill in with starting values
wf.a = n1.a[0]
wf.b = n1.b[0]

r = np.random.random(wf.W.shape)
wf.W = 0.01*r + 0.00*wf.W

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

t = trainer_ti.Trainer(h)

wf, elist = t.train(wf,state,1000,101,gamma_fun, file='Outputs/test', out_freq=20)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)
s.run(nruns)