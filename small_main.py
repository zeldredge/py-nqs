import heisenberg1d
import nqs
import sampler
import trainer
import numpy as np

nruns = 100

wf = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.wf")
np.random.seed(5291992)
r = np.random.random(wf.W.shape)
wf.W = 0*wf.W + 0.1*r


h = heisenberg1d.Heisenberg1d(40, 1)
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)

s = sampler.Sampler(wf, h)
s.run(nruns)

state = s.state
wf.init_lt(state)

t = trainer.Trainer(h)
wf = t.train(wf,state,10000,5,10**-3)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)
s.run(nruns)
