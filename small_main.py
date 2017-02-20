import heisenberg1d
import nqs
import sampler
import trainer
import numpy as np

nruns = 1000

wf = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.wf")
h = heisenberg1d.Heisenberg1d(40, 1)
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)
t = trainer.Trainer(h)

#h = ising1d.Ising1d(40,1)
#s = sampler.Sampler(wf, h)
#s.run(nruns)
