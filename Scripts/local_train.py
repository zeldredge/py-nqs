import heisenberg1d
import ising1d
import fermionhop1d
import nqs
import trainer
import sampler
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
nruns = 100
k = 1
gam = .01
nspins = 10
h = heisenberg1d.Heisenberg1d(nspins,1)
#h = ising1d.Ising1d(40,0.5)
#h = fermionhop1d.FermionHop(nspins,1)

wf = nqs.NqsLocal(nspins, 1, k)  # A local NQS instance

wf.W = 0.1*np.random.random(wf.W.shape) + 0j # Fill in with starting values
wf.a = 0.1*np.random.random(wf.a.shape) + 0j
wf.b = 0.1*np.random.random(wf.b.shape) + 0j


base_array = np.concatenate(
                (np.ones(int(1)), -1 * np.ones(int(nspins-1))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)

s = sampler.Sampler(wf, h)
s.run(nruns)

state = s.state
wf.init_lt(state)

def gamma_fun(p):
    return gam

t = trainer.TrainerLocal(h, cores=4)

wf, elist = t.train(wf,state,nruns,101,gamma_fun, file='../Outputs/Ising1d05', out_freq=0)

s = sampler.Sampler(wf, h)
s.run(nruns)
end = time.time()
print("Elapsed time: {}".format(end - start))
plt.plot(elist)