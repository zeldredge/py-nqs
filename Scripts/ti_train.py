import heisenberg1d
import ising1d
import xyz
import nqs
import trainer
import sampler
import numpy as np
import matplotlib.pyplot as plt
import time
import fermionhop1d


start = time.time()
nruns = 1000
nspins = 10
nsteps = 100
m = True

wf = nqs.NqsTI(nspins, 1)  # A translation invariant NQS instance

wf.Wreduced = 0.1*np.random.random(wf.Wreduced.shape) + 0j # Fill in with starting values
if type(wf.a) == int:
    wf.a = 0.1*np.random.uniform() + 0j
else:
    wf.a = 0.1*np.random.uniform()*np.ones(wf.a.shape) + 0j
wf.breduced = 0.1*np.random.random(wf.breduced.shape) + 0j

h = ising1d.Ising1d(10,0.5)
#h = heisenberg1d.Heisenberg1d(10,1)
h = xyz.XYZ(10,(-1,-1,0))
#h = fermionhop1d.FermionHop(nspins,-2)
base_array = -1*np.ones(nspins)
base_array[0:nspins // 2] *= -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array
wf.init_lt(state)

s = sampler.Sampler(wf, h, mag0 = m)
s.run(nruns, init_state = state)

state = s.state
wf.init_lt(state)

def gamma_fun(p):
    return .05
    #return .99**p

t = trainer.build_trainer(wf,h)

wf, elist = t.train(wf,state,nruns,nsteps,gamma_fun, file='../Outputs/Ising1d05', out_freq=0)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)
s.run(nruns)

end = time.time()
print('Time Elapsed: {}'.format(end - start))
plt.plot(elist)