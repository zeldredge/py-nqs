import nqs
import numpy as np
import sampler
import heisenberg1d
import ising1d
import time
import trainer

# Script to get and save local NQS solutions of the TFI h = 0.5 model

#we create the local NQS instance

wf = nqs.NqsLocal(40, 2, 1)
wf.W = 0.1 * np.random.random(wf.W.shape) + 0j  # Fill in with starting values
wf.a = 0.1 * np.random.random(wf.a.shape) + 0j
wf.b = 0.1 * np.random.random(wf.b.shape) + 0j

# Now begin testing outputs
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

wf.init_lt(state)

h = ising1d.Ising1d(40, 0.5)

def gamma_fun(p):
    return .01

t = trainer.TrainerLocal(h)

wf, elist = t.train(wf, state, 1000, 201, gamma_fun, file='Outputs/Ising05_k=2_', out_freq=200)
