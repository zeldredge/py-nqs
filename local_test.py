import nqs
import numpy as np
import sampler
import heisenberg1d
import time
import trainer

# Script to test the local NQS class I wrote against the normal class

n1 = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.npz")  # A vanilla NQS instance

n1.W = 0.1 * np.random.random(n1.W.shape) + 0j  # Fill in with starting values
n1.a = 0.1 * np.random.random(n1.a.shape) + 0j
n1.b = 0.1 * np.random.random(n1.b.shape) + 0j

# To "localize" the above array, create a tridiagonal array and multiply elementwise
tridiag = np.diag(np.ones(40,dtype=complex)) + np.diag(np.ones(39), -1) + np.diag(np.ones(39), +1)
n1.W = tridiag * n1.W
n = n1.W.T

# Now we create the local NQS instance

n2 = nqs.NqsLocal(40,1,1)
n2.a = n1.a
n2.b = n1.b
indices = np.array([-1,0,1])
for i in range(40):
    n2.W[i][0] = n[i][(i+indices) % n1.nv]


# Now begin testing outputs
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

n2.init_lt(state)
n1.init_lt(state)

flips = np.array(np.random.choice(np.arange(n1.nv), 2))

lv_match = np.all(np.isclose(n1.log_val(state), n2.log_val(state)))
lp_match = np.all(np.isclose(n1.log_pop(state, flips), n2.log_pop(state, flips)))
print("Log_val matches: {}".format(lv_match))
print("Log_pop matches: {}".format(lp_match))

if not (lv_match and lp_match):
    exit()

nruns = 1000
h = heisenberg1d.Heisenberg1d(40, 1)

print("Sampling n1 ...")

start_time = time.time()
s1 = sampler.Sampler(n1, h)
s1.run(nruns)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

print("Sampling n2 ...")

start_time = time.time()
s2 = sampler.Sampler(n2, h)
s2.run(nruns)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

def gamma_fun(p):
    return .01

t = trainer.TrainerLocal(h)

n2, elist = t.train(n2,state,100,101,gamma_fun, file='Outputs/test', out_freq=20)

#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(n2, h)
s.run(nruns)