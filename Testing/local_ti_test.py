import nqs
import numpy as np
import sampler
import heisenberg1d
import time
from scipy.linalg import hankel

density = 1
n1 = nqs.NqsTI(40, density)  # A translation invariant NQS instance with alpha = 1
n2 = nqs.NqsLocalTI(40, density, 1) # A local TI NQS instance with same alpha

# Randomize the TI case
n1.W = 0.1*np.random.random(n1.W.shape) + 0j
n1.a = 0.1*np.random.uniform(0) + 0j
n1.b = 0.1*np.random.random(n1.b.shape) + 0j


# Now we localize the TI NQS instance by creating the array (1, 1, 0, 0.... 1) and multiplying elementwise
localizer = np.zeros(40)
localizer[0:2] = 1
localizer[-1] = 1
n1.W *= localizer

# Now feed all the TI parameters to local TI NQS
n2.a = n1.a
n2.b = n1.b
for a in range(density):
    n2.Wloc[a] = n1.W[a][np.arange(-1,2)]

# Now begin testing outputs
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

n1.init_lt(state)
n2.init_lt(state)

flips = np.array(np.random.choice(np.arange(n1.nv), 2))

print("Log_val matches: {}".format(np.all(np.isclose(n1.log_val(state), n2.log_val(state)))))
print("Log_pop matches: {}".format(np.all(np.isclose(n1.log_pop(state, flips), n2.log_pop(state, flips)))))

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