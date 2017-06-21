import nqs
import numpy as np
import sampler
import heisenberg1d
import time
from scipy.linalg import hankel

n1 = nqs.Nqs(40,1)
n1 = n1.load_parameters("./Outputs/evolution_ti_0.npz")  # a full, normal nqs without translation invariance
n2 = nqs.NqsTI(n1.nv, 1)  # A translation invariant NQS instance with alpha = 2

# Now check whether nqs is TI
wti = np.all(np.isclose(n1.W, hankel(n1.W[:][0], n1.W[-1])))
ati = np.all(np.isclose(n1.a, n1.a[0]))
bti = np.all(np.isclose(n1.b, n1.b[0]))

if wti and ati and bti:
    print("Vanilla NQS is TI: confirmed")

else:
    raise ValueError("Vanilla NQS is NOT TI!")

# If NQS is TI, let's continue by constructing the TI version
n2.W[0] = n1.W[0]
n2.W[1] = np.zeros(n1.W[0].shape)
n2.a = n1.a[0]
n2.b = np.array([n1.b[0], 0])

# Now begin testing outputs
base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

n2.init_lt(state)
n1.init_lt(state)

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