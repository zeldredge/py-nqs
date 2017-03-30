import nqs
import numpy as np
import sampler
import heisenberg1d
import time
import trainer

# Script to test the symmetric NQS class I wrote against the class that is hardcoded translation invariant

n1 = nqs.NqsTI(40, 1)  # A translation invariant NQS instance

n1.W = 0.1 * np.random.random(n1.W.shape) + 0j # Fill in with starting values
n1.a = 0.1 * np.random.uniform() + 0j
n1.b = 0.1 * np.random.random(n1.b.shape) + 0j

shift_group = np.array([np.roll(np.identity(40), i, axis=1) for i in range(40)])

n2 = nqs.NqsSymmetric(40, 1, shift_group)
n2.W = np.copy(n1.W)
n2.a =np.array([n1.a])
n2.b = np.copy(n1.b)

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


def gamma_fun(p):
    return .01

print("Training n1...")
t1 = trainer.TrainerTI(h)
n1,elist = t1.train(n1,state,100,101,gamma_fun, file='../Outputs/test', out_freq=20)

print("Training n2...")
t2 = trainer.TrainerSymmetric(h)
n2,elist = t2.train(n2,state,100,101,gamma_fun, file='../Outputs/test', out_freq=20)

print("Testing training on density-2 symmetric case...")
n2 = nqs.NqsSymmetric(40, 2, shift_group)
n2.W = 0.1*np.random.random(n2.W.shape) + 0j
n2.a = 0.1*np.random.random(n2.a.shape) + 0j
n2.b = 0.1*np.random.random(n2.b.shape) + 0j

base_array = np.concatenate(
                (np.ones(int(20)), -1 * np.ones(int(20))))  # make an array of half 1, half -1
state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

n2.init_lt(state)

n2,elist = t2.train(n2,state,100,101,gamma_fun, file='../Outputs/test', out_freq=20)