import nqs
import numpy as np
import ising1d
import sampler
from math import *
import scipy.optimize
import matplotlib.pyplot as plt

def log_to_fit(l,delta,gamma):
    return (1 - delta/2)*np.log(l) + gamma

nspins = 60
nsamples = 10000
alpha = 2

annqs = nqs.NqsTI(nspins,alpha)
annqs.load_parameters('../Outputs/'+str(nspins)+'_Ising10_ti_200.npz')
h = ising1d.Ising1d(nspins,1.0)

sampler = sampler.Sampler(annqs,h, quiet = False)
sampler.run(nsamples)

states = np.zeros((nsamples,nspins))

for i in range(nsamples):
    for j in range(nspins):
        sampler.move()
    states[i] = sampler.state

data = np.zeros(nspins)

for size in range(nspins):
    size_data = np.array([])
    for state in states:
        size_data = np.append(size_data, np.sum(state[np.arange(size)]))
    size_data = np.histogram(size_data, np.arange(-nspins,nspins+1))[0]
    size_data = size_data/np.sum(size_data)
    size_data = size_data[size_data > 0]
    data[size] = -np.sum(size_data * np.log(size_data))

fit, cov = scipy.optimize.curve_fit(log_to_fit, np.arange(1,nspins+1), data)

plt.plot(np.arange(1,nspins+1), data, marker='o', linestyle = 'none', label = 'data')
plt.plot(np.arange(1,nspins+1), log_to_fit(np.arange(1,nspins+1), fit[0], fit[1]), label = 'fit')
plt.xlabel('Subsystem Size')
plt.ylabel('Entropy of Magnetization Distribution')
plt.legend(loc = 2)
plt.text(16,1,'Delta: {0:.3f} \nGamma: {1:.3f}'.format(*fit))
print("Fit parameters:\n")
print("Delta = {0} \n".format(fit[0]))
print("Gamma = {0} \n".format(fit[1]))