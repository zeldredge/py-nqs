import nqs
import numpy as np
import ising1d
import sampler
from math import *
import scipy.optimize
import matplotlib.pyplot as plt

def log_to_fit(l,delta,gamma):
    return (1 - delta/2)*np.log(l) + gamma

dlist = []
glist = []

nspins = 40
training_step = 500
nsamples = 50000
alpha = 2

annqs = nqs.NqsTI(nspins,alpha)
annqs.load_parameters('../Outputs/'+str(nspins)+'_alpha='+str(alpha)+'_Ising10_ti_'+str(training_step)+'.npz')
h = ising1d.Ising1d(nspins,1.0)

s = sampler.Sampler(annqs,h, quiet = False)
s.run(nsamples)

states = np.zeros((nsamples,nspins))

for i in range(nsamples):
    for j in range(nspins):
        s.move()
    states[i] = s.state

data = np.zeros(nspins)

for size in range(nspins):
    size_data = np.array([])
    for state in states:
        size_data = np.append(size_data, np.sum(state[:size+1]))
    size_x, size_p = np.unique(size_data, return_counts = True)
    size_p = size_p/np.sum(size_p)
    data[size] = -np.sum(size_p[size_p > 0] * np.log(size_p[size_p > 0]))
    std = np.std(size_data)
    h = np.histogram(std*size_data, std*size_x)

fit, cov = scipy.optimize.curve_fit(log_to_fit, np.arange(1,nspins+1), data)
plt.gcf().clear()
plt.plot(np.arange(1,nspins+1), data, marker='o', linestyle = 'none', label = 'data')
plt.plot(np.arange(1,nspins+1), log_to_fit(np.arange(1,nspins+1), fit[0], fit[1]), label = 'fit')
plt.xlabel('Subsystem Size')
plt.ylabel('Entropy of Magnetization Distribution')
plt.legend(loc = 2)
plt.text(16,1,'Delta: {0:.3f} \nGamma: {1:.3f}'.format(*fit))
print("Fit parameters:\n")
print("Delta = {0} \n".format(fit[0]))
print("Gamma = {0} \n".format(fit[1]))
dlist.append(fit[0])
glist.append(fit[1])
plt.figure()
#plt.plot(std*h[0]/np.sum(h[0]), marker = 'o')
plt.hist(size_data, bins = np.arange(-nspins,nspins+2), normed = True)
#
#plt.figure()
#plt.plot([20, 40, 60, 80, 100, 120, 140, 160],dlist)