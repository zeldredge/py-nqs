import nqs
import numpy as np
import sampler
import observables
import time
import matplotlib.pyplot as plt
import distances
import ising1d

#Script to generate the data -- sigmax(t) for all our time-evolved wave functions
nsteps = 200
data_rate = 25
nruns = 1000
talk_rate = 25
h = ising1d.Ising1d(40,1.0)

## Now that we have all the wavefunctions generated, find the <sigma(x)> at each one

#Fully connected translation-invariant
print("Fully connected ANNQS")
sxnonloc = []
nonlocerr = []
wf = nqs.NqsTI(40,1)
start_time = time.time()
for t in np.arange(0, nsteps, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('../Outputs/evolution_ti_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sxnonloc.append(40*s.estav)
    err = distances.get_rdist(wf,nruns,h)/distances.get_ddist(wf,h,.01,nruns)
    nonlocerr.append(err)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

#1-local
sx1 = []
loc1err = []
print("1-local ANNQS")
wf = nqs.NqsLocal(40,1,1)
start_time = time.time()
for t in np.arange(0, nsteps, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('../Outputs/evolution_k=1_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sx1.append(40*s.estav)
    err = distances.get_rdist(wf, nruns, h) / distances.get_ddist(wf,h,.01,nruns)
    loc1err.append(err)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

#2-local
print("2-local ANNQS")
sx2 = []
loc2err = []
wf = nqs.NqsLocal(40,2,1)
start_time = time.time()
for t in np.arange(0, nsteps, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('../Outputs/evolution_k=2_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sx2.append(40*s.estav)
    err = distances.get_rdist(wf,nruns,h)/distances.get_ddist(wf,h,.01,nruns)
    loc2err.append(err)

print("time elapsed: {:.2f}s".format(time.time() - start_time))

np.savez('../Outputs/sxdata.npz',sxnonloc,sx1,sx2,nonlocerr,loc1err,loc2err)

plt.figure()
xpts = np.arange(0, nsteps, data_rate)
plt.ylabel(r'$\langle \sigma_x \rangle (t) $')
plt.xlabel('Time')
plt.plot(xpts,sxnonloc,label='Fully-connected')
plt.plot(xpts,sx1,label='1-local')
plt.plot(xpts,sx2,label='2-local')
plt.legend()


plt.figure()
plt.ylabel(r'Residuals')
plt.xlabel('Time')
plt.plot(xpts,nonlocerr,label='Fully-connected')
plt.plot(xpts,loc1err,label='1-local')
plt.plot(xpts,loc2err,label='2-local')
plt.legend()
