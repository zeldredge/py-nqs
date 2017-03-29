import nqs
import numpy as np
import sampler
import observables
import time
import matplotlib.pyplot as plt

#Script to generate the data -- sigmax(t) for all our time-evolved wave functions
nsteps = 400
data_rate = 2
nruns = 10000
talk_rate = 25

## Now that we have all the wavefunctions generated, find the <sigma(x)> at each one

#Fully connected translation-invariant
print("Fully connected ANNQS")
sxnonloc = []
wf = nqs.NqsTI(40,1)
start_time = time.time()
for t in np.arange(0, nsteps, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('./Outputs/evolution_ti_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sxnonloc.append(40*s.estav)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

#1-local
sx1 = []
print("1-local ANNQS")
wf = nqs.NqsLocal(40,1,1)
start_time = time.time()
for t in np.arange(0, nsteps // 2, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('./Outputs/evolution_k=1_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sx1.append(40*s.estav)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

#2-local
print("2-local ANNQS")
sx2 = []
wf = nqs.NqsLocal(40,2,1)
start_time = time.time()
for t in np.arange(0, nsteps, data_rate):
    if t % talk_rate == 0:
        print('t = {}'.format(t))
    wf.load_parameters('./Outputs/evolution_k=2_'+str(t)+'.npz')
    s = sampler.Sampler(wf, observables.Sigmax(40,1), opname='transverse polarization')
    s.run(nruns)
    sx2.append(40*s.estav)

print("time elapsed: {:.2f}s".format(time.time() - start_time))

np.savez('./Outputs/sxdata.npz',sxnonloc,sx1,sx2,)

plt.figure()
xpts = np.arange(0,2.00,.01)
plt.ylabel(r'$\langle \sigma_x \rangle (t) $')
plt.xlabel('Time')
plt.plot(xpts,sxnonloc,label='Fully-connected')
plt.plot(np.arange(0, 1.00,.01),sx1,label='1-local')
plt.plot(xpts,sx2,label='2-local')
plt.legend()
plt.ylim(0,0.7)