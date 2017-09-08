import numpy as np
import nqs
import ising1d
import distances
import matplotlib.pyplot as plt
from observables import *
import sampler

n1 = nqs.NqsTI(10, 1)

h = ising1d.Ising1d(10, 1.0)

dlist = []
rlist = []

#for i in range(0,200,5):
#    n1.load_parameters('../Outputs/10SpinEvolve/evolution_ti_'+str(i)+'.npz')
#    ddist = distances.get_ddist(n1, h, .01, 1000)
#    rdist = distances.get_rdist(n1, 1000, h)
#    dlist.append(ddist)
#    rlist.append(rdist)

h = Sigmax(10,1)

n1.load_parameters('../Outputs/10_Ising00_ti_200.npz')
s = sampler.Sampler(n1,h,quiet = False)
s.run(1000)

ddist = distances.get_ddist(n1,h,.01,1000)
rdist = distances.get_rdist(n1,1000,h)

print("ddist = {}".format(ddist))
print("rdist = {}".format(rdist))
print("rdist/ddist = {} \n".format(rdist/ddist))

plt.plot(dlist)
plt.plot(rlist)
plt.figure()
plt.plot([i/j for i, j in zip(rlist,dlist)])