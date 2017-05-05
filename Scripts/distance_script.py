import numpy as np
import nqs
import ising1d
import distances

n1 = nqs.NqsTI(40, 1)
n1.load_parameters('../Outputs/Ising05_ti_200.npz')
h = ising1d.Ising1d(40, 0.5)

ddist = distances.get_ddist(n1, h, .01, 100)
rdist = distances.get_rdist(n1, 100, h)
print("ddist = {}".format(ddist))
print("rdist = {}".format(rdist))
print("rdist/ddist = {}".format(rdist/ddist))

