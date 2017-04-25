import numpy as np
import nqs
import ising1d
import distances

n1 = nqs.NqsTI(40, 1)
n1.load_parameters('../Outputs/evolution100.npz')
h = ising1d.Ising1d(40, 1.0)

print("ddist = ", distances.get_ddist(n1, h, .01, 100))
print("rdist = ", distances.get_rdist(n1, 100, h))

