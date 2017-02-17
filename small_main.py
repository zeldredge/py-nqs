import sampler
import nqs
import heisenberg1d
import heisenberg2d
import ising1d
import sys

nruns = 10000

wf = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.wf")
h = heisenberg1d.Heisenberg1d(40, 1)
#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)

s.run(nruns)
