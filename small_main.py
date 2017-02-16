import sampler
import nqs
import heisenberg1d
import heisenberg2d
import ising1d
import sys

nruns = 100

wf = nqs.Nqs("./Ground/Ising1d_40_1_1.wf")
#h = heisenberg2d.Heisenberg2d(100, 1, 10)
h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)

s.run(nruns)
