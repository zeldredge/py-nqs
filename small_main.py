import sampler
import nqs
import heisenberg1d
import sys

nruns = 100

wf = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.wf")
h = heisenberg1d.Heisenberg1d(40, 1)
s = sampler.Sampler(wf, h)

s.run(nruns)
