import sampler
import nqs
import heisenberg1d
import sys

nruns = int(sys.argv[1])

wf = nqs.Nqs("../Ground/Heisenberg1d_40_1_1.wf")
h = heisenberg1d.Heisenberg1d(40, 1)
s = sampler.Sampler(wf, h)

s.Run(nruns)
