import heisenberg1d

import nqs
import sampler

nruns = 1000

wf = nqs.Nqs("./Ground/Heisenberg1d_40_1_1.wf")
h = heisenberg1d.Heisenberg1d(40, 1)
#h = ising1d.Ising1d(40,1)
s = sampler.Sampler(wf, h)

s.run(nruns)
