import numpy as np
import trainer
import ising1d
import nqs
import sampler
import evolver

### INITIALIZATION

nruns = 1000
wf = nqs.NqsTI(40, 1)  # Set up a translation-invariant neural network
wf.load_parameters('./Outputs/Ising1d05100.npz')  # Load this pre-trained ANNQS

### TIME EVOLVE
h = ising1d.Ising1d(40, 1.0)
evo = evolver.Evolver(h)
wf = evo.evolve(wf, .01, 100, symmetry="ti")