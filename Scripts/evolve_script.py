import numpy as np
import trainer
import ising1d
import nqs
import sampler
import evolver
import observables

nsteps = 400

### INITIALIZATION
#wf = nqs.NqsLocal(40, 2, 1)  # Set up a translation-invariant neural network
#wf.load_parameters('./Outputs/Ising05_k=2_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
#h = ising1d.Ising1d(40, 1.0)
#evo = evolver.Evolver(h)
#wf = evo.evolve(wf, .01, nsteps + 1, symmetry="local", file='./Outputs/evolution_k=2_', print_freq=100, out_freq=1, batch_size=1000)

### INITIALIZATION
#wf = nqs.NqsLocal(40, 1, 1)  # Set up a translation-invariant neural network
#wf.load_parameters('./Outputs/Ising05_k=1_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
#h = ising1d.Ising1d(40, 1.0)
#evo = evolver.Evolver(h)
#wf = evo.evolve(wf, .01, nsteps + 1, symmetry="local", file='./Outputs/evolution_k=1_', print_freq=100, out_freq=1, batch_size=1000)

### INITIALIZATION
wf = nqs.NqsTI(40, 1)  # Set up a translation-invariant neural network
wf.load_parameters('./Outputs/Ising05_ti_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
h = ising1d.Ising1d(40, 1.0)
evo = evolver.Evolver(h)
wf = evo.evolve(wf, .01, nsteps + 1, symmetry="ti", file='./Outputs/evolution_ti_', print_freq=25, out_freq=1, batch_size=1000)
