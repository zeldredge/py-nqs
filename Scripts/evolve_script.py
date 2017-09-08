import numpy as np
import trainer
import ising1d
import nqs
import sampler
import evolver
import observables

nsteps = 400

### INITIALIZATION
wf = nqs.NqsLocal(10, 2, 1)  # Set up a translation-invariant neural network
wf.load_parameters('../Outputs/10_Ising05_2loc_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
h = ising1d.Ising1d(10, 1.0)
evo = evolver.Evolver(h)
wf = evo.evolve(wf, .01, nsteps + 1, symmetry="local", file='../Outputs/10SpinEvolve/evolution_2loc_', print_freq=25, out_freq=1, batch_size=1000)

### INITIALIZATION
wf = nqs.NqsLocal(10, 1, 1)  # Set up a translation-invariant neural network
wf.load_parameters('../Outputs/10_Ising05_1loc_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
h = ising1d.Ising1d(10, 1.0)
evo = evolver.Evolver(h)
wf = evo.evolve(wf, .01, nsteps + 1, symmetry="local", file='../Outputs/10SpinEvolve/evolution_1loc_', print_freq=25, out_freq=1, batch_size=1000)

### INITIALIZATION
wf = nqs.NqsTI(10, 1)  # Set up a translation-invariant neural network
wf.load_parameters('../Outputs/10_Ising05_ti_200.npz')  # Load this pre-trained ANNQS

## TIME EVOLVE
h = ising1d.Ising1d(10, 1.0)
evo = evolver.Evolver(h)
wf = evo.evolve(wf, .01, nsteps + 1, symmetry="ti", file='../Outputs/10SpinEvolve/evolution_ti_', print_freq=25, out_freq=1, batch_size=1000)
