# py-nqs
Python code for Neural Network Quantum States in the Carleo, Troyer style

FILES
small_main.py: constructs all the needed classes and calculates the energy on one of the included wfs

nqs.py: class which implements the neural network representation, has functions which can return the wf amplitudes

sampler.py: class which performs Monte Carlo sampling and energy calculation

heisenberg1d.py: class representation of the 1d Heisenberg anti-ferromagnet, with adjustable number of spins and Jz
 By inputting a state, can acquire a list of all states that state is connected to and the corresponding matrix elements


TO-DO LIST:
- Implement other Hamiltonians besides the heisenberg1d
- Implement file output (writestates)
- Create a unified main script to minimize the hardcoded arguments
- Find areas where the code is particularly slow and see if these can be sped up via vectorization
    -if not, can we push off to C++ code for e.g., sampling?
- Work with more arbitrary weights (i.e., initialize a random w.f.)
- Training
    -Obviously this is actually more than one thing