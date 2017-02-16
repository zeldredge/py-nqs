import math
from nqs import *


class Sampler:
    def __init__(self, wf, hamilt):
        # Input args:
        # wf: an instance of the nqs class holding the boltzmann machine
        # hamilt: a Hamiltonian (i.e., class that can return connected states)
        # nflips: number of flips to attempt each time we try to make a move
        self.wf = wf
        self.nspins = self.wf.nv
        self.hamiltonian = hamilt
        self.energy = []
        self.writestates = False
        self.flips = []
        self.nmoves = 0
        self.accepted = 0
        self.state = np.ones(self.nspins)
        self.out = None  # eventually, file object
        self.nflips = 0

    def rand_spins(self, mag0=True):  # Random spin flips, current max is two
        # mag0 tells the program whether to keep total magnetization at zero or not
        # This function generates a set of flips (stored in class variables),
        # and then checks to see whether it's a good set, returning True or False
        # Bad sets either don't respect mag0 or are the same flip twice
        self.flips = np.random.randint(0, self.nspins, self.nflips)  # Get a nflips-length list of flips

        if self.nflips == 2:
            if not mag0:
                return self.flips[0] != self.flips[1]
            if mag0:
                return self.state[self.flips[0]] != self.state[self.flips[1]]

        return True

    def init_random_state(self, mag0=True):

        if not mag0:  # if we don't enforce magnetization = 0, it is easy
            self.state = np.random.choice([-1, 1], self.nspins)  # make a bunch of random -1, 1

        if mag0:  # if we do, need to be cleverer
            if self.nspins % 2 != 0:
                raise ValueError('Need even number of spins to have zero magnetization!')
            base_array = np.concatenate(
                (np.ones(int(self.nspins / 2)), -1 * np.ones(int(self.nspins / 2))))  # make an array of half 1, half -1
            self.state = np.random.permutation(base_array)  # return a random permutation of the half 1, half-1 array

    def reset_av(self):
        self.nmoves = 0
        self.accepted = 0

    def acceptance(self):
        return self.accepted / self.nmoves

    def move(self):
        if self.rand_spins():  # check if random spins is ok
            accept_prob = abs(self.wf.pop(self.state, self.flips)) ** 2  # acceptance probability of this flip

            if accept_prob > np.random.random():  # Metropolis-Hastings
                self.wf.update_lt(self.state, self.flips)  # Update the wavefunction look-up tables

                for flip in self.flips:  # Update the state in the sampler
                    self.state[flip] *= -1

                self.accepted += 1  # Update count of accepted moves

        self.nmoves += 1  # Update count of moves

    def set_file_states(self, filename):  # If we are outputting the Markov Chain to file as we go, open that up
        self.writestates = True
        self.out = open(filename, 'w')

    def write_state(self):
        self.out.write(self.state)

    def measure_energy(self):
        en = 0 + 0j

        mel, flips = self.hamiltonian.find_conn(self.state)

        for i in range(len(flips)):
            if i == None: pass
            en += self.wf.pop(self.state, flips[i]) * mel[i]

        self.energy.append(en)

    # Now, the function that runs the Monte Carlo sweeping
    # nsweeps = number of sweeps to do
    # thermfactor = fraction of sweeps to discard as initial thermalization
    # Sweepfactor = number of single flips per sweep
    # nflipss = number of flips to make per move, automatically 1 or 2 depending on Hamiltonian if the input is -1

    def run(self, nsweeps, thermfactor=.1, sweepfactor=1, nflipss=-1):
        self.nflips = nflipss
        if self.nflips == -1:  # Fix the number of flips
            self.nflips = self.hamiltonian.minflips

        # Some input checks, same as Carleo ran
        if self.nflips not in [1, 2]:
            raise ValueError('Nflips must be 1 or 2')

        if thermfactor > 1 or thermfactor < 0:
            raise ValueError('Invalid thermfactor')

        if nsweeps < 50:
            raise ValueError('Use more sweeps (>50)')

        print("Starting Monte Carlo sampling, nsweeps = {}".format(nsweeps))

        self.init_random_state()  # Get the random state
        self.wf.init_lt(self.state)  # Initialize the look-up tables

        self.reset_av()  # Reset variables

        print("Beginning thermalization...")  # Thermalize
        for thermsweep in range(math.floor(thermfactor * nsweeps)):
            for spinmove in range(sweepfactor * self.nspins):
                self.move()
        print("Thermalization done.")

        self.reset_av()

        print("Sweeping...")  # Normal sweeps
        for sweep in range(nsweeps):
            for spinmove in range(sweepfactor * self.nspins):
                self.move()
                if self.writestates:
                    self.write_state()
            self.measure_energy()
        print("Sweeping done. Acceptance rate was = {}".format(self.acceptance()))
        self.output_energy()

    def output_energy(self):
        nblocks = 50
        blocksize = math.floor(len(self.energy) / nblocks)
        enmean = 0
        enmeansq = 0
        enmean_unblocked = 0
        enmeansq_unblocked = 0

        for i in range(nblocks):
            eblock = 0
            for j in range(i * blocksize, (i + 1) * blocksize):
                eblock += self.energy[j].real
                assert (j < len(self.energy))
                delta = self.energy[j].real - enmean_unblocked
                enmean_unblocked += delta / (j + 1)
                delta2 = self.energy[j].real - enmean_unblocked
                enmeansq_unblocked += delta * delta2

            eblock /= blocksize
            delta = eblock - enmean
            enmean += delta / (i + 1)
            delta2 = eblock - enmean
            enmeansq += delta * delta2

        enmeansq /= (nblocks - 1)
        enmeansq_unblocked /= (nblocks * blocksize - 1)
        estav = enmean / self.nspins
        esterror = sqrt(enmeansq / nblocks) / self.nspins
        esterror = esterror.real

        print("Estimated average energy per spin: {0} +/- {1}".format(estav, esterror))
        print("Error estimated with binning analysis consisting of {0} bins".format(nblocks))
        print("Block size is {}".format(blocksize))
        print("Estimated autocorrelation time is {}".format(0.5 * blocksize * enmeansq / enmeansq_unblocked))
