import trainer
import sampler


class Evolver:
    def __init__(self, h):
        self.h = h
        self.deltat = .01

    def evolve(self, wf, deltat, ntsteps, batch_size=100, symmetry='None'):
        # Function to evolve the wavefunction forward in time
        # Since we can view time evolution as being like the original training but with imaginary steps,
        # we take advantage of the existing training code
        self.deltat = deltat
        t = trainer.Trainer(self.h, reg_list=(0, 0, 0))  # Default, will be reassigned if we get a symmetry arg

        if symmetry == "local":
            t = trainer.TrainerLocal(self.h, reg_list=(0, 0, 0))  # note no regulator for time evolution

        if symmetry == "ti":
            t = trainer.TrainerTI(self.h, reg_list=(0, 0, 0))

        if symmetry == "general":
            t = trainer.TrainerSymmetric(self.h, reg_list=(0, 0, 0))

        s = sampler.Sampler(wf, self.h)  # To determine the starting state, we initialize a sampler
        s.run(5000)
        init_state = s.state

        wf, elist = t.train(wf, init_state, batch_size, ntsteps, self.gamma, print_freq=1,
                            file='./Outputs/time_evol', out_freq=0)

        return wf

    def gamma(self,p):
        return 1j*self.deltat