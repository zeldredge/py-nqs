import numpy as np
from cmath import *

class Nqs:
    def __init__(self, filename):
        #Initializing a bunch of variables. Not so necessary in python! but doing it anyway
        self.W = np.zeros((1,1)) #neural network weights (matrix of W_ij)
        self.a = np.zeros(1) #neural network visible bias (vector)
        self.b = np.zeros(1) #neural network hidden bias (vector)
        self.nh = 0 #number of hidden spins (to be determined from file)
        self.nv = 0 #number of visible spins (to be determiend from file)

        self.Lt = np.zeros(1) #look-up table for angles

        self.log2 = log(2) #Apparently easier to precompute this?

        #Ok, now get all the parameters
        self.LoadParameters(filename)

    def LogVal(self,state): #computes the logarithm of the wavefunction in a particular state
        #Just uses the formula in C1 with the logarithm used to take a sum
        rbm = 0 + 0j #Start with this as a zero complex number
        rbm = sum([self.a[v]*state[v] for v in range(self.nv)]) #add all visible biases
        #The two sums below: inner sum is over all v (each hidden unit accounts for all of its visible connections)
        #outer sum is over all h (each cosh in the product)
        rbm = rbm + sum([lncosh(sum([b[h] + self.W[v][h]*state[v] for v in range(self.nv)])) for h in range(self.nh)])
        return rbm

    #Next function is LogPoP, computes Log Psi'/Psi, wher Psi' is the state with certain flipped spins
    #Look-up table sare used for speed; the vector flips tells us which are flipped

    def LogPoP(self,state,flips):
        if len(flips) == 0: #No flips? We out
            return 0

        if np.all(flips == flips[0]): #If it's this one that means no flips
            return 0

        logpop = 0 + 0j #Initialize the variable

        #This is the change due to visible biases
        logpop = logpop - sum([self.a[flip]*2.0*state[flip] for flip in flips])
        
        #This is the change due to the interaction weights
        logpop = logpop + sum([ lncosh(self.Lt[h] - sum([2*self.W[flip][h]*state[flip] for flip in flips])) - lncosh(self.Lt[h]) for h in range(self.nh)])
        
        return logpop

    def PoP(self,state,flips): #This does the obvious
        return exp(self.LogPoP(state,flips))

    def InitLt(self,state): #Initialize lookup tables
        self.Lt = np.zeros(self.nh) #See eqn C7
        self.Lt = [self.b[h] + sum([self.W[v][h]*state[v] for v in range(self.nv)]) for h in range(self.nh)]
        return None

    def UpdateLt(self,state,flips): #Update lookup tables after flips
        if len(flips) == 0: #Again, if no flips, leave
            return None
        for h in range(self.nh):
            self.Lt[h] -= sum([2*state[flip]*self.W[flip][h] for flip in flips])
        return None

    def LoadParameters(self,filename):
        with open(filename,'r') as f:
            self.nv = int(f.readline())
            self.nh = int(f.readline())

            self.a = [CToPyComplex(f.readline()) for i in range(self.nv)] #had to write a function to parse the C++ complex output, which is (real, imag)
            self.b = [CToPyComplex(f.readline()) for i in range(self.nh)]
            self.W = [[CToPyComplex(f.readline()) for i in range(self.nh)] for j in range(self.nv)]

            #a = [complex(f.readline()) for i in range(self.nv)]
            #b = [complex(f.readline()) for i in range(self.nh)]
            #W = [complex(f.readline()) for i in range(self.nh*self.nv)]
            print("NQS loaded from file: {}".format(filename))
            print("N_visbile = {0}      N_hidden = {1}".format(self.nv,self.nh))

    def Nspins(self): #This function exists for some reason, and I don't want to break anything
        return self.nv

def CToPyComplex(instring):
    coordinates = instring.translate({ord(c) : None for c in '()\n'}) #strip out parentheses and newline
    coordinates = coordinates.split(",") #split the coordinates into two strings at the comma
    outnum = float(coordinates[0]) + 1j*float(coordinates[1])
    return outnum

def lncosh(x): #I don't really understand why they write a more complicated function than this -- I think this should work though
    if abs(x) <= 12:
        return log(np.cosh(x)) 
    else:
        return abs(x) - log(2) #The large x limit
