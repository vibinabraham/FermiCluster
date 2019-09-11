import numpy as np
import scipy 
import scipy.sparse
import copy as cp
from Hamiltonian import *
from davidson import *
from helpers import *

from ci_string import *
from hubbard_fn import *

def fill_H(h,v,e_nuc=0,e_core=0):
    H = Hamiltonian()
    H.e_nuc = e_nuc
    H.e_core = e_core
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = v
    return H

np.random.seed(2)

n_orb = 4
U = 1.
beta = 1.

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

H = fill_H(h,g)
basis = {}
print(" do CI for each particle number block")
for na in range(n_orb+1):
    for nb in range(n_orb+1):
        n_roots = 4 
        ci = ci_solver()
        ci.algorithm = "direct"
        ci.init(H,na,nb,n_roots)
        print(ci)
        ci.run()
        basis[(na,nb)] = ci.results_v

a_IJp = compute_tdm_a(n_orb, (1,2),(2,2),basis)
