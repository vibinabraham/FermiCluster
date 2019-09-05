import numpy as np
import scipy 
import itertools as it


class Cluster(object):

    def __init__(self,bl, oei, tei):

        self.orb_list = bl
        self.so_orb_list = sum([[2*i, 2*i+1] for i in bl], [])


        self.oei  = oei
        self.tei  = tei
        self.n_orb = oei.shape[0]
        self.n_so_orb = 2 * self.n_orb
        self.dim_tot = 2**(2*self.n_orb)
        self.Ham = 0
        self.n_range = 0 
        self.n_orb_start = 0 
        self.n_orb_stop = 0 
        self.g_spin = 0
        self.h_spin = 0
        self.n_elec = 0


        self.tdm = {}
        self.doubles_tdm_p2 = {}
        self.doubles_tdm_m2 = {}

        self.h_eff = {}

        self.block_states = {}
        self.block_states_2 = {}

        
    def init(self,n_range):

        self.n_range = n_range
        #self.n_elec  = n_elec
        #self.n_start = n_range[0] 
        #self.n_stop = n_range[-1]   
        #self.n_so_start = 2*self.n_start
        #self.n_so_stop = 2*self.n_start+2*len(self.n_range)

        self.so_range = 2*len(n_range)

        #integrals
        self.h_spin = np.kron(self.oei,np.eye(2))


        g_spin = np.kron(self.tei,np.eye(2))
        g_spin = np.kron(g_spin.T,np.eye(2))

        self.g_spin = g_spin.swapaxes(1,2)
        
        self.t_spin = np.kron(self.oei,np.eye(2))

    def read_block_states(self,ci_eval,ci_vec,n_a,n_b,ind):
        self.block_states[n_a,n_b,ind] = ci_vec

    def read_block_states_2(self,ci_eval,ci_vec,n_a,n_b):
        self.block_states_2[n_a,n_b] = ci_vec

