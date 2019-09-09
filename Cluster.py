import numpy as np
import scipy 
import itertools as it


class Cluster(object):

    def __init__(self,bl):

        self.orb_list = bl

        self.n_orb = len(bl)
        self.dim_tot = 2**(2*self.n_orb)
        self.block_states = {}
        self.tdm_a = {}

        
    def init(self,n_range):
        #have to add more stuff
        self.n_range = n_range

    def read_block_states(self,ci_eval,ci_vec,n_a,n_b):
        self.block_states[n_a,n_b] = ci_vec

    def read_tdms(self,mat,string,n_a,n_b):
        self.tdm_a[string,n_a,n_b] = mat

