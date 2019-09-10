import numpy as np
import scipy
import itertools as it

from ci_python.ci_string import *
from ci_python.Hamiltonian import *
from ci_python.davidson import *

class Cluster(object):

    def __init__(self,idx,bl):
        """
        input: 
            bl  = a list of spatial orbital indices
            idx = index of the cluster

        data:
            self.block_states:      dict connecting (na,nb) quantum numbers to ndarray
                                    self.block_states[(3,4)] = ndarray(determinant, cluster state)

            self.ops:               dict holding matrices of operators
                                    e.g., self.ops['Aab'][(Ina,Inb),(Jna,Jnb)] = ndarray(vecI,vecJ,p,q,r)
        """
        self.idx = idx 
        self.orb_list = bl

        self.n_orb = len(bl)    
        self.dim_tot = 2**(2*self.n_orb)    # total size of hilbert space
        self.dim = self.dim_tot             # size of basis         
        self.block_states = {}
        self.tdm_a = {}
        self.ops    = {}

    def __len__(self):
        return len(self.orb_list) 
    def __str__(self):
        has = ""
        for si in self.orb_list:
            has = has + "%03i|"%si
        if len(self.orb_list) == 0:
            has = "|"
        return "IDX%03i:DIM%04i:%s" %(self.idx, self.dim, has)

    def add_operator(self,op):
        if op in self.ops:
            return
        else:
            self.ops[op] = []
    
    def build_op_matrices(self):
        ci = ci_solver()
        
        H = Hamiltonian()
        ci.algorithm = "direct"
        na = 1
        nb = 1
        n_roots = 10
        ci.init(H,na,nb,n_roots)
        print(ci)
        ci.run()
 
        ob.tdms["ca_aa"] = tools.compute_tdm_ca_aa(ci,ci)
        ob.tdms["ca_bb"] = tools.compute_tdm_ca_bb(ci,ci)
        for op in self.ops:
            print(op)

        # start out by building a single annihilation operator

        
###################################################################################################################

    def read_block_states(self, vecs, n_a, n_b):
        self.block_states[n_a,n_b] = vecs
        #todo assert number of rows == dimension suggested by (n_a,n_b)
        assert(vecs.shape[0]>=vecs.shape[1])
        assert(n_a<=self.n_orb and n_a >= 0)
        assert(n_b<=self.n_orb and n_b >= 0)


    def read_tdms(self,mat,string,n_a,n_b):
        self.tdm_a[string,n_a,n_b] = mat


    def read_ops(self,tens,string,Ina,Inb,Jna,Jnb):
        """
        input:
            tens = ndarray(vecI,vecJ,p,q,r,...)
            string = operator class:
                    e.g., "Aab" for creation(alpha), annihilation(alpha), annihilation(beta)
            Ina,Inb = quantum numbers for ket
            Jna,Jnb = quantum numbers for bra
        """
        self.ops[string][(Ina,Inb),(Jna,Jnb)] = tens
