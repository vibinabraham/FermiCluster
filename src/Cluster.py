import numpy as np
import scipy
import itertools as it


class Cluster(object):

    def __init__(self,bl):
        """
        input: bl is a list of spatial orbital indices

        data:
        self.block_states:      dict connecting (na,nb) quantum numbers to ndarray
                                self.block_states[(3,4)] = ndarray(determinant, cluster state)

        self.ops:               dict holding matrices of operators
                                e.g., self.ops['Aab'][(Ina,Inb),(Jna,Jnb)] = ndarray(vecI,vecJ,p,q,r)
        """
        self.orb_list = bl

        self.n_orb = len(bl)
        self.dim_tot = 2**(2*self.n_orb)
        self.block_states = {}
        self.tdm_a = {}
        self.ops    = {}


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
        self.ops[string][(Ina,Inb),(Jna,Jnb))] = tens
