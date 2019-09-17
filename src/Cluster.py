import numpy as np
import scipy
import itertools as it

from ci_string import *
from Hamiltonian import *
from davidson import *

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
                                    keys = strings denoting operator A/B creation a/b annihilation
                                    values = dicts of keys=(Na_bra, Nb_bra, Na_ket, Nb_ket) 
                                                      with values being the tensor representation
                                    e.g., self.ops = {
                                            'Aab':{  [(0,0),(0,1)]:D(I,J,p,q,r) 
                                                     [(0,1),(0,2)]:D(I,J,p,q,r) 
                                                     ...
                                                  }
                                            'Aa' :{  [(0,0),(0,0)]:D(I,J,p,q,r) 
                                                     [(0,1),(0,1)]:D(I,J,p,q,r) 
                                                     ...
                                                  }
                                            'aa' :{  [(0,0),(3,0)]:D(I,J,p,q,r) 
                                                     [(1,1),(3,1)]:D(I,J,p,q,r) 
                                                     ...
                                                  }
                                            ...

        """
        self.idx = idx 
        self.orb_list = bl

        self.n_orb = len(bl)    
        self.dim_tot = 2**(2*self.n_orb)    # total size of hilbert space
        self.dim = self.dim_tot             # size of basis         
        self.basis = {}                     # organized as basis = { (na,nb):v(IJ,s), (na',nb'):v(IJ,s), ...} 
                                            #       where IJ is alpha,beta string indices
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
            self.ops[op] = {}
    
    def form_eigbasis_from_local_operator(self,local_op):
        h = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        for t in local_op.terms:
            if t.ops[0] == "Aa":
                h = t.ints 
            if t.ops[0] == "AAaa":
                v = t.ints 
        
        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h
        H.V = v
        self.basis = {}
        print(" Do CI for each particle number block")
        for na in range(self.n_orb+1):
            for nb in range(self.n_orb+1):
                n_roots = 100 
                ci = ci_solver()
                ci.algorithm = "direct"
                ci.init(H,na,nb,n_roots)
                print(ci)
                ci.run()
                #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                self.basis[(na,nb)] = ci.results_v
         
    def build_op_matrices(self):
        """
        build all operators needed
        """

        #  a, A 
        for na in range(1,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                self.ops['a'][(na-1,nb),(na,nb)] = build_annihilation(self.n_orb, (na-1,nb),(na,nb),self.basis)
                self.ops['A'][(na,nb),(na-1,nb)] = cp.deepcopy(np.swapaxes(self.ops['a'][(na-1,nb),(na,nb)],0,1))
                # note:
                #   I did a deepcopy instead of reference. This increases memory requirements and 
                #   basis transformation costs, but simplifies later manipulations. Later I need to 
                #   remove the redundant storage by manually handling the transpositions from a to A
        
        #  b, B 
        for na in range(0,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                self.ops['b'][(na,nb-1),(na,nb)] = build_annihilation(self.n_orb, (na,nb-1),(na,nb),self.basis)
                self.ops['B'][(na,nb),(na,nb-1)] = cp.deepcopy(np.swapaxes(self.ops['b'][(na,nb-1),(na,nb)],0,1))
                # note:
                #   I did a deepcopy instead of reference. This increases memory requirements and 
                #   basis transformation costs, but simplifies later manipulations. Later I need to 
                #   remove the redundant storage by manually handling the transpositions from a to A

        #  Aa 
        for na in range(1,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                A = self.ops['A'][(na,nb),(na-1,nb)]  # B[ I(N,N'),J(N,N-1'), p]
                a = self.ops['a'][(na-1,nb),(na,nb)]  # b[ K(N-1,N'),L(N,N'), q]
                self.ops['Aa'][(na,nb),(na,nb)] = np.einsum('ijp,jlq->ilpq',A,a)
        #  Bb 
        for na in range(0,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                B = self.ops['B'][(na,nb),(na,nb-1)]  # B[ I(N,N'),J(N,N'-1), p]
                b = self.ops['b'][(na,nb-1),(na,nb)]  # b[ K(N,N'-1),L(N,N'), q]
                self.ops['Bb'][(na,nb),(na,nb)] = np.einsum('ijp,jlq->ilpq',B,b)

        #Add remaining operators ....



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
