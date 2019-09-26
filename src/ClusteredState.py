import numpy as np
import scipy
import itertools as it
import copy as cp
from collections import OrderedDict
from helpers import *

class ClusteredState(OrderedDict):
    """
    class for organizing information of a state(s) on a clustered space

    e.g., C[(3,2),(2,3),(4,4)][(4,78,56)] = c[1:10] <- ndarray

             a b   a b   a b    I,J,K
             ^number of electrons in each cluster
                                ^local state labels
                                             ^global state labels
    """
    def __init__(self,clusters):
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.n_orb = 0
        for ci,c in enumerate(self.clusters):
            self.n_orb += c.n_orb

        self.data = OrderedDict()

        #self._len = 0
    
    def init(self,fock_config):
        """
        Initialize to ground state of specified
        fock_config (# of a/b electrons in each cluster)
        """
        assert(len(fock_config)==self.n_clusters)
        self[fock_config] = OrderedDict()
        
        self[fock_config][tuple([0]*self.n_clusters)] = np.ones((1,))
        #self.data[fock_config][tuple([0]*self.n_clusters)] = np.ndarray([1.],dtype='float')
   
    def items(self):
        return self.data.items()
    def keys(self):
        return self.data.keys()
    def __getitem__(self,fock_config):
        return self.data[fock_config]
    def __setitem__(self,fock_config,c):
        self.data[fock_config]=c
        #self._len += len(c)
    def __len__(self):
        l = 0
        for b,c in self.data.items():
            l += len(c)
        return l 

    def fblock(self,b):
        return self.data[b]
    def fblocks(self):
        return self.data.keys()
    def add_fockblock(self,block):
        if block not in self.data:
            self.data[block] = OrderedDict()

    def expand_to_full_space(self):
        """
        expand basis to full space
        """
        
        na = 0
        nb = 0
        # do something here
        #for c in self.clusters:


    def print(self):
        """ Pretty print """
        print(" Total Length %5i:" %len(self))
        for f in self.data:
            print(" Dim %4i fock_space: "%len(self.data[f]), f)
            #for config, value in self.data[f].items():
            #    print("%20s"%str(config),end="")
            #    #print_row(value)
            #    print(" ",value)
    
    def print_configs(self):
        """ Pretty print """
        for f in self.data:
            if len(self.data[f]) == 0:
                continue
            print(" Dim %4i fock_space: "%len(self.data[f]), f)
            for config, value in self.data[f].items():
                print("%20s"%str(config),end="")
                #print_row(value)
                print(" %12.8f"%value)
