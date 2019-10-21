import numpy as np
import scipy
import itertools
import copy as cp
from collections import OrderedDict,abc
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
       
        self[fock_config][tuple([0]*self.n_clusters)] = 1 
        #self[fock_config][tuple([0]*self.n_clusters)] = [0] 
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


    def __iter__(self):
        self._curr_fock_idx = 0
        self._curr_conf_idx = 0
        self._curr_idx = -1 
        self._max_iter = len(self)
        self._fock_iter = iter(self.data)
        self._curr_fock = next(self._fock_iter) 
        self._conf_iter = iter(self.data[self._curr_fock])
        return self

#    def __next__(self):
#        if self._curr_idx < self._max_iter-1:
#            self._curr_idx += 1
#            #next(self._conf_iter)
#            try:
#                curr_conf = next(self._conf_iter)
#                return self._curr_fock, curr_conf 
#            except StopIteration:
#                self._curr_fock = next(self._curr_fock)
#                self._conf_iter = iter(self._fock_iter)
#                return self._curr_fock, next(self._conf_iter)
#        else:
#            raise StopIteration

    def __next__(self):
       
        try:
            curr_conf = next(self._conf_iter)
            return self._curr_fock, curr_conf, self.data[self._curr_fock][curr_conf]
        except StopIteration:
            try:
                self._curr_fock = next(self._fock_iter)
                self._conf_iter = iter(self.data[self._curr_fock])
                # Check to make sure that there are no empty fock_spaces
                # as that would require a recursive check
                # If this assert fails, either make this function recursive
                # or simply run "self.prune_empty_fock_spaces()"
                assert(len(self.data[self._curr_fock])>0)
                curr_conf = next(self._conf_iter)
                return self._curr_fock, curr_conf, self.data[self._curr_fock][curr_conf]
            except StopIteration:
                raise StopIteration

    def __len__(self):
        l = 0
        for b,c in self.data.items():
            l += len(c)
        return l
    def copy(self):
        """
        Create a deep copy of this state
        """
        new = ClusteredState(self.clusters)
        new.data = cp.deepcopy(self.data)
        return new
    def get_vector(self):
        """
        return a ndarray vector for the state
        """
        v = np.zeros((len(self),1))
        idx = 0
        for fockspace,configs in self.items():
            for config,coeff in configs.items():
                v[idx] = coeff
                idx += 1
        return v
    def set_vector(self,v):
        """
        input a ndarray vector for the state and update coeffs
        """
        idx = 0
        for fock,configs in self.data.items():
            for config,coeffs in configs.items():
                self.data[fock][config] = v[idx]
                idx += 1
        return 
    def clip(self,thresh):
        """
        delete values smaller than thresh
        """
        idx_to_keep = []
        idx = 0
        for fockspace,configs in self.items():
            for config,coeff in list(configs.items()):
                if abs(coeff) < thresh:
                    del self.data[fockspace][config]
                else:
                    idx_to_keep.append(idx)
                idx += 1

        self.prune_empty_fock_spaces()
        return idx_to_keep
    
    def prune_empty_fock_spaces(self):
        """
        remove fock_spaces that don't have any configurations 
        """
        for fockspace,configs in list(self.items()):
            if len(configs) == 0:
                del self.data[fockspace]
        return 

    def zero(self):
        for fock,configs in self.data.items():
            for config,coeffs in configs.items():
                self.data[fock][config] = 0
    def fblock(self,b):
        return self.data[b]
    def fblocks(self):
        return self.data.keys()
    def add_fockspace(self,block):
        """
        Add a fock space to the current state basis
        """
        if block not in self.data:
            self.data[block] = OrderedDict()

    def expand_each_fock_space(self):
        """
        expand basis to full space
        """
        # {{{
        print("\n Make each Fock-Block the full space")
        # create full space for each fock block defined
        for fblock,configs in self.items():
            dims = []
            for c in self.clusters:
                # get number of vectors for current fock space
                dims.append(range(c.basis[fblock[c.idx]].shape[1]))
            for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                self[fblock][newconfig] = 0 
        return
# }}}

    def expand_to_full_space(self):
        """
        expand basis to full space
        """
        # {{{
        # do something here
        #for c in self.clusters:
        print("\n Expand to full space")
        ns = []
        na = 0
        nb = 0
        for fblock,configs in self.items():
            for c in fblock:
                na += c[0]
                nb += c[1]
            break
    
        for c in self.clusters:
            nsi = []
            for nai in range(c.n_orb+1):
                for nbi in range(c.n_orb+1):
                    nsi.append((nai,nbi))
            ns.append(nsi)
        for newfock in itertools.product(*ns):
            nacurr = 0
            nbcurr = 0
            for c in newfock:
                nacurr += c[0]
                nbcurr += c[1]
            if nacurr == na and nbcurr == nb:
                self.add_fockspace(newfock) 
    
    
        print("\n Make each Fock-Block the full space")
        # create full space for each fock block defined
        for fblock,configs in self.items():
            dims = []
            for c in self.clusters:
                # get number of vectors for current fock space
                dims.append(range(c.basis[fblock[c.idx]].shape[1]))
            for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                self[fblock][newconfig] = 0 
        return
# }}}

    def add(self,other):
        """
        add clusteredstate vector coefficients to self
        """
        for fockspace,configs in other.items():
            if fockspace not in self.fblocks():
                self.add_fockspace(fockspace)
            for config,coeff in configs.items():
                if config in self.data[fockspace]:
                    self.data[fockspace][config] += coeff
                else:
                    self.data[fockspace][config] = coeff
        return 
    
    def add_basis(self,other):
        """
        add the configuration space to the self
        """
        for fockspace,configs in other.items():
            if fockspace not in self.fblocks():
                self.add_fockspace(fockspace)
            for config,coeff in configs.items():
                if config not in self.data[fockspace]:
                    self.data[fockspace][config] = 0
        return 

    def norm(self):
        """
        Compute norm of state
        """
        norm = 0
        for fockspace,config,coeff in self:
            norm += coeff*coeff
        return norm

    def normalize(self):
        """
        Normalize state
        """
        norm = self.norm()
        for fockspace,config,coeff in self:
            self[fockspace][config] = coeff/np.sqrt(norm)
        return

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
            print(" Dim %4i fock_space: "%len(f),end='')
            [print(" Cluster %-2i(%ia:%ib) "%(fii,fi[0],fi[1]),end='') for fii,fi in enumerate(f)] 
            print()
            for config, value in self.data[f].items():
                print("%20s"%str(config),end="")
                #print_row(value)
                print(" %12.8f"%value)
                #[print(" %12.8f"%value) for value in values]
