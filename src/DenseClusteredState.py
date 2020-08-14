import numpy as np
import scipy
import itertools
import copy as cp
from collections import OrderedDict,abc
from helpers import *

class DenseClusteredState(OrderedDict):
    """
    class for organizing information of a state(s) on a clustered space

    e.g., C[(3,2),(2,3),(4,4)] = c[s,I,J,K] <- ndarray

                a b   a b   a b    
                                    number of electrons in each cluster
                I,J,K:  local state labels
                s:      is the global state index
    """
    def __init__(self):
        self.n_clusters = 0
        self.n_orb = 0

        self.data = OrderedDict()

    
    def init(self,clusters,fock_config):
        """
        Initialize to ground state of specified
        fock_config (# of a/b electrons in each cluster)
        """
        self.n_clusters = len(clusters)
        self.n_orb = 0
        states_per_cluster = []
        for ci,c in enumerate(clusters):
            self.n_orb += c.n_orb
            states_per_cluster.append(c.basis[fock_config[ci]].shape[1])

        self.data = OrderedDict()

        assert(len(fock_config)==self.n_clusters)

        self[fock_config] = np.zeros(states_per_cluster) 
        self[fock_config][tuple([0]*self.n_clusters)] = 1 


    def items(self):
        return self.data.items()
    def keys(self):
        return self.data.keys()
    def __getitem__(self,fock_config):
        return self.data[fock_config]
    def __setitem__(self,fock_config,c):
        self.data[fock_config]=c


    def __len__(self):
        l = 0
        for b,c in self.data.items():
            l += c.size
        return l
    def copy(self):
        """
        Create a deep copy of this state
        """
        return cp.deepcopy(self) 
    def get_vector(self):
        """
        return a ndarray vector for the state
        """
        v = [] 
        for fockspace,configs in self.items():
            v.append(configs.flatten())
        return np.array(v)
    def get_fspace_vector(self,fspace):
        """
        return a ndarray for the coefficients associated with fockspace=fspace 
        """
        v = np.zeros((len(self[fspace]),1))
        idx = 0
        for config,coeff in self[fspace].items():
            v[idx] = coeff
            idx += 1
        return v


    ##################### resume here..... 


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
    def clip(self,thresh, max=None):
        """
        delete values smaller than thresh, and return the list of kept indices
        and optionally larger than or equal to max
        """
        idx_to_keep = []
        idx = 0
        if max==None:
            for fockspace,configs in self.items():
                for config,coeff in list(configs.items()):
                    if abs(coeff) < thresh:
                        del self.data[fockspace][config]
                    else:
                        idx_to_keep.append(idx)
                    idx += 1
        else:
            assert( max > thresh)
            for fockspace,configs in self.items():
                for config,coeff in list(configs.items()):
                    if abs(coeff) < thresh or abs(coeff) >= max:
                        del self.data[fockspace][config]
                    else:
                        idx_to_keep.append(idx)
                    idx += 1

        self.prune_empty_fock_spaces()
        return idx_to_keep

    def add_single_excitonic_states(self,clusters):
        for fspace in self.data.keys():
            config = [0]*self.n_clusters
            for ci in clusters:
                fock_i = fspace[ci.idx]
                new_config = cp.deepcopy(config)
                for cii in range(ci.basis[fock_i].shape[1]):
                    new_config[ci.idx] = cii
                    self[fspace][tuple(new_config)] = 0 

    
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

    def compute_size_of_space(self,clusters):
        """
        Compute the size of hilbert space accessible with given cluster basis 
        """
        # {{{
        ns = []
        na = 0
        nb = 0
        size = 0
        for fblock,configs in self.items():
            for c in fblock:
                na += c[0]
                nb += c[1]
            break
    
        for c in clusters:
            nsi = []
            for fspace in c.basis:
                nsi.append(fspace)
            ns.append(nsi)
        for newfock in itertools.product(*ns):
            nacurr = 0
            nbcurr = 0
            for c in newfock:
                nacurr += c[0]
                nbcurr += c[1]
            if nacurr == na and nbcurr == nb:
                size += self.compute_max_fock_space_dim(newfock,clusters)
        return size 
# }}}

    def compute_max_fock_space_dim(self,fspace,clusters):
        """
        Compute the max number of configurations in a given fock space
        """
        # {{{
        dim = 1
        for c in clusters:
            # get number of vectors for current fock space
            dim *= c.basis[fspace[c.idx]].shape[1]
        return dim
# }}}
    
    def expand_each_fock_space(self,clusters):
        """
        expand basis to full space
        """
        # {{{
        print("\n Make each Fock-Block the full space")
        # create full space for each fock block defined
        for fblock,configs in self.items():
            dims = []
            for c in clusters:
                # get number of vectors for current fock space
                dims.append(range(c.basis[fblock[c.idx]].shape[1]))
            for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                self[fblock][newconfig] = 0 
        return
# }}}

    def expand_to_full_space(self,clusters):
        """
        expand basis to full space defined by cluster basis
        """
        # {{{
        # do something here
        #for c in clusters:
        print("\n Expand to full space")
        ns = []
        na = 0
        nb = 0
        for fblock,configs in self.items():
            for c in fblock:
                na += c[0]
                nb += c[1]
            break
    
        for c in clusters:
            nsi = []
            for fspace in c.basis:
                nsi.append(fspace)
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
            for c in clusters:
                #print(c, fblock)
                # get number of vectors for current fock space
                dims.append(range(c.basis[fblock[c.idx]].shape[1]))
            for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                self[fblock][newconfig] = 0 
        return
# }}}

    def expand_to_random_space(self,clusters,seed=2,thresh=.1):
        """
        expand basis to random space defined by cluster basis
        """
        # {{{
        # do something here
        #for c in clusters:
        print("\n Expand to random space")
        ns = []
        na = 0
        nb = 0
        np.random.seed(seed)
        for fblock,configs in self.items():
            for c in fblock:
                na += c[0]
                nb += c[1]
            break
    
        for c in clusters:
            nsi = []
            for fspace in c.basis:
                nsi.append(fspace)
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
            for c in clusters:
                #print(c, fblock)
                # get number of vectors for current fock space
                dims.append(range(c.basis[fblock[c.idx]].shape[1]))
                
            for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                if np.random.rand() < thresh:
                    self[fblock][newconfig] = 0 
        self.prune_empty_fock_spaces()
        print(" Size of random basis:", len(self))
        return
# }}}

    def randomize_vector(self,seed=None):
        """
        randomize coefficients in defined space 
        """
        # {{{
        np.random.seed(seed)
        for fockspace,config,coeff in self:
            self[fockspace][config] = np.random.normal() - .5
        self.normalize()
        return
    # }}}

    def scale(self,x):
        """
        multiply vector coefficients by scalar 
        """
        for fock,conf,coeff in self:
            self[fock][conf] *= x
        return 

    def add(self,other,scalar=1):
        """
        add clusteredstate vector coefficients to self
        multiplying other by optional scalar a
        self = a * other
        """
        for fockspace,configs in other.items():
            if fockspace in self.fblocks():
                for config,coeff in configs.items():
                    if config in self.data[fockspace]:
                        self.data[fockspace][config] += scalar * coeff
                    else:
                        self.data[fockspace][config] = scalar * coeff
            else:
                self.data[fockspace] = other[fockspace] 
        return 

    def dot(self,other):
        """
        Compute dot product of state with other

        loop is over self, so use the dot function belonging to the shortest vector
        """
        dot = 0
        for fockspace,configs in self.items():
            try:
                fock2 = other[fockspace]
                for config,coeff in configs.items():
                    try: 
                        dot += coeff * fock2[config]
                    except KeyError:
                        continue
            except KeyError:
                continue
        return dot
    
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
        #for fockspace,config,coeff in self:
        #    norm += coeff*coeff
        for fockspace,configs in self.items():
            for config,coeff in configs.items():
                norm += coeff*coeff
        return np.sqrt(norm)

    def normalize(self):
        """
        Normalize state
        """
        norm = self.norm()
        for fockspace,configs in self.items():
            for config,coeff in configs.items():
                self[fockspace][config] = coeff/norm
        return

    def print(self, thresh=1e-3):
        """ Pretty print """
        print(" --------------------------------------------------")
        print(" ---------- Fockspaces in state ------: Dim = %5i"%len(self))
        print(" --------------------------------------------------")
        print(" Printing contributions greater than: ", thresh)
        print()
        print(" %-20s%-20s%-20s"%("Weight", "# Configs", "Fock space")) 
        print(" %-20s%-20s%-20s"%("-------", "---------", "----------")) 
        for f in self.data:
            prob = 0
            for config, coeff in self.data[f].items():
                prob += coeff*coeff 
            if prob > thresh:
                print(" %-20.3f%-20i%-s"%(prob,len(self.data[f]), f))
            #print(" Dim %4i  Weight %8.1f  fock_space: "%(len(self.data[f]),prob*100), f)
            #for config, value in self.data[f].items():
            #    print("%20s"%str(config),end="")
            #    #print_row(value)
            #    print(" ",value)
        print(" --------------------------------------------------")
        
    def print_configs(self):
        """ Pretty print """
        for f in self.data:
            if len(self.data[f]) == 0:
                continue
            print(" Dim %4i fock_space: "%len(self.data[f]),end='')
            [print(" Cluster %-2i(%i:%i) "%(fii,fi[0],fi[1]),end='') for fii,fi in enumerate(f)] 
            print()
            for config, value in self.data[f].items():
                print("%20s"%str(config),end="")
                #print_row(value)
                print(" %12.8f"%value)
                #[print(" %12.8f"%value) for value in values]
    
    def analyze(self, print_thresh=1e-2):
        """
        Print out analysis of wavefunction
        """
        print()
        print(" -----------------------------------------------------")
        print(" Analysis of wavefunction")
        print(" -----------------------------------------------------")
        print(" Total Length: %5i" %len(self))
        self.clip(1e-12)
        print(" Total Length: %5i (clipped)" %len(self), 1e-12)
        l1 = 0
        #diag_l2 = 0
        for f in self.fblocks():
            print(" Fock Space Config: ",f)
            f_weight = 0
            f_l1 = 0
            for c in self[f]:
                f_weight += self[f][c]*self[f][c]
                f_l1 += abs(self[f][c])
                if abs(self[f][c]) > print_thresh:
                    print("%20s"%str(c),end="")
                    #print_row(value)
                    print(" %12.8f"%self[f][c])
                
                
                #diag = True
                #for ci in range(1,len(c)):
                #    if c[ci-1] != c[ci]:
                #        diag = False
                #        break
                #if diag:
                #    diag_l2 += self[f][c] * self[f][c]
                    
            l1 += f_l1
            print("     Population: %12.8f" %f_weight)
            print("     Dimension : %12i" %len(self[f]))
            print("     L1:         %12.8f" %f_l1)
        
        print(" ")
        print(" Length of vector:................... %-8i" % len(self))
        print(" Number of Fock space configurations: %-8i" % len(self.fblocks()))
        print(" Total L1:........................... %-8.2f" %l1)
        print(" -----------------------------------------------------")

            #for config, value in self.data[f].items():
            #    print("%20s"%str(config),end="")
            #    #print_row(value)
            #    print(" ",value)

