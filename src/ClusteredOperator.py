import numpy as np
import scipy
import itertools as it
import copy as cp
from collections import OrderedDict
from helpers import *


class ClusteredTerm:
    def __init__(self, delta, ops, ints):
        """
        input:
            delta = list of change of Na,Nb,state
                    e.g., [(-1,-1,1),(1,1,1),(0,0,0)] means alpha and beta transition 
                        from cluster 1 to 2, cluster 3 is diagonal
            ops   = list of operators
                    e.g., ["ab","AB",""]
            
            ints  = tensor containing the integrals for this block
                    e.g., ndarray([p,q,r,s]) where p,q are in 1 and r,s are in 2
        """
        assert(len(ops)==len(delta))
        self.delta = delta
        self.ops = ops
        self.ints = ints

    def __str__(self):
        out = ""
        for d in self.delta:
            out += " %2i"%d[0]
            out += " %2i"%d[1]
            out += " %2i"%d[2]
            out += ":"
        out += "|"
        for o in self.ops:
            if o == "":
                out += " ____,"
            else:
                out += " %4s,"%o
        return out


class ClusteredOperator:
    """
    Defines a fermionic operator which can act on multiple clusters

    data:
    self.terms = dict of operators: transition list -> operator list -> integral tensor
                self.ints[[(delNa, delNb, I==J), (delNa, delNb, I==J), ...]["","","Aab","",B,...] = ndarray(p,q,r,s)
                with p,q,r on cluster 2, and s on cluster 4
                ^this needs cleaned up
    """
    def __init__(self,clusters):
        self.n_clusters = 0
        self.clusters = clusters 
        self.terms = OrderedDict() 
        self.n_orb = 0
        for ci,c in enumerate(self.clusters):
            self.n_orb += c.n_orb

    def add_1b_terms(self,h):
        assert(len(h.shape)==2)
        assert(h.shape[0]==self.n_orb)
        assert(h.shape[1]==self.n_orb)

        delta_tmp = []
        ops_tmp = []
        for ci in self.clusters:
            delta_tmp.append([0,0,0])
            ops_tmp.append("")
        
        for ci in self.clusters: 
            for cj in self.clusters: 
                delta_a = list(cp.deepcopy(delta_tmp)) #alpha hopping
                delta_b = list(cp.deepcopy(delta_tmp)) #beta hopping
                ops_a = cp.deepcopy(ops_tmp) #alpha hopping
                ops_b = cp.deepcopy(ops_tmp) #beta hopping
                delta_a[ci.idx][2] = 1  #not diagonal
                delta_a[cj.idx][2] = 1  #not diagonal
                delta_b[ci.idx][2] = 1  #not diagonal
                delta_b[cj.idx][2] = 1  #not diagonal
                
                delta_a[ci.idx][0] += 1  #not diagonal
                delta_a[cj.idx][0] -= 1  #not diagonal
                delta_b[ci.idx][1] += 1  #not diagonal
                delta_b[cj.idx][1] -= 1  #not diagonal
                ops_a[ci.idx] += "A"
                ops_a[cj.idx] += "a"
                ops_b[ci.idx] += "B"
                ops_b[cj.idx] += "b"
                hij = h[ci.orb_list,:][:,cj.orb_list]
             
                delta_a = tuple([tuple(i) for i in delta_a])
                delta_b = tuple([tuple(i) for i in delta_b])
                try:
                    self.terms[delta_a].append(ClusteredTerm(delta_a, ops_a, hij))
                except:
                    self.terms[delta_a] = [ClusteredTerm(delta_a, ops_a, hij)]
                try:
                    self.terms[delta_b].append(ClusteredTerm(tuple(delta_b), ops_b, hij))
                except:
                    self.terms[delta_b] = [ClusteredTerm(tuple(delta_b), ops_b, hij)]
       
        print(self.print_terms_header())
        for ti,t in self.terms.items():
            for tt in t:
                print(tt)
                print_mat(tt.ints)
    
    def print_terms_header(self):
        """
        print header with labels for printing term
        """
        out = ""
        for di,d in enumerate(self.clusters):
            out += "%2i________"%di
        out += "\n"
        for di,d in enumerate(self.clusters):
            out += " Δa"
            out += " Δb"
            out += ":"
            out += "   "
        out += "|"
        for oi,o in enumerate(self.clusters):
            out += " %4i,"%oi
        return out


    def add_ops_to_clusters(self):
        """
        go through list of terms and make sure each cluster is prepared to build the necessary operators
        """
        for t in self.terms:
            for tt in self.terms[t]:
                for opi,op in enumerate(tt.ops):
                    if op != "":
                        self.clusters[opi].add_operator(op)

if __name__== "__main__":
    term = ClusteredTerm()
