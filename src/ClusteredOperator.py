import numpy as np
import scipy
import itertools as it
import copy as cp
from collections import OrderedDict
from helpers import *

class LocalOperator:
    def __init__(self, cluster):
        self.cluster = cluster
        self.terms = []
    def add_term(self,term):
        self.terms.append(term)
    def build_matrix_dumb1(self):
        np.zeros([self.cluster.dim_tot, self.cluster.dim_tot])

class ClusteredTerm:
    def __init__(self, delta, ops, ints, clusters):
        """
        input:
            delta = list of change of Na,Nb,state
                    e.g., [(-1,-1),(1,1),(0,0)] means alpha and beta transition
                        from cluster 1 to 2, cluster 3 is fock diagonal
            ops   = list of operators
                    e.g., ["ab","AB",""]

            ints  = tensor containing the integrals for this block
                    e.g., ndarray([p,q,r,s]) where p,q are in 1 and r,s are in 2

        data:
            active: list of clusters which have non-identity operators
                    this includes fock-diagonal couplings,
                    e.g., ["Aa","","Bb"] would have active = [0,2]
        """
        assert(len(ops)==len(delta))
        self.clusters = clusters
        self.n_clusters = len(self.clusters)
        self.delta = delta
        self.ops = ops
        self.ints = ints
        self.sign = 1
        self.active = [] 
        assert(len(self.ints.shape) == 2 or len(self.ints.shape) == 4) 
        if len(self.ints.shape) == 2:
            self.ints_inds = 'ab'
        elif len(self.ints.shape) == 4:
            self.ints_inds = 'abcd'


    def __str__(self):
        #assert(len(self.delta)==2)
        out = ""
        for d in self.delta:
            out += " %2i"%d[0]
            out += " %2i"%d[1]
            out += ":"
        out += "|"
        if self.sign == 1:
            out += "+"
        elif self.sign == -1:
            out += "-"
        else:
            print("wtf?")
            exit()
        for o in self.ops:
            if o == "":
                out += " ____,"
            else:
                out += " %4s,"%o
        return out

    def get_active_clusters(self):
        """
        return list of clusters active in term
        """
        return self.active
    
    def matrix_element(self,fock_bra,bra,fock_ket,ket):
        """
        Compute the matrix element between <fock1,config1|H|fock2,config2>
        where fock is the 'fock-block' of bra. This is just a specification
        of the particle number space of each cluster. Eg., 
        ((2,3),(4,3),(2,2)) would have 3 clusters with 2(3), 4(3), 2(2) 
        alpha(beta) electrons, respectively. 

        Args:
            fock_bra (tuple(tuple)): fock-block for bra
            bra (tuple): cluster state configuration within specified fock block
            fock_ket (tuple(tuple)): fock-block for ket 
            ket (tuple): cluster state configuration within specified fock block
        Returns:
            matrix element. <IJK...|Hterm|LMN...>, where IJK, and LMN
            are the state indices for clusters 1, 2, and 3, respectively, in the 
            particle number blocks specified by fock_bra and fock_ket.
        """
        for ci in range(self.n_clusters):
            if (bra[ci]!=ket[ci]) and (ci not in self.active):
                return 0
        
        # <bra|term|ket>    = <IJK|o1o2o3|K'J'I'>
        #                   = <I|o1|I'><J|o2|J'><K|o3|K'> 
        #print(bra,ket,self)
        #print(self.ints.shape)
    
        mats = []
        state_sign = 1
        #print(self.ints.shape)
        for oi,o in enumerate(self.ops):
            #print(self.clusters[oi].ops[o][(fock_bra[oi],fock_ket[oi])][:,:,0])
            #print("dens:")
            #print(self.clusters[oi].ops[o][(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi],:])
            #print("ints:")
            #print(self.ints)
            if o == '':
                continue
            if len(o) == 1 or len(o) == 3:
                print(o)
                for cj in range(self.n_clusters,oi):
                    state_sign *= (-1)**(fock_ket[cj][0]+fock_ket[cj][1])
                    print(state_sign)
                #exit()
            #print(o) 
            #print(self.clusters[oi].ops[o].keys())
            try:
                d = self.clusters[oi].ops[o][(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
            except:
                return 0
            mats.append(d)
            #print(self.clusters[oi].ops[o][tuple([].extend(fock_bra[oi])).extend(fock_ket[oi]))].shape)

        #print("ints:")
        #print(self.ints)
        me = 0.0
        mats_inds = ""
        idx = 0
        for mi,m in enumerate(mats):
            for i in range(len(m.shape)):
                mats_inds += self.ints_inds[idx]
                idx += 1
            mats_inds += ","
        string = mats_inds + self.ints_inds + "->"
        print("state_sign:", state_sign)
        if len(mats) == 1:
            #print('huh: ', huh, self.sign*np.einsum(string,mats[0],self.ints))
            #return self.sign*np.einsum(string,mats[0],self.ints)
            me = self.sign*np.einsum(string,mats[0],self.ints) * state_sign
        elif len(mats) == 2:
            #print('mats: ', mats)
            #print('ints: ', self.ints)
            #print('huh: ', huh, self.sign*np.einsum(string,mats[0],mats[1],self.ints))
            me = self.sign*np.einsum(string,mats[0],mats[1],self.ints) * state_sign
            #return self.sign*np.einsum(string,mats[0],mats[1],self.ints)
        elif len(mats) == 0:
            return 0 
        else:
            print("NYI")
            exit()
        #print(" huh: %2i"%huh, "<",fock_bra, bra, "|H|", (fock_ket,ket), ">", "%12.8f"%me, self.active)
        #print(self)
        return me
class ClusteredOperator:
    """
    Defines a fermionic operator which can act on multiple clusters

    data:
    self.terms = dict of operators: transition list -> operator list -> integral tensor
                self.ints[[(delNa, delNb), (delNa, delNb), ...]["","","Aab","",B,...] = ndarray(p,q,r,s)

                delNa = bra(Na) - ket(Na)
                with p,q,r on cluster 2, and s on cluster 4
                ^this needs cleaned up
    """
    def __init__(self,clusters):
        self.n_clusters = len(clusters)
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
            delta_tmp.append([0,0])
            ops_tmp.append("")

        for ci in self.clusters:
            for cj in self.clusters:
                delta_a = list(cp.deepcopy(delta_tmp)) #alpha hopping
                delta_b = list(cp.deepcopy(delta_tmp)) #beta hopping
                ops_a = cp.deepcopy(ops_tmp) #alpha hopping
                ops_b = cp.deepcopy(ops_tmp) #beta hopping

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
                term_a = ClusteredTerm(delta_a, ops_a, hij, self.clusters)
                term_b = ClusteredTerm(delta_b, ops_b, hij, self.clusters)

                if ci.idx==cj.idx:
                    term_a.active = [ci.idx]
                    term_b.active = [ci.idx]
                else:
                    term_a.active = [ci.idx,cj.idx]
                    term_b.active = [ci.idx,cj.idx]
                if cj.idx < ci.idx:
                    #term_a.sign = -1
                    #term_b.sign = -1
                    term_a.ints = 1.0*np.transpose(term_a.ints, axes=(1,0))
                    term_b.ints = 1.0*np.transpose(term_b.ints, axes=(1,0))
                    #term_a.ints = np.transpose(term_a.ints, axes=(0,1))
                    #term_b.ints = np.transpose(term_b.ints, axes=(0,1))


                try:
                    self.terms[delta_a].append(term_a)
                except:
                    self.terms[delta_a] = [term_a]
                try:
                    self.terms[delta_b].append(term_b)
                except:
                    self.terms[delta_b] = [term_b]

        print(self.print_terms_header())
        for ti,t in self.terms.items():
            for tt in t:
                print(tt)
                #print_mat(tt.ints)

    def print_terms_header(self):
        """
        print header with labels for printing term
        """
        out = ""
        for di,d in enumerate(self.clusters):
            out += "%2i_____"%di
        out += "\n"
        for di,d in enumerate(self.clusters):
            out += " Δa"
            out += " Δb"
            out += ":"
        out += "|"
        out += " "
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

    def extract_local_operator(self,cluster_idx):
        op = LocalOperator(self.clusters[cluster_idx])
        for t in self.terms:
            for tt in self.terms[t]:
                active = tt.get_active_clusters()
                if len(active) == 1 and active[0] == cluster_idx:

                    term = cp.deepcopy(tt)
                    term.ops = [term.ops[cluster_idx]]
                    term.delta = [term.delta[cluster_idx]]
                    op.add_term(term)
        return op



if __name__== "__main__":
    term = ClusteredTerm()
