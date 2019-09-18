import numpy as np
import scipy
import itertools as it
import copy as cp
from collections import OrderedDict
from helpers import *

import countswaps

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

        self.contract_string = ""
        
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
        #out += "    "+self.contract_string
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
                for cj in range(oi):
                    state_sign *= (-1)**(fock_ket[cj][0]+fock_ket[cj][1])
                    #print(state_sign)
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
        if len(mats) == 1:
            me = self.sign*np.einsum(self.contract_string,mats[0],self.ints) * state_sign
        elif len(mats) == 2:
            me = self.sign*np.einsum(self.contract_string,mats[0],mats[1],self.ints) * state_sign
        elif len(mats) == 3:
            me = self.sign*np.einsum(self.contract_string,mats[0],mats[1],mats[2],self.ints) * state_sign
        elif len(mats) == 4:
            me = self.sign*np.einsum(self.contract_string,mats[0],mats[1],mats[2],mats[3],self.ints) * state_sign
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
        """
        Add terms of the form h_{pq}\hat{a}^\dagger_p\hat{a}_q

        input:
        h is a square matrix NxN, where N is the number of spatial orbitals
        """
# {{{
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
                    term_a.sign = -1
                    term_b.sign = -1
                    term_a.ints = 1.0*np.transpose(term_a.ints, axes=(1,0))
                    term_b.ints = 1.0*np.transpose(term_b.ints, axes=(1,0))
                    #term_a.ints = np.transpose(term_a.ints, axes=(0,1))
                    #term_b.ints = np.transpose(term_b.ints, axes=(0,1))

                if cj.idx == ci.idx:
                    term_a.contract_string = "pq,pq->"
                    term_b.contract_string = "pq,pq->"
                else:
                    term_a.contract_string = "p,q,pq->"
                    term_b.contract_string = "p,q,pq->"

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
# }}}

    def add_2b_terms(self,v):
        """
        Add terms of the form v_{pqrs}\hat{a}^\dagger_p\hat{a}^\dagger_q\hat{a}_s\hat{a}_r

        input:
        v is a square matrix NxNxNxN, where N is the number of spatial orbitals
        """
# {{{
        assert(len(v.shape)==4)
        assert(v.shape[0]==self.n_orb)
        assert(v.shape[1]==self.n_orb)
        assert(v.shape[2]==self.n_orb)
        assert(v.shape[3]==self.n_orb)

        delta_tmp = []
        ops_tmp = []
        for ci in self.clusters:
            delta_tmp.append([0,0])
            ops_tmp.append("")

        for ci in self.clusters:
            for cj in self.clusters:
                for ck in self.clusters:
                    for cl in self.clusters:
                        delta_aa = list(cp.deepcopy(delta_tmp)) 
                        delta_bb = list(cp.deepcopy(delta_tmp)) 
                        delta_ab = list(cp.deepcopy(delta_tmp)) 
                        delta_ba = list(cp.deepcopy(delta_tmp)) 
                        ops_aa = cp.deepcopy(ops_tmp) #alpha hopping
                        ops_ab = cp.deepcopy(ops_tmp) #beta hopping
                        ops_ba = cp.deepcopy(ops_tmp) #alpha hopping
                        ops_bb = cp.deepcopy(ops_tmp) #beta hopping
                        
                        delta_aa[ci.idx][0] += 1  #not diagonal
                        delta_aa[cj.idx][0] += 1  #not diagonal
                        delta_aa[ck.idx][0] -= 1  #not diagonal
                        delta_aa[cl.idx][0] -= 1  #not diagonal

                        delta_ab[ci.idx][0] += 1  #not diagonal
                        delta_ab[cj.idx][1] += 1  #not diagonal
                        delta_ab[ck.idx][1] -= 1  #not diagonal
                        delta_ab[cl.idx][0] -= 1  #not diagonal

                        delta_ba[ci.idx][1] += 1  #not diagonal
                        delta_ba[cj.idx][0] += 1  #not diagonal
                        delta_ba[ck.idx][0] -= 1  #not diagonal
                        delta_ba[cl.idx][1] -= 1  #not diagonal

                        delta_bb[ci.idx][1] += 1  #not diagonal
                        delta_bb[cj.idx][1] += 1  #not diagonal
                        delta_bb[ck.idx][1] -= 1  #not diagonal
                        delta_bb[cl.idx][1] -= 1  #not diagonal

                        ops_aa[ci.idx] += "A"
                        ops_aa[cj.idx] += "A"
                        ops_aa[ck.idx] += "a"
                        ops_aa[cl.idx] += "a"
                        
                        ops_ab[ci.idx] += "A"
                        ops_ab[cj.idx] += "B"
                        ops_ab[ck.idx] += "b"
                        ops_ab[cl.idx] += "a"
                        
                        ops_ba[ci.idx] += "B"
                        ops_ba[cj.idx] += "A"
                        ops_ba[ck.idx] += "a"
                        ops_ba[cl.idx] += "b"
                        
                        ops_bb[ci.idx] += "B"
                        ops_bb[cj.idx] += "B"
                        ops_bb[ck.idx] += "b"
                        ops_bb[cl.idx] += "b"
                        
                        clusters_idx = [ci.idx,cj.idx,ck.idx,cl.idx]
                        indices = ['p','q','r','s']
                        sorted_idx = np.argsort(clusters_idx)
                        sorted_clusters_idx = [clusters_idx[s] for s in sorted_idx]
                        #print(sorted_clusters_idx) 

                        vijkl = v[ci.orb_list,:,:,:][:,cj.orb_list,:,:][:,:,ck.orb_list,:][:,:,:,cl.orb_list]
                        vijkl = 1*np.transpose(vijkl,axes=sorted_idx)
                        
                        contract_string = indices[0]
                        for si in range(1,4):
                            if sorted_clusters_idx[si] == sorted_clusters_idx[si-1]:
                                contract_string += indices[si]
                            else:
                                contract_string += ","+indices[si]
                        contract_string += ",pqrs->"
                        #print("contract_string",contract_string)


                        delta_aa = tuple([tuple(i) for i in delta_aa])
                        delta_ab = tuple([tuple(i) for i in delta_ab])
                        delta_ba = tuple([tuple(i) for i in delta_ba])
                        delta_bb = tuple([tuple(i) for i in delta_bb])
                        
                        term_aa = ClusteredTerm(delta_aa, ops_aa, vijkl, self.clusters)
                        term_ab = ClusteredTerm(delta_ab, ops_ab, vijkl, self.clusters)
                        term_ba = ClusteredTerm(delta_ba, ops_ba, vijkl, self.clusters)
                        term_bb = ClusteredTerm(delta_bb, ops_bb, vijkl, self.clusters)
                       
                        term_aa.active = list(set([ci.idx,cj.idx,ck.idx,cl.idx]))
                        term_ab.active = list(set([ci.idx,cj.idx,ck.idx,cl.idx]))
                        term_ba.active = list(set([ci.idx,cj.idx,ck.idx,cl.idx]))
                        term_bb.active = list(set([ci.idx,cj.idx,ck.idx,cl.idx]))
                        
#                        if cj.idx < ci.idx:
#                            term_a.sign = -1
#                            term_b.sign = -1
#                            term_a.ints = 1.0*np.transpose(term_a.ints, axes=(1,0))
#                            term_b.ints = 1.0*np.transpose(term_b.ints, axes=(1,0))
               
                        
                        nswaps = countswaps.countSwaps([ci.idx,cj.idx,ck.idx,cl.idx],4)

                        if nswaps%2==0:
                            sign = 1
                        else: 
                            sign = -1
                       
                        term_aa.sign = sign
                        term_ab.sign = sign
                        term_ba.sign = sign
                        term_bb.sign = sign
                        
                        term_aa.contract_string = contract_string
                        term_ab.contract_string = contract_string
                        term_ba.contract_string = contract_string
                        term_bb.contract_string = contract_string
                       
                        #print(term_bb, [ci.idx,cj.idx,ck.idx,cl.idx])
                       
                        #todo:
                        #   create einsum contraction strings here and store for each term

                        try:
                            self.terms[delta_aa].append(term_aa)
                        except:
                            self.terms[delta_aa] = [term_aa]
                        
                        try:
                            self.terms[delta_ab].append(term_ab)
                        except:
                            self.terms[delta_ab] = [term_ab]
                        
                        try:
                            self.terms[delta_ba].append(term_ba)
                        except:                
                            self.terms[delta_ba] = [term_ba]
                        
                        try:
                            self.terms[delta_bb].append(term_bb)
                        except:                
                            self.terms[delta_bb] = [term_bb]
                        


        print(self.print_terms_header())
        for ti,t in self.terms.items():
            print(ti)
            for tt in t:
                print(tt)
                #print_mat(tt.ints)
# }}}

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
