import numpy as np
import scipy
import itertools as it
import copy as cp
from collections import OrderedDict
from helpers import *
import opt_einsum as oe

import countswaps
from ClusteredState import *

class LocalOperator:
    def __init__(self, cluster):
        self.cluster = cluster
        self.terms = []
    def add_term(self,term):
        self.terms.append(term)
    def build_matrix_dumb1(self):
        np.zeros([self.cluster.dim_tot, self.cluster.dim_tot])

    def combine_common_terms(self,iprint=0):
        """
        combine common terms        
        """
# {{{
        if iprint > 0:
            print(self.print_terms_header())
            for ti,t in self.terms.items():
                print(ti)
                for tt in t:
                    print(tt,tt.contract_string)
        unique = []
        for opi_idx, opi in enumerate(self.terms):
            found = False
            for opj_idx,opj in enumerate(unique):
                if opj.ops == opi.ops and opi.contract_string ==  opi.contract_string:
                    opj.ints += opi.ints
                    found = True
            if found == False:
                unique.append(opi)
        self.terms = unique
# }}}



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
        #self.sign = 1
        self.active = [] 
        
        if len(self.ints.shape) == 0:
            self.ints_inds = ''
        elif len(self.ints.shape) == 2:
            self.ints_inds = 'ab'
        elif len(self.ints.shape) == 4:
            self.ints_inds = 'abcd'
        else:
            print(" Problem with integral tensor")
            exit()

        self.contract_string = ""
        self.contract_string_matvec = ""
        
    def __str__(self):
        #assert(len(self.delta)==2)
        out = ""
        for d in self.delta:
            out += " %2i"%d[0]
            out += " %2i"%d[1]
            out += ":"
        out += "|"
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
    
    def matrix_element(self,fock_bra,bra,fock_ket,ket, opt_einsum=True):
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
        # {{{
        for ci in range(self.n_clusters):
            if (bra[ci]!=ket[ci]) and (ci not in self.active):
                return 0
     
        assert(len(fock_bra) == len(fock_ket))
        assert(len(fock_ket) == len(bra))
        assert(len(bra) == len(ket))
        assert(len(ket) == self.n_clusters)

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
                do = self.clusters[oi].ops[o]
                #do = self.clusters[oi].ops[o][(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
            except:
                print(" Couldn't find:", self)
                exit()
                return 0
            try:
                d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
            except:
                #print(" Couldn't find:", self)
                return 0
            mats.append(d)
            #print(self.clusters[oi].ops[o][tuple([].extend(fock_bra[oi])).extend(fock_ket[oi]))].shape)

        #print(self.clusters[self.active[0]])
        #print(mats[0])
        #print(mats[1])
        me = 0.0
        if len(mats) == 0:
            return 0 
    
        #print('sign2:',state_sign)
        #print()
        # todo:
        #    For some reason, precompiled contract expression is slower than direct einsum - figure this out
        #me = self.contract_expression(*mats) * state_sign
        me = np.einsum(self.contract_string,*mats,self.ints, optimize=opt_einsum) * state_sign
        
        return me
# }}}


    def diag_matrix_element(self,fock,config, opt_einsum=True):
        """
        Compute the diagonal matrix element between <fock,config|H|fock,config>
        where fock is the fockspace of bra. This is just a specification
        of the particle number space of each cluster. Eg., 
        ((2,3),(4,3),(2,2)) would have 3 clusters with 2(3), 4(3), 2(2) 
        alpha(beta) electrons, respectively. 

        Args:
            fock (tuple(tuple)): fockspace
            config (tuple): cluster state configuration within specified fockspace configuration
        Returns:
            matrix element. <IJK...|Hterm|IJK...>, where IJK, and LMN
            are the state indices for clusters 1, 2, and 3, respectively
        """
        # {{{
        assert(len(config) == self.n_clusters)

        mats = []
        # state sign is always 1 here, since an even number of creation/annihilation operators can only 
        # contribute to diagonal
        
        state_sign = 1
        n_active = 0
        for oi,o in enumerate(self.ops):
            if o == '':
                continue
            n_active+=1
        if n_active <= 1:
            return 0

        for oi,o in enumerate(self.ops):
            if o == '':
                continue
            try:
                do = self.clusters[oi].ops[o]
            except KeyError:
                print(" Couldn't find:", self)
                exit()
            try:
                d = do[(fock[oi],fock[oi])][config[oi],config[oi]] #D(I,J,:,:...)
            except KeyError:
                return 0
            mats.append(d)

        if len(mats) == 0:
            return 0 
        me = 0.0
        me = np.einsum(self.contract_string,*mats,self.ints, optimize=opt_einsum)
        
        return me
# }}}


    def effective_cluster_operator(self, cluster_idx, fock_bra, bra, fock_ket, ket):
        """
        Compute the cluster density matrix between <fock1,config1|H|fock2,config2>
        where fock is the 'fock-block' of bra. This is just a specification
        of the particle number space of each cluster. Eg., 
        ((2,3),(4,3),(2,2)) would have 3 clusters with 2(3), 4(3), 2(2) 
        alpha(beta) electrons, respectively. 

        Args:
            cluster_idx: index of cluster to extract effective operator for
            fock_bra (tuple(tuple)): fock-block for bra
            bra (tuple): cluster state configuration within specified fock block
            fock_ket (tuple(tuple)): fock-block for ket 
            ket (tuple): cluster state configuration within specified fock block
        Returns:
            matrix element. <IJK...|Hterm|LMN...>, where IJK, and LMN
            are the state indices for clusters 1, 2, and 3, respectively, in the 
            particle number blocks specified by fock_bra and fock_ket.
        """
        # {{{
        for ci in range(self.n_clusters):
            if (bra[ci]!=ket[ci]) and (ci not in self.active):
                return 0
     
        assert(len(fock_bra) == len(fock_ket))
        assert(len(fock_ket) == len(bra))
        assert(len(bra) == len(ket))
        assert(len(ket) == self.n_clusters)

        # <bra|term|ket>    = <IJK|o1o2o3|K'J'I'>
        #                   = <I|o1|I'><J|o2|J'><K|o3|K'> 
        #print(bra,ket,self)
        #print(self.ints.shape)
    
        mats = []
        state_sign = 1
        if self.ops[cluster_idx] == '':
            return
        
        new_opstr = ""
        str_to_delete = -1 
        for oi,o in enumerate(self.ops):
            if o == '':
                continue
            if oi == cluster_idx:
                new_opstr = o
                str_to_delete = oi
                continue
            if len(o) == 1 or len(o) == 3:
                for cj in range(oi):
                    state_sign *= (-1)**(fock_ket[cj][0]+fock_ket[cj][1])
            try:
                do = self.clusters[oi].ops[o]
            except:
                print(" Couldn't find:", self)
                exit()
                return 0
            try:
                d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
            except:
                return 0
            mats.append(d)

        tmp = [x.strip() for x in self.contract_string.split(',')]
        output_str = tmp[str_to_delete]
        tmp.remove(tmp[str_to_delete])
        new_contract_string = ",".join(tmp) + output_str
        
        print(self, self.contract_string, new_contract_string)
        if len(mats) > 0:
            me = np.einsum(new_contract_string,*mats,self.ints) * state_sign
        elif len(mats) == 0:
            return 0 
        else:
            print("Error")
            exit()
        return me
# }}}


class LocalClusteredTerm(ClusteredTerm):
    """
    This is a special Clustered Term which only has operators on one cluster at a time
    """
    def __init__(self, delta, ops, clusters):
        super().__init__(delta, ops, np.empty([]), clusters)

    
    def matrix_element(self,fock_bra,bra,fock_ket,ket,opt_einsum=None):
        """
        Compute the matrix element between <fock1,config1|H|fock2,config2>
        where fock is the 'fock-block' of bra. This is just a specification
        of the particle number space of each cluster. Eg., 
        ((2,3),(4,3),(2,2)) would have 3 clusters with 2(3), 4(3), 2(2) 
        alpha(beta) electrons, respectively. 
        
        For this Local term, it just needs to return the matrix element of the stored operator

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
        # {{{
        for ci in range(self.n_clusters):
            if (bra[ci]!=ket[ci]) and (ci not in self.active):
                return 0
     
        assert(len(fock_bra) == len(fock_ket))
        assert(len(fock_ket) == len(bra))
        assert(len(bra) == len(ket))
        assert(len(ket) == self.n_clusters)
        assert(len(self.active) == 1)

        ci = self.active[0]
        return self.clusters[ci].ops['H'][(fock_bra[ci],fock_ket[ci])][bra[ci],ket[ci]]
# }}}
    
    def diag_matrix_element(self,fock,config,opt_einsum=None):
        return self.matrix_element(fock,config,fock,config) 





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
    def __init__(self,clusters, core_energy=0.0):
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.terms = OrderedDict()
        self.local_terms = []
        self.n_orb = 0
        self.core_energy = core_energy 
        for ci,c in enumerate(self.clusters):
            self.n_orb += c.n_orb

    def add_local_terms(self, opstr="H"):
        """
        Add terms of the form h_{pq}\hat{a}^\dagger_p\hat{a}_q

        input:
        h is a square matrix NxN, where N is the number of spatial orbitals
        """
# {{{

        self.local_terms = []
        delta_tmp = []
        ops_tmp = []
        for ci in self.clusters:
            delta_tmp.append([0,0])
            ops_tmp.append("")
        delta = tuple([tuple(i) for i in delta_tmp])

        for ci in self.clusters:

            ops = cp.deepcopy(ops_tmp) 
            ops[ci.idx] += opstr 
            
            term = LocalClusteredTerm(delta, ops, self.clusters)

            term.active = [ci.idx]

            try:
                self.terms[delta].append(term)
            except:
                self.terms[delta] = [term]

# }}}

    def add_1b_terms(self,h):
        """
        Add terms of the form h_{pq}\hat{a}^\dagger_p\hat{a}_q

        input:
        h is a square matrix NxN, where N is the number of spatial orbitals
        """
# {{{
        
        try:
            assert(len(h.shape)==2)
            assert(h.shape[0]==self.n_orb)
            assert(h.shape[1]==self.n_orb)
        except AssertionError:
            print(h.shape)
            print(self.n_orb)
            raise AssertionError

        delta_tmp = []
        ops_tmp = []
        for ci in self.clusters:
            delta_tmp.append([0,0])
            ops_tmp.append("")

        for ci in self.clusters:
            for cj in self.clusters:
                if ci == cj:
                    continue
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

                if  not np.any(hij):
                    continue
                
                delta_a = tuple([tuple(i) for i in delta_a])
                delta_b = tuple([tuple(i) for i in delta_b])
                term_a = ClusteredTerm(delta_a, ops_a, hij, self.clusters)
                term_b = ClusteredTerm(delta_b, ops_b, hij, self.clusters)

                if ci.idx==cj.idx:
                    term_a.active = [ci.idx]
                    term_b.active = [ci.idx]
                elif ci.idx < cj.idx:
                    term_a.active = [ci.idx,cj.idx]
                    term_b.active = [ci.idx,cj.idx]
                elif ci.idx > cj.idx:
                    term_a.active = [cj.idx,ci.idx]
                    term_b.active = [cj.idx,ci.idx]
                
                if cj.idx < ci.idx:
                    term_a.ints = -1.0*np.transpose(term_a.ints, axes=(1,0))
                    term_b.ints = -1.0*np.transpose(term_b.ints, axes=(1,0))

                if cj.idx == ci.idx:
                    term_a.contract_string = "pq,pq->"
                    term_b.contract_string = "pq,pq->"
                    term_a.contract_string_matvec = "xpq,pq->x"
                    term_b.contract_string_matvec = "xpq,pq->x"
                else:
                    term_a.contract_string = "p,q,pq->"
                    term_b.contract_string = "p,q,pq->"
                    term_a.contract_string_matvec = "xp,yq,pq->xy"
                    term_b.contract_string_matvec = "xp,yq,pq->xy"
                
                
                #build Contract expression
                shapes = []
                if cj.idx == ci.idx:
                    shapes.append([ci.n_orb,ci.n_orb])
                elif ci.idx < cj.idx:
                    shapes.append([ci.n_orb])
                    shapes.append([cj.n_orb])
                elif cj.idx < ci.idx:
                    shapes.append([cj.n_orb])
                    shapes.append([ci.n_orb])

                shapes.append(term_a.ints)
                
                term_a.contract_expression = oe.contract_expression(term_a.contract_string,*shapes,constants=[len(shapes)-1])
                term_b.contract_expression = oe.contract_expression(term_b.contract_string,*shapes,constants=[len(shapes)-1])
                
                try:
                    self.terms[delta_a].append(term_a)
                except:
                    self.terms[delta_a] = [term_a]
                try:
                    self.terms[delta_b].append(term_b)
                except:
                    self.terms[delta_b] = [term_b]

        #print(self.print_terms_header())
        for ti,t in self.terms.items():
            for tt in t:
                print(tt)
                #print_mat(tt.ints)
# }}}

    def add_2b_terms(self,v,iprint=0):
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
                        if ci == cj and ci == ck and ci == cl:
                            continue
                       
                        count = {}
                        for i in [ci.idx, cj.idx, ck.idx, cl.idx]:
                            count[i] = 0
                        for i in [ci.idx, cj.idx, ck.idx, cl.idx]:
                            count[i] += 1
                        if len(count) > 2:
                            continue
                        skip = 0
                        for key,val in count.items():
                            if val > 2:
                                skip = 1
                        if skip == 1:
                            continue

                        #if ci != cj or ck != cl:
                        #    continue
                        delta_aa = list(cp.deepcopy(delta_tmp)) 
                        delta_bb = list(cp.deepcopy(delta_tmp)) 
                        delta_ab = list(cp.deepcopy(delta_tmp)) 
                        delta_ba = list(cp.deepcopy(delta_tmp)) 
                        #print("\n===========================================================")
                        #print("ci.idx, cj.idx, ck.idx, cl.idx",ci.idx, cj.idx, ck.idx, cl.idx)
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
                        shapes_list = [ci.n_orb,cj.n_orb,ck.n_orb,cl.n_orb]
                        ops_aa_list = ['A','A','a','a']
                        ops_ab_list = ['A','B','b','a']
                        ops_ba_list = ['B','A','a','b']
                        ops_bb_list = ['B','B','b','b']
                        cont_indices1 = ['i','j','k','l'] #density indices
                        cont_indices2 = ['i','l','j','k'] #integral indices in chemists notation
                        sorted_idx = np.argsort(clusters_idx,kind='stable')
                       
                        
                        shapes_list = [shapes_list[s] for s in sorted_idx]
                        ops_aa_list = [ops_aa_list[s] for s in sorted_idx]
                        ops_ab_list = [ops_ab_list[s] for s in sorted_idx]
                        ops_ba_list = [ops_ba_list[s] for s in sorted_idx]
                        ops_bb_list = [ops_bb_list[s] for s in sorted_idx]
                        cont_indices1 = [cont_indices1[s] for s in sorted_idx]
                        #cont_indices2 = [cont_indices2[s] for s in sorted_idx]
                        clusters_idx = [clusters_idx[s] for s in sorted_idx]
                        
                       
                        # get sign from reordering operators
                        sign = 1
                        nswaps = countswaps.countSwaps([ci.idx,cj.idx,ck.idx,cl.idx],4)
                        if nswaps%2!=0:
                            sign = -1
                       
                        #print('indices:', [ci.idx,cj.idx,ck.idx,cl.idx], cont_indices1, sign)
                        
                        # i'j'kl<ij|lk> = i'j'kl(il|jk)
                        
                        #vijkl = sign*v[ci.orb_list,:,:,:][:,cl.orb_list,:,:][:,:,cj.orb_list,:][:,:,:,ck.orb_list]
                        vijkl = sign*.5*v[ci.orb_list,:,:,:][:,cl.orb_list,:,:][:,:,cj.orb_list,:][:,:,:,ck.orb_list]
                        
                        if  not np.any(vijkl):
                            continue
                        
                        #vijkl = np.transpose(vijkl,axes=sorted_idx) # sort 
                    
                        sorted_idx_ints = np.argsort([ci.idx,cl.idx,cj.idx,ck.idx],kind='stable')
                        cont_indices2 = [cont_indices2[s] for s in sorted_idx_ints]
                        vijkl = np.transpose(vijkl,axes=sorted_idx_ints) # sort 
                        #print(vijkl.shape)


                        #   Group the indices by cluster
                        str_dict = OrderedDict() 
                        for idx in range(4):
                            str_dict[clusters_idx[idx]] = ""

                        state_indices = ['w','x','y','z']
                        for idx in range(4): 
                            str_dict[clusters_idx[idx]] += cont_indices1[idx] 
                       
                        #print(str_dict)
                        
                        contract_string = ""
                        contract_string_matvec = ""
                        tmp = 0
                        for stringi,string in str_dict.items():
                            contract_string += string + ","
                            contract_string_matvec += state_indices[tmp] + string + ","
                            tmp+=1
                        

                        
                        contract_string += cont_indices2[0] +cont_indices2[1] +cont_indices2[2] +cont_indices2[3] + "->"
                        contract_string_matvec += cont_indices2[0] +cont_indices2[1] +cont_indices2[2] +cont_indices2[3] + "->"
                       
                         
                        for stringi,string in enumerate(str_dict.items()):
                            contract_string_matvec += state_indices[stringi] 
                        
                        delta_aa = tuple([tuple(i) for i in delta_aa])
                        delta_ab = tuple([tuple(i) for i in delta_ab])
                        delta_ba = tuple([tuple(i) for i in delta_ba])
                        delta_bb = tuple([tuple(i) for i in delta_bb])
                        
                        term_aa = ClusteredTerm(delta_aa, ops_aa, vijkl, self.clusters)
                        term_ab = ClusteredTerm(delta_ab, ops_ab, vijkl, self.clusters)
                        term_ba = ClusteredTerm(delta_ba, ops_ba, vijkl, self.clusters)
                        term_bb = ClusteredTerm(delta_bb, ops_bb, vijkl, self.clusters)
                       
                        term_aa.active = sorted(list(set([ci.idx,cj.idx,ck.idx,cl.idx])))
                        term_ab.active = sorted(list(set([ci.idx,cj.idx,ck.idx,cl.idx])))
                        term_ba.active = sorted(list(set([ci.idx,cj.idx,ck.idx,cl.idx])))
                        term_bb.active = sorted(list(set([ci.idx,cj.idx,ck.idx,cl.idx])))
                        
                        
                        term_aa.contract_string = contract_string
                        term_ab.contract_string = contract_string
                        term_ba.contract_string = contract_string
                        term_bb.contract_string = contract_string
                        
                        term_aa.contract_string_matvec = contract_string_matvec
                        term_ab.contract_string_matvec = contract_string_matvec
                        term_ba.contract_string_matvec = contract_string_matvec
                        term_bb.contract_string_matvec = contract_string_matvec
                      
                      
                        shapes = []
                        shapes_seen = 0
                        for stringi,string in str_dict.items():
                            #print(string,[10]*len(string),vijkl.shape)
                            shape = []
                            for i in range(len(string)):
                                shape.append(shapes_list[shapes_seen])
                                shapes_seen += 1
                            shapes.append(shape)
                        shapes.append(vijkl)
                        #exit()

                        #uncomment the following to use the optimized contraction strings in opteinsum
                        if 0:
                            term_aa.contract_expression = oe.contract_expression(term_aa.contract_string,*shapes,constants=[len(shapes)-1])
                            term_ab.contract_expression = oe.contract_expression(term_ab.contract_string,*shapes,constants=[len(shapes)-1])
                            term_ba.contract_expression = oe.contract_expression(term_ba.contract_string,*shapes,constants=[len(shapes)-1])
                            term_bb.contract_expression = oe.contract_expression(term_bb.contract_string,*shapes,constants=[len(shapes)-1])
                            #print(term_aa.contract_expression)
                            #print(term_bb, [ci.idx,cj.idx,ck.idx,cl.idx])
                       
                        try:
                            self.terms[delta_aa].append(term_aa)
                        except:
                            self.terms[delta_aa] = [term_aa]
                        
                        try:
                            self.terms[delta_ab].append(term_ab)
                        except:
                            self.terms[delta_ab] = [term_ab]
                        
#                        if len(term_ba.active) > 1:
#                            try:
#                                self.terms[delta_ba].append(term_ba)
#                            except:                
#                                self.terms[delta_ba] = [term_ba]
                        try:
                            self.terms[delta_ba].append(term_ba)
                        except:                
                            self.terms[delta_ba] = [term_ba]
                        
                        try:
                            self.terms[delta_bb].append(term_bb)
                        except:                
                            self.terms[delta_bb] = [term_bb]

        if iprint > 2:
            print(self.print_terms_header())
            for ti,t in self.terms.items():
                #print(ti)
                for tt in t:
                    print(tt)
# }}}

    def combine_common_terms(self,iprint=1):
        """
        Combine identical terms     
        """
# {{{
        if iprint > 0:
            print(self.print_terms_header())
            for ti,t in self.terms.items():
                print(ti)
                for tt in t:
                    print(tt,tt.contract_string)
        print("NYI")
        exit()
# }}}

    def print_terms_header(self):
        """
        print header with labels for printing term
        """
        # {{{
        out = ""
        for di,d in enumerate(self.clusters):
            out += "%2i_____"%di
        out += "\n"
        for di,d in enumerate(self.clusters):
            out += " Δa"
            out += " Δb"
            out += ":"
        out += "|"
        for oi,o in enumerate(self.clusters):
            out += " %4i,"%oi
        return out
# }}}


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
        """
        Extract Local operator, considering only terms which are completely contained inside cluster
        """
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

    def extract_local_embedded_operator(self,cluster_idx,rdm):
        """
        Extract Local operator, considering only terms which are completely contained inside cluster,
        integrating over the rest of the system.

        H^A = \sum_{pq \in A}}\left(h_{pq} + \sum_B\sum_{rs\in B}\mel{pr\vert qs}P_{rs}\right) \hat{p}^\dagger\hat{q}
                + \sum_{pqrs \in A} \mel{pr\vert qs}\hat{p}^\dagger\hat{r}^\dagger\hat{s}\hat{r}
        
        input:
            1rdm: <p'q> matrix
        
        """
        op = LocalOperator(self.clusters[cluster_idx])
        for t in self.terms:
            for tt in self.terms[t]:
                active = tt.get_active_clusters()
                if len(active) == 1 and active[0] == cluster_idx:

                    term = cp.deepcopy(tt)
                    term.ops = [term.ops[cluster_idx]]
                    term.delta = [term.delta[cluster_idx]]
                    op.add_term(term)
                elif len(active) == 2:
                    if active[0] == cluster_idx:
                        if tt.ops[0] == 'Aa' or tt.ops[0] == 'Bb':
                            tti = tt.ops[0]
                            ttj = tt.ops[1]
                            
                            term = cp.deepcopy(tt)
                            term.ops = [tti]
                            term.delta = [(0,0)]
                            term.ints = np.einsum('pqrs,rs->pq',term.ints,densites[ttj])
                            print(term)
                            op.add_term(term)

        return op
    
    def extract_local_embedded_operator2(self,cluster_idx,fock_state,config):
        """
        Extract a local operator embedded in a direct product state

            e.g. p'q<r's>(pr|qs)

        input:
            cluster_idx: index of cluster 
            fock_state:  fockstate configuration of the clusters
            config    :  state configuration within that fockstate
        """
        op = LocalOperator(self.clusters[cluster_idx])
        for t in self.terms:
            for tt in self.terms[t]:
                active = tt.get_active_clusters()
                if len(active) == 1 and active[0] == cluster_idx:

                    term = cp.deepcopy(tt)
                    term.ops = [term.ops[cluster_idx]]
                    term.delta = [term.delta[cluster_idx]]
                    op.add_term(term)

                doterm = False

                new_str = ""
                if len(active) == 2 and active[0] == cluster_idx:
                    if tt.ops[0] == 'Aa' or tt.ops[0] == 'Bb':
                        i,j = 0,1
                        doterm = True
                        new_str = tt.contract_string[3:12] +  tt.contract_string[0:2]
                elif len(active) == 2 and active[1] == cluster_idx:
                    if tt.ops[1] == 'Aa' or tt.ops[1] == 'Bb':
                        i,j = 1,0
                        doterm = True
                        new_str = tt.contract_string[3:12] +  tt.contract_string[0:2]
                else:
                    continue
               
                #print(tt,new_str)
                if doterm:
                    fock_j = fock_state[active[j]]
                    cj = self.clusters[active[j]]
                    conf_j = config[active[j]]
                    print(tt)
                    try:
                        dens_j = cj.ops[tt.ops[j]]
                    except KeyError:
                        print("Error: ", tt.ops[j], " apparently not in cluster:",active[j]," have you built the basis yet?")
                        exit()
                    
                    try:
                        dens_j = dens_j[(fock_j, fock_j)][conf_j, conf_j]
                        #print(tt,new_str, dens_j.shape)
                        new_v = np.einsum(new_str, dens_j, tt.ints)
                        #new_str2 = new_str[9:11] + "," + new_str[3:5] + "->"
                        
                        term = cp.deepcopy(tt)
                        term.ops = [term.ops[cluster_idx]]
                        term.delta = [(0,0)]
                        term.contract_string = "ab,ab->"    
                        if new_str[9] == new_str[3]:
                            term.ints = new_v
                        elif new_str[9] == new_str[4]:
                            term.ints = new_v.T
                        else:
                            print(" Error:")
                            exit()
                        #print(term, term.contract_string)
                        op.add_term(term)
                    except KeyError:
                        #must not exist for state
                        pass

        return op



if __name__== "__main__":
    term = ClusteredTerm()
