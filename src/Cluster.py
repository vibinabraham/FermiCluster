import numpy as np
import scipy
import itertools as it

import opt_einsum as oe
import time
from ci_string import *
from Hamiltonian import *
from davidson import *
from helpers import *
from myfci import *

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
        
        self.energies = {}                  # Diagonal of local operators

    def __len__(self):
        return len(self.orb_list) 
    def __str__(self):
        has = ""
        for si in self.orb_list:
            has = has + "%03i|"%si
        if len(self.orb_list) == 0:
            has = "|"
        return "IDX%03i:DIM%04i:%s" %(self.idx, self.dim, has)

    def possible_fockspaces(self, delta_elec=None):
        """
        Get list of possible fock spaces accessible to the cluster

        delta_elec      :   (ref_alpha, ref_beta, delta) allows restrictions to fock spaces
                                based on a delta from some reference occupancy (ref_alpha, ref_beta)
        """
        # {{{
      
        if delta_elec != None:
            assert(len(delta_elec) == 3)
            ref_a = delta_elec[0]
            ref_b = delta_elec[1]
            delta = delta_elec[2]
            
        fspaces = []
        for na in range(self.n_orb+1):
            for nb in range(self.n_orb+1):
                if delta_elec != None:
                    if abs(na-ref_a)+abs(nb-ref_b) > delta:
                        continue
                fspaces.append([na,nb])
        return fspaces
        # }}}
    
    def add_operator(self,op):
        if op in self.ops:
            return
        else:
            self.ops[op] = {}

    def form_fockspace_eigbasis(self, hin, vin, spaces, max_roots=1000, rdm1_a=None, rdm1_b=None, ecore=0, iprint=0):
        """
        Get matrices of local Hamiltonians embedded in the system 1rdm and diagonalize to form
        the cluster basis vectors. This is the main step in CMF

        This function creates entries in the self.ops dictionary for self.ops['H_mf']
        This data is used to obtain the cluster 
        energies for an MP2 like correction rather than a EN correction.

        """
# {{{
        for f in spaces:
            assert(len(f)==2)

        if len(f) == 0:
            print(" No spaces requested - certainly a mistake")
            exit()

        print(" Build basis vectors for the following fockspaces:")
        for f in spaces:
            print("    Alpha:Beta = %4i %-4i" %(f[0],f[1]))

        h = np.zeros([self.n_orb]*2)
        f = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                h[pidx,qidx] = hin[p,q]
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                for ridx,r in enumerate(self.orb_list):
                    for sidx,s in enumerate(self.orb_list):
                        v[pidx,qidx,ridx,sidx] = vin[p,q,r,s]


        if rdm1_a is not None and rdm1_b is not None:
            print(" Compute single particle embedding potential")
            denv_a = 1*rdm1_a
            denv_b = 1*rdm1_b
            for pidx,p in enumerate(self.orb_list):
                for qidx,q in enumerate(range(rdm1_a.shape[0])):
                    denv_a[p,q] = 0
                    denv_b[p,q] = 0
                    denv_a[q,p] = 0
                    denv_b[q,p] = 0
           
            if iprint>0:
                print(" Environment 1RDM:")
                print_mat(denv_a+denv_b)
            print(" Trace of env 1RDM: %12.8f" %np.trace(denv_a + denv_b))
            print(" Compute energy of 1rdm:")
            e1 =  np.trace(hin @ rdm1_a )
            e1 += np.trace(hin @ rdm1_b )
            e2 =  np.einsum('pqrs,pq,rs->',vin,rdm1_a,rdm1_a)
            e2 -= np.einsum('pqrs,ps,qr->',vin,rdm1_a,rdm1_a)
            
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_b,rdm1_b)
            e2 -= np.einsum('pqrs,ps,qr->',vin,rdm1_b,rdm1_b)
            
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_a,rdm1_b)
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_b,rdm1_a)
            #e += np.einsum('pqrs,pq,rs->',vin,d,d)
           
            e = e1 + .5*e2
            print(" E: %12.8f" %(e+ecore))
           
            fa =  hin*0 
            fb =  hin*0
            fa += np.einsum('pqrs,pq->rs',vin,denv_a)
            fa += np.einsum('pqrs,pq->rs',vin,denv_b)
            fa -= np.einsum('pqrs,ps->qr',vin,denv_a)
            fb += np.einsum('pqrs,pq->rs',vin,denv_b)
            fb += np.einsum('pqrs,pq->rs',vin,denv_a)
            fb -= np.einsum('pqrs,ps->qr',vin,denv_b)

        
            for pidx,p in enumerate(self.orb_list):
                for qidx,q in enumerate(self.orb_list):
                    f[pidx,qidx] = .5*(fa[p,q] + fb[p,q])
           

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h + f
        H.V = v
        H.ecore = ecore
        self.basis = {}
        self.ops['H_mf'] = {}
       
        for na,nb in spaces:
            ci = ci_solver()
            ci.algorithm = "direct"
            ci.init(H,na,nb,max_roots)
            print(ci)
            Hci = ci.run()
            #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
            fock = (na,nb)
            self.basis[fock] = ci.results_v
            self.ops['H_mf'][(fock,fock)] = ci.results_v.T @ Hci @ ci.results_v
    # }}}

    #remove:
    def form_eigbasis_from_ints(self,hin,vin,max_roots=1000, max_elec=None, min_elec=0, rdm1_a=None, rdm1_b=None, ecore=0):
        """
        grab integrals acting locally and form eigenbasis by FCI

        rdm1 is the spin summed density matrix
        """
# {{{
        h = np.zeros([self.n_orb]*2)
        f = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                h[pidx,qidx] = hin[p,q]
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                for ridx,r in enumerate(self.orb_list):
                    for sidx,s in enumerate(self.orb_list):
                        v[pidx,qidx,ridx,sidx] = vin[p,q,r,s]


        if rdm1_a is not None and rdm1_b is not None:
            print(" Compute single particle embedding potential")
            denv_a = 1*rdm1_a
            denv_b = 1*rdm1_b
            for pidx,p in enumerate(self.orb_list):
                for qidx,q in enumerate(range(rdm1_a.shape[0])):
                    denv_a[p,q] = 0
                    denv_b[p,q] = 0
                    denv_a[q,p] = 0
                    denv_b[q,p] = 0
           
            print(" Environment 1RDM:")
            print_mat(denv_a+denv_b)
            print(" Trace of env 1RDM: %12.8f" %np.trace(denv_a + denv_b))
            print(" Compute energy of 1rdm:")
            e1 =  np.trace(hin @ rdm1_a )
            e1 += np.trace(hin @ rdm1_b )
            e2 =  np.einsum('pqrs,pq,rs->',vin,rdm1_a,rdm1_a)
            e2 -= np.einsum('pqrs,ps,qr->',vin,rdm1_a,rdm1_a)
            
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_b,rdm1_b)
            e2 -= np.einsum('pqrs,ps,qr->',vin,rdm1_b,rdm1_b)
            
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_a,rdm1_b)
            e2 += np.einsum('pqrs,pq,rs->',vin,rdm1_b,rdm1_a)
            #e += np.einsum('pqrs,pq,rs->',vin,d,d)
           
            e = e1 + .5*e2
            print(" E: %12.8f" %(e+ecore))
           
            fa =  hin*0 
            fb =  hin*0
            fa += np.einsum('pqrs,pq->rs',vin,denv_a)
            fa += np.einsum('pqrs,pq->rs',vin,denv_b)
            fa -= np.einsum('pqrs,ps->qr',vin,denv_a)
            fb += np.einsum('pqrs,pq->rs',vin,denv_b)
            fb += np.einsum('pqrs,pq->rs',vin,denv_a)
            fb -= np.einsum('pqrs,ps->qr',vin,denv_b)

        
            for pidx,p in enumerate(self.orb_list):
                for qidx,q in enumerate(self.orb_list):
                    f[pidx,qidx] = .5*(fa[p,q] + fb[p,q])
           

        if max_elec == None:
            max_elec == self.n_orb

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h + f
        H.V = v
        self.basis = {}
        print(" Do CI for each particle number block")
        for na in range(self.n_orb+1):
            for nb in range(self.n_orb+1):
                if (na+nb) > max_elec or (na+nb) < min_elec:
                    continue
                ci = ci_solver()
                ci.algorithm = "direct"
                ci.init(H,na,nb,max_roots)
                print(ci)
                Hci = ci.run()
                #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                self.basis[(na,nb)] = ci.results_v
                #self.Hci[(na,nb)] = ci.results_v.T @ Hci @ ci.results_v
    # }}}
    
    #remove:
    def form_eigbasis_from_local_operator(self,local_op,max_roots=1000,ratio = 1,s2_shift=False):
        """
        grab integrals acting locally and form eigenbasis by FCI
        """
# {{{
        h = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        for t in local_op.terms:
            if t.ops[0] == "Aa":
                h += t.ints 
            if t.ops[0] == "AAaa":
                v = 2*t.ints 

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h
        H.V = v
        self.basis = {}
        print(" Do CI for each particle number block")
        for na in range(self.n_orb+1):
            for nb in range(self.n_orb+1):

                dim = calc_nchk(self.n_orb, na)*calc_nchk(self.n_orb, nb)
                max_roots = int(((ratio * dim) //1) +1 )

                ci = ci_solver()
                ci.algorithm = "direct"
                ci.init(H,na,nb,max_roots)
                print(ci)
                Hci = ci.run()
                #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                self.basis[(na,nb)] = ci.results_v
                #self.Hci[(na,nb)] = Hci
                #print(ci.results_v)
                if s2_shift == True:
                    S2 = form_S2(self.n_orb,na,nb)

                    eva,evec = np.linalg.eigh(Hci + 0.3 * S2)
                    self.basis[(na,nb)] = evec


                if 0: #basis deteminant ordering not same yet, so cant use pyscf 
                    from pyscf import gto, scf, ao2mo, fci, cc
                    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
                    cisolver = fci.direct_spin1.FCI()
                    e, civ = cisolver.kernel(h, v, h.shape[1], (na,nb), ecore=0,nroots=max_roots)
                    print(e)
                    civ = np.array(civ)
                    civ = civ.reshape(dim,dim)
                    self.basis[(na,nb)] = civ
                    #self.basis[(na,nb)] = np.eye(civ.shape[0])

                if 0:
                    ci.init(H,na,nb,dim)
                    print(ci.results_v[:,-max_roots:] )
                    ci.run()
                    self.basis[(na,nb)] = ci.results_v[:,-max_roots:]
                    # }}}

    #remove:
    def form_eigbasis_from_local_operator_nanb(self,local_op,n_max,max_roots=1000,ratio = 1):
        """
        grab integrals acting locally and form eigenbasis by FCI
        """
# {{{
        h = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        for t in local_op.terms:
            if t.ops[0] == "Aa":
                h += t.ints 
            if t.ops[0] == "AAaa":
                v = 2*t.ints 

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h
        H.V = v
        self.basis = {}
        print(" Do CI for each particle number block")
        for na in range(n_max+1):
            for nb in range(n_max+1):
                if na+nb < n_max:
                    dim = calc_nchk(self.n_orb, na)*calc_nchk(self.n_orb, nb)
                    #max_roots = int(((ratio * dim) //1) +1 )

                    ci = ci_solver()
                    ci.algorithm = "direct"
                    ci.init(H,na,nb,max_roots)
                    print(ci)
                    Hci = ci.run()
                    #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                    self.basis[(na,nb)] = ci.results_v
                    #self.Hci[(na,nb)] = Hci
# }}}
                    
            
    def rotate_basis(self,U):
        """
        Rotate cluster's basis using U, which is an dictionary mapping fock spaces to unitary rotation matrices.
        rotate basis, and all associated operators
        """
# {{{
        for fspace,mat in U.items():
            self.basis[fspace] = self.basis[fspace] @ mat
            #self.Hci[fspace] = mat.T @ self.Hci[fspace] @ mat 
            #self.Hci[fspace] = self.basis[fspace].T @ self.Hci[fspace] @ self.basis[fspace]
        #print(" Build all operators:")
        #self.build_op_matrices()
        for op,fspace_deltas in self.ops.items():
            for fspace_delta,tdm in fspace_deltas.items():
                fspace_l = fspace_delta[0]
                fspace_r = fspace_delta[1]
                #if fspace_l in U:
                #    Ul = U[fspace_l]
                #    self.ops[op][fspace_delta] = np.einsum('pq,pr...->qr...',Ul,self.ops[op][fspace_delta])
                #if fspace_r in U:
                #    Ur = U[fspace_r]
                #    self.ops[op][fspace_delta] = np.einsum('rs,pr...->ps...',Ur,self.ops[op][fspace_delta])
                if fspace_l in U and fspace_r in U:
                    Ul = U[fspace_l]
                    Ur = U[fspace_r]
                    try:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,pr...->qs...',Ul,Ur,self.ops[op][fspace_delta], optimize=True)
                    except ValueError:
                        print("Error: Rotate basis failed for term: ", op, " fspace_delta: ", fspace_delta) 
                        print(Ul.shape)
                        print(Ur.shape)
                        print(self.ops[op][fspace_delta].shape)
                        exit()
   # }}}
    
    
    def get_ops(self):
        return self.ops

    def get_op(self,opstr):
        return self.ops[opstr]

    
    def get_op_mel(self,opstr,fI,fJ,I,J):
        return self.ops[opstr][(fI,fJ)][I,J,:]


    def build_local_terms(self,hin,vin):
        start = time.time()
        self.ops['H'] = {}
        """
        grab integrals acting locally and form precontracted operator in current eigenbasis
        """
# {{{
        h = np.zeros([self.n_orb]*2)
        f = np.zeros([self.n_orb]*2)
        v = np.zeros([self.n_orb]*4)
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                h[pidx,qidx] = hin[p,q]
        
        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                for ridx,r in enumerate(self.orb_list):
                    for sidx,s in enumerate(self.orb_list):
                        v[pidx,qidx,ridx,sidx] = vin[p,q,r,s]



        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h
        H.V = v
       
        for fock in self.basis:
            ci = ci_solver()
            ci.algorithm = "direct"
            na = fock[0]
            nb = fock[1]
            ci.init(H,na,nb,1)
            #print(ci)
            self.ops['H'][(fock,fock)] = ci.build_H_matrix(self.basis[fock])
            #print(" GS Energy: %12.8f" %self.ops['H'][(fock,fock)][0,0])
    # }}}
        

        stop = time.time()
        print(" Time spent TDM 0: %12.2f" %(stop-start))


    def build_op_matrices(self, iprint=1):
        """
        build all operators needed
        """
# {{{
        start = time.time()
        self.ops['A'] = {}
        self.ops['a'] = {}
        self.ops['B'] = {}
        self.ops['b'] = {}
        self.ops['Aa'] = {}
        self.ops['Bb'] = {}
        self.ops['Ab'] = {}
        self.ops['Ba'] = {}
        #self.ops['AAaa'] = {}
        #self.ops['BBbb'] = {}
        #self.ops['ABba'] = {}
        #self.ops['BAab'] = {}
        self.ops['AA'] = {}
        self.ops['BB'] = {}
        self.ops['AB'] = {}
        self.ops['BA'] = {}
        self.ops['aa'] = {}
        self.ops['bb'] = {}
        self.ops['ba'] = {}
        self.ops['ab'] = {}
        self.ops['AAa'] = {}
        self.ops['BBb'] = {}
        self.ops['ABb'] = {}
        self.ops['BAa'] = {}
        self.ops['BAb'] = {}
        self.ops['ABa'] = {}
        self.ops['Aaa'] = {}
        self.ops['Bbb'] = {}
        self.ops['Aab'] = {}
        self.ops['Bba'] = {}
        self.ops['Aba'] = {}
        self.ops['Bab'] = {}

        start_tot = time.time()


        #  a, A
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['a'][(na-1,nb),(na,nb)] = build_annihilation(self.n_orb, (na-1,nb),(na,nb),self.basis)
                    self.ops['A'][(na,nb),(na-1,nb)] = cp.deepcopy(np.swapaxes(self.ops['a'][(na-1,nb),(na,nb)],0,1))
                except KeyError:
                    continue
                # note:
                #   I did a deepcopy instead of reference. This increases memory requirements and 
                #   basis transformation costs, but simplifies later manipulations. Later I need to 
                #   remove the redundant storage by manually handling the transpositions from a to A
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 1: %12.2f" %(stop-start))

        #  b, B 
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['b'][(na,nb-1),(na,nb)] = build_annihilation(self.n_orb, (na,nb-1),(na,nb),self.basis)
                    self.ops['B'][(na,nb),(na,nb-1)] = cp.deepcopy(np.swapaxes(self.ops['b'][(na,nb-1),(na,nb)],0,1))
                except KeyError:
                    continue
                # note:
                #   I did a deepcopy instead of reference. This increases memory requirements and 
                #   basis transformation costs, but simplifies later manipulations. Later I need to 
                #   remove the redundant storage by manually handling the transpositions from a to A
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 2: %12.2f" %(stop-start))

        #  Aa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['Aa'][(na,nb),(na,nb)] = build_ca_ss(self.n_orb, (na,nb),(na,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 3: %12.2f" %(stop-start))
        
        #  Bb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bb'][(na,nb),(na,nb)] = build_ca_ss(self.n_orb, (na,nb),(na,nb),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 4: %12.2f" %(stop-start))

               


        #  Ab,Ba
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Ab'][(na,nb-1),(na-1,nb)] = build_ca_os(self.n_orb, (na,nb-1),(na-1,nb),self.basis,'ab')
                    self.ops['Ba'][(na-1,nb),(na,nb-1)] = build_ca_os(self.n_orb, (na-1,nb),(na,nb-1),self.basis,'ba')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 5: %12.2f" %(stop-start))

        
        """       
        #  AAaa,BBbb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['AAaa'][(na,nb),(na,nb)] = build_ccaa_ss(self.n_orb, (na,nb),(na,nb),self.basis,'a')
                    self.ops['BBbb'][(na,nb),(na,nb)] = build_ccaa_ss(self.n_orb, (na,nb),(na,nb),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 6: %12.2f" %(stop-start))



        #  ABba
        #  BAab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ABba'][(na,nb),(na,nb)] = build_ccaa_os(self.n_orb, (na,nb),(na,nb),self.basis,'abba')
                    self.ops['BAab'][(na,nb),(na,nb)] = build_ccaa_os(self.n_orb, (na,nb),(na,nb),self.basis,'baab')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 7: %12.2f" %(stop-start))
        """

        
        #  AA
        start = time.time()
        for na in range(2,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['aa'][(na-2,nb),(na,nb)] = build_aa_ss(self.n_orb, (na-2,nb),(na,nb),self.basis,'a')

                    temp = cp.deepcopy(np.swapaxes(self.ops['aa'][(na-2,nb),(na,nb)],0,1))
                    self.ops['AA'][(na,nb),(na-2,nb)] = cp.deepcopy(np.swapaxes(temp,2,3))
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 8: %12.2f" %(stop-start))

        # BB
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(2,self.n_orb+1):
                try:
                    self.ops['bb'][(na,nb-2),(na,nb)] = build_aa_ss(self.n_orb, (na,nb-2),(na,nb),self.basis,'b')

                    temp = cp.deepcopy(np.swapaxes(self.ops['bb'][(na,nb-2),(na,nb)],0,1))
                    self.ops['BB'][(na,nb),(na,nb-2)] = cp.deepcopy(np.swapaxes(temp,2,3))
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM 9: %12.2f" %(stop-start))



        # Ab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ab'][(na-1,nb-1),(na,nb)] = build_aa_os(self.n_orb, (na-1,nb-1),(na,nb),self.basis,'ab')
                    self.ops['ba'][(na-1,nb-1),(na,nb)] = build_aa_os(self.n_orb, (na-1,nb-1),(na,nb),self.basis,'ba')

                    temp = cp.deepcopy(np.swapaxes(self.ops['ba'][(na-1,nb-1),(na,nb)],0,1))
                    self.ops['AB'][(na,nb),(na-1,nb-1)] = cp.deepcopy(np.swapaxes(temp,2,3))

                    temp = cp.deepcopy(np.swapaxes(self.ops['ab'][(na-1,nb-1),(na,nb)],0,1))
                    self.ops['BA'][(na,nb),(na-1,nb-1)] = cp.deepcopy(np.swapaxes(temp,2,3))
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM10: %12.2f" %(stop-start))

        
               
        #  AAa #   have to fix the swapaxes
        start = time.time()
        for na in range(2,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['AAa'][(na,nb),(na-1,nb)] = build_cca_ss(self.n_orb, (na,nb),(na-1,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM11: %12.2f" %(stop-start))


        #  BBb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(2,self.n_orb+1):
                try:
                    self.ops['BBb'][(na,nb),(na,nb-1)] = build_cca_ss(self.n_orb, (na,nb),(na,nb-1),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM12: %12.2f" %(stop-start))


        #  ABb
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ABb'][(na,nb),(na-1,nb)] = build_cca_os(self.n_orb, (na,nb),(na-1,nb),self.basis,'abb')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM13: %12.2f" %(stop-start))

        #  BAa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['BAa'][(na,nb),(na,nb-1)] = build_cca_os(self.n_orb, (na,nb),(na,nb-1),self.basis,'baa')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM14: %12.2f" %(stop-start))


        #  ABa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ABa'][(na,nb),(na,nb-1)] = build_cca_os(self.n_orb, (na,nb),(na,nb-1),self.basis,'aba')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM15: %12.2f" %(stop-start))

        #  BAb
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['BAb'][(na,nb),(na-1,nb)] = build_cca_os(self.n_orb, (na,nb),(na-1,nb),self.basis,'bab')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM16: %12.2f" %(stop-start))


        #  Aaa
        start = time.time()
        for na in range(2,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['Aaa'][(na-1,nb),(na,nb)] = build_caa_ss(self.n_orb, (na-1,nb),(na,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM17: %12.2f" %(stop-start))

        
        #  Bbb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(2,self.n_orb+1):
                try:
                    self.ops['Bbb'][(na,nb-1),(na,nb)] = build_caa_ss(self.n_orb, (na,nb-1),(na,nb),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM18: %12.2f" %(stop-start))

        
        #  Aab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Aab'][(na,nb-1),(na,nb)] = build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aab')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM19: %12.2f" %(stop-start))


        #  Bba
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bba'][(na-1,nb),(na,nb)] = build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bba')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM20: %12.2f" %(stop-start))


        #  Aba
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Aba'][(na,nb-1),(na,nb)] = build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aba')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM21: %12.2f" %(stop-start))


        #  Bab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bab'][(na-1,nb),(na,nb)] = build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bab')
                except KeyError:
                    continue
        stop = time.time()
        if iprint>0:
            print(" Time spent TDM22: %12.2f" %(stop-start))

        

        if iprint>0:
            print(" Swapping axes to get contiguous data")
        start = time.time()
        for o in self.ops:
            for f in self.ops[o]:
                self.ops[o][f] = np.ascontiguousarray(self.ops[o][f])
                #self.ops[o][f] = np.ascontiguousarray(np.swapaxes(self.ops[o][f],0,1))
        stop = time.time()
        if iprint>0:
            print(" Time spent making data contiguous: %12.2f" %(stop-start))


        stop_tot = time.time()
        print(" Time spent building TDMs Total %12.2f" %(stop_tot-start_tot))
# }}}


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
    
