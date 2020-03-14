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
        self.Hci = {}

    def __len__(self):
        return len(self.orb_list) 
    def __str__(self):
        has = ""
        for si in self.orb_list:
            has = has + "%03i|"%si
        if len(self.orb_list) == 0:
            has = "|"
        return "IDX%03i:DIM%05i:%s" %(self.idx, self.dim, has)

    def add_operator(self,op):
        if op in self.ops:
            return
        else:
            self.ops[op] = {}
    
    def form_eigbasis_from_ints(self,hin,vin,max_roots=1000, rdm1_a=None, rdm1_b=None, ecore=0):
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
           
            print(" 1 particle potential from environment")
            print_mat(f)
#            e = 0
#            n_alpha = int(np.trace(d))
#            n_beta = int(np.trace(d))
#            for i in range(n_alpha):
#                e += hin[i,i]
#            for i in range(n_beta):
#                e += hin[i,i]
#            for i in range(n_alpha):
#                for j in range(i,n_alpha):
#                    e += vin[i,i,j,j] - vin[i,j,j,i]
#            for i in range(n_beta):
#                for j in range(i,n_beta):
#                    e += vin[i,i,j,j] - vin[i,j,j,i]
#            for i in range(n_alpha):
#                for j in range(n_beta):
#                    e += vin[i,i,j,j] 
#            
#            print(" E:       %12.8f" %(e))
#            print(" E+ecore: %12.8f" %(e+ecore))

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h + f
        H.V = v
        self.basis = {}
        print(" Do CI for each particle number block")
        for na in range(self.n_orb+1):
            for nb in range(self.n_orb+1):
                ci = ci_solver()
                ci.algorithm = "direct"
                ci.init(H,na,nb,max_roots)
                print(ci)
                Hci = ci.run()
                #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                self.basis[(na,nb)] = ci.results_v
                self.Hci[(na,nb)] = ci.results_v.T @ Hci @ ci.results_v
    # }}}
    
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
                self.Hci[(na,nb)] = Hci
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
                    self.Hci[(na,nb)] = Hci
# }}}
                    
            
    def rotate_basis(self,U):
        """
        Rotate cluster's basis using U, which is an dictionary mapping fock spaces to unitary rotation matrices.
        rotate basis, and all associated operators
        """
# {{{

        for fspace,mat in U.items():
            np.testing.assert_allclose(mat.T @ mat, np.eye(mat.shape[1]), atol=1e-14)

        for fspace,mat in U.items():
            if mat.shape[1] == 0:
                #print(" Delete fockspace: ", fspace, " from cluster:", self.idx)
                del self.basis[fspace]
                try:
                    del self.Hci[fspace]
                except KeyError:
                    pass
            else:
                self.basis[fspace] = self.basis[fspace] @ mat
                try:
                    self.Hci[fspace] = mat.T @ self.Hci[fspace] @ mat 
                except KeyError:
                    pass

        for op,fspace_deltas in self.ops.items():
            to_remove = []
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
                    if Ul.shape[1] == 0 or Ur.shape[1]== 0:
                        to_remove.append(fspace_delta)
                    else:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,pr...->qs...',Ul,Ur,self.ops[op][fspace_delta], optimize=True)
                elif fspace_l in U and fspace_r not in U:
                    Ul = U[fspace_l]
                    if Ul.shape[1] == 0:
                        to_remove.append(fspace_delta)
                    else:
                        self.ops[op][fspace_delta] = np.einsum('pq,pr...->qr...',Ul,self.ops[op][fspace_delta], optimize=True)
                elif fspace_l not in U and fspace_r in U:
                    Ur = U[fspace_r]
                    if Ur.shape[1]== 0:
                        to_remove.append(fspace_delta)
                    else:
                        self.ops[op][fspace_delta] = np.einsum('rs,pr...->ps...',Ur,self.ops[op][fspace_delta], optimize=True)
            for f_trash in to_remove:
                #print(" Delete fockspace transition: ", f_trash, " from operator", op, " on cluster:", self.idx)
                del self.ops[op][f_trash]
    # }}}
    
    
    def get_ops(self):
        return self.ops

    def get_op(self,opstr):
        return self.ops[opstr]

    
    def get_op_mel(self,opstr,fI,fJ,I,J):
        return self.ops[opstr][(fI,fJ)][I,J,:]


    def build_op_matrices(self):
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
        self.ops['AAaa'] = {}
        self.ops['BBbb'] = {}
        self.ops['ABba'] = {}
        self.ops['BAab'] = {}
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
        print(" Time spent TDM 1: %12.2f" %(stop-start), flush=True)

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
        print(" Time spent TDM 2: %12.2f" %(stop-start), flush=True)

        #  Aa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['Aa'][(na,nb),(na,nb)] = build_ca_ss(self.n_orb, (na,nb),(na,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM 3: %12.2f" %(stop-start), flush=True)
        
        #  Bb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bb'][(na,nb),(na,nb)] = build_ca_ss(self.n_orb, (na,nb),(na,nb),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM 4: %12.2f" %(stop-start), flush=True)

               


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
        print(" Time spent TDM 5: %12.2f" %(stop-start), flush=True)

        
               
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
        print(" Time spent TDM 6: %12.2f" %(stop-start), flush=True)



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
        print(" Time spent TDM 7: %12.2f" %(stop-start), flush=True)


        
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
        print(" Time spent TDM 8: %12.2f" %(stop-start), flush=True)

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
        print(" Time spent TDM 9: %12.2f" %(stop-start), flush=True)



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
        print(" Time spent TDM10: %12.2f" %(stop-start), flush=True)

        
               
        #  AAa #   have to fix the swapaxes
        start = time.time()
        for na in range(2,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['AAa'][(na,nb),(na-1,nb)] = build_cca_ss(self.n_orb, (na,nb),(na-1,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM11: %12.2f" %(stop-start), flush=True)


        #  BBb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(2,self.n_orb+1):
                try:
                    self.ops['BBb'][(na,nb),(na,nb-1)] = build_cca_ss(self.n_orb, (na,nb),(na,nb-1),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM12: %12.2f" %(stop-start), flush=True)


        #  ABb
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ABb'][(na,nb),(na-1,nb)] = build_cca_os(self.n_orb, (na,nb),(na-1,nb),self.basis,'abb')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM13: %12.2f" %(stop-start), flush=True)

        #  BAa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['BAa'][(na,nb),(na,nb-1)] = build_cca_os(self.n_orb, (na,nb),(na,nb-1),self.basis,'baa')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM14: %12.2f" %(stop-start), flush=True)


        #  ABa
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['ABa'][(na,nb),(na,nb-1)] = build_cca_os(self.n_orb, (na,nb),(na,nb-1),self.basis,'aba')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM15: %12.2f" %(stop-start), flush=True)

        #  BAb
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['BAb'][(na,nb),(na-1,nb)] = build_cca_os(self.n_orb, (na,nb),(na-1,nb),self.basis,'bab')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM16: %12.2f" %(stop-start), flush=True)


        #  Aaa
        start = time.time()
        for na in range(2,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                try:
                    self.ops['Aaa'][(na-1,nb),(na,nb)] = build_caa_ss(self.n_orb, (na-1,nb),(na,nb),self.basis,'a')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM17: %12.2f" %(stop-start), flush=True)

        
        #  Bbb
        start = time.time()
        for na in range(0,self.n_orb+1):
            for nb in range(2,self.n_orb+1):
                try:
                    self.ops['Bbb'][(na,nb-1),(na,nb)] = build_caa_ss(self.n_orb, (na,nb-1),(na,nb),self.basis,'b')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM18: %12.2f" %(stop-start), flush=True)

        
        #  Aab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Aab'][(na,nb-1),(na,nb)] = build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aab')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM19: %12.2f" %(stop-start), flush=True)


        #  Bba
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bba'][(na-1,nb),(na,nb)] = build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bba')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM20: %12.2f" %(stop-start), flush=True)


        #  Aba
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Aba'][(na,nb-1),(na,nb)] = build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aba')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM21: %12.2f" %(stop-start), flush=True)


        #  Bab
        start = time.time()
        for na in range(1,self.n_orb+1):
            for nb in range(1,self.n_orb+1):
                try:
                    self.ops['Bab'][(na-1,nb),(na,nb)] = build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bab')
                except KeyError:
                    continue
        stop = time.time()
        print(" Time spent TDM22: %12.2f" %(stop-start), flush=True)

        

        print(" Swapping axes to get contiguous data")
        start = time.time()
        for o in self.ops:
            for f in self.ops[o]:
                self.ops[o][f] = np.ascontiguousarray(self.ops[o][f])
                #self.ops[o][f] = np.ascontiguousarray(np.swapaxes(self.ops[o][f],0,1))
        stop = time.time()
        print(" Time spent making data contiguous: %12.2f" %(stop-start), flush=True)

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


###################################################################################################################
#           Helper functions for Cluster objects
###################################################################################################################


def join_bases(ci,cj):
    """
    Take two clusters, ci and cj as input and return a child cluster cij, containing all the orbitals, 
    and with a set of basis vectors comprised as direct product of the parent clusters.

    The the new cluster's hilbert space spans the exact same space as the original two clusters.
    """
# {{{

    def calc_linear_index(config, no):
        """
        Return linear index for lexically ordered __config string
        """
        
        lin_index = 0
        v_prev = -1
    
        for i in range(len(config)):
            v = config[i]
            M = no - v
            N = len(config) - i - 1
            w = v - v_prev - 1
            #todo: change mchn from function call to data lookup!
            for j in range(0,w):
                lin_index += calc_nchk(M+j,N)
            v_prev = v
        return lin_index
    print(" Join the basis vectors for clusters: %4i and %-4i " %(ci.idx,cj.idx),flush=True)
    tmp = []
    tmp.extend(ci.orb_list)
    tmp.extend(cj.orb_list)
    cij = Cluster(0,tmp)
    no = ci.n_orb + cj.n_orb
    dimer_map = dict()
    start = time.time()
    for fi,vi in ci.basis.items():
        #print("a",fi)
        for fj,vj in cj.basis.items():
            ket_aj = ci_string(cj.n_orb, fj[0])
            ket_bj = ci_string(cj.n_orb, fj[1])
            #print(" Combine the following fock spaces: %6i x %-6i = %8i: " %(vi.shape[1], vj.shape[1],vi.shape[1]*vj.shape[1] ), fi, fj, flush=True)
            for fii in fi:
                for fjj in fj:
                    if (fii,fjj) in dimer_map.keys():
                        continue
                    ket_i = ci_string(ci.n_orb, fii)
                    ket_j = ci_string(cj.n_orb, fjj)
                    
                    dimer_map[(fii,fjj)] = np.zeros((ket_i.max(), ket_j.max()),dtype=int)

                    ket_i.reset()
                    for I in range(ket_i.max()): 
                        ket_j.reset()
                        for J in range(ket_j.max()): 
                            new_conf = []
                            new_conf.extend(ket_i.config())
                            new_conf.extend([ci.n_orb + i for i in ket_j.config()])
                            #new_conf.append(ket_aj.config())
                            #print(ket_ai.max(), ket_aj.max(), new_conf)
                            #print("d      ", new_conf)
                            #print(fi, fj, I, J, new_conf, ' -> ', calc_linear_index(new_conf, no))
                            dimer_map[(fii,fjj)][I,J] = calc_linear_index(new_conf, no)
                            ket_j.incr()
                        ket_i.incr()
    stop = time.time()
    print(" Time spent on %-20s: %12.2f" %('dimer_map', (stop-start)))
    #for f in dimer_map:
    #    print(f)
    #    print_mat(dimer_map[f])

    start = time.time()
    dimer_dim = 0
    for fi,vi in ci.basis.items():

        Nia = calc_nchk(ci.n_orb,fi[0])
        Nib = calc_nchk(ci.n_orb,fi[1])
        for fj,vj in cj.basis.items():

            print(" Combine the following fock spaces: %6i x %-6i = %8i: " %(vi.shape[1], vj.shape[1],vi.shape[1]*vj.shape[1] ), fi, fj, flush=True)

            Nja = calc_nchk(cj.n_orb,fj[0])
            Njb = calc_nchk(cj.n_orb,fj[1])
            
            #print(fi,fj)
            vij = np.kron(vi,vj)
            

            Nija = calc_nchk(no,fi[0]+fj[0])
            Nijb = calc_nchk(no,fi[1]+fj[1])
            Nij = Nija * Nijb 
            Mij = vi.shape[1] * vj.shape[1] 
            Vij = np.zeros((Nij, Mij))

            print("  ", vi.shape, "x", vj.shape, " = ", vij.shape, " --> ", Vij.shape, flush=True)

            assert(vij.shape[0] == Nia*Nib*Nja*Njb)
            assert(Vij.shape[0] == Nija*Nijb)

            for ia in range(Nia):
                for ib in range(Nib):
                    iab = ib + ia*Nib
                    
                    for ja in range(Nja):
                        for jb in range(Njb):
                            jab = jb + ja*Njb
                           
                            Ia =  dimer_map[(fi[0],fj[0])][ia,ja]
                            Ib =  dimer_map[(fi[1],fj[1])][ib,jb]
                       
                            #print(" Below:")
                            #print(Ia,Ib,Ib+Ia*Nijb, Nija*Nijb, Vij.shape)
                            #print(Vij[Ib + Ia*Nijb,:])
                            #print(iab,jab,jab+iab*Nja*Njb, Nia*Nib*Nja*Njb, vij.shape)
                            #print(vij[iab + jab*Nia*Nib,:]) 
                            Vij[Ib + Ia*Nijb,:] = vij[jab + iab*Nja*Njb,:]
                            #Vij[Ib + Ia*Nijb,:] = vij[iab + jab*Nia*Nib,:]

            fij = (fi[0]+fj[0], fi[1]+fj[1])
            if fij in cij.basis.keys():
                cij.basis[fij] = np.hstack((cij.basis[fij], Vij))
            else:
                cij.basis[fij] = Vij
            dimer_dim += vij.shape[1]
    
    stop = time.time()
    print(" Time spent on %-20s: %12.2f" %('join basis', (stop-start)), flush=True)

    #for f in cij.basis:
    #    print(f)
    #    #print_mat(cij.basis[f].T @ cij.basis[f])
    #    print_mat(cij.basis[f])

    return cij
# }}}
