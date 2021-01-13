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
from pyscf_helper import *
import tools

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
        self.cs_solver = 0   # 0 for defaul,1 for pyscf

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

    def form_fockspace_eigbasis(self, hin, vin, spaces, max_roots=1000, rdm1_a=None, rdm1_b=None, ecore=0, iprint=0, subspace=False,cs_solver=0):
        """
        Get matrices of local Hamiltonians embedded in the system 1rdm and diagonalize to form
        the cluster basis vectors. This is the main step in CMF

        This function creates entries in the self.ops dictionary for self.ops['H_mf']
        Hlocal is the same operator but stored in the full determinant basis (we'll probably want to remove this soon)
        This data is used to obtain the cluster
        energies for an MP2 like correction rather than a EN correction.

        subspace    :   Should the cluster state rotations only mix states within the current cluster basis?
                        Generally this would be set to False. However, sometimes one might want to find the CMF basis
                        inside of an already truncated basis, i.e., doing cleaner MP2 like perturbations after doing
                        a schmidt-truncation

        """
# {{{
        for f in spaces:
            assert(len(f)==2)

        self.cs_solver = cs_solver

        if len(spaces) == 0:
            print(" No spaces requested - certainly a mistake")
            exit()

        print(" Build basis vectors for the following fockspaces:")
        for f in spaces:
            print("    Alpha:Beta = %4i %-4i" %(f[0],f[1]))

        if rdm1_a is None:
            rdm1_a = np.zeros_like(hin)
        if rdm1_b is None:
            rdm1_b = np.zeros_like(hin)

        Eenv,h,v = tools.build_1rdm_dressed_integrals(hin,vin,self.orb_list,rdm1_a,rdm1_b)
        print("ENVVV",Eenv)
        print()
        print(h)

        if iprint>0:
            print(self)

        #local integrals
        hc = np.zeros([self.n_orb]*2)
        vc = np.zeros([self.n_orb]*4)

        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                hc[pidx,qidx] = hin[p,q]

        for pidx,p in enumerate(self.orb_list):
            for qidx,q in enumerate(self.orb_list):
                for ridx,r in enumerate(self.orb_list):
                    for sidx,s in enumerate(self.orb_list):
                        vc[pidx,qidx,ridx,sidx] = vin[p,q,r,s]


        if self.cs_solver== 1:
            from pyscf import fci
            if subspace == True:
                print("Can't do with pyscf")
                exit()
            else:
                self.basis = {}
            self.ops['H_mf'] = {}
            self.ops['H'] = {}
            self.ops['Aa'] = {}
            self.ops['Bb'] = {}
            for na,nb in spaces:
                fock = (na,nb)
                norb = h.shape[0]
                
                #Hci = fci.direct_spin1.pspace(h, v, h.shape[0], fock, np=10000)[1]
                #l,C = np.linalg.eigh(Hci)
                #sort_ind = np.argsort(l)
                #l = l[sort_ind]
                #C = C[:,sort_ind]
                #C = C[:,:max_roots]
                #self.basis[fock] = C
                #self.ops['H_mf'][(fock,fock)] = C.T @ Hci @ C

                cisolver = fci.direct_spin0.FCI()
                cisolver.max_cycle = 200
                cisolver.conv_tol = 1e-13
                efci, ci = cisolver.kernel(h, v, norb, fock, ecore=0,nroots =1,verbose=100)
                fci_dim = ci.shape[0]*ci.shape[1]
                DA,DB = cisolver.make_rdm1s(ci, norb, fock)

                self.ops['Aa'][(na,nb),(na,nb)] = DA.reshape(1,1,DA.shape[0],DA.shape[1])
                self.ops['Bb'][(na,nb),(na,nb)] = DB.reshape(1,1,DA.shape[0],DA.shape[1])
                self.ops['H_mf'][(fock,fock)] = efci.reshape(1,1)

                #local
                h2eff = cisolver.absorb_h1e(hc, vc, norb, fock, .5)
                ci1 = cisolver.contract_2e(h2eff, ci, norb, fock)
                e = np.einsum('pq,pq',ci1,ci).reshape(1,1)
                self.ops['H'][(fock,fock)] = e
                print("FFCI%16.12f"%efci)

                self.basis[fock] = ci.reshape(ci.shape[0]*ci.shape[1],1)
                #print("fock",fock,self.ops['H'][(fock,fock)])

        if self.cs_solver== 0:
            H = Hamiltonian()
            H.S = np.eye(h.shape[0])
            H.C = H.S
            H.t = h
            H.V = v
            H.ecore = ecore
            self.ops['H_mf'] = {}

            if subspace == False:
                self.basis = {}

            self.Hlocal = {}
            for na,nb in spaces:
                #print(na,nb)
                fock = (na,nb)
                ci = ci_solver()
                ci.init(H,na,nb,max_roots)

                if subspace:
                    ci.algorithm = "direct" # this is probably not needed here
                    if fock not in self.basis:
                        continue
                    Hci = ci.build_H_matrix(self.basis[fock])
                    #print(" Nick: ", self.idx, fock, self.basis[fock].shape,flush=True)
                    if Hci.shape[0] > 1 and Hci.shape[0] > ci.n_roots:
                        l,C = scipy.sparse.linalg.eigsh(Hci, ci.n_roots,which='SA')
                        sort_ind = np.argsort(l)
                        l = l[sort_ind]
                        C = C[:,sort_ind]
                    elif Hci.shape[0] > 1 and Hci.shape[0] <= ci.n_roots:
                        l,C = np.linalg.eigh(Hci)
                        sort_ind = np.argsort(l)
                        l = l[sort_ind]
                        C = C[:,sort_ind]
                    elif Hci.shape[0] == 1:
                        l = [Hci[0,0]]
                        C = np.array([[1.0]])
                    else:
                        print(" Problem with Hci dimension")
                        exit()
                    self.basis[fock] = self.basis[fock] @ C
                    self.ops['H_mf'][(fock,fock)] = C.T @ Hci @ C 
                    #self.Hlocal[fock] =  Hci 

                else:
                    if ci.full_dim > 10000:
                        ci.thresh = 1e-5
                        ci.init(H,na,nb,1)
                        ci.algorithm = "davidson"
                    else:
                        ci.algorithm = "direct"
                    
                    
                    Hci = ci.run()
                    #Hci = ci.run(s2=True)
                    #self.basis[(na,nb)] = np.eye(ci.results_v.shape[0])
                    if iprint>0:
                        print(" %4ia %4ib"%(na,nb))
                        for i,ei in enumerate(ci.results_e):
                            print(" Local State %5i: Local E: %12.8f Embedded E: %12.8f Total E: %12.8f" %(i, ei, ei+Eenv, ei+ecore+Eenv))
                    fock = (na,nb)
                    
                    C = ci.results_v
                    self.basis[fock] = C
                    if ci.algorithm == "davidson":
                        #self.Hlocal[fock] =  Hci
                        sigma = Hci
                        self.ops['H_mf'][(fock,fock)] = C.T @ sigma
                    
                    
                    elif ci.algorithm == "direct":
                        self.Hlocal[fock] =  Hci
                        self.ops['H_mf'][(fock,fock)] = C.T @ Hci @ C
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
                Hci = ci.build_H_matrix(self.basis[fock])
                print(" Nick: ", self.idx, fock, self.basis[fock].shape,flush=True)
                if Hci.shape[0] > 1 and Hci.shape[0] > ci.n_roots:
                    l,C = scipy.sparse.linalg.eigsh(Hci, ci.n_roots,which='SA')
                    sort_ind = np.argsort(l)
                    l = l[sort_ind]
                    C = C[:,sort_ind]
                elif Hci.shape[0] > 1 and Hci.shape[0] <= ci.n_roots:
                    l,C = np.linalg.eigh(Hci)
                    sort_ind = np.argsort(l)
                    l = l[sort_ind]
                    C = C[:,sort_ind]
                elif Hci.shape[0] == 1:
                    l = [Hci[0,0]]
                    C = np.array([[1.0]])
                else:
                    print(" Problem with Hci dimension")
                    exit()
                self.basis[fock] = self.basis[fock] @ C
                self.ops['H_mf'][(fock,fock)] = C.T @ Hci @ C
                self.Hlocal[fock] =  Hci

            else:
                print(ci)
                Hci = ci.run()
                if iprint>0:
                    for i,ei in enumerate(ci.results_e):
                        print(" Local State %5i: Local E: %12.8f Embedded E: %12.8f Total E: %12.8f" %(i, ei, ei+Eenv, ei+ecore+Eenv))

                C = ci.results_v
                #if np.amax(np.abs(C.T@C - np.eye(C.shape[1]))) > 1e-14:
                #    S = C.T @ C
                #    S = scipy.linalg.inv( scipy.linalg.sqrtm(S))
                #    C = C@S
                #    print(np.amax(np.abs(C.T@C - np.eye(C.shape[1]))))
                #    assert(np.amax(np.abs(C.T@C - np.eye(C.shape[1]))) < 1e-14)
                self.basis[fock] = C
                self.Hlocal[fock] =  Hci
                self.ops['H_mf'][(fock,fock)] = C.T @ Hci @ C
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
                    o = self.ops[op][fspace_delta]
                    if len(o.shape) == 2:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,pr->qs',Ul,Ur,o, optimize=True)
                    elif len(o.shape) == 3:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,prt->qst',Ul,Ur,o, optimize=True)
                    elif len(o.shape) == 4:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,prtu->qstu',Ul,Ur,o, optimize=True)
                    elif len(o.shape) == 5:
                        self.ops[op][fspace_delta] = np.einsum('pq,rs,prtuv->qstuv',Ul,Ur,o, optimize=True)
                    else:
                        print("Wrong dimension")
                        assert(1==0)
                elif fspace_l in U and fspace_r not in U:
                    Ul = U[fspace_l]
                    o = self.ops[op][fspace_delta]
                    if len(o.shape) == 2:
                        self.ops[op][fspace_delta] = np.einsum('pq,ps->qs',Ul,o, optimize=True)
                    elif len(o.shape) == 3:
                        self.ops[op][fspace_delta] = np.einsum('pq,pst->qst',Ul,o, optimize=True)
                    elif len(o.shape) == 4:
                        self.ops[op][fspace_delta] = np.einsum('pq,pstu->qstu',Ul,o, optimize=True)
                    elif len(o.shape) == 5:
                        self.ops[op][fspace_delta] = np.einsum('pq,pstuv->qstuv',Ul,o, optimize=True)
                    else:
                        print("Wrong dimension")
                        assert(1==0)
                elif fspace_l not in U and fspace_r in U:
                    Ur = U[fspace_r]
                    o = self.ops[op][fspace_delta]
                    if len(o.shape) == 2:
                        self.ops[op][fspace_delta] = np.einsum('rs,qr->qs',Ur,o, optimize=True)
                    elif len(o.shape) == 3:
                        self.ops[op][fspace_delta] = np.einsum('rs,qrt->qst',Ur,o, optimize=True)
                    elif len(o.shape) == 4:
                        self.ops[op][fspace_delta] = np.einsum('rs,qrtu->qstu',Ur,o, optimize=True)
                    elif len(o.shape) == 5:
                        self.ops[op][fspace_delta] = np.einsum('rs,qrtuv->qstuv',Ur,o, optimize=True)
                    else:
                        print("Wrong dimension")
                        assert(1==0)
                    #try:
                    #    self.ops[op][fspace_delta] = np.einsum('pq,rs,pr...->qs...',Ul,Ur,self.ops[op][fspace_delta], optimize=True)
                    #except ValueError:
                    #    print("Error: Rotate basis failed for term: ", op, " fspace_delta: ", fspace_delta)
                    #    print(Ul.shape)
                    #    print(Ur.shape)
                    #    print(self.ops[op][fspace_delta].shape)
                    #    exit()
   # }}}

    def check_basis_orthogonality(self,thresh=1e-14):
        for fspace,mat in self.basis.items():
            I = mat.T @ mat
            try:
                if mat.shape[1] > 0:
                    assert(np.amax(np.abs(I - np.eye(I.shape[0]))) < thresh)
            except AssertionError:
                print(" WARNING: in check_basis:")
                print(" Cluster:", self)
                print(" Fockspace:", fspace)
                print(" Orthogonality Error: ",np.amax(np.abs(I - np.eye(I.shape[0]))))

    #NYI
    def grow_basis_by_coupling(self, rdms=None):
       # {{{
        print("NYI")
        exit()
        #  Aa
        start = time.time()
        for fock in self.basis:
            v1 = self.basis[fock]

            dimX = v1.shape[0] - v1.shape[1] # how many vectors can we add?
            if dimX == 0:
                continue

            # get basis for the orthogonal compliment to our current space
            v1 = np.eye(v1.shape[0]) - v1 @ v1.T
            v1,s1,u1 = np.linalg.svd(v1)
            v1 = v1[:,:dimX]
            basis1 = {}
            basis1[fock] = v1

            print()
            print(self.basis[fock].shape)
            print(v1.shape)

            GIipq = build_ca_ss_tdm(self.n_orb, fock, fock, self.basis, basis1, 'a')

#            if rdms != None:
#                #first contract with density matrix diagonal
#                print(rdms[fock])
#                GIpq = np.einsum('Iipq,i->Ipq', GIipq, rdms[fock])
#                G = np.einsum('Iipq,Jipq->IJ', GIipq, GIipq)
#
#            else:
#                G = np.einsum('Iipq,Jipq->IJ', GIipq, GIipq)
#            print(G.shape)

            G = np.einsum('Iipq,Jipq->IJ', GIipq, GIipq)
            print(G.shape)

            l,U = np.linalg.eigh(G)
            idx = l.argsort()
            l = l[idx]
            U = U[:,idx]
            for ii,i in enumerate(l):
                print(" %4i %12.8f"%(ii+1,i))
# }}}


    def grow_basis_by_energy(self, hin, vin, max_roots=None, max_energy=None, rdm1_a=None, rdm1_b=None):
        """
        Grow the cluster basis for each fock space currently present

            this works by building local hamiltonian, H (or mean-field H if rdm1_a/b are provided)
            in the orthogonal complement of the current space, then diagonalizes this, and then adds the rest of the space
            allowed by limitations imposed by max_roots or max_energy.

            max_roots   : maximum number of new vectors to add
            max_energy  : max energy of new vectors to add
        """
        # {{{
        start = time.time()

        if rdm1_a is None:
            rdm1_a = np.zeros(hin.shape)
        if rdm1_b is None:
            rdm1_b = np.zeros(hin.shape)

        Eenv,h,v = tools.build_1rdm_dressed_integrals(hin,vin,self.orb_list,rdm1_a,rdm1_b)

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

            v1 = self.basis[fock]

            dimX = v1.shape[0] - v1.shape[1] # how many vectors can we add?
            if dimX == 0:
                continue

            # get basis for the orthogonal compliment to our current space
            v1 = np.eye(v1.shape[0]) - v1 @ v1.T
            v1,s1,u1 = np.linalg.svd(v1)
            v1 = v1[:,:dimX]
            s1 = s1[:dimX]


            #print()

            #HX = v1.T @ self.Hlocal[fock] @ v1
            HX = ci.build_H_matrix(v1)

            l,U = np.linalg.eigh(HX)
            idx = l.argsort()
            l = l[idx]
            U = U[:,idx]

            if max_roots != None:
                if max_roots < len(l):
                    assert(max_roots>=0)
                    l = l[:max_roots]
                    U = U[:,:max_roots]

            if max_energy != None:
                nkeep = 0
                for li in range(len(l)):
                    if l[li] <= max_energy:
                        nkeep += 1
                l = l[:nkeep]
                U = U[:,:nkeep]

            U = v1@U


            v2 = np.hstack((self.basis[fock],U))
            self.basis[fock] = v2

            assert(np.amax(v2.T @ v2 - np.eye(v2.shape[1])) < 1e-14)

        # since we have made all the operators  invalid - remove the data so it will trigger
        # an error if we try to use it before rebuilding
        self.ops = {}

            # The mean-field hamiltonian (H_mf) lives in the cluster basis while
            # the same operator (Hlocal) lives in the determinant basis
            # this doesn't get updated with all the other operators, so update this quantity here
            #self.ops['H_mf'][(fock,fock)] = v2.T @ self.Hlocal[fock] @ v2

            #print(U.shape)
            #for ii,i in enumerate(l):
            #    print(" %4i %12.8f"%(ii+1,i))
# }}}


    def get_ops(self):
        return self.ops

    def get_op(self,opstr):
        return self.ops[opstr]


    def get_op_mel(self,opstr,fI,fJ,I,J):
        return self.ops[opstr][(fI,fJ)][I,J,:]


    def build_local_terms(self,hin,vin):
        start = time.time()
        #self.ops['H'] = {}
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

        if self.cs_solver == 1:
            # already done in fockspace build
            htemp = 1


            

        if self.cs_solver == 0:
            self.ops['H'] = {}
            H = Hamiltonian()
            H.S = np.eye(h.shape[0])
            H.C = H.S
            H.t = h
            H.V = v

            for fock in self.basis:

                ci = ci_solver()
                na = fock[0]
                nb = fock[1]
                ci.init(H,na,nb,1)
                if ci.full_dim > 10000:
                    ci.thresh = 1e-5
                    ci.algorithm = "davidson"
                    self.ops['H'][(fock,fock)] = ci.build_H_matrix(self.basis[fock])
                    self.ops['H'][(fock,fock)] = .5*(self.ops['H'][(fock,fock)] + self.ops['H'][(fock,fock)].T)
                else:
                    ci.algorithm = "direct"
                    #print(ci)
                    self.ops['H'][(fock,fock)] = ci.build_H_matrix(self.basis[fock])
                    self.ops['H'][(fock,fock)] = .5*(self.ops['H'][(fock,fock)] + self.ops['H'][(fock,fock)].T)
                #print(" GS Energy: %12.8f" %self.ops['H'][(fock,fock)][0,0])
                #print("fock",fock,Hci)

        stop = time.time()
        print(" Time spent TDM 0: %12.2f" %(stop-start))
    # }}}


    def build_effective_cmf_hamiltonian(self,hin,vin,rdm1_a,rdm1_b):
        """
        grab integrals acting locally and form precontracted operator in current eigenbasis
        """
        start = time.time()
        self.ops['H_mf'] = {}
        # {{{
        Eenv,h,v = tools.build_1rdm_dressed_integrals(hin,vin,self.orb_list,rdm1_a,rdm1_b)

        H = Hamiltonian()
        H.S = np.eye(h.shape[0])
        H.C = H.S
        H.t = h
        H.V = v

        for fock in self.basis:
            ci = ci_solver()
            na = fock[0]
            nb = fock[1]
            ci.init(H,na,nb,1)
            self.ops['H_mf'][(fock,fock)] = ci.build_H_matrix(self.basis[fock])
        # }}}


        stop = time.time()
        print(" Time spent TDM 0: %12.2f" %(stop-start))


    def build_op_matrices(self, iprint=0):
        """
        build all operators needed
        """
# {{{

        assert(self.cs_solver == 0)
        start = time.time()
        self.ops['A'] = {}
        self.ops['a'] = {}
        self.ops['B'] = {}
        self.ops['b'] = {}
        self.ops['Aa'] = {}
        self.ops['Bb'] = {}
        self.ops['Ab'] = {}
        self.ops['Ba'] = {}
        #self.ops['tAAaa'] = {}
        #self.ops['tBBbb'] = {}
        #self.ops['tABba'] = {}
        #self.ops['tBAab'] = {}
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
        self.ops['Aaa'] = {}
        self.ops['Bbb'] = {}
        self.ops['ABb'] = {}
        self.ops['BAa'] = {}
        self.ops['BAb'] = {}
        self.ops['ABa'] = {}
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
                    self.ops['Aab'][(na,nb-1),(na,nb)] = 0*build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aab')
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
                    self.ops['Bba'][(na-1,nb),(na,nb)] = 0*build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bba')
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
                    self.ops['Aba'][(na,nb-1),(na,nb)] = 0*build_caa_os(self.n_orb, (na,nb-1),(na,nb),self.basis,'aba')
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
                    self.ops['Bab'][(na-1,nb),(na,nb)] = 0*build_caa_os(self.n_orb, (na-1,nb),(na,nb),self.basis,'bab')
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


        self.ops['AAa'] = {}
        self.ops['BBb'] = {}
        self.ops['Aaa'] = {}
        self.ops['Bbb'] = {}
        self.ops['ABb'] = {}
        self.ops['BAa'] = {}
        self.ops['BAb'] = {}
        self.ops['ABa'] = {}
        self.ops['Aab'] = {}
        self.ops['Bba'] = {}
        self.ops['Aba'] = {}
        self.ops['Bab'] = {}
        stop_tot = time.time()
        print(" Time spent building TDMs Total %12.2f" %(stop_tot-start_tot))
# }}}

    def build_op_matrices_cmf(self, iprint=0):
        """
        build all operators needed
        """
# {{{
        if self.cs_solver == 0:
            self.ops['Aa'] = {}
            self.ops['Bb'] = {}

            fspace = list(self.basis.keys())[0]
            vec =    list(self.basis.values())[0]


            start_tot = time.time()

            #  Aa
            start = time.time()
            self.ops['Aa'][fspace,fspace] = build_ca_ss(self.n_orb, fspace,fspace,self.basis,'a')
            stop = time.time()
            if iprint>0:
                print(" Time spent TDM 3: %12.2f" %(stop-start))

            #  Bb
            start = time.time()
            self.ops['Bb'][fspace,fspace] = build_ca_ss(self.n_orb, fspace,fspace,self.basis,'b')
            stop = time.time()
            if iprint>0:
                print(" Time spent TDM 4: %12.2f" %(stop-start))



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


        if self.cs_solver == 1:
            # already done in fockspace code
            htemp = 1

# }}}

    def form_schmidt_basis(self,h,g,Da,Db,thresh_orb=1e-8,thresh_schmidt=1e-3,thresh_ci=1e-5,iprint=1,ecore=0,do_embedding=True):
        """
        thresh_orb      :   threshold for determining how many bath orbitals to include
        thresh_schmidt  :   threshold for determining how many singular vectors to include for cluster basis
        """
        # {{{
        print()
        print(" Form Schmidt-style basis for ", self)
        D = Da+Db
        #K = np.einsum('pqrs,rs->pq',g,D)
        K = np.einsum('pqrs,ps->qr',g,D)

        na_tot = int(round(np.trace(Da)))
        nb_tot = int(round(np.trace(Db)))
        active = self.orb_list
        backgr = [i for i in range(h.shape[0]) if i not in active ]

        print(" Active:", active)
        print(" Backgr:", backgr)
        K2 = np.zeros_like(K)
        K2 = np.zeros((self.n_orb, D.shape[0]-self.n_orb))

        for pi,p in enumerate(active):
            for qi,q in enumerate(backgr):
                K2[pi,qi] = K[p,q]

        print("K2")
        print(K2)
        U,s,V = np.linalg.svd(K2)
        print("  %16s"%("Sing. Val."))
        nkeep = 0
        for si in s:
            if si > thresh_orb:
                nkeep += 1
                print("  %16.12f"%si)
            else:
                print("  %16.12f*"%si)
        C = np.zeros_like(h)
        for pi,p in enumerate(active):
            for qi,q in enumerate(active):
                #C[p,qi] = U[pi,qi]
                if pi==qi:
                    C[p,qi] = 1
        for pi,p in enumerate(backgr):
            for qi,q in enumerate(backgr):
                C[p,qi+len(active)] = V[qi,pi]

        Cfrag = C[:,0:self.n_orb]
        Cbath = C[:,self.n_orb:self.n_orb+nkeep]
        Cenvt = C[:,self.n_orb+nkeep:]

        # add these to the class so we can visualize them later
        self.C_schmidt = C

        print("Cfrag")
        print(Cfrag)
        print(" NElec: %12.8f"%(np.trace(Cfrag.T@(Da+Db)@Cfrag)))
        print("Cbath")
        print(Cbath)
        print(" NElec: %12.8f"%(np.trace(Cbath.T@(Da+Db)@Cbath)))
        print("Cenv")
        print(Cenvt)
        print(" NElec: %12.8f"%(np.trace(Cenvt.T@(Da+Db)@Cenvt)))
        #print("U")
        #print(U)
        #print("V")
        #print(V)
        #print("C")
        #print(C)
        #print("K")
        #print(K)
        K2 = C.T @ K @ C
        #print("K2")
        #print(K2)
        Da2 = C.T @ Da @ C
        Db2 = C.T @ Db @ C
        na = np.trace(Da2[0:self.n_orb+nkeep,0:self.n_orb+nkeep])
        nb = np.trace(Db2[0:self.n_orb+nkeep,0:self.n_orb+nkeep])

        print(" Number of electrons in Fragment+Bath system:")
        print("   Alpha: %12.8f"%(na))
        print("   Beta : %12.8f"%(nb))

        # Modify environment density to have integer number of electrons
        #nelec_envt = round(np.trace(Cenv.T@(Da+Db)@Cenv))
        #print(" Change in electron count in environment: %12.8f"%(nelec_envt - np.trace(Cenv.T@(Da+Db)@Cenv)))
        #print(" Create embedding potential using %3i electrons."%nelec_envt)
        denvt_a = Cenvt@Cenvt.T @ Da @ Cenvt@ Cenvt.T
        denvt_b = Cenvt@Cenvt.T @ Db @ Cenvt@ Cenvt.T
        na_envt = np.trace(denvt_a)
        nb_envt = np.trace(denvt_b)
        print(" Number of electrons in Environment system:")
        print("   Alpha: %12.8f"%(na_envt))
        print("   Beta : %12.8f"%(nb_envt))

        na_envt = int(round(np.trace(Cenvt.T@(Da)@Cenvt)))
        nb_envt = int(round(np.trace(Cenvt.T@(Db)@Cenvt)))
        #print(" Change in electron count in environment: %12.8f"%(nelec_envt - np.trace(Cenv.T@(Da+Db)@Cenv)))
        na_actv = self
        print(" Create embedding potential using %3i electrons."%(na_envt+nb_envt))


        # rotate integrals to current subspace basis
        denvt_a = C.T @ denvt_a @ C
        denvt_b = C.T @ denvt_b @ C
        h2 = C.T @ h @ C
        g2 = np.einsum("pqrs,pl->lqrs",g,C)
        g2 = np.einsum("lqrs,qm->lmrs",g2,C)
        g2 = np.einsum("lmrs,rn->lmns",g2,C)
        g2 = np.einsum("lmns,so->lmno",g2,C)

        # find closest idempotent density for the environment
        if do_embedding:
            if Cenvt.shape[1]>0:
                #eigenvalue
                n,U = np.linalg.eigh(denvt_a)
                idx = n.argsort()[::-1]
                n = n[idx]
                U = U[:,idx]
                print(n)
                #we shouldn't ever have zero eigevanlues in our keep space
                for i in range(nkeep):
                    assert(n[i]>1e-14)
                denvt_a = U[:,0:na_envt] @ U[:,0:na_envt].T

                n,U = np.linalg.eigh(denvt_b)
                idx = n.argsort()[::-1]
                n = n[idx]
                U = U[:,idx]

                #we shouldn't ever have zero eigevanlues in our keep space
                for i in range(nkeep):
                    assert(n[i]>1e-14)
                denvt_b = U[:,0:nb_envt] @ U[:,0:nb_envt].T

            Eenv,h2,g2 = tools.build_1rdm_dressed_integrals(h2,g2,range(Cfrag.shape[1]+Cbath.shape[1]),denvt_a,denvt_b)
        else:
            denvt_a *= 0
            denvt_b *= 0
            Eenv,h2,g2 = tools.build_1rdm_dressed_integrals(h2,g2,range(Cfrag.shape[1]+Cbath.shape[1]),denvt_a,denvt_b)


        print(" Number of electrons in Environment system:")
        print("   Alpha: %12.8f"%(np.trace(denvt_a)))
        print("   Beta : %12.8f"%(np.trace(denvt_b)))

        na_actv = na_tot - na_envt
        nb_actv = nb_tot - nb_envt
        print(" Number of electrons in Fragment+Bath:")
        print("   Alpha: %12i"%(na_actv))
        print("   Beta : %12i"%(na_actv))


        H = Hamiltonian()
        H.S = np.eye(h2.shape[0])
        H.C = H.S
        H.t = h2
        H.V = g2

        ci = ci_solver()
        ci.init(H,na_actv,nb_actv,1)
        ci.thresh = thresh_ci
        print(ci,flush=True)
        if ci.full_dim > 10000:
            ci.algorithm = "davidson"
        else:
            ci.algorithm = "direct"
        Hci = ci.run()

        if iprint>0:
            for i,ei in enumerate(ci.results_e):
                print(" Local State %5i: Local E: %12.8f Embedded E: %12.8f Total E: %12.8f" %(i, ei, ei+Eenv, ei+ecore+Eenv))

        # since we have made all the operators  invalid - remove the data so it will trigger
        # an error if we try to use it before rebuilding
        self.ops = {}

        self.basis = ci.svd_state(len(active),nkeep, thresh=thresh_schmidt)
        if iprint>0:
            print(" We will have these fock spaces present")
            for na,nb in self.basis:
                print(na,nb)
        # }}}


    def split(self,fock,n_orbs1,h,g,thresh_schmidt=0):
        """
        Split cluster into two new clusters, using the SVD of the ground state (of fock space sector = fock) as the new cluster states,
        then build the operators
        """
        assert(n_orbs1<self.n_orb)
        n_orbs2 = self.n_orb-n_orbs1
        H = Hamiltonian()
        H.C = np.eye(self.n_orb)
        ci = ci_solver()
        ci.init(H,fock[0],fock[1],1)
        ci.results_v = self.basis[fock][:,0]

        c1 = Cluster(-1,self.orb_list[:n_orbs1])
        c2 = Cluster(-1,self.orb_list[n_orbs1:])
        print(c1)
        print(c2)
        c1.basis, c2.basis = ci.svd_state(n_orbs1, n_orbs2, thresh=thresh_schmidt, both=True)
        # build cluster basis and operator matrices using CMF optimized density matrices
        #c1.build_op_matrices()
        #c1.build_local_terms(h,g)
        #c2.build_op_matrices()
        #c2.build_local_terms(h,g)
        return c1, c2



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
