import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys


def run_fci_pyscf( h, g, nelec, ecore=0):
# {{{
    # FCI
    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
    cisolver = fci.direct_spin1.FCI()
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,verbose=4)
    fci_dim = ci.shape[0]*ci.shape[1]
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    print("FCI %10.8f"%(efci))
    return efci,fci_dim
# }}}

def run_hci_pyscf( h, g, nelec, ecore=0, select_cutoff=5e-4, ci_cutoff=5e-4):
# {{{
    #heat bath ci
    from pyscf import mcscf
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = ci_cutoff
    ehci, civec = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,verbose=4)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    print("HCI %10.8f"%(ehci))
    return ehci,hci_dim
# }}}

def init_pyscf(molecule,charge,spin,basis_set,orb_basis='scf',cas=False,cas_nstart=None,cas_nstop=None,cas_nel=None):
# {{{
    from pyscf import gto, scf, ao2mo, molden, lo
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                      Using Pyscf:")
    print(" ---------------------------------------------------------")
    print("                                                          ")

    mol = gto.Mole()
    mol.atom = molecule

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis_set
    mol.build()


    #SCF 
    mf = scf.RHF(mol).run()
    #C = mf.mo_coeff #MO coeffs
    enu = mf.energy_nuc()

    #orbitals and lectrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b
    

    if cas == True:
        cas_norb = cas_nstop - cas_nstart
        from pyscf import mcscf
        assert(cas_nstart != None)
        assert(cas_nstop != None)
        assert(cas_nel != None)
    else:
        cas_nstart = 0
        cas_nstop = n_orb
        cas_nel = nel


    ##READING INTEGRALS FROM PYSCF
    E_nu = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nu)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nu))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))



    ##AO 2 MO Transformation: orb_basis or scf
    if orb_basis == 'scf':
        print("\nUsing Canonical Hartree Fock orbitals...\n")
        C = cp.deepcopy(mf.mo_coeff)

    elif orb_basis == 'boys_no_mix':
        print("\nUsing localised orbitals...\n")
        Cl = lo.orth.orth_ao(mol)
        #Cl = lo.boys(mol).kernel(mf.mo_coeff[:,0:4], verbose=4)
        #cl_o = lo.Boys(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        #cl_v = lo.Boys(mol, mf.mo_coeff[:,n_a:]).kernel()

        cl_o = lo.Boys(mol).kernel(mf.mo_coeff[:,:n_a])
        cl_v = lo.Boys(mol).kernel(mf.mo_coeff[:,n_a:])

        C = np.column_stack((cl_o,cl_v))
        #end


    elif orb_basis == 'lowdin':
        assert(cas == False)
        S = mol.intor('int1e_ovlp_sph')
        print("Using lowdin orthogonalized orbitals")

        C = lowdin(S)
        #end

    elif orb_basis == 'boys':
        cl_c = mf.mo_coeff[:, :cas_nstart]
        cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop:]
        C = np.column_stack((cl_c, cl_a, cl_v))
        print(cl_a)

    molden.from_mo(mol, 'h8.molden', C)
    if cas == True:
        mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
        h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  #core core orbs to form ecore and eff
        h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym='s4',compact=False).reshape(4 * ((cas_norb), )) 
        print(h1e_cas)
        return h1e_cas,h2e_cas,ecore
    elif cas==False:
        h = C.T.dot(mf.get_hcore()).dot(C)
        g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
        print(h)
        return h, g, enu
# }}}

def lowdin(S):
# {{{
    print("Using lowdin orthogonalized orbitals")
    #forming S^-1/2 to transform to A and B block.
    sal, svec = np.linalg.eigh(S)
    idx = sal.argsort()[::-1]
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal**-0.5
    sal = np.diagflat(sal)
    X = svec @ sal @ svec.T
    return X
# }}}

def reorder_integrals(idx,h,g):
# {{{
    h = h[:,idx] 
    h = h[idx,:] 

    g = g[:,:,:,idx] 
    g = g[:,:,idx,:] 
    g = g[:,idx,:,:] 
    g = g[idx,:,:,:] 
    return h,g
# }}}

def e1_order(h,cut_off):
# {{{
    hnew = np.absolute(h)
    hnew[hnew < cut_off] = 0
    np.fill_diagonal(hnew, 0)
    print(hnew)
    import scipy.sparse
    idx = scipy.sparse.csgraph.reverse_cuthill_mckee(
        scipy.sparse.csr_matrix(hnew))
    print(idx)
    idx = idx 
    hnew = h[:, idx]
    hnew = hnew[idx, :]
    print(hnew)
    return idx
# }}}
