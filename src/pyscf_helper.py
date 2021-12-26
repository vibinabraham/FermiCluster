import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

class PyscfHelper(object):
    """
    Pyscf is used to generate 
    integrals etc for the TPSCI program. this is a class which keeps info about pyscf mol info and HF stuff
    """

    def __init__(self):

        self.mf  = None
        self.mol = None

        self.h      = None
        self.g      = None
        self.n_orb  = None
        #self.na     = 0
        #self.nb     = 0
        self.ecore  = 0
        self.C      = None
        self.S      = None
        self.J      = None
        self.K      = None

    def init(self,molecule,charge,spin,basis_set,orb_basis='scf',cas=False,cas_nstart=None,cas_nstop=None,cas_nel=None,loc_nstart=None,loc_nstop=None,
            scf_conv_tol=1e-14):
    # {{{
        import pyscf
        from pyscf import gto, scf, ao2mo, molden, lo
        pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
        #PYSCF inputs
        print(" ---------------------------------------------------------")
        print("                      Using Pyscf:")
        print(" ---------------------------------------------------------")
        print("                                                          ")

        mol = gto.Mole()
        mol.atom = molecule

        mol.max_memory = 1000 # MB
        mol.symmetry = True
        mol.charge = charge
        mol.spin = spin
        mol.basis = basis_set
        mol.build()
        print("symmertry")
        print(mol.topgroup)

        #SCF 

        #mf = scf.RHF(mol).run(init_guess='atom')
        mf = scf.RHF(mol).run(conv_tol=scf_conv_tol)
        #C = mf.mo_coeff #MO coeffs
        enu = mf.energy_nuc()
        
        print(" SCF Total energy: %12.8f" %mf.e_tot) 
        print(" SCF Elec  energy: %12.8f" %(mf.e_tot-enu))
        print(mf.get_fock())
        print(np.linalg.eig(mf.get_fock())[0])
        
        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, mf.mo_coeff)
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mf.mo_energy[i]))

        #orbitals and lectrons
        n_orb = mol.nao_nr()
        n_b , n_a = mol.nelec 
        nel = n_a + n_b
        self.n_orb = mol.nao_nr()


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

        ##AO 2 MO Transformation: orb_basis or scf
        if orb_basis == 'scf':
            print("\nUsing Canonical Hartree Fock orbitals...\n")
            C = cp.deepcopy(mf.mo_coeff)
            print("C shape")
            print(C.shape)

        elif orb_basis == 'lowdin':
            assert(cas == False)
            S = mol.intor('int1e_ovlp_sph')
            print("Using lowdin orthogonalized orbitals")

            C = lowdin(S)
            #end

        elif orb_basis == 'boys':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'boys2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.ER(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ibmo':
            loc_vstop =  loc_nstop - n_a
            print(loc_vstop)

            mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
            mo_vir = mf.mo_coeff[:,mf.mo_occ==0]
            c_core = mo_occ[:,:loc_nstart]
            iao_occ = lo.iao.iao(mol, mo_occ[:,loc_nstart:])
            iao_vir = lo.iao.iao(mol, mo_vir[:,:loc_vstop])
            c_out  = mo_vir[:,loc_vstop:]

            # Orthogonalize IAO
            iao_occ = lo.vec_lowdin(iao_occ, mf.get_ovlp())
            iao_vir = lo.vec_lowdin(iao_vir, mf.get_ovlp())

            #
            # Method 1, using Knizia's alogrithm to localize IAO orbitals
            #
            '''
            Generate IBOS from orthogonal IAOs
            '''
            ibo_occ = lo.ibo.ibo(mol, mo_occ[:,loc_nstart:], iaos = iao_occ)
            ibo_vir = lo.ibo.ibo(mol, mo_vir[:,:loc_vstop], iaos = iao_vir)

            C = np.column_stack((c_core,ibo_occ,ibo_vir,c_out))

        else: 
            print("Error:NO orbital basis defined")

        molden.from_mo(mol, 'orbitals.molden', C)

        if cas == True:
            print(C.shape)
            print(cas_norb)
            print(cas_nel)
            mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
            h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  #core core orbs to form ecore and eff
            h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym='s4',compact=False).reshape(4 * ((cas_norb), )) 
            print(h1e_cas)
            print(h1e_cas.shape)
            #return h1e_cas,h2e_cas,ecore,C,mol,mf
            self.h = h1e_cas
            self.g = h2e_cas
            self.ecore = ecore
            self.mf = mf
            self.mol = mol
            self.C = cp.deepcopy(C[:,cas_nstart:cas_nstop])
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C

            #HF density
            if orb_basis == 'scf':
                #C = C[:,cas_nstart:cas_nstop]
                D = mf.make_rdm1(mo_coeff=C)
                S = mf.get_ovlp()
                sal, svec = np.linalg.eigh(S)
                idx = sal.argsort()[::-1]
                sal = sal[idx]
                svec = svec[:, idx]
                sal = sal**-0.5
                sal = np.diagflat(sal)
                X = svec @ sal @ svec.T
                C_ao2mo = np.linalg.inv(X) @ C
                Cocc = C_ao2mo[:, :n_a]
                D = Cocc @ Cocc.T
                DMO = C_ao2mo.T   @ D @ C_ao2mo
                
                #only for cas space 
                DMO = DMO[cas_nstart:cas_nstop,cas_nstart:cas_nstop]
                self.dm_aa = DMO
                self.dm_bb = DMO
                print("DENSITY")
                print(self.dm_aa.shape)

            if 0:
                h = C.T.dot(mf.get_hcore()).dot(C)
                g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
                const,heff = get_eff_for_casci(cas_nstart,cas_nstop,h,g)
                print(heff)
                print("const",const)
                print("ecore",ecore)
                
                idx = range(cas_nstart,cas_nstop)
                h = h[:,idx] 
                h = h[idx,:] 
                g = g[:,:,:,idx] 
                g = g[:,:,idx,:] 
                g = g[:,idx,:,:] 
                g = g[idx,:,:,:] 

                self.ecore = const
                self.h = h + heff
                self.g = g 


        elif cas==False:
            h = C.T.dot(mf.get_hcore()).dot(C)
            g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
            print(h)
            #return h, g, enu, C,mol,mf
            self.h = h
            self.g = g
            self.ecore = enu
            self.mf = mf
            self.mol = mol
            self.C = C
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C

            #HF density
            if orb_basis == 'scf':
                D = mf.make_rdm1(mo_coeff=None)
                S = mf.get_ovlp()
                sal, svec = np.linalg.eigh(S)
                idx = sal.argsort()[::-1]
                sal = sal[idx]
                svec = svec[:, idx]
                sal = sal**-0.5
                sal = np.diagflat(sal)
                X = svec @ sal @ svec.T
                C_ao2mo = np.linalg.inv(X) @ C
                Cocc = C_ao2mo[:, :n_a]
                D = Cocc @ Cocc.T
                DMO = C_ao2mo.T   @ D @ C_ao2mo
                self.dm_aa = DMO
                self.dm_bb = DMO
                print("DENSITY")
                print(self.dm_aa)
    # }}}

def run_fci_pyscf( h, g, nelec, ecore=0,nroots=1, conv_tol=None, max_cycle=None):
# {{{
    # FCI
    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
    cisolver = fci.direct_spin1.FCI()
    if max_cycle != None:
        cisolver.max_cycle = max_cycle 
    if conv_tol != None:
        cisolver.conv_tol = conv_tol 
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
    fci_dim = ci.shape[0]*ci.shape[1]
    d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
    print(" PYSCF 1RDM: ")
    occs = np.linalg.eig(d1)[0]
    [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    with np.printoptions(precision=6, suppress=True):
        print(d1)
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    #for i in range(0,nroots):
    #    print("FCI %10.8f"%(efci[i]))
    #exit()
    #fci_dim =1
            
    return efci,fci_dim
# }}}

def run_hci_pyscf( h, g, nelec, ecore=0, select_cutoff=5e-4, ci_cutoff=5e-4,nroots=1):
# {{{
    #heat bath ci
    from pyscf import mcscf
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = ci_cutoff
    ehci, civec = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,verbose=4,nroots=nroots)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    print("HCI %10.8f"%(ehci))
    #for i in range(0,nroots):
    #    print("HCI %10.8f"%(ehci[i]))
    #hci_dim = 1
    return ehci,hci_dim
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
    hnew = hnew[:, idx]
    hnew = hnew[idx, :]
    print("New order")
    print(hnew)
    return idx
# }}}

def ordering(pmol,cas,cas_nstart,cas_nstop,loc_nstart,loc_nstop,ordering='hcore'):
# {{{
    loc_range = np.array(list(range(loc_nstart-cas_nstart,loc_nstop-cas_nstart)))
    #cas_range = range(cas_nstart,cas_nstop)
    out_range = np.array(list(range(loc_nstop-cas_nstart,cas_nstop-cas_nstart)))
    print(loc_range)
    print(out_range)

    h = cp.deepcopy(pmol.h)
    print(h)
    if ordering == 'hcore':
        print("Bonding Active Space")
        hl = h[:,loc_range]
        hl = hl[loc_range,:]
        print(hl)
        idl = e1_order(hl,cut_off = 1e-2)

        ho = h[:,out_range]
        ho = ho[out_range,:]
        print("Virtual Active Space")
        ido = e1_order(ho,cut_off = 1e-2)

        idl = idl 
        ido = ido + loc_nstop - cas_nstart 

    print(idl)
    print(ido)
    idx = np.append(idl,ido)
    print(idx)
    return idx
    # }}}

def mulliken_ordering(mol,norb,C):
# {{{
    """
    pyscf mulliken
    """
    S = mol.intor('int1e_ovlp_sph')
    mulliken = np.zeros((mol.natm,norb))
    for i in range(0,norb):
        Cocc = C[:,i].reshape(C.shape[0],1)
        temp = Cocc @ Cocc.T @ S   
        for m,lb in enumerate(mol.ao_labels()):
            #print(lb)
            v1,v2,v3 = lb.split()
            #print(v1)
            mulliken[int(v1),i] += temp[m,m]
    print(mulliken)
    return mulliken
# }}}

def block_order_mulliken(n_blocks,n_orb,mulliken,atom_block): 
# {{{
    blocks = [[] for i in range(n_blocks)]
    for i in range(0,n_orb):
        atom = mulliken[:,i].argmax(axis=0)

        for ind,bl in enumerate(atom_block):
            if atom in bl:
                #print(ind)
                blocks[ind].append(i)
        print(blocks)
    return blocks
# }}}

def get_eff_for_casci(n_start,n_stop,h,g):
# {{{
    const = 0
    for i in range(0,n_start):
        const += 2 * h[i,i]
        for j in range(0,n_start):
            const += 2 * g[i,i,j,j] -  g[i,j,i,j]

    eff = np.zeros((n_stop - n_start,n_stop - n_start))

    for l in range(n_start,n_stop):
        L = l - n_start
        for m in range(n_start,n_stop):
            M = m - n_start
            for j in range(0,n_start):
                eff[L,M] += 2 * g[l,m,j,j] -  g[l,j,j,m]
    return const, eff
# }}}

def get_eff_for_casci_orblist(focc_list,cas_list,h,g):
# {{{

    cas_dim = len(cas_list)
    const = 0
    for i in focc_list:
        const += 2 * h[i,i]
        for j in focc_list:
            const += 2 * g[i,i,j,j] -  g[i,j,i,j]

    eff = np.zeros((cas_dim,cas_dim))
    for L,l in enumerate(cas_list):
        for M,m in enumerate(cas_list):
            for j in focc_list:
                eff[L,M] += 2 * g[l,m,j,j] -  g[l,j,j,m]
    return const, eff
# }}}

def ordering_diatomics(mol,C,basis_set):
# {{{
    ##DZ basis diatomics reordering with frozen 1s

    if basis_set == '6-31g':
        orb_type = ['s','pz','px','py']
    elif basis_set == 'ccpvdz':
        orb_type = ['s','pz','dz','px','dxz','py','dyz','dx2-y2','dxy']
    else:
        print("clustering not general yet")
        exit()

    ref = np.zeros(C.shape[1]) 

    ## Find dimension of each space
    dim_orb = []
    for orb in orb_type:
        print("Orb type",orb)
        idx = 0
        for label in mol.ao_labels():
            if orb in label:
                #print(label)
                idx += 1

        ##frozen 1s orbitals
        if orb == 's':
            idx -= 2 
        dim_orb.append(idx)
        print(idx)
    

    new_idx = []
    ## Find orbitals corresponding to each orb space
    for i,orb in enumerate(orb_type):
        print("Orbital type:",orb)
        from pyscf import mo_mapping
        s_pop = mo_mapping.mo_comps(orb, mol, C)
        #print(s_pop)
        ref += s_pop
        cas_list = s_pop.argsort()[-dim_orb[i]:]
        cas_list = np.sort(cas_list)
        print('cas_list', np.array(cas_list))
        new_idx.extend(cas_list) 
        #print(orb,' population for active space orbitals', s_pop[cas_list])

    ao_labels = mol.ao_labels()
    #idx = mol.search_ao_label(['N.*s'])
    #for i in idx:
    #    print(i, ao_labels[i])
    print(ref)
    print(new_idx)
    for label in mol.ao_labels():
        print(label)

    assert(len(new_idx) == len(set(new_idx)))
    return new_idx
# }}}

def get_pi_space(mol,mf,cas_norb,cas_nel,local=True,p3=False):
# {{{
    from pyscf import mcscf, mo_mapping, lo, ao2mo
    # find the 2pz orbitals using mo_mapping
    ao_labels = ['C 2pz']

    # get the 3pz and 2pz orbitals
    if p3:
        ao_labels = ['C 2pz','C 3pz']
        cas_norb = 2 * cas_norb

    pop = mo_mapping.mo_comps(ao_labels, mol, mf.mo_coeff)
    cas_list = np.sort(pop.argsort()[-cas_norb:])  #take the 2z orbitals and resort in MO order
    print('Population for pz orbitals', pop[cas_list])
    mo_occ = np.where(mf.mo_occ>0)[0]
    focc_list = list(set(mo_occ)-set(cas_list))
    focc = len(focc_list)

    # localize the active space
    if local:
        cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_list]).kernel(verbose=4)
        C = mf.mo_coeff 
        C[:,cas_list] = cl_a
    else:
        C = mf.mo_coeff
        mo_energy = mf.mo_energy[cas_list]
        J,K = mf.get_jk()
        K  = K[cas_list,:][:,cas_list]
        print(K)

        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, C[:,cas_list])
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))

    # reorder the orbitals to get docc,active,vir ordering  (Note:sort mo takes orbital ordering from 1)
    mycas = mcscf.CASCI(mf, cas_norb, cas_nel)
    C = mycas.sort_mo(cas_list+1,mo_coeff=C)
    np.save('C.npy',C)

    # Get the active space integrals and the frozen core energy
    h, ecore = mycas.get_h1eff(C)
    g = ao2mo.kernel(mol,C[:,focc:focc+cas_norb], aosym = 's4', compact = False).reshape(4*((cas_norb),))
    C = C[:,focc:focc+cas_norb]  #only carrying the active sapce orbs
    return h,ecore,g,C
# }}}
