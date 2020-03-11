import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
from numpy.linalg import norm


from ci_string import *

def get_dimer_index(ket1, ket2):    
    ket = ci_string(ket1.no+ket2.no, ket1.ne+ket2.ne)
    print(ket)

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

def join_bases(ci,cj):
# {{{
    print(" Join the basis vectors for clusters: %4i and %-4i " %(ci.idx,cj.idx))
    tmp = []
    tmp.extend(ci.orb_list)
    tmp.extend(cj.orb_list)
    cij = Cluster(0,tmp)
    no = ci.n_orb + cj.n_orb
    dimer_map = dict()
    for fi,vi in ci.basis.items():
        #print("a",fi)
        for fj,vj in cj.basis.items():
            ket_aj = ci_string(cj.n_orb, fj[0])
            ket_bj = ci_string(cj.n_orb, fj[1])
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
    #for f in dimer_map:
    #    print(f)
    #    print_mat(dimer_map[f])

    dimer_dim = 0
    for fi,vi in ci.basis.items():

        Nia = calc_nchk(ci.n_orb,fi[0])
        Nib = calc_nchk(ci.n_orb,fi[1])
        for fj,vj in cj.basis.items():


            Nja = calc_nchk(cj.n_orb,fj[0])
            Njb = calc_nchk(cj.n_orb,fj[1])
            
            #print(fi,fj)
            vij = np.kron(vi,vj)
            

            Nija = calc_nchk(no,fi[0]+fj[0])
            Nijb = calc_nchk(no,fi[1]+fj[1])
            Nij = Nija * Nijb 
            Mij = vi.shape[1] * vj.shape[1] 
            Vij = np.zeros((Nij, Mij))


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
    print(" Joined clusters:")
    print(" ", ci)
    print(" ", cj)
    print(" to form:")
    print(" ", cij)

    #for f in cij.basis:
    #    print(f)
    #    #print_mat(cij.basis[f].T @ cij.basis[f])
    #    print_mat(cij.basis[f])

    return cij
# }}}

if __name__ == "__main__":


    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000
    
    from pyscf import gto, scf, mcscf, ao2mo
    
    r0 = 1.5
    
    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)
    
    charge = 0
    spin  = 0
    basis_set = '6-31g'
    
    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10
    
    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,7],[3,6],[4,5]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    
    
    
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    C = pmol.C
    K = pmol.K
    mol = pmol.mol
    mo_energy = pmol.mf.mo_energy
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb
    
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    print(" FCI: %12.8f (elec)" %(efci-ecore)) 
    print(" FCI: %12.8f (total)" %efci) 
    do_fci = 1
    do_hci = 1
    do_tci = 1
    
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    #clustered_ham.add_1b_terms(cp.deepcopy(h))
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.add_2b_terms(cp.deepcopy(g))
    #clustered_ham.combine_common_terms(iprint=1)
    
    
    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        #cmf(clustered_ham, ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=4)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=50,dm_guess=(dm_aa,dm_bb),diis=True)
        cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=2,dm_guess=(dm_aa,dm_bb),diis=True)
    else:
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            ci.form_eigbasis_from_ints(h,g,max_roots=1)
        
            print(" Build new operators for cluster ",ci.idx)
            ci.build_op_matrices()
  
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(" Energy of reference TPS: %12.8f (elec)"%(edps))
    print(" Energy of reference TPS: %12.8f (total)"%(edps+ecore))
  
    ci_vector.expand_to_full_space()
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))


    print(" ---------------- Combine -------------------")
    c12 = join_bases(clusters[0], clusters[1]) 
    new_clusters = [c12]
    new_clusters.extend(clusters[2:])
    [print(i) for i in new_clusters]
    clusters = new_clusters 
    for ci in range(len(clusters)):
        clusters[ci].idx = ci
    init_fspace = ((3, 3), (1, 1), (1, 1))
    

    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(cp.deepcopy(h))
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(cp.deepcopy(g))
    #clustered_ham.combine_common_terms(iprint=1)
    
    do_cmf = 0
    if do_cmf:
        # Get CMF reference
        cmf(clustered_ham, ci_vector, h, g, max_iter=10)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=5,dm_guess=(dm_aa,dm_bb),diis=True)

    print(" Build cluster operators")
    [ci.build_op_matrices() for ci in clusters]
    
    edps2 = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(" Energy of reference TPS: %12.8f (elec)"%(edps2))
    print(" Energy of reference TPS: %12.8f (total)"%(edps2+ecore))
    
    ci_vector.expand_to_full_space()
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))

    
    assert(abs(e0-e0_new) < 1e-8)
