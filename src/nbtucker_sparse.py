import math
import sys
import numpy as np
import scipy
import itertools
import copy as cp
from helpers import *
import opt_einsum as oe
import tools
import time

from ClusteredOperator import *
from ClusteredState import *
from Cluster import *

from ham_build import *

def compute_rspt2_correction(ci_vector, clustered_ham, e0, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        #h0v(clustered_ham,ci_vector)
        #exit()
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    #pt_vector.prune_empty_fock_spaces()
    
    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
    print(" Variance: %12.8f" % var,flush=True)
    
    
    print("Dim of PT space %4d"%len(pt_vector))
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    print("Dim of PT space %4d"%len(pt_vector))
    
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])
    
    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector))
    print(" Dimension of PT space: ", len(pt_vector))
    print(" Compute Denominator",flush=True)
    # compute diagonal for PT2
    start = time.time()

    #pt_vector.prune_empty_fock_spaces()
        
       
    if nproc==1:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
    else:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        #Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
    #pr.disable()
    #pr.print_stats(sort='time')
    end = time.time()
    print(" Time spent in demonimator: ", end - start)

    denom = 1/(e0 - Hd)
    pt_vector_v = pt_vector.get_vector()
    pt_vector_v.shape = (pt_vector_v.shape[0])

    e2 = np.multiply(denom,pt_vector_v)
    pt_vector.set_vector(e2)
    e2 = np.dot(pt_vector_v,e2)

    print(" PT2 Energy Correction = %12.8f" %e2)
    return e2,pt_vector
# }}}


def compute_lcc2_correction(ci_vector, clustered_ham, e0, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        #h0v(clustered_ham,ci_vector)
        #exit()
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    #pt_vector.prune_empty_fock_spaces()
    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
    print(" Variance: %12.8f" % var,flush=True)
    
    print("Dim of PT space %4d"%len(pt_vector))
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    print("Dim of PT space %4d"%len(pt_vector))
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])
    
    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector))
    print(" Dimension of PT space: ", len(pt_vector))
    print(" Compute Denominator",flush=True)
    # compute diagonal for PT2
    start = time.time()

    #pt_vector.prune_empty_fock_spaces()
        
       
    if nproc==1:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        Hd2 = build_hamiltonian_diagonal(clustered_ham, pt_vector)
    else:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        Hd2 = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
    #pr.disable()
    #pr.print_stats(sort='time')
    end = time.time()
    print(" Time spent in demonimator: ", end - start)

    denom = 1/(e0 - Hd)
    pt_vector_v = pt_vector.get_vector()
    pt_vector_v.shape = (pt_vector_v.shape[0])

    e2 = np.multiply(denom,pt_vector_v)
    pt_vector.set_vector(e2)
    e2 = np.dot(pt_vector_v,e2)

    print(" PT2 Energy Correction = %12.8f" %e2)

    print("E DPS %16.8f"%e0)
    #ci_vector.add(pt_vector)
    H = build_full_hamiltonian(clustered_ham, pt_vector)
    np.fill_diagonal(H,0)

    denom.shape = (denom.shape[0],1)
    v1 = pt_vector.get_vector()
    for j in range(0,10):
        print("v1",v1.shape)
        v2 = H @ v1
        #print("v2",v2.shape)
        #print("denom",denom.shape)
        v2 = np.multiply(denom,v2)
        #print("afrer denom",v2.shape)
        e3 = np.dot(pt_vector_v,v2)
        print(e3.shape)
        print("PT2",e2)
        print("PT3",e3[0])
        v1 = v2
    pt_vector.set_vector(v2)

    return e3[0],pt_vector
# }}}


def build_h0(clustered_ham,ci_vector,pt_vector):
    """
    Build hamiltonian diagonal in basis in ci_vector as difference of cluster energies as in RSPT
    """
# {{{
    clusters = clustered_ham.clusters
    Hd = np.zeros((len(pt_vector)))
    
    E0 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0, opt_einsum=True)[0,0]
    assert(len(ci_vector)==1)
    print("E0%16.8f"%E0)

    #idx = 0
    #Hd1 = np.zeros((len(pt_vector)))
    #for f,c,v in pt_vector:
    #    e0_X = 0
    #    for ci in clustered_ham.clusters:
    #        e0_X += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]]
    #    Hd1[idx] = e0_X
    #    idx += 1


    idx = 0
    Hd = np.zeros((len(pt_vector)))
    for ci_fspace, ci_configs in ci_vector.items():
        for ci_config, ci_coeff in ci_configs.items():
            for fockspace, configs in pt_vector.items():
                #print("FS",fockspace)
                active = []
                inactive = []
                for ind,fs in enumerate(fockspace):
                    if fs != ci_fspace[ind]:
                        active.append(ind)
                    else:
                        inactive.append(ind)
                            
                delta_fock= tuple([(fockspace[ci][0]-ci_fspace[ci][0], fockspace[ci][1]-ci_fspace[ci][1]) for ci in range(len(clusters))])
                #print("active",active)
                #print("active",delta_fock)
                #print(tuple(np.array(list(fockspace))[inactive]))

                for config, coeff in configs.items():
                    delta_fock= tuple([(0,0) for ci in range(len(clusters))])

                    diff = tuple(x-y for x,y in zip(ci_config,config))
                    #print("CI",ci_config)
                    for x in inactive:
                        if diff[x] != 0  and x not in active:
                            active.append(x)
                    #print("PT",config)
                    #print("d ",diff)
                    #print("ACTIVE",active)

                    Hd[idx] = E0
                    for cidx in active:
                        fspace = fockspace[cidx]
                        conf = config[cidx]
                        #print(" Cluster: %4d  Fock Space:%s config:%4d Energies %16.8f"%(cidx,fspace,conf,clusters[cidx].energies[fspace][conf]))
                        #Hd[idx] += clusters[cidx].energies[fockspace[cidx]][config[cidx]]
                        #Hd[idx] -= clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]

                        Hd[idx] += clusters[cidx].ops['H_mf'][(fockspace[cidx],fockspace[cidx])][config[cidx],config[cidx]]
                        Hd[idx] -= clusters[cidx].ops['H_mf'][(ci_fspace[cidx],ci_fspace[cidx])][ci_config[cidx],ci_config[cidx]]
                        #print("-Cluster: %4d  Fock Space:%s config:%4d Energies %16.8f"%(cidx,ci_fspace[cidx],ci_config[cidx],clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]))
                    # for EN:
                    #for term in terms:
                    #    Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
                    #    print(term.active)


                    # for RS
                    #Hd[idx] = E0 - term.active  
                    idx += 1

    return Hd

    # }}}

def cepa(clustered_ham,ci_vector,pt_vector,cepa_shift):
# {{{
    ts = 0

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    #H00 = H[tb0.start:tb0.stop,tb0.start:tb0.stop]
    
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[ts]
  
    Ec = 0.0

    #cepa_shift = 'aqcc'
    #cepa_shift = 'cisd'
    #cepa_shift = 'acpf'
    #cepa_shift = 'cepa0'
    cepa_mit = 1
    cepa_mit = 100
    for cit in range(0,cepa_mit): 

        #Hdd = cp.deepcopy(H[tb0.stop::,tb0.stop::])
        Hdd = build_full_hamiltonian(clustered_ham,pt_vector,iprint=0)

        shift = 0.0
        if cepa_shift == 'acpf':
            shift = Ec * 2.0 / n_blocks
            #shift = Ec * 2.0 / n_sites
        elif cepa_shift == 'aqcc':
            shift = (1.0 - (n_blocks-3.0)*(n_blocks - 2.0)/(n_blocks * ( n_blocks-1.0) )) * Ec
        elif cepa_shift == 'cisd':
            shift = Ec
        elif cepa_shift == 'cepa0':
            shift = 0

        Hdd += -np.eye(Hdd.shape[0])*(E0 + shift)
        #Hdd += -np.eye(Hdd.shape[0])*(E0 + -0.220751700895 * 2.0 / 8.0)
        
        
        #Hd0 = H[tb0.stop::,tb0.start:tb0.stop].dot(V0[:,ts])
        H0d = build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0)
        Hd0 = H0d.T
        Hd0 = Hd0.dot(V0[:,ts])
        
        #Cd = -np.linalg.inv(Hdd-np.eye(Hdd.shape[0])*E0).dot(Hd0)
        #Cd = np.linalg.inv(Hdd).dot(-Hd0)
        
        Cd = np.linalg.solve(Hdd, -Hd0)
        
        print(" CEPA(0) Norm  : %16.12f"%np.linalg.norm(Cd))
        
        V0 = V0[:,ts]
        V0.shape = (V0.shape[0],1)
        Cd.shape = (Cd.shape[0],1)
        C = np.vstack((V0,Cd))
        H00d = np.insert(H0d,0,E0,axis=1)
        
        #E = V0[:,ts].T.dot(H[tb0.start:tb0.stop,:]).dot(C)
        E = V0[:,ts].T.dot(H00d).dot(C)
        
        cepa_last_vectors = C  
        cepa_last_values  = E
        
        print(" CEPA(0) Energy: %16.12f"%E)
        
        if abs(E-E0 - Ec) < 1e-10:
            print("Converged")
            break
        Ec = E - E0
    print("Ec %16.8f"%Ec)
    return Ec[0]
# }}}

def build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0):
    """
    Build hamiltonian in basis of two different clustered states
    """
# {{{
    clusters = clustered_ham.clusters
    H0d = np.zeros((len(ci_vector),len(pt_vector)))
    
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_ri, fock_r in enumerate(pt_vector.data):
                configs_r = pt_vector[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])

                try:
                    terms = clustered_ham.terms[delta_fock]
                except KeyError:
                    shift_r += len(configs_r) 
                    continue 

                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri


                    #print("FOC",fock_l,fock_r)
                    #print("con",config_l,config_r)
                    #print(shift_r,config_ri)
                    #print("idx",idx_l,idx_r)
                    for term in terms:
                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        H0d[idx_l,idx_r] += me
                    

                shift_r += len(configs_r) 
        shift_l += len(configs_l)
    return H0d
# }}}

def compute_cisd_correction(ci_vector, clustered_ham, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[0]

    if nproc==1:
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])
    
    ci_vector.add(pt_vector)
       
    if nproc==1:
        H = build_full_hamiltonian(clustered_ham, ci_vector)
    else:
        H = build_full_hamiltonian(clustered_ham, ci_vector)
    e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v = v[:,0]
    e0 = e[0]
    e = e[0]
    Ec = e - E0

    print(" CISD Energy Correction = %12.8f" %Ec)
    return Ec
# }}}

def truncated_ci(clustered_ham, ci_vector, pt_vector=None, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)

    if pt_vector==None:
        H = build_full_hamiltonian(clustered_ham, ci_vector)
        e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
        idx = e.argsort()
        e = e[idx]
        v = v[:,idx]
        v = v[:,0]
        e0 = e[0]
        e = e[0]
    else:
        H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
        E0,V0 = np.linalg.eigh(H00)
        E0 = E0[0]

        ci_vector.add(pt_vector)

        H = build_full_hamiltonian(clustered_ham, ci_vector)
        e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
        idx = e.argsort()
        e = e[idx]
        v = v[:,idx]
        v = v[:,0]
        e0 = e[0]
        e = e[0]

        Ec = e - E0

        print(" Truncated CI Correction = %12.8f" %Ec)
    ci_vector.set_vector(v)
    return e,ci_vector
# }}}

def expand_doubles(ci_vector,clusters):
# {{{

    ci_vector.print_configs()

    for fspace in ci_vector.keys():
        for ci in clusters:
            for cj in clusters:
                if cj.idx != ci.idx:

                    #same fock space
                    nfs = fspace
                    fock_i = nfs[ci.idx]
                    fock_j = nfs[cj.idx]
                    dims = [[0] for ca in range(len(clusters))]
                    dims[ci.idx] = range(1,ci.basis[fock_i].shape[1])
                    dims[cj.idx] = range(1,cj.basis[fock_j].shape[1])
                    for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                        ci_vector[nfs][newconfig] = 0 

                    # alpha excitation

                    new_fspace_a = [list(fs) for fs in fspace]
                    new_fspace_a[ci.idx][0] += 1
                    new_fspace_a[cj.idx][0] -= 1
                    new_fspace_a = tuple( tuple(fs) for fs in new_fspace_a)

                    good = True
                    for c in clusters:
                        if min(new_fspace_a[c.idx]) < 0 or max(new_fspace_a[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_a)
                        ci_vector.add_fockspace(new_fspace_a)
                        nfs = new_fspace_a
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 

                    # beta excitation

                    new_fspace_b = [list(fs) for fs in fspace]
                    new_fspace_b[ci.idx][1] += 1
                    new_fspace_b[cj.idx][1] -= 1
                    new_fspace_b = tuple( tuple(fs) for fs in new_fspace_b)

                    good = True
                    for c in clusters:
                        if min(new_fspace_b[c.idx]) < 0 or max(new_fspace_b[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_b)
                        ci_vector.add_fockspace(new_fspace_b)
                        nfs = new_fspace_b
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 

                    new_fspace_aa = [list(fs) for fs in fspace]
                    new_fspace_aa[ci.idx][0] += 2
                    new_fspace_aa[cj.idx][0] -= 2
                    new_fspace_aa = tuple( tuple(fs) for fs in new_fspace_aa)

                    good = True
                    for c in clusters:
                        if min(new_fspace_aa[c.idx]) < 0 or max(new_fspace_aa[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_aa)
                        ci_vector.add_fockspace(new_fspace_aa)
                        nfs = new_fspace_aa
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 

                    new_fspace_bb = [list(fs) for fs in fspace]
                    new_fspace_bb[ci.idx][1] += 2
                    new_fspace_bb[cj.idx][1] -= 2
                    new_fspace_bb = tuple( tuple(fs) for fs in new_fspace_bb)
                    print(new_fspace_bb)

                    good = True
                    for c in clusters:
                        if min(new_fspace_bb[c.idx]) < 0 or max(new_fspace_bb[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_bb)
                        ci_vector.add_fockspace(new_fspace_bb)
                        nfs = new_fspace_bb
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 

                    new_fspace_ab = [list(fs) for fs in fspace]
                    new_fspace_ab[ci.idx][0] += 1
                    new_fspace_ab[cj.idx][0] -= 1
                    new_fspace_ab[ci.idx][1] += 1
                    new_fspace_ab[cj.idx][1] -= 1
                    new_fspace_ab = tuple( tuple(fs) for fs in new_fspace_ab)
                    print("AB",new_fspace_ab)

                    good = True
                    for c in clusters:
                        if min(new_fspace_ab[c.idx]) < 0 or max(new_fspace_ab[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_ab)
                        ci_vector.add_fockspace(new_fspace_ab)
                        nfs = new_fspace_ab
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 

                    new_fspace_ab = [list(fs) for fs in fspace]
                    new_fspace_ab[ci.idx][0] += 1
                    new_fspace_ab[cj.idx][0] -= 1
                    new_fspace_ab[ci.idx][1] -= 1
                    new_fspace_ab[cj.idx][1] += 1
                    new_fspace_ab = tuple( tuple(fs) for fs in new_fspace_ab)
                    print("AB",new_fspace_ab)

                    good = True
                    for c in clusters:
                        if min(new_fspace_ab[c.idx]) < 0 or max(new_fspace_ab[c.idx]) > c.n_orb:
                            good = False
                            break
                    if good == False:
                        p = 1
                    else:
                        print(new_fspace_ab)
                        ci_vector.add_fockspace(new_fspace_ab)
                        nfs = new_fspace_ab
                        fock_i = nfs[ci.idx]
                        fock_j = nfs[cj.idx]
                        dims = [[0] for ca in range(len(clusters))]
                        dims[ci.idx] = range(ci.basis[fock_i].shape[1])
                        dims[cj.idx] = range(cj.basis[fock_j].shape[1])
                        for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
                            ci_vector[nfs][newconfig] = 0 


    ci_vector.print_configs()
    print(len(ci_vector))
    #ci_vector.add_single_excitonic_states()
    #ci_vector.expand_each_fock_space()
    #ci_vector.print()
    print(len(ci_vector))
    ci_vector.print_configs()

    """
    for fspace in ci_vector.keys():
        config = [0]*len(self.clusters)
        for ci in self.clusters:
            fock_i = fspace[ci.idx]
            new_config = cp.deepcopy(config)
            for cii in range(ci.basis[fock_i].shape[1]):
                new_config[ci.idx] = cii
                self[fspace][tuple(new_config)] = 0 
    """
    return ci_vector

# }}}

def truncated_pt2(clustered_ham,ci_vector,pt_vector,method = 'mp2',inf=False):
# {{{
    """ method: mp2,mplcc2,en2,enlcc, use the inf command to do infinite order PT when u have the full H"""

    clusters = clustered_ham.clusters

    ts = 0
    print("len CI",len(ci_vector))
    print("len PT",len(pt_vector))

    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    print("Dim of PT space %4d"%len(pt_vector))

    pt_dim = len(pt_vector)
    ci_dim = len(ci_vector)
    pt_order = 500
    
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])

    H0d = build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0)

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    Hdd = build_full_hamiltonian(clustered_ham,pt_vector,iprint=0)
    print(H00)
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[ts]

    if method == 'en2' or method == 'enlcc':
        Hd = build_hamiltonian_diagonal(clustered_ham,pt_vector)
        np.fill_diagonal(Hdd,0)

    elif method == 'mp2' or method == 'mplcc':
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        #Hd += 10
        for i in range(0,Hdd.shape[0]):
            Hdd[i,i] -= (Hd[i])
    else:
        print("Method not found")

    print("E0 %16.8f"%E0)
    print(Hdd)
    
    R0 = 1/(E0 - Hd)
    print(R0)

    v1 = np.multiply(R0,H0d)
    pt_vector.set_vector(v1.T)

    print(v1.shape)
    print(H0d.shape)

    e2 = H0d @ v1.T
    print(e2)

    v_n = np.zeros((pt_dim,pt_order+1))   #list of PT vectors
    E_mpn = np.zeros((pt_order+1))         #PT energy

    v_n[: ,0] = v1
    E_mpn[0] = e2
    E_corr = 0

    print(" %6s  %16s  %16s "%("Order","Correction","Energy"))
    #print(" %6i  %16.8f  %16.8f "%(1,first_order_E[0,s],E_mpn[0]))

    E_corr = E_mpn[0] 
    print(" %6i  %16.8f  %16.8f "%(2,E_mpn[0],E_corr))

    if method == 'enlcc' or method == 'mplcc':

        Eold = E_corr
        for i in range(1,pt_order-1):
            h1 = Hdd @ v_n[:,i-1]

            v_n[:,i] = h1.reshape(pt_dim)
            if inf ==True:
                for k in range(0,i):
                    v_n[:,i] -= np.multiply(E_mpn[k-1],v_n[:,(i-k-1)].reshape(pt_dim))

            v_n[:,i] = np.multiply(R0,v_n[:,i])
            E_mpn[i] = H0d @ v_n[:,i].T
            #print(E_mpn)
            E_corr += E_mpn[i]
            print(" %6i  %16.8f  %16.8f "%(i+2,E_mpn[i],E_corr))

            pt_vector.set_vector(v_n[:,i])
            if abs(E_corr - Eold) < 1e-10:
                print("LCC:%16.8f "%E_corr)
                break
            else:
                Eold = E_corr
                #v_n[:,i] = 0.8 * v_n[:,i] + 0.2 * v_n[:,i-1]

    elif method == 'en2' or method == 'mp2':
        print("MP2:%16.8f "%E_corr)
     
    return E_corr, pt_vector
# }}}

def pt2infty(clustered_ham,ci_vector,pt_vector,form_H=True,nproc=None):
    """ 
    The DMBPT infty equivalent for TPS methods. equvalent to CEPA/ LCCSD
    Input:
        clustered_ham:  clustered_ham for the system
        ci_vector: single CMF state for now
        pt_vector: the pt_vector generated using compute_pt2_correction function 
    Output:
        The correlation energy using cepa
        pt_vector: which is the updated LCC vector
    """
# {{{

    clusters = clustered_ham.clusters

    ts = 0
    print("len CI",len(ci_vector))
    print("len PT",len(pt_vector))

    pt_dim = len(pt_vector)
    ci_dim = len(ci_vector)
    pt_order = 500
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])

    H0d = build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0)

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    
    if form_H:
        print("Storage for H %8.4f GB"%(pt_dim*pt_dim*8/10e9))
        if pt_dim > 60000:
            print("Memory for just storing H is approx 29 GB")
            exit()
        Hdd = build_full_hamiltonian_parallel2(clustered_ham,pt_vector,iprint=0,nproc=nproc)
        np.fill_diagonal(Hdd,0)

    print(H00)
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[ts]

    Hd = build_hamiltonian_diagonal(clustered_ham,pt_vector)

    print("E0 %16.8f"%E0)
    
    R0 = 1/(E0 - Hd)
    print(R0)

    v1 = np.multiply(R0,H0d)
    pt_vector.set_vector(v1.T)

    print(v1.shape)
    print(H0d.shape)

    e2 = H0d @ v1.T
    print(e2)

    v_n = np.zeros((pt_dim,pt_order+1))   #list of PT vectors
    E_mpn = np.zeros((pt_order+1))         #PT energy

    v_n[: ,0] = v1
    E_mpn[0] = e2
    E_corr = 0

    print(" %6s  %16s  %16s "%("Order","Correction","Energy"))
    #print(" %6i  %16.8f  %16.8f "%(1,first_order_E[0,s],E_mpn[0]))

    E_corr = E_mpn[0] 
    print(" %6i  %16.8f  %16.8f "%(2,E_mpn[0],E_corr))

    Eold = E_corr
    for i in range(1,pt_order-1):
        #h1 = Hdd @ v_n[:,i-1]

        if form_H:
            h1 = Hdd @ v_n[:,i-1]

        else:
            sigma = build_sigma(clustered_ham,pt_vector,iprint=0, opt_einsum=True)
            h1 = sigma.get_vector()

        v_n[:,i] = h1.reshape(pt_dim)

        #for k in range(0,i):
        #    v_n[:,i] -= np.multiply(E_mpn[k-1],v_n[:,(i-k-1)].reshape(pt_dim))

        v_n[:,i] = np.multiply(R0,v_n[:,i])
        E_mpn[i] = H0d @ v_n[:,i].T
        #print(E_mpn)
        E_corr += E_mpn[i]
        print(" %6i  %16.8f  %16.8f "%(i+2,E_mpn[i],E_corr))

        pt_vector.set_vector(v_n[:,i])
        if abs(E_corr - Eold) < 1e-10:
            print("LCC:%16.8f "%E_corr)
            break
        else:
            Eold = E_corr
            #v_n[:,i] = 0.8 * v_n[:,i] + 0.2 * v_n[:,i-1]

    return E_corr, pt_vector
# }}}

def build_sigma(clustered_ham,ci_vector,iprint=0, opt_einsum=True):
    """
    Form the sigma vector using the EN zero order hamiltonian
    Cannot be used for davidson since this is for H0 of EN partitioning(diagonal of H is 0)
    """
# {{{
    clusters = clustered_ham.clusters
    sigma = np.zeros(len(ci_vector))
    ci_v = ci_vector.get_vector()
    
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_ri, fock_r in enumerate(ci_vector.data):
                configs_r = ci_vector[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                if fock_ri<fock_li:
                    shift_r += len(configs_r) 
                    continue
                try:
                    terms = clustered_ham.terms[delta_fock]
                except KeyError:
                    shift_r += len(configs_r) 
                    continue 
                
                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri
                    if idx_r<idx_l:
                        continue
                    
                    for term in terms:
                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        if idx_r == idx_l:
                            me = 0
                        sigma[idx_l] += me  * ci_v[idx_r]
                        if idx_r>idx_l:
                            sigma[idx_r] += me  * ci_v[idx_l]

                        #print(" %4i %4i = %12.8f"%(idx_l,idx_r,me),"  :  ",config_l,config_r, " :: ", term)
                shift_r += len(configs_r) 
        shift_l += len(configs_l)

    sigma_vec = ci_vector.copy()
    sigma_vec.set_vector(sigma)
    return sigma_vec
# }}}

