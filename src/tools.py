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

from matvec import *
from ham_build import *


def build_brdm(ci_vector, ci):
    """
    Build block reduced density matrix for Cluster ci
    """
    # {{{
    rdms = OrderedDict()
    for fspace, configs in ci_vector.items():
        #print()
        #print("fspace:",fspace)
        #print()
        curr_dim = ci.basis[fspace[ci.idx]].shape[1]
        rdm = np.zeros((curr_dim,curr_dim))
        for configi,coeffi in configs.items():
            for cj in range(curr_dim):
                configj = list(cp.deepcopy(configi))
                configj[ci.idx] = cj
                configj = tuple(configj)
                #print(configi,configj,configi[ci.idx],configj[ci.idx])
                try:
                    #print(configi,configj,configi[ci.idx],configj[ci.idx],coeffi,configs[configj])
                    rdm[configi[ci.idx],cj] += coeffi*configs[configj]
                    #print(configi[ci.idx],cj,rdm[configi[ci.idx],cj])
                except KeyError:
                    pass
        try:
            rdms[fspace[ci.idx]] += rdm 
        except KeyError:
            rdms[fspace[ci.idx]] = rdm 

    return rdms
# }}}


def build_brdm_diagonal(ci_vector, ci_idx, clusters):
    """
    Build diagonal of block reduced density matrix for cluster ci_idx
    """
    # {{{
    ci = clusters[ci_idx]
    rdms = OrderedDict()
    for fspace, configs in ci_vector.items():
        #print()
        #print("fspace:",fspace)
        #print()
        curr_dim = ci.basis[fspace[ci_idx]].shape[1]
        rd = np.zeros((curr_dim))
        for configi,coeffi in configs.items():
            try:
                rd[configi[ci_idx]] += coeffi*coeffi
            except KeyError:
                pass
        try:
            rdms[fspace[ci_idx]] += rd 
        except KeyError:
            rdms[fspace[ci_idx]] = rd 

    return rdms
# }}}


def do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, max_iter_cmf=10, do_pt2=True):
    """
    Sort the cluster pairs based on how much correlation energy is recovered when combined
    """
# {{{
    dimer_energies = {}
    init_dim = 1
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    for ci,c in enumerate(clusters):
        init_dim = init_dim * calc_nchk(c.n_orb,init_fspace[ci][0])
        init_dim = init_dim * calc_nchk(c.n_orb,init_fspace[ci][1])
    
    for i in range(len(blocks)):
        for j in range(i+1,len(blocks)):
            if len(blocks[i]) + len(blocks[j]) > max_cluster_size:
                continue
            
            new_block = []
            new_block.extend(blocks[i])
            new_block.extend(blocks[j])
            new_block = sorted(new_block)
            new_blocks = [new_block]
            new_init_fspace = [(init_fspace[i][0]+init_fspace[j][0],init_fspace[i][1]+init_fspace[j][1])]
            for k in range(len(blocks)):
                if k!=i and k!=j:
                    new_blocks.append(blocks[k])
                    new_init_fspace.append(init_fspace[k])
            new_init_fspace = tuple(new_init_fspace)
            
            new_clusters = []
            for ci,c in enumerate(new_blocks):
                new_clusters.append(Cluster(ci,c))
            
            new_ci_vector = ClusteredState()
            new_ci_vector.init(new_clusters,new_init_fspace)
         
            ## unless doing PT2, make sure new dimension is greater than 1
            if do_pt2 == False:
                dim = 1
                for ci,c in enumerate(new_clusters):
                    dim = dim * calc_nchk(c.n_orb,new_init_fspace[ci][0])
                    dim = dim * calc_nchk(c.n_orb,new_init_fspace[ci][1])
                if dim <= init_dim:
                    continue
            
            print(" Clusters:")
            [print(ci) for ci in new_clusters]
            
            new_clustered_ham = ClusteredOperator(new_clusters)
            print(" Add 1-body terms")
            new_clustered_ham.add_1b_terms(cp.deepcopy(h))
            print(" Add 2-body terms")
            new_clustered_ham.add_2b_terms(cp.deepcopy(g))
            #clustered_ham.combine_common_terms(iprint=1)
           
           
            # Get CMF reference
            print(" Let's do CMF for blocks %4i:%-4i"%(i,j))
            e_curr,converged = cmf(new_clustered_ham, new_ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=10, max_nroots=1)
           
            if do_pt2:
                e2, v = compute_pt2_correction(new_ci_vector, new_clustered_ham, e_curr)
                print(" PT2 Energy Total      = %12.8f" %(e_curr+e2))
               
                e_curr += e2

            print(" Pairwise-CMF(%i,%i) Energy = %12.8f" %(i,j,e_curr))
            dimer_energies[(i,j)] = e_curr
    
    import operator
    dimer_energies = OrderedDict(sorted(dimer_energies.items(), key=lambda x: x[1]))
    for d in dimer_energies:
        print(" || %10s | %12.8f" %(d,dimer_energies[d]))
  
    pairs = list(dimer_energies.keys())
    if len(pairs) == 0:
        return blocks, init_fspace

    #target_pair = next(iter(dimer_energies))
    target_pair = pairs[0]

    i = target_pair[0]
    j = target_pair[1]
    print(target_pair)
    new_block = []
    new_block.extend(blocks[i])
    new_block.extend(blocks[j])
    new_blocks = [new_block]
    new_init_fspace = [(init_fspace[i][0]+init_fspace[j][0],init_fspace[i][1]+init_fspace[j][1])]
    for k in range(len(blocks)):
        if k!=i and k!=j:
            new_blocks.append(blocks[k])
            new_init_fspace.append(init_fspace[k])
    print(" This is the new clustering")
    print(" | %12.8f" %dimer_energies[(i,j)], new_blocks)
    new_init_fspace = tuple(new_init_fspace)
    print(new_init_fspace)
    print()
    return new_blocks, new_init_fspace
# }}}

def compute_pt2_correction(ci_vector, clustered_ham, e0, 
        thresh_asci     = 0,
        thresh_search   = 0,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()

        print(" Compute Matrix Vector Product:", flush=True)
        profile = 0
        if profile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        start = time.time()
        if nbody_limit != 4:
            print(" Warning: nbody_limit set to %4i, resulting PT energies are meaningless" %nbody_limit)

        if matvec==1:
            pt_vector = matvec1_parallel1(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==3:
            pt_vector = matvec1_parallel3(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==4:
            pt_vector = matvec1_parallel4(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search,
                    nbody_limit=nbody_limit,
                    batch_size=batch_size, shared_mem=shared_mem)
        stop = time.time()
        print(" Time spent in matvec: %12.2f" %( stop-start))
        #exit()
       
        e0_curr = ci_vector.dot(pt_vector)/asci_vector.dot(asci_vector) 
        print(" Zeroth-order energy: %12.8f Error in E0: %12.8f" %(e0_curr, e0_curr - e0)) 

        if profile:
            pr.disable()
            pr.print_stats(sort='time')
        
        pt_vector.prune_empty_fock_spaces()


        var = pt_vector.dot(pt_vector) - e0*e0
        print(" Variance:          %12.8f" % var,flush=True)
        tmp = ci_vector.dot(pt_vector)
        var = pt_vector.dot(pt_vector) - tmp*tmp
        print(" Variance Subspace: %12.8f" % var,flush=True)


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

        print(" Norm of CI vector = %12.8f" %ci_vector.norm())
        print(" Dimension of CI space: ", len(ci_vector))
        print(" Dimension of PT space: ", len(pt_vector))
        print(" Compute Denominator",flush=True)
        #exit()
        pt_vector.prune_empty_fock_spaces()
            
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()


        # Build Denominator
        if pt_type == 'en':
            start = time.time()
            if nproc==1:
                Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
            else:
                Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
            #pr.disable()
            #pr.print_stats(sort='time')
            end = time.time()
            print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
            
            denom = 1/(e0 - Hd)
        elif pt_type == 'mp':
            start = time.time()
            # get barycentric MP zeroth order energy
            e0_mp = 0
            for f,c,v in ci_vector:
                for ci in clustered_ham.clusters:
                    e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v
            
            print(" Zeroth-order MP energy: %12.8f" %e0_mp, flush=True)

            #   This is not really MP once we have rotated away from the CMF basis.
            #   H = F + (H - F), where F = sum_I F(I)
            #
            #   After Tucker basis, we just use the diagonal of this fock operator. 
            #   Not ideal perhaps, but better than nothing at this stage
            denom = np.zeros(len(pt_vector))
            idx = 0
            for f,c,v in pt_vector:
                e0_X = 0
                for ci in clustered_ham.clusters:
                    e0_X += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]]
                denom[idx] = 1/(e0_mp - e0_X)
                idx += 1
            end = time.time()
            print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
        
        pt_vector_v = pt_vector.get_vector()
        pt_vector_v.shape = (pt_vector_v.shape[0])

        e2 = np.multiply(denom,pt_vector_v)
        pt_vector.set_vector(e2)
        e2 = np.dot(pt_vector_v,e2)
        
        ecore = clustered_ham.core_energy
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2, pt_vector
# }}}

def compute_pt2_correction_lowmem_serial(ci_vector, clustered_ham, e0, 
        thresh_asci     = 0,
        thresh_search   = 1e-12,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        opt_einsum      = True,
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction using low-memory (slow) version")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        

        # get barycentric MP zeroth order energy
        e0_mp = 0
        for f,c,v in ci_vector:
            for ci in clustered_ham.clusters:
                e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v

        print(" Compute Matrix Vector Product:", flush=True)

        e2 = 0
        #Computei C(A)<A|H|X>DX<X|H|B>C(B) directly
        clusters = clustered_ham.clusters
        n_clusters = len(clusters)
        v = asci_vector
        for fock_l in v.fblocks(): 
            for fock_r in v.fblocks(): 
                confs_r = v[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                for terms_l in clustered_ham.terms:
                    for terms_r in clustered_ham.terms:
                        do_term = True 
                        
                        for ci in range(n_clusters):
                            #if delta_fock[ci][0] != terms_l[ci][0] + terms_r[ci][0] or delta_fock[ci][1] != terms_l[ci][1] + terms_r[ci][1]:
                            if fock_l[ci][0]+terms_l[ci][0] != fock_r[ci][0]+terms_r[ci][0] or fock_l[ci][1]+terms_l[ci][1] != fock_r[ci][1]+terms_r[ci][1]:
                                do_term = False
                                break

                        if do_term == False:
                            continue
        
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        # now go work
                        #print()
                        #print(do_term)
                        #print("fock_l :",fock_l)
                        #print("fock_r :",fock_r)

                        fock_x = [(fock_l[ci][0]+terms_l[ci][0],fock_l[ci][1]+terms_l[ci][1]) for ci in range(n_clusters)]
                        #assert(fock_x == [(fock_r[ci][0]+terms_r[ci][0],fock_r[ci][1]+terms_r[ci][1]) for ci in range(n_clusters)])
                       

                        # Check to make sure fock_x has an acceptable number of electrons
                        #
                        do_term1 = True
                        for ci in range(n_clusters):
                            if (fock_x[ci][0] < 0) or (fock_x[ci][1] < 0) or (fock_x[ci][0] > clusters[ci].n_orb) or (fock_x[ci][1] > clusters[ci].n_orb):
                                do_term1 = False
                        if do_term1 == False:
                            continue
                       
                        #print("fock_x :", fock_x)

                        #[print(t) for t in clustered_ham.terms[terms_r]]
                        #[print(t) for t in clustered_ham.terms[terms_l]]
                        configs_x = {}
                        for conf_r in v[fock_r]:
                            coeff = v[fock_r][conf_r]
                        
                            for term in clustered_ham.terms[terms_r]:
                                
                                # do local terms separately
                                if len(term.active) == 1:
                                    #start2 = time.time()
                                    
                                    ci = term.active[0]
                                        
                                    tmp = clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    
                                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])].shape[0])
                                    
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        if abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_x:
                                                configs_x[spi] = tmp[sp_idx] 
                                            else:
                                                configs_x[spi] += tmp[sp_idx] 
                                    #stop2 = time.time()
                            
                            
                                else:
                                    if len(term.active)>nbody_limit:
                                        continue
                                    #print(" term: ", term)
                                    state_sign = 1
                                    for oi,o in enumerate(term.ops):
                                        if o == '':
                                            continue
                                        if len(o) == 1 or len(o) == 3:
                                            for cj in range(oi):
                                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                                        
                                    opii = -1
                                    mats = []
                                    good = True
                                    for opi,op in enumerate(term.ops):
                                        if op == "":
                                            continue
                                        opii += 1
                                        ci = clusters[opi]
                                        try:
                                            oi = ci.ops[op][(fock_x[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                            mats.append(oi)
                                        except KeyError:
                                            good = False
                                            break
                                    if good == False:
                                        continue                        
                                    if len(mats) == 0:
                                        continue
                                    
                                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                                    
                                    
                                    #stop2 = time.time()
                                    
                                    
                                    #v_coeff = v[fock_r][conf_r]
                                    #tmp = state_sign * tmp.ravel() * v_coeff
                                    tmp = state_sign * tmp.ravel() * coeff 
                                    
                                    _abs = abs
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    for cacti,cact in enumerate(term.active):
                                        new_configs[cact] = range(mats[cacti].shape[0])
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                                        if _abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_x:
                                                configs_x[spi] = tmp[sp_idx] 
                                            else:
                                                configs_x[spi] += tmp[sp_idx] 
                            
                            #
                            # C(A)<A| H(fock) | X> now completed
                            #
                            #   now remove from configs in variational space from X 
                            print(len(configs_x))
                            fock_x = tuple(fock_x)
                            if fock_x in v.fblocks():
                                for config,coeff in v[fock_x].items():
                                    if config in configs_x:
                                        del configs_x[config]
                                        #print(" Remove:", config)
                            #[print(i,j) for i,j in configs_x.items()]
                            

                            #
                            #   Now form denominator
                            #
                            if pt_type == 'en':
                                print(" NYI!")
                                exit()
                            elif pt_type == 'mp':
                                start = time.time()
                                #   This is not really MP once we have rotated away from the CMF basis.
                                #   H = F + (H - F), where F = sum_I F(I)
                                #
                                #   After Tucker basis, we just use the diagonal of this fock operator. 
                                #   Not ideal perhaps, but better than nothing at this stage
                                for c in configs_x.keys():
                                    e0_X = 0
                                    for ci in clustered_ham.clusters:
                                        e0_X += ci.ops['H_mf'][(fock_x[ci.idx],fock_x[ci.idx])][c[ci.idx],c[ci.idx]]
                                    
                                    configs_x[c] /= e0_mp - e0_X
                                end = time.time()
                                #print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
                            
                            #
                            # C(A)<A| H(fock) | X> / delta E(X) now completed
                            #
            

                            # Now get C(B)<B|H|X>
                           
                            #HBX = {}
                            for conf_x,coef_x in configs_x.items():
                                #HBX[conf_x] = 0
                                for conf_l,coef_l in v[fock_l].items():
                                    #print(conf_l)
                                    for term in clustered_ham.terms[terms_r]:
                                        e2 += coef_x * term.matrix_element(fock_x,conf_x,fock_l,conf_l) * coef_l
                             
        print(" PT2 Energy Correction %18.12f"%e2)



    
        ecore = clustered_ham.core_energy
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2
# }}}

def compute_pt2_correction_lowmem(ci_vector, clustered_ham_in, e0, 
        thresh_asci     = 0,
        thresh_search   = 1e-12,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        opt_einsum      = True,
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction using low-memory (slow) version")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        
        
        # get barycentric MP zeroth order energy
        e0_mp = 0
        for f,c,v in ci_vector:
            for ci in clustered_ham_in.clusters:
                e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v

        import ray
        import tools_para
        if nproc==None:
            ray.init(object_store_memory=shared_mem)
        else:
            ray.init(num_cpus=nproc, object_store_memory=shared_mem)
    
        # precompute screening indices for operator mats
        screen = 1e-4        
        for ci in clustered_ham_in.clusters:
            ci.ops_screen = {}
            for o in ci.ops:
                ci.ops_screen[o] = {} 
                for f1,f2 in ci.ops[o]:
                    oi = ci.ops[o][(f1,f2)]
                    ci.ops_screen[o][(f1,f2)] = []
                    for I in range(oi.shape[1]):
                        nz_curr = []
                        for J in range(oi.shape[0]):
                            if len(oi.shape) > 2:
                                if np.amax(np.abs(oi[J,I,:])) > screen:
                                    #ci.ops_screen[o][(f1,f2)][I].append(J)
                                    nz_curr.append(J)
                            else:
                                if np.amax(np.abs(oi[J,I])) > screen:
                                    #ci.ops_screen[o][(f1,f2)][I].append(J)
                                    nz_curr.append(J)
                        ci.ops_screen[o][(f1,f2)].append(nz_curr)
                        #print(len(ci.ops_screen[o][(f1,f2)][I]) / oi.shape[0])
        

        h_id    = ray.put(clustered_ham_in)
        v_id    = ray.put(ci_vector)



        def matvec_update_with_new_configs(coeff_tensor, new_configs, configs, active, thresh_search=1e-12):
           # {{{
            nactive = len(active) 
           
            _abs = abs
        
        
            config_curr = [i[0] for i in new_configs]
            count = 0
            if nactive==2:
       
                for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                    try:
                        config_curr[active[0]] = new_configs[active[0]][I[0]] 
                        config_curr[active[1]] = new_configs[active[1]][I[1]] 
                    except:
                        print()
                        print(" Tensor: ", coeff_tensor.shape)
                        print(" Nz Idx: ", I)
                        print(" new_co: ", new_configs)
                        print(" active: ", active,flush=True)
                        exit()
                    key = tuple(config_curr)
                    if key not in configs:
                        configs[key] = coeff_tensor[I[0],I[1]]
                    else:
                        configs[key] += coeff_tensor[I[0],I[1]]
                    #count += 1
                #print(" nb2: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
        
                            
            elif nactive==3:
        
                for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                    config_curr[active[0]] = new_configs[active[0]][I[0]] 
                    config_curr[active[1]] = new_configs[active[1]][I[1]] 
                    config_curr[active[2]] = new_configs[active[2]][I[2]] 
                    key = tuple(config_curr)
                    if key not in configs:
                        configs[key] = coeff_tensor[I[0],I[1],I[2]]
                    else:
                        configs[key] += coeff_tensor[I[0],I[1],I[2]]
                    #count += 1
                #print(" nb3: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
        
            elif nactive==4:
        
                for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                    config_curr[active[0]] = new_configs[active[0]][I[0]] 
                    config_curr[active[1]] = new_configs[active[1]][I[1]] 
                    config_curr[active[2]] = new_configs[active[2]][I[2]] 
                    config_curr[active[3]] = new_configs[active[3]][I[3]] 
                    key = tuple(config_curr)
                    if key not in configs:
                        configs[key] = coeff_tensor[I[0],I[1],I[2],I[3]]
                    else:
                        configs[key] += coeff_tensor[I[0],I[1],I[2],I[3]]
                    #count += 1
                #print(" nb4: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
        
            else:
                # local terms should trigger a fail since they are handled separately 
                print(" Wrong value in update_with_new_configs")
                exit()
       
            #print(" Size of ndarray:   ", sys.getsizeof(coeff_tensor))
            #print(" Size of dictionary:", sys.getsizeof(configs))
        
            return 
        # }}}


        @ray.remote
        def parallel_work3(inp):
        # {{{
            fock_l  = inp[0]
            pt_type = inp[1]
            e0_mp   = inp[2]
            e2_worker = 0
            h = ray.get(h_id)
            v = ray.get(v_id)
            screen = 1e-4
            
            todo = 0
            for fock_r in v.fblocks():
                if fock_l >= fock_r:
                    todo += 1
            
            fidx = 0
            for fock_r in v.fblocks():
                if fock_l >= fock_r:
                    print('%7i/%-7i'%(fidx,todo),flush=True)
                    fidx += 1
                    scale = 1.0
                    if fock_l > fock_r:
                        scale = 2.0
                    
                    opt_einsum = True
                    clusters = h.clusters
                    
                    confs_r = v[fock_r]
                   
                    # find pairs of terms which transition to the same fock space for X
                    for terms_l in h.terms:
                        fock_x = [(fock_l[ci][0]+terms_l[ci][0],fock_l[ci][1]+terms_l[ci][1]) for ci in range(n_clusters)]

                        terms_r = tuple([(fock_x[ci][0] - fock_r[ci][0], fock_x[ci][1] - fock_r[ci][1]) for ci in range(len(clusters))])
                      
                        if terms_r not in h.terms:
                            continue
                        
                    
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        # Check to make sure fock_x has an acceptable number of electrons
                        do_term1 = True
                        for ci in range(n_clusters):
                            if (fock_x[ci][0] < 0) or (fock_x[ci][1] < 0) or (fock_x[ci][0] > clusters[ci].n_orb) or (fock_x[ci][1] > clusters[ci].n_orb):
                                do_term1 = False
                        if do_term1 == False:
                            continue
                        
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        assert(fock_x == [(fock_r[ci][0]+terms_r[ci][0],fock_r[ci][1]+terms_r[ci][1]) for ci in range(n_clusters)])
                        
                        #   We now need to go compute the contribution
                        #       <fock_l|terms_l|fock_X> DX <fock_X|terms_r|fock_r>
                        #
                        #       we will compute the two numerators separately
                        #           <fock_X|terms_l|fock_l>
                        #           <fock_X|terms_r|fock_r>
                        #
                        #       then add the denominator only for necessary elements
                        configs_xl = {}
                        configs_xr = {}
                        
                    
                        #
                        #   
                        #           <fock_X|terms_l|fock_l>
                        for conf_l in v[fock_l]:
                            coeff = v[fock_l][conf_l]
                            # {{{
                            for term in h.terms[terms_l]:
                                 
                                # do local terms separately
                                if len(term.active) == 1:
                                    #start2 = time.time()
                                    
                                    ci = term.active[0]
                                        
                                    tmp = clusters[ci].ops['H'][(fock_x[ci],fock_l[ci])][:,conf_l[ci]] * coeff 
                                    
                                    new_configs = [[i] for i in conf_l] 
                                    
                                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_l[ci])].shape[0])
                                    
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        if abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_xl:
                                                configs_xl[spi] = tmp[sp_idx] 
                                            else:
                                                configs_xl[spi] += tmp[sp_idx] 
                                    #stop2 = time.time()
                            
                            
                                else:
                                    state_sign = 1
                                    for oi,o in enumerate(term.ops):
                                        if o == '':
                                            continue
                                        if len(o) == 1 or len(o) == 3:
                                            for cj in range(oi):
                                                state_sign *= (-1)**(fock_l[cj][0]+fock_l[cj][1])
                                        
                                    nonzeros = []
                                    opii = -1
                                    mats = []
                                    good = True
                                    for opi,op in enumerate(term.ops):
                                        if op == "":
                                            continue
                                        opii += 1
                                        ci = clusters[opi]
                                        try:
                                            oi = ci.ops[op][(fock_x[ci.idx],fock_l[ci.idx])][:,conf_l[ci.idx],:]
                                        except KeyError:
                                            good = False
                                            break
                                       

#                                        nonzeros_curr = []
#                                        for K in range(oi.shape[0]):
#                                            if np.amax(np.abs(oi[K,:])) > screen:
#                                            #if np.amax(np.abs(oi[K,:])) > thresh_search/10:
#                                                nonzeros_curr.append(K)
                                        nonzeros_curr = ci.ops_screen[op][(fock_x[ci.idx],fock_l[ci.idx])][conf_l[ci.idx]]
                                        if len(nonzeros_curr) == 0:
                                            good = False
                                            break
                                        if len(nonzeros_curr)/oi.shape[0] < .5:
                                            oinz = oi[nonzeros_curr,:]
                                            mats.append(oinz)
                                            nonzeros.append(nonzeros_curr)
                                        else:
                                            mats.append(oi)
                                            nonzeros.append(range(oi.shape[0]))

                                    if good == False:
                                        continue                        
                                    if len(mats) == 0:
                                        continue
                                    
                                    I = term.ints * state_sign * coeff 
                                    tmp = np.einsum(term.contract_string_matvec, *mats, I, optimize=opt_einsum)
                                    #tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                                    
                                    new_configs = [[i] for i in conf_l] 
                                    for cacti,cact in enumerate(term.active):
                                        new_configs[cact] = nonzeros[cacti] 
                                        
                                    matvec_update_with_new_configs(tmp, new_configs, configs_xl, term.active, thresh_search)
                                    
                                    #stop2 = time.time()
                                    
                                    
#                                    tmp = state_sign * tmp.ravel() * coeff 
#                                    
#                                    _abs = abs
#                                    
#                                    new_configs = [[i] for i in conf_l] 
#                                    for cacti,cact in enumerate(term.active):
#                                        new_configs[cact] = range(mats[cacti].shape[0])
#                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
#                                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
#                                        if _abs(tmp[sp_idx]) > thresh_search:
#                                            if spi not in configs_xl:
#                                                configs_xl[spi] = tmp[sp_idx] 
#                                            else:
#                                                configs_xl[spi] += tmp[sp_idx] 
#                                        #try:
#                                        #    configs_xl[spi] += tmp[sp_idx] 
#                                        #except KeyError:
#                                        #    configs_xl[spi] = tmp[sp_idx] 
# }}}               
                        
                        #
                        #   
                        #           <fock_X|terms_r|fock_r>
                        for conf_r in v[fock_r]:
                            coeff = v[fock_r][conf_r]
                            # {{{
                            for term in h.terms[terms_r]:
                                 
                                # do local terms separately
                                if len(term.active) == 1:
                                    #start2 = time.time()
                                    
                                    ci = term.active[0]
                                        
                                    tmp = clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    
                                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])].shape[0])
                                    
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        if abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_xr:
                                                configs_xr[spi] = tmp[sp_idx] 
                                            else:
                                                configs_xr[spi] += tmp[sp_idx] 
                                    #stop2 = time.time()
                            
                            
                                else:
                                    state_sign = 1
                                    for oi,o in enumerate(term.ops):
                                        if o == '':
                                            continue
                                        if len(o) == 1 or len(o) == 3:
                                            for cj in range(oi):
                                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                                        
                                    nonzeros = []
                                    opii = -1
                                    mats = []
                                    good = True
                                    for opi,op in enumerate(term.ops):
                                        if op == "":
                                            continue
                                        opii += 1
                                        ci = clusters[opi]
                                        try:
                                            oi = ci.ops[op][(fock_x[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                        except KeyError:
                                            good = False
                                            break
                                       
                                        #nonzeros_curr = []
                                        #for K in range(oi.shape[0]):
                                        #    if np.amax(np.abs(oi[K,:])) > screen:
                                        #    #if np.amax(np.abs(oi[K,:])) > thresh_search/10:
                                        #        nonzeros_curr.append(K)
                                        # only do sparse when sparsity is greater than .5
                                        nonzeros_curr = ci.ops_screen[op][(fock_x[ci.idx],fock_r[ci.idx])][conf_r[ci.idx]]
                                        if len(nonzeros_curr) == 0:
                                            good = False
                                            break
                                        if len(nonzeros_curr)/oi.shape[0] < .5:
                                            oinz = oi[nonzeros_curr,:]
                                            mats.append(oinz)
                                            nonzeros.append(nonzeros_curr)
                                        else:
                                            mats.append(oi)
                                            nonzeros.append(range(oi.shape[0]))

                                    if good == False:
                                        continue                        
                                    if len(mats) == 0:
                                        continue
                                    
                                    I = term.ints * state_sign * coeff 
                                    tmp = np.einsum(term.contract_string_matvec, *mats, I, optimize=opt_einsum)
                                    #tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    for cacti,cact in enumerate(term.active):
                                        new_configs[cact] = nonzeros[cacti] 
                                        
                                    matvec_update_with_new_configs(tmp, new_configs, configs_xr, term.active, thresh_search)
                                    
                                    #stop2 = time.time()
                                    
                                    
#                                    #v_coeff = v[fock_r][conf_r]
#                                    #tmp = state_sign * tmp.ravel() * v_coeff
#                                    tmp = state_sign * tmp.ravel() * coeff 
#                                    
#                                    _abs = abs
#                                    
#                                    new_configs = [[i] for i in conf_r] 
#                                    for cacti,cact in enumerate(term.active):
#                                        new_configs[cact] = range(mats[cacti].shape[0])
#                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
#                                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
#                                        #try:
#                                        #    configs_xr[spi] += tmp[sp_idx] 
#                                        #except KeyError:
#                                        #    configs_xr[spi] = tmp[sp_idx] 
#                                        if _abs(tmp[sp_idx]) > thresh_search:
#                                            if spi not in configs_xr:
#                                                configs_xr[spi] = tmp[sp_idx] 
#                                            else:
#                                                configs_xr[spi] += tmp[sp_idx] 
# }}}               
                        
                        #
                        #
                        #   now remove from configs in variational space from X 
                        fock_x = tuple(fock_x)
                        if fock_x in v.fblocks():
                            for config,coeff in v[fock_x].items():
                                if config in configs_xl:
                                    del configs_xl[config]
                                if config in configs_xr:
                                    del configs_xr[config]
                        
                        #
                        #
                        #   Now compute correction
                        #
                        #   since we will loop over x, lets loop first over the smallest vector
                        if len(configs_xl) <= len(configs_xr):
                            configs_x1 = configs_xl
                            configs_x2 = configs_xr
                        else:
                            configs_x2 = configs_xl
                            configs_x1 = configs_xr
                        
                        for config1 in configs_x1.keys():
                            if config1 in configs_x2:
                    
                                #   form denominator
                                if pt_type == 'en':
                                    print(" NYI!")
                                    exit()
                                elif pt_type == 'mp':
                                    start = time.time()
                                    #   This is not really MP once we have rotated away from the CMF basis.
                                    #   H = F + (H - F), where F = sum_I F(I)
                                    #
                                    #   After Tucker basis, we just use the diagonal of this fock operator. 
                                    #   Not ideal perhaps, but better than nothing at this stage
                                    e0_X = 0
                                    for ci in h.clusters:
                                        e0_X += ci.ops['H_mf'][(fock_x[ci.idx],fock_x[ci.idx])][config1[ci.idx],config1[ci.idx]]
                                    
                                    #print("%12.8f, %12.8f"%(configs_x1[config1] , configs_x2[config1] ))
                                    #print( scale * configs_x1[config1] * configs_x2[config1] / (e0_mp - e0_X))
                                    e2_worker += scale * configs_x1[config1] * configs_x2[config1] / (e0_mp - e0_X)
                                    end = time.time()
                                    #print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
                            
                            
            return e2_worker# }}}
          
        #loop over fock blocks and set up jobs
        jobs = []
        e2 = 0
        #Computei C(A)<A|H|X>DX<X|H|B>C(B) directly
        clusters = clustered_ham_in.clusters
        n_clusters = len(clusters)
        v = asci_vector
        for fock_l in v.fblocks(): 
            inp = [fock_l, pt_type, e0_mp]
            jobs.append(inp)
                             

       
        print(" Number of ray jobs: ", len(jobs))
        #result_ids = [tools_para.parallel_work2.remote(i) for i in jobs]
        result_ids = [parallel_work3.remote(i) for i in jobs]
       
        e2 = 0
        start = time.time()
        while len(result_ids): 
            done_id, result_ids = ray.wait(result_ids) 
            e2 += ray.get(done_id[0])
            print(".",end='',flush=True)
        stop = time.time()
        print()
        print(" Time spent in pt2 energy: %12.2f" %( stop - start), flush=True)
        
        #out = ray.get(result_ids)
        #e2 = 0
        #for o in out:
        #    e2 += o 
        

        ray.shutdown()

    
        ecore = clustered_ham_in.core_energy
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2
# }}}

def compute_pt2_correction_lowmem_save(ci_vector, clustered_ham_in, e0, 
        thresh_asci     = 0,
        thresh_search   = 1e-12,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        opt_einsum      = True,
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction using low-memory (slow) version")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        

        # get barycentric MP zeroth order energy
        e0_mp = 0
        for f,c,v in ci_vector:
            for ci in clustered_ham_in.clusters:
                e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v

        import ray
        import tools_para
        if nproc==None:
            ray.init(object_store_memory=shared_mem)
        else:
            ray.init(num_cpus=nproc, object_store_memory=shared_mem)
    
        h_id    = ray.put(clustered_ham_in)
        v_id    = ray.put(ci_vector)


        @ray.remote
        def parallel_work3(inp):
        # {{{
            fock_l  = inp[0]
            pt_type = inp[1]
            e0_mp   = inp[2]
            e2_worker = 0
            h = ray.get(h_id)
            v = ray.get(v_id)
            
            thresh_search = 1e-12
            opt_einsum = True
            clusters = h.clusters
        
            for fock_r in v.fblocks(): 
                confs_r = v[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                for terms_l in clustered_ham_in.terms:
                    for terms_r in clustered_ham_in.terms:
                        do_term = True 
                        
                        for ci in range(n_clusters):
                            #if delta_fock[ci][0] != terms_l[ci][0] + terms_r[ci][0] or delta_fock[ci][1] != terms_l[ci][1] + terms_r[ci][1]:
                            if fock_l[ci][0]+terms_l[ci][0] != fock_r[ci][0]+terms_r[ci][0] or fock_l[ci][1]+terms_l[ci][1] != fock_r[ci][1]+terms_r[ci][1]:
                                do_term = False
                                break
                
                        if do_term == False:
                            continue
                
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        # now go work
                        fock_x = [(fock_l[ci][0]+terms_l[ci][0],fock_l[ci][1]+terms_l[ci][1]) for ci in range(n_clusters)]
                        assert(fock_x == [(fock_r[ci][0]+terms_r[ci][0],fock_r[ci][1]+terms_r[ci][1]) for ci in range(n_clusters)])
                       
                
                        # Check to make sure fock_x has an acceptable number of electrons
                        #
                        do_term1 = True
                        for ci in range(n_clusters):
                            if (fock_x[ci][0] < 0) or (fock_x[ci][1] < 0) or (fock_x[ci][0] > clusters[ci].n_orb) or (fock_x[ci][1] > clusters[ci].n_orb):
                                do_term1 = False
                        if do_term1 == False:
                            continue
                        
                        for conf_r in v[fock_r]:
                            configs_x = {}
                            coeff = v[fock_r][conf_r]
                            
                            for term in h.terms[terms_r]:
                                 
                                # do local terms separately
                                if len(term.active) == 1:
                                    #start2 = time.time()
                                    
                                    ci = term.active[0]
                                        
                                    tmp = clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    
                                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])].shape[0])
                                    
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        if abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_x:
                                                configs_x[spi] = tmp[sp_idx] 
                                            else:
                                                configs_x[spi] += tmp[sp_idx] 
                                    #stop2 = time.time()
                            
                            
                                else:
                                    state_sign = 1
                                    for oi,o in enumerate(term.ops):
                                        if o == '':
                                            continue
                                        if len(o) == 1 or len(o) == 3:
                                            for cj in range(oi):
                                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                                        
                                    opii = -1
                                    mats = []
                                    good = True
                                    for opi,op in enumerate(term.ops):
                                        if op == "":
                                            continue
                                        opii += 1
                                        ci = clusters[opi]
                                        try:
                                            oi = ci.ops[op][(fock_x[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                            mats.append(oi)
                                        except KeyError:
                                            good = False
                                            break
                                    if good == False:
                                        continue                        
                                    if len(mats) == 0:
                                        continue
                                    
                                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                                    
                                    
                                    #stop2 = time.time()
                                    
                                    
                                    #v_coeff = v[fock_r][conf_r]
                                    #tmp = state_sign * tmp.ravel() * v_coeff
                                    tmp = state_sign * tmp.ravel() * coeff 
                                    
                                    _abs = abs
                                    
                                    new_configs = [[i] for i in conf_r] 
                                    for cacti,cact in enumerate(term.active):
                                        new_configs[cact] = range(mats[cacti].shape[0])
                                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                                        if _abs(tmp[sp_idx]) > thresh_search:
                                            if spi not in configs_x:
                                                configs_x[spi] = tmp[sp_idx] 
                                            else:
                                                configs_x[spi] += tmp[sp_idx] 
                            
                            #
                            # C(A)<A| H(fock) | X> now completed
                            #
                            #   now remove from configs in variational space from X 
                            #print(len(configs_x))
                            fock_x = tuple(fock_x)
                            if fock_x in v.fblocks():
                                for config,coeff in v[fock_x].items():
                                    if config in configs_x:
                                        del configs_x[config]
                                        #print(" Remove:", config)
                            #[print(i,j) for i,j in configs_x.items()]
                            
                            
                            #
                            #   Now form denominator
                            #
                            if pt_type == 'en':
                                print(" NYI!")
                                exit()
                            elif pt_type == 'mp':
                                start = time.time()
                                #   This is not really MP once we have rotated away from the CMF basis.
                                #   H = F + (H - F), where F = sum_I F(I)
                                #
                                #   After Tucker basis, we just use the diagonal of this fock operator. 
                                #   Not ideal perhaps, but better than nothing at this stage
                                for c in configs_x.keys():
                                    e0_X = 0
                                    for ci in h.clusters:
                                        e0_X += ci.ops['H_mf'][(fock_x[ci.idx],fock_x[ci.idx])][c[ci.idx],c[ci.idx]]
                                    
                                    configs_x[c] /= e0_mp - e0_X
                                end = time.time()
                                #print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
                            
                            #
                            # C(A)<A| H(fock) | X> / delta E(X) now completed
                            #
                            
                            
                            # Now get C(B)<B|H|X>
                           
                            #print('fock_x:',fock_x)
                            #print(configs_x)
                            #HBX = {}
                            delta_fock2= tuple([(fock_x[ci][0]-fock_l[ci][0], fock_x[ci][1]-fock_l[ci][1]) for ci in range(len(clusters))])
                            for conf_x,coef_x in configs_x.items():
                                #HBX[conf_x] = 0
                                for conf_l,coef_l in v[fock_l].items():
                                    #print("a:")
                                    #print(fock_x, conf_x)
                                    #print(fock_l, conf_l)
                                    for term in h.terms[delta_fock2]:
                                        e2_worker += coef_x * term.matrix_element(fock_x,conf_x,fock_l,conf_l) * coef_l
                            
            return e2_worker# }}}
          
        #loop over fock blocks and set up jobs
        jobs = []
        e2 = 0
        #Computei C(A)<A|H|X>DX<X|H|B>C(B) directly
        clusters = clustered_ham_in.clusters
        n_clusters = len(clusters)
        v = asci_vector
        for fock_l in v.fblocks(): 
            inp = [fock_l, pt_type, e0_mp]
            jobs.append(inp)
                             

       
        print(" Number of ray jobs: ", len(jobs))
        #result_ids = [tools_para.parallel_work2.remote(i) for i in jobs]
        result_ids = [parallel_work3.remote(i) for i in jobs]
       
        e2 = 0
        start = time.time()
        while len(result_ids): 
            done_id, result_ids = ray.wait(result_ids) 
            e2 += ray.get(done_id[0])
            print(".",end='',flush=True)
        stop = time.time()
        print()
        print(" Time spent in pt2 energy: %12.2f" %( stop - start), flush=True)
        
        #out = ray.get(result_ids)
        #e2 = 0
        #for o in out:
        #    e2 += o 
        

        ray.shutdown()

    
        ecore = clustered_ham_in.core_energy
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2
# }}}

def compute_pt2_correction_lowmem_old(ci_vector, clustered_ham_in, e0, 
        thresh_asci     = 0,
        thresh_search   = 1e-12,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        opt_einsum      = True,
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction using low-memory (slow) version")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        

        # get barycentric MP zeroth order energy
        e0_mp = 0
        for f,c,v in ci_vector:
            for ci in clustered_ham_in.clusters:
                e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v

        import ray
        import tools_para
        if nproc==None:
            ray.init(object_store_memory=shared_mem)
        else:
            ray.init(num_cpus=nproc, object_store_memory=shared_mem)
    
        h_id    = ray.put(clustered_ham_in)
        v_id    = ray.put(ci_vector)


        @ray.remote
        def parallel_work3(inp):
        # {{{
            fock_r  = inp[0]
            fock_l  = inp[1]
            fock_x  = inp[2]
            terms_r = inp[3]
            terms_l = inp[4]
            pt_type = inp[5]
            e0_mp   = inp[6]
            e2_worker = 0
            h = ray.get(h_id)
            v = ray.get(v_id)
            
            thresh_search = 1e-12
            opt_einsum = True
            clusters = h.clusters
        
            for conf_r in v[fock_r]:
                coeff = v[fock_r][conf_r]
                configs_x = {}
                
                for term in h.terms[terms_r]:
                     
                    # do local terms separately
                    if len(term.active) == 1:
                        #start2 = time.time()
                        
                        ci = term.active[0]
                            
                        tmp = clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        
                        new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])].shape[0])
                        
                        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                            if abs(tmp[sp_idx]) > thresh_search:
                                if spi not in configs_x:
                                    configs_x[spi] = tmp[sp_idx] 
                                else:
                                    configs_x[spi] += tmp[sp_idx] 
                        #stop2 = time.time()
                
                
                    else:
                        state_sign = 1
                        for oi,o in enumerate(term.ops):
                            if o == '':
                                continue
                            if len(o) == 1 or len(o) == 3:
                                for cj in range(oi):
                                    state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                            
                        opii = -1
                        mats = []
                        good = True
                        for opi,op in enumerate(term.ops):
                            if op == "":
                                continue
                            opii += 1
                            ci = clusters[opi]
                            try:
                                oi = ci.ops[op][(fock_x[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                mats.append(oi)
                            except KeyError:
                                good = False
                                break
                        if good == False:
                            continue                        
                        if len(mats) == 0:
                            continue
                        
                        tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                        
                        
                        #stop2 = time.time()
                        
                        
                        #v_coeff = v[fock_r][conf_r]
                        #tmp = state_sign * tmp.ravel() * v_coeff
                        tmp = state_sign * tmp.ravel() * coeff 
                        
                        _abs = abs
                        
                        new_configs = [[i] for i in conf_r] 
                        for cacti,cact in enumerate(term.active):
                            new_configs[cact] = range(mats[cacti].shape[0])
                        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                            #print(" New config: %12.8f" %tmp[sp_idx], spi)
                            if _abs(tmp[sp_idx]) > thresh_search:
                                if spi not in configs_x:
                                    configs_x[spi] = tmp[sp_idx] 
                                else:
                                    configs_x[spi] += tmp[sp_idx] 
                
                #
                # C(A)<A| H(fock) | X> now completed
                #
                #   now remove from configs in variational space from X 
                #print(len(configs_x))
                fock_x = tuple(fock_x)
                if fock_x in v.fblocks():
                    for config,coeff in v[fock_x].items():
                        if config in configs_x:
                            del configs_x[config]
                            #print(" Remove:", config)
                #[print(i,j) for i,j in configs_x.items()]
                
                
                #
                #   Now form denominator
                #
                if pt_type == 'en':
                    print(" NYI!")
                    exit()
                elif pt_type == 'mp':
                    start = time.time()
                    #   This is not really MP once we have rotated away from the CMF basis.
                    #   H = F + (H - F), where F = sum_I F(I)
                    #
                    #   After Tucker basis, we just use the diagonal of this fock operator. 
                    #   Not ideal perhaps, but better than nothing at this stage
                    for c in configs_x.keys():
                        e0_X = 0
                        for ci in h.clusters:
                            e0_X += ci.ops['H_mf'][(fock_x[ci.idx],fock_x[ci.idx])][c[ci.idx],c[ci.idx]]
                        
                        configs_x[c] /= e0_mp - e0_X
                    end = time.time()
                    #print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
                
                #
                # C(A)<A| H(fock) | X> / delta E(X) now completed
                #
                
                
                # Now get C(B)<B|H|X>
                
                #HBX = {}
                for conf_x,coef_x in configs_x.items():
                    #HBX[conf_x] = 0
                    for conf_l,coef_l in v[fock_l].items():
                        #print(conf_l)
                        for term in h.terms[terms_r]:
                            e2_worker += coef_x * term.matrix_element(fock_x,conf_x,fock_l,conf_l) * coef_l
                
                return e2_worker# }}}
          
        #loop over fock blocks and set up jobs
        jobs = []
        e2 = 0
        #Computei C(A)<A|H|X>DX<X|H|B>C(B) directly
        clusters = clustered_ham_in.clusters
        n_clusters = len(clusters)
        v = asci_vector
        for fock_l in v.fblocks(): 
            for fock_r in v.fblocks(): 
                confs_r = v[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                for terms_l in clustered_ham_in.terms:
                    for terms_r in clustered_ham_in.terms:
                        do_term = True 
                        
                        for ci in range(n_clusters):
                            #if delta_fock[ci][0] != terms_l[ci][0] + terms_r[ci][0] or delta_fock[ci][1] != terms_l[ci][1] + terms_r[ci][1]:
                            if fock_l[ci][0]+terms_l[ci][0] != fock_r[ci][0]+terms_r[ci][0] or fock_l[ci][1]+terms_l[ci][1] != fock_r[ci][1]+terms_r[ci][1]:
                                do_term = False
                                break

                        if do_term == False:
                            continue
        
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        # now go work
                        fock_x = [(fock_l[ci][0]+terms_l[ci][0],fock_l[ci][1]+terms_l[ci][1]) for ci in range(n_clusters)]
                        assert(fock_x == [(fock_r[ci][0]+terms_r[ci][0],fock_r[ci][1]+terms_r[ci][1]) for ci in range(n_clusters)])
                       

                        # Check to make sure fock_x has an acceptable number of electrons
                        #
                        do_term1 = True
                        for ci in range(n_clusters):
                            if (fock_x[ci][0] < 0) or (fock_x[ci][1] < 0) or (fock_x[ci][0] > clusters[ci].n_orb) or (fock_x[ci][1] > clusters[ci].n_orb):
                                do_term1 = False
                        if do_term1 == False:
                            continue
                       
                        #print("fock_x :", fock_x)

                        #[print(t) for t in clustered_ham.terms[terms_r]]
                        #[print(t) for t in clustered_ham.terms[terms_l]]
                        
                        inp = [fock_r, fock_l, fock_x, terms_r, terms_l, pt_type, e0_mp]
                        #inp = [fock_r, fock_l, fock_x, terms_r, terms_l, h_id, v_id, pt_type, e0_mp]
                        jobs.append(inp)
                        
                        #for conf_r in v[fock_r]:
                        #    coeff = v[fock_r][conf_r]
                        #    inp = [conf_r, fock_r, coeff, fock_l, fock_x, terms_r, terms_l, pt_type, e0_mp]
                        #    #inp = [conf_r, fock_r, coeff, fock_l, fock_x, terms_r, terms_l, h_id, v_id, pt_type, e0_mp]
                        #    jobs.append(inp)
                        #    #e2 += parallel_work(inp)
                             
        print(" PT2 Energy Correction %18.12f"%e2)

       
        print(" Number of ray jobs: ", len(jobs))
        #result_ids = [tools_para.parallel_work2.remote(i) for i in jobs]
        result_ids = [parallel_work3.remote(i) for i in jobs]
       
        e2 = 0
        start = time.time()
        while len(result_ids): 
            done_id, result_ids = ray.wait(result_ids) 
            e2 += ray.get(done_id[0])
            print(".",end='',flush=True)
        stop = time.time()
        print()
        print(" Time spent in pt2 energy: %12.2f" %( stop - start), flush=True)
        
        #out = ray.get(result_ids)
        #e2 = 0
        #for o in out:
        #    e2 += o 
        

        ray.shutdown()

    
        ecore = clustered_ham_in.core_energy
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2
# }}}

def compute_pt2_correction_lowmem_pathos(ci_vector, clustered_ham_in, e0, 
        thresh_asci     = 0,
        thresh_search   = 1e-12,
        pt_type         = 'en',
        nbody_limit     = 4,
        matvec          = 4,
        batch_size      = 1,
        shared_mem      = 3e9, #1GB holds clustered_ham
        opt_einsum      = True,
        nproc           = None): 
    # {{{
        print()
        print(" Compute PT2 Correction using low-memory (slow) version")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |thresh_asci    : ", thresh_asci    )
        print("     |matvec         : ", matvec         )
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        

        # get barycentric MP zeroth order energy
        e0_mp = 0
        for f,c,vv in ci_vector:
            for ci in clustered_ham_in.clusters:
                e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * vv * vv

  



        v = ci_vector
        h = clustered_ham_in
         
        def parallel_work(inp):
        # {{{
            conf_r  = inp[0]
            fock_r  = inp[1]
            coeff   = inp[2]
            fock_l  = inp[3]
            fock_x  = inp[4]
            terms_r = inp[5]
            terms_l = inp[6]
            pt_type = inp[7]
            e0_mp   = inp[8]
            e2_worker = 0
            
            thresh_search = 1e-12
            opt_einsum = True
            clusters = h.clusters
            configs_x = {}
        
            for term in h.terms[terms_r]:
                 
                # do local terms separately
                if len(term.active) == 1:
                    #start2 = time.time()
                    
                    ci = term.active[0]
                        
                    tmp = clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    
                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_x[ci],fock_r[ci])].shape[0])
                    
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        if abs(tmp[sp_idx]) > thresh_search:
                            if spi not in configs_x:
                                configs_x[spi] = tmp[sp_idx] 
                            else:
                                configs_x[spi] += tmp[sp_idx] 
                    #stop2 = time.time()
            
            
                else:
                    state_sign = 1
                    for oi,o in enumerate(term.ops):
                        if o == '':
                            continue
                        if len(o) == 1 or len(o) == 3:
                            for cj in range(oi):
                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                        
                    opii = -1
                    mats = []
                    good = True
                    for opi,op in enumerate(term.ops):
                        if op == "":
                            continue
                        opii += 1
                        ci = clusters[opi]
                        try:
                            oi = ci.ops[op][(fock_x[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                            mats.append(oi)
                        except KeyError:
                            good = False
                            break
                    if good == False:
                        continue                        
                    if len(mats) == 0:
                        continue
                    
                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                    
                    
                    #stop2 = time.time()
                    
                    
                    #v_coeff = v[fock_r][conf_r]
                    #tmp = state_sign * tmp.ravel() * v_coeff
                    tmp = state_sign * tmp.ravel() * coeff 
                    
                    _abs = abs
                    
                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                        if _abs(tmp[sp_idx]) > thresh_search:
                            if spi not in configs_x:
                                configs_x[spi] = tmp[sp_idx] 
                            else:
                                configs_x[spi] += tmp[sp_idx] 
            
            #
            # C(A)<A| H(fock) | X> now completed
            #
            #   now remove from configs in variational space from X 
            #print(len(configs_x))
            fock_x = tuple(fock_x)
            if fock_x in v.fblocks():
                for config,coeff in v[fock_x].items():
                    if config in configs_x:
                        del configs_x[config]
                        #print(" Remove:", config)
            #[print(i,j) for i,j in configs_x.items()]
            
        
            #
            #   Now form denominator
            #
            if pt_type == 'en':
                print(" NYI!")
                exit()
            elif pt_type == 'mp':
                start = time.time()
                #   This is not really MP once we have rotated away from the CMF basis.
                #   H = F + (H - F), where F = sum_I F(I)
                #
                #   After Tucker basis, we just use the diagonal of this fock operator. 
                #   Not ideal perhaps, but better than nothing at this stage
                for c in configs_x.keys():
                    e0_X = 0
                    for ci in h.clusters:
                        e0_X += ci.ops['H_mf'][(fock_x[ci.idx],fock_x[ci.idx])][c[ci.idx],c[ci.idx]]
                    
                    configs_x[c] /= e0_mp - e0_X
                end = time.time()
                #print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
            
            #
            # C(A)<A| H(fock) | X> / delta E(X) now completed
            #
            
        
            # Now get C(B)<B|H|X>
            
            #HBX = {}
            for conf_x,coef_x in configs_x.items():
                #HBX[conf_x] = 0
                for conf_l,coef_l in v[fock_l].items():
                    #print(conf_l)
                    for term in h.terms[terms_r]:
                        e2_worker += coef_x * term.matrix_element(fock_x,conf_x,fock_l,conf_l) * coef_l
       
            #print(".",end='',flush=True)
            return e2_worker# }}}

          
        #loop over fock blocks and set up jobs
        jobs = []
        e2 = 0
        #Computei C(A)<A|H|X>DX<X|H|B>C(B) directly
        clusters = clustered_ham_in.clusters
        n_clusters = len(clusters)
        v = asci_vector
        for fock_l in v.fblocks(): 
            for fock_r in v.fblocks(): 
                confs_r = v[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                for terms_l in clustered_ham_in.terms:
                    for terms_r in clustered_ham_in.terms:
                        do_term = True 
                        
                        for ci in range(n_clusters):
                            #if delta_fock[ci][0] != terms_l[ci][0] + terms_r[ci][0] or delta_fock[ci][1] != terms_l[ci][1] + terms_r[ci][1]:
                            if fock_l[ci][0]+terms_l[ci][0] != fock_r[ci][0]+terms_r[ci][0] or fock_l[ci][1]+terms_l[ci][1] != fock_r[ci][1]+terms_r[ci][1]:
                                do_term = False
                                break

                        if do_term == False:
                            continue
        
                        # at this point we know that terms_l and term_r can connect fock_l and fock_r to the same fock_X
                        # now go work
                        fock_x = [(fock_l[ci][0]+terms_l[ci][0],fock_l[ci][1]+terms_l[ci][1]) for ci in range(n_clusters)]
                        assert(fock_x == [(fock_r[ci][0]+terms_r[ci][0],fock_r[ci][1]+terms_r[ci][1]) for ci in range(n_clusters)])
                       

                        # Check to make sure fock_x has an acceptable number of electrons
                        #
                        do_term1 = True
                        for ci in range(n_clusters):
                            if (fock_x[ci][0] < 0) or (fock_x[ci][1] < 0) or (fock_x[ci][0] > clusters[ci].n_orb) or (fock_x[ci][1] > clusters[ci].n_orb):
                                do_term1 = False
                        if do_term1 == False:
                            continue
                       
                        #print("fock_x :", fock_x)

                        #[print(t) for t in clustered_ham.terms[terms_r]]
                        #[print(t) for t in clustered_ham.terms[terms_l]]
                        configs_x = {}
                        for conf_r in v[fock_r]:
                            coeff = v[fock_r][conf_r]
                       
                        
                            inp = [conf_r, fock_r, coeff, fock_l, fock_x, terms_r, terms_l, pt_type, e0_mp]
                            jobs.append(inp)
                            #e2 += parallel_work(inp)
                             

        # run the jobs:

        import multiprocessing as mp
        from pathos.multiprocessing import ProcessingPool as Pool
        
        if nproc == None:
            pool = Pool()
        else:
            pool = Pool(processes=nproc)
    
        start = time.time()
        print(" Number of parallel jobs: ", len(jobs), flush=True)
        out = pool.map(parallel_work, jobs)
        stop = time.time()
       
        e2 = 0
        for o in out:
            e2 += o 
        


        pool.close()
        pool.join()
        pool.clear()
    
        ecore = clustered_ham_in.core_energy
        print(" Time spent in pt2 energy: %12.2f" %( stop - start), flush=True)
        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        return e2
# }}}

def extrapolate_pt2_correction(ci_vector, clustered_ham, e0, 
        nsteps          = 10,
        start           = 1,
        stop            = 1e-4,
        thresh_search   = 0,
        pt_type         = 'en',
        scale           = 'log', 
        matvec          = 3,
        nproc           = None): 
    # {{{
        print()
        print(" Extrapolate PT2 Correction")
        print("     |pt_type        : ", pt_type        )
        print("     |thresh_search  : ", thresh_search  )
        print("     |start          : ", start     )
        print("     |stop           : ", stop      )
        print("     |nsteps         : ", nsteps         )
        print("     |scale          : ", scale        )
        print(" NYI!")
        exit()
   
        print(" E0: ", e0)
        if scale=='log':
            stepsize = np.log((start - stop)/nsteps) 
            #stepsize = -(np.log(start) - np.log(stop))/nsteps 
            print(" Stepsize: ", stepsize) 
            asci1 = start
            asci2 = start*np.exp(stepsize)
            steps = []
            for asci_iter in range(nsteps):
                steps.append((asci1, asci2))
                asci1 *= np.exp(stepsize)
                asci2 *= np.exp(stepsize)
        else:
            print("NYI")
            exit()

        
        pt_vector = ClusteredState()
        count = 0
        for asci1,asci2 in steps:
            asci_v = ci_vector.copy()
            asci_v.clip(asci2, max=asci1)
            count += len(asci_v)
            print(" Collect configs between %12.2e and %12.2e: Size: %7i Norm: %12.8f" %( asci1, asci2, len(asci_v), asci_v.norm()))
            
            if len(asci_v) == 0:
                continue
            print(" Compute Matrix Vector Product:", flush=True)
            
            start = time.time()
            if matvec == 1:
                pt_vector_curr = matvec1_parallel1(clustered_ham, asci_v, nproc=nproc, thresh_search=thresh_search)
            elif matvec == 2:
                pt_vector_curr = matvec1_parallel2(clustered_ham, asci_v, nproc=nproc, thresh_search=thresh_search)
            elif matvec == 3:
                pt_vector_curr = matvec1_parallel3(clustered_ham, asci_v, nproc=nproc, thresh_search=thresh_search)
            elif matvec == 4:
                pt_vector_curr = matvec1_parallel4(clustered_ham, asci_v, nproc=nproc, thresh_search=thresh_search)
            else:
                print(" wrong matvec")
                exit()
            #pt_vector_curr = matvec1_parallel3(clustered_ham, asci_v, nproc=nproc, thresh_search=thresh_search)
            pt_vector.add(pt_vector_curr)
            stop = time.time()
            print(" Time spent in matvec: %12.2f" %( stop-start))
          
            e0_curr = ci_vector.dot(pt_vector) 
            print(" Zeroth-order energy: %12.8f " %e0_curr) 
            
            pt_vector.prune_empty_fock_spaces()
            
            tmp = ci_vector.dot(pt_vector)
            var = pt_vector.dot(pt_vector) - tmp*tmp
            print(" Variance Subspace: %12.8f" % var,flush=True)
            
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
            
            print(" Norm of CI vector = %12.8f" %ci_vector.norm())
            print(" Dimension of CI space: ", len(ci_vector))
            print(" Dimension of PT space: ", len(pt_vector))
            print(" Compute Denominator",flush=True)
            #exit()
            pt_vector.prune_empty_fock_spaces()
                
            
            # Build Denominator
            if pt_type == 'en':
                start = time.time()
                if nproc==1:
                    Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
                else:
                    Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
                end = time.time()
                print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
                
                denom = 1/(e0 - Hd)
            elif pt_type == 'mp':
                start = time.time()
                # get barycentric MP zeroth order energy
                e0_mp = 0
                for f,c,v in ci_vector:
                    for ci in clustered_ham.clusters:
                        e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v
                
                print(" Zeroth-order MP energy: %12.8f" %e0_mp, flush=True)
            
                #   This is not really MP once we have rotated away from the CMF basis.
                #   H = F + (H - F), where F = sum_I F(I)
                #
                #   After Tucker basis, we just use the diagonal of this fock operator. 
                #   Not ideal perhaps, but better than nothing at this stage
                denom = np.zeros(len(pt_vector))
                idx = 0
                for f,c,v in pt_vector:
                    e0_X = 0
                    for ci in clustered_ham.clusters:
                        e0_X += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]]
                    denom[idx] = 1/(e0_mp - e0_X)
                    idx += 1
                end = time.time()
                print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
            
            pt_vector_v = pt_vector.get_vector()
            pt_vector_v.shape = (pt_vector_v.shape[0])
            
            e2 = np.multiply(denom,pt_vector_v)
            e2 = np.dot(pt_vector_v,e2)
            
            ecore = clustered_ham.core_energy
            print(" PT2 Energy Correction = %12.8f" %e2)
            print(" PT2 Energy Total      = %12.8f" %(e0+e2+ecore))

        assert(count == len(ci_vector))

        return e2, pt_vector
# }}}

def run_hierarchical_sci(h,g,blocks,init_fspace,dimer_threshold,ecore):
    """
    compute a dimer calculation and figure out what states to retain for a larger calculation
    """
# {{{
    fclusters = []
    findex_list = {}
    for ci,c in enumerate(blocks):
        fclusters.append(Cluster(ci,c))
        #findex_list[c] = {}

    n_blocks = len(blocks)

    for ca in range(0,n_blocks):
        for cb in range(ca+1,n_blocks):
            f_idx = [ca,cb]
            s_blocks = [blocks[ca],blocks[cb]]
            s_fspace = ((init_fspace[ca]),(init_fspace[cb]))
            print("Blocks:",ca,cb)
            print(s_blocks)
            print(s_fspace)

            idx = [j for sub in s_blocks for j in sub]
            # h2
            h2 = h[:,idx] 
            h2 = h2[idx,:] 
            print(h2)
            g2 = g[:,:,:,idx] 
            g2 = g2[:,:,idx,:] 
            g2 = g2[:,idx,:,:] 
            g2 = g2[idx,:,:,:] 

            #do not want clusters to be wierdly indexed.
            print(len(s_blocks[0]))
            print(len(s_blocks[1]))
            s_blocks = [range(0,len(s_blocks[0])),range(len(s_blocks[0]),len(s_blocks[0])+len(s_blocks[1]))]

            s_clusters = []
            for ci,c in enumerate(s_blocks):
                s_clusters.append(Cluster(ci,c))

            #Cluster States initial guess
            ci_vector = ClusteredState()
            ci_vector.init(s_clusters,(s_fspace))
            ci_vector.print_configs()
            print(" Clusters:")
            [print(ci) for ci in s_clusters]

            #Clustered Hamiltonian
            clustered_ham = ClusteredOperator(s_clusters)
            print(" Add 1-body terms")
            clustered_ham.add_1b_terms(h2)
            print(" Add 2-body terms")
            clustered_ham.add_2b_terms(g2)

            do_cmf = 0
            if do_cmf:
                # Get CMF reference
                cmf(clustered_ham, ci_vector, h2, g2, max_iter=10,max_nroots=50)
            else:
                # Get vaccum reference
                for ci_idx, ci in enumerate(s_clusters):
                    print()
                    print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
                    ci.form_eigbasis_from_ints(h2,g2,max_roots=50)
                    print(" Build these local operators")
                    print(" Build mats for cluster ",ci.idx)
                    ci.build_op_matrices()

            ci_vector.expand_to_full_space()
            H = build_full_hamiltonian(clustered_ham, ci_vector)
            vguess = ci_vector.get_vector()
            #e,v = scipy.sparse.linalg.eigsh(H,1,v0=vguess,which='SA')
            e,v = scipy.sparse.linalg.eigsh(H,1,which='SA')
            idx = e.argsort()
            e = e[idx]
            v = v[:,idx]
            v0 = v[:,0]
            e0 = e[0]
            print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))
            ci_vector.zero()
            ci_vector.set_vector(v0)
            ci_vector.print_configs()

            for fspace, configs in ci_vector.data.items():
                for ci_idx, ci in enumerate(s_clusters):
                    print("fspace",fspace[ci_idx])
                    print("ci basis old\n",ci.basis[fspace[ci_idx]])
                    vec = ci.basis[fspace[ci_idx]]
                    #print(configs.items())
                    idx = []
                    for configi,coeffi in configs.items():
                        #print(configi[ci_idx],coeffi)
                        if abs(coeffi) > dimer_threshold:
                            if configi[ci_idx] not in idx:
                                idx.append(configi[ci_idx])
                    print("IDX of Cluster")
                    print(ci_idx,f_idx[ci_idx])
                    print(idx)
                    try:
                        findex_list[f_idx[ci_idx],fspace[ci_idx]] = sorted(list(set(findex_list[f_idx[ci_idx],fspace[ci_idx]]) | set(idx)))
                        #ci.cs_idx[fspace[ci_idx]] =  sorted(list(set(ci.cs_idx[fspace[ci_idx]]) | set(idx)))
                    except:
                        #print(findex_list[ci_idx][fspace[ci_idx]])
                        findex_list[f_idx[ci_idx],fspace[ci_idx]] = sorted(idx)
                        #ci.cs_idx[fspace[ci_idx]] =  sorted(idx) 

                    ### TODO 
                    # first : have to save these indices in fcluster obtect and not the s_cluster. so need to change that,
                    # second: have to move the rest of the code in the block to outside pair loop. loop over fspace
                    #           look at indices kept for fspace and vec also is in fspace. and then prune it.

                    print(vec.shape)
                    print(findex_list[f_idx[ci_idx],fspace[ci_idx]])
                    vec = vec[:,findex_list[f_idx[ci_idx],fspace[ci_idx]]]
                    #vec = vec[:,idx]
                    print("ci basis new\n")
                    print(vec)
                    fclusters[f_idx[ci_idx]].basis[fspace[ci_idx]] = vec
                    #print(ci.basis[fspace[ci_idx]])
                    print("Fock indices")
                    print(findex_list)

    for ci_idx, ci in enumerate(fclusters):
        ci.build_op_matrices()
        #print(findex_list[ci_idx])
    print("     *====================================================================.")
    print("     |         Tensor Product Selected Configuration Interaction          |")
    print("     *====================================================================*")

    #Cluster States initial guess
    ci_vector = ClusteredState()
    ci_vector.init(fclusters,(init_fspace))
    print(" Clusters:")
    [print(ci) for ci in fclusters]


    #Clustered Hamiltonian
    clustered_ham = ClusteredOperator(fclusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    ci_vector, pt_vector, etci, etci2,l  = tpsci_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-6, thresh_ci_clip=5e-4,asci_clip=0)
    #ci_vector, pt_vector, etci, etci2  = tp_cipsi(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-10, thresh_ci_clip=5e-6,asci_clip=0.01)

    print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
    print(" TPSCI(2):       %12.8f      Dim:%6d" % (etci2+ecore,len(pt_vector)))
# }}}
