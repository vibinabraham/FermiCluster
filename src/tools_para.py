import numpy as np
import itertools 
import time
import ray

@ray.remote
def parallel_work(inp):
# {{{
    conf_r  = inp[0]
    fock_r  = inp[1]
    coeff   = inp[2]
    fock_l  = inp[3]
    fock_x  = inp[4]
    terms_r = inp[5]
    terms_l = inp[6]
    h_id    = inp[7]
    v_id    = inp[8]
    pt_type = inp[9]
    e0_mp   = inp[10]
    e2_worker = 0
    h = ray.get(h_id)
    v = ray.get(v_id)
    
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
