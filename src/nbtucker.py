from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *


def hv2(clustered_ham,ci_vector,term_thresh=1e-12):
    """
    Compute the action of H onto a sparse trial vector v at 2nd order nb
    returns a pt tensor object that does not have the 3 body interactions. 
    
    Currently used to generate the pt_space

    """
# {{{
    clusters = clustered_ham.clusters
    sigma = OrderedDict() 


    for fock_ri, fock_r in enumerate(ci_vector.fblocks()):

        sigma[fock_r] = OrderedDict()
        for terms in clustered_ham.terms:

            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue

            active = []
            inactive = []
            for ind,fs in enumerate(fock_l):
                if fs != fock_r[ind]:
                    active.append(ind)
                else:
                    inactive.append(ind)
            if len(active) == 2 or len(active) == 0:
            #if len(active) <=2:

                if fock_l not in sigma.keys():
                    sigma[fock_l] = OrderedDict()

                print("FOCK",fock_l)
                print(" terms: ", terms)
                for term in clustered_ham.terms[terms]:
                    print(" term: ", term)
                    #if (active == term.active and len(active) ==2) or (len(active) ==0 and len(term.active) ==2)  or (len(active) == 2 and len(term.active) ==3) :
                    if (active == term.active and len(active) ==2) or (len(active) ==0 and len(term.active) ==2)  :
                        state_sign = 1
                        for oi,o in enumerate(term.ops):
                            if o == '':
                                continue
                            if len(o) == 1 or len(o) == 3:
                                for cj in range(oi):
                                    state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                            
                        print("ACTIVE",active)
                        for conf_ri, conf_r in enumerate(ci_vector[fock_r]):
                            print("CONF",conf_r)
                            
                            opii = -1
                            mats = []
                            good = True
                            print("contract_string       :", term.contract_string)
                            print("contract_string_matvec:", term.contract_string_matvec)
                            print("ACTIVE:",term.active)
                            for opi,op in enumerate(term.ops):
                                if op == "":
                                    continue
                                opii += 1
                                #print(opi,term.active)
                                ci = clusters[opi]
                                #ci = clusters[term.active[opii]]
                                try:
                                    oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                    mats.append(oi)
                                except KeyError:
                                    good = False
                                    break
                            if good == False:
                                continue                        
                                #break
                           
                            if len(mats) == 0:
                                continue

                            tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
                            #print("tmp",tmp.shape)
                            print(tmp.shape)
                            print(tmp)
                            if fock_r == fock_l:
                                print("REMOVING THE REF space config")
                                tmp  = np.delete(tmp,0,axis=0)
                                tmp  = np.delete(tmp,0,axis=1)

                            #v_coeff = ci_vector[fock_r][conf_r]

                            #tmp = state_sign * tmp.ravel() * v_coeff
                            tmp = state_sign * tmp
                            print(tmp)

                            #assert(len(term.active)==2)
                            print(term.active)

                            #sigma[fock_r][cidx,cjdx] = tmp
                            if len(term.active) ==3:
                                cidx = term.active[0]
                                cjdx = term.active[1]
                                ckdx = term.active[2]
                                try:
                                    sigma[fock_l][cidx,cjdx] += tmp[:,:,0]
                                    sigma[fock_l][cidx,ckdx] += tmp[:,0,:]
                                    sigma[fock_l][cjdx,ckdx] += tmp[0,:,:]
                                except KeyError:
                                    sigma[fock_l][cidx,cjdx] = tmp[:,:,0]
                                    sigma[fock_l][cidx,ckdx] = tmp[:,0,:]
                                    sigma[fock_l][cjdx,ckdx] = tmp[0,:,:]
                                
                            else:
                                cidx =term.active[0]
                                cjdx =term.active[1]
                                try:
                                    sigma[fock_l][cidx,cjdx] += tmp
                                except KeyError:
                                    sigma[fock_l][cidx,cjdx] = tmp


    #print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    return sigma
# }}}

def build_block_hamiltonian_2(clustered_ham,ci_vector,pt_tensor,iprint=0):
    """
    Build hamiltonian in basis of H0d for double excited space
    """
# {{{
    clusters = ci_vector.clusters
    #H0d = np.zeros((len(ci_vector),len(pt_vector)))
    H0d = OrderedDict()
    
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_r, fock_item in pt_tensor.items():
                #configs_r = pt_tensor[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                print(delta_fock)
                H0d[fock_r] = OrderedDict()
                for pair_key, pair_tensor in fock_item.items():
                    #print(pair_tensor)
                    H0d[fock_r][pair_key] = np.zeros_like(pair_tensor)

                    try:
                        terms = clustered_ham.terms[delta_fock]
                    except KeyError:
                        #shift_r += len(configs_r) 
                        continue 

                    if fock_l != fock_r:
                        for term in terms:
                            print("matvec",term.contract_string_matvec,term.active)
                            #print(term.contract_string)
                            H0d[fock_r][pair_key] += nb02_block(term,fock_l,fock_r)
                    elif fock_l == fock_r:
                        for term in terms:
                            print("PAIRRR KRY",pair_key)
                            if term.active ==list(pair_key):
                                print("matvec",term.contract_string_matvec,term.active)
                                H0d[fock_r][pair_key] += nb02_block(term,fock_l,fock_r)

    return H0d
# }}}

def nb02_block(term,fock_bra,fock_ket,iprint=0):
    """
    Compute the nb02 matrix for two fock spaces for each hamiltonian term
    """
# {{{
    mats = []
    bra = [0]*len(term.clusters)
    state_sign = 1
    print(term.contract_string)
    for oi,o in enumerate(term.ops):
        if o == '':
            continue
        if len(o) == 1 or len(o) == 3:
            for cj in range(oi):
                state_sign *= (-1)**(fock_ket[cj][0]+fock_ket[cj][1])
        try:
            do = term.clusters[oi].ops[o]
        except:
            print(" Couldn't find:", term)
            exit()
            return 0
        try:
            d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],:]  #D(I,J,:,:...)
        except:
            return 0
        mats.append(d)

    me = 0.0
    if len(mats) == 0:
        return 0 
  
    [print(m.shape) for m in mats]
    print("Ints")
    print(term.ints.shape)
    print()
    me = np.einsum(term.contract_string_matvec,*mats,term.ints) * state_sign
    print("final", me.shape)

    if len(term.active) == 3:
        for opi,op in enumerate(mats):
            print(op.shape)
            if len(op.shape) == 3:
                print("active",term.active[opi])
                me = np.rollaxis(me, opi)[0,:,:]

    if fock_bra == fock_ket and len(term.active) ==2:
        #me = me[1:,1:]
        me  = np.delete(me,0,axis=0)
        me  = np.delete(me,0,axis=1)
        
    return me
# }}}


def build_h0_diag2(clustered_ham,ci_vector,pt_tensor):
    """
    Build hamiltonian diagonal in basis in ci_vector as difference of cluster energies as in RSPT
    """
# {{{
    clusters = ci_vector.clusters
    
    ts = 0
    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    #H00 = H[tb0.start:tb0.stop,tb0.start:tb0.stop]
    
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[ts]
    print()

    Hdiag = OrderedDict()
   
    idx = 0
    for ci_fspace, ci_configs in ci_vector.items():
        for ci_config, ci_coeff in ci_configs.items():
            for fockspace, configs in pt_tensor.items():
                Hdiag[fockspace] = OrderedDict()
                print("FS",fockspace)
                active = []
                inactive = []
                for ind,fs in enumerate(fockspace):
                    if fs != ci_fspace[ind]:
                        active.append(ind)
                    else:
                        inactive.append(ind)
                            
                delta_fock= tuple([(fockspace[ci][0]-ci_fspace[ci][0], fockspace[ci][1]-ci_fspace[ci][1]) for ci in range(len(clusters))])
                

                for config, coeff in configs.items():
                    print(config)
                    print(coeff)
                    delta_fock= tuple([(0,0) for ci in range(len(clusters))])

                    
                    coeff = E0 * np.ones((coeff.shape[0],coeff.shape[1]))
                    cidx = config[0]
                    cjdx = config[1]
                    if len(active) == 2:
                        for i in range(coeff.shape[0]):
                            for j in range(coeff.shape[1]):
                                coeff[i,j] += clusters[cidx].energies[fockspace[cidx]][i]
                                coeff[i,j] += clusters[cjdx].energies[fockspace[cjdx]][j]
                                coeff[i,j] -= clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]
                                coeff[i,j] -= clusters[cjdx].energies[ci_fspace[cjdx]][ci_config[cjdx]]
                    if len(active) == 0:
                        for i in range(0,coeff.shape[0]):
                            for j in range(0,coeff.shape[1]):
                                coeff[i,j] += clusters[cidx].energies[fockspace[cidx]][i+1]
                                coeff[i,j] += clusters[cjdx].energies[fockspace[cjdx]][j+1]
                                coeff[i,j] -= clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]
                                coeff[i,j] -= clusters[cjdx].energies[ci_fspace[cjdx]][ci_config[cjdx]]

                    Hdiag[fockspace][cidx,cjdx] = coeff
                        
    return Hdiag

    # }}}

def pt_doubles_IJ(clustered_ham,ci_vector):
# {{{
    ts = 0
    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    print(H00)
    E0,V0 = np.linalg.eigh(H00)
    e0 = E0[ts]
    #pt space generate
    sigma = hv2(clustered_ham,ci_vector)

    #hv matrix
    sigma = build_block_hamiltonian_2(clustered_ham,ci_vector,sigma,iprint=0)
    #print_vec(sigma)

    #diag
    Hdiag = build_h0_diag2(clustered_ham,ci_vector,sigma)
    
    #denom  v1  e2
    denom = OrderedDict()
    v1 = OrderedDict()
    emp2 = 0
    for k, v in  Hdiag.items():
        denom[k] = OrderedDict()
        v1[k] = OrderedDict()
        for t, c in v.items():
            denom[k][t] = 1/(e0-Hdiag[k][t])
            v1[k][t] = np.multiply(sigma[k][t],denom[k][t])
            emp2 += np.einsum('ij,ij',v1[k][t],sigma[k][t])
    return emp2
# }}}


def print_vec(sigma):
    """
    print the ordered dict for doubles excited methods
    """
# {{{
    fdim = 0
    for k, v in sigma.items():
        #print(k)
        #print(" fock_space: ",k,end='')
        [print(" Cluster %-2i(%ia:%ib) "%(fii,fi[0],fi[1]),end='') for fii,fi in enumerate(k)] 
        for config, value in v.items():
            print("Excited Clusters:",config,end="")
            dim = np.prod(value.shape)
            print(" Dim:%4d     "%(dim))
            fdim += dim
            for ni in range(value.shape[0]):
                for nj in range(value.shape[1]):
                    print("%4d %4d %16.8f"%(ni,nj,value[ni,nj]))
# }}}

def flatten_vec(sigma):
    """
    get the vector in numoy array form doubles excited methods
    """
# {{{
    fdim = 0
    for k, v in sigma.items():
        for config, value in v.items():
            dim = np.prod(value.shape)
            fdim += dim
    print(fdim)

    vec = np.zeros(fdim)
    fdim = 0
    for k, v in sigma.items():
        #print(k)
        #print(" fock_space: ",k,end='')
        for config, value in v.items():
            tmp = value.ravel()
            dim = np.prod(value.shape)
            vec[fdim:fdim+dim] = tmp
            dim = np.prod(value.shape)
            fdim += dim
    return vec
# }}}
