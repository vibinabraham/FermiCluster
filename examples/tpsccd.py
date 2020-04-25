import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from tpsci import *
from pyscf_helper import *
from nbtucker import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=6, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues


def hv2_neutral(clustered_ham,ci_vector,term_thresh=1e-12):
    """
    Compute the action of H onto a sparse trial vector v at 2nd order nb
    returns a pt tensor object that does not have the 3 body interactions. 
    
    Currently used to generate the pt_space in neutral space

    """
# {{{
    clusters = clustered_ham.clusters
    sigma = OrderedDict() 


    for fock_ri, fock_r in enumerate(ci_vector.fblocks()):

        sigma[fock_r] = OrderedDict()
        for terms in clustered_ham.terms:

            fock_l = fock_r
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
                    if  (len(active) ==0 and len(term.active) ==2)  :
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

def expand_doubles_neutral(ci_vector):
# {{{
    clusters = ci_vector.clusters

    ci_vector.print_configs()

    for fspace in ci_vector.keys():
        for ci in ci_vector.clusters:
            for cj in ci_vector.clusters:
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

    ci_vector.print_configs()
    print(len(ci_vector))
    #ci_vector.add_single_excitonic_states()
    #ci_vector.expand_each_fock_space()
    #ci_vector.print()
    print(len(ci_vector))
    ci_vector.print_configs()

    return ci_vector

# }}}

def build_dd_block(clustered_ham,ci_vector,pt_tensor,iprint=0):
    """
    Build hamiltonian in basis of two different clustered states
    """
# {{{
    clusters = ci_vector.clusters
    Hdd = OrderedDict()
    Hnew = OrderedDict()
    print(ci_vector.keys())
    
    for fock_l, fock_item_l in pt_tensor.items():
        for fock_r, fock_item_r in pt_tensor.items():
            delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
            print(delta_fock)
            Hdd[fock_l,fock_r] = OrderedDict()
            Hnew[fock_l,fock_r] =  OrderedDict()
            for pair_key_l, pair_tensor_l in fock_item_l.items():
                for pair_key_r, pair_tensor_r in fock_item_r.items():
                    #print(pair_tensor_l.shape)
                    #print(pair_tensor_r.shape)
                    dim = pair_tensor_l.shape  + pair_tensor_r.shape
                    #print(dim)
                    Hdd[fock_l,fock_r][pair_key_l,pair_key_r] = np.zeros(dim)
                    Hnew[fock_l,fock_r][pair_key_l,pair_key_r] = np.zeros((9,9))
                    H = 0
                    H0d = np.zeros(9)

                    try:
                        terms = clustered_ham.terms[delta_fock]
                    except KeyError:
                        #shift_r += len(configs_r) 
                        continue 

                    #for term in terms:
                    #    print(len(term.active))
                    if fock_l != fock_r:
                        for term in terms:
                            print("matvec",term.contract_string_matvec,term.active)
                            #print(term.contract_string)
                            #Hdd[fock_r][pair_key] += nb22_block(term,fock_l,fock_r)
                            temp = nb22_block(term,fock_l,fock_r)
                    elif fock_l == fock_r:
                        if pair_key_l == pair_key_r:
                            print("fl=fr")
                            for term in terms:
                                if term.active ==list(pair_key_r):
                                    Hdd[fock_l,fock_r][pair_key_l,pair_key_r] += nb22_block(term,fock_l,fock_r)
                                    Hnew[fock_l,fock_r][pair_key_l,pair_key_r] +=  nb22_block(term,fock_l,fock_r).reshape(9,9)

                                    #print("Added term:",pair_key_l)
                                    #print(Hnew[fock_l,fock_r][pair_key_l,pair_key_r] )

                                elif len(term.active) ==1 and term.active[0] in list(pair_key_r):
                                    print("ACTIVE 1-body:")
                                    #print(term.active)
                                    #print(term.contract_string)
                                    #print(term.contract_string_matvec)
                                    me = nb22_block_diagonal(term,fock_l,fock_r)
                                    me  = np.delete(me,0,axis=0)

                                    if term.active[0] == list(pair_key_r)[0]:
                                        I2 = np.ones(pair_tensor_l.shape[0])
                                        me = np.kron(me,I2) 
                                        
                                    elif term.active[0] == list(pair_key_r)[1]:
                                        I1 = np.ones(pair_tensor_l.shape[0])
                                        me = np.kron(I1,me) 
                                    #print("Diagonal")
                                    H0d += me
                                    #print(H0d)
                                    tmp = np.eye(9)
                                    np.fill_diagonal(tmp, me)
                                    Hnew[fock_l,fock_r][pair_key_l,pair_key_r] += tmp
                                    Hdd[fock_l,fock_r][pair_key_l,pair_key_r] += tmp.reshape(dim)
                        print("Added term both:",pair_key_l)
                        print(Hnew[fock_l,fock_r][pair_key_l,pair_key_r] )
                        #Hdd[fock_l,fock_r][pair_key_l,pair_key_r] = Hnew[fock_l,fock_r][pair_key_l,pair_key_r].reshape(dim)

    return Hdd
# }}}

def nb22_block(term,fock_bra,fock_ket,iprint=0):
# {{{
    mats = []
    bra = [0]*len(term.clusters)
    state_sign = 1
    print("NEW TERM",term.contract_string)
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
            d = do[(fock_bra[oi],fock_ket[oi])]  #D(I,J,:,:...)
        except:
            return 0
        mats.append(d)

    me = 0.0
    if len(mats) == 0:
        return 0 
  
    #[print(m.shape) for m in mats]
    #print("Ints")
    #print(term.ints.shape)
    #print()
    #me = np.einsum(term.contract_string_mat,*mats,term.ints) * state_sign

    if len(term.active) != 3:

        arr = list(term.contract_string_matvec)
        arr.insert(0, 'z')
        for ai, an in enumerate(arr):
            if an == ',':
                arr.insert(ai+1, 'v')
                break        

        for ai, an in enumerate(arr):
            if an == '>':
                arr.insert(ai+1, 'zv')
                break        

        #arr.insert(-1, 'v')
        #print(arr)
        contract_string_mat = ''.join(map(str, arr))

        me = np.einsum(contract_string_mat,*mats,term.ints) * state_sign
        print(contract_string_mat)
        #print(me)

    elif len(term.active) == 3:

        # only take ground state for p'q term
        for opi,op in enumerate(mats):
            if len(op.shape) ==4:
                mats[opi] =  op[0,0,:,:]

        arr = list(term.contract_string)
        lstr = []
        pair_list = ['xy','zw']
        tmp = 0
        shift = 0
        for opi,op in enumerate(mats):
            if len(op.shape) ==3:
                lstr.append(pair_list[tmp])
                
                arr.insert(shift,pair_list[tmp])

                tmp += 1
                shift += len(op.shape)

            elif len(op.shape) ==2:
                shift += len(op.shape)+1


        arr.extend(lstr)
        #print(arr)
        contract_string_mat = ''.join(map(str, arr))
        print(contract_string_mat)

        me = np.einsum(contract_string_mat,*mats,term.ints) * state_sign

    if fock_bra == fock_ket and len(term.active) ==2:
        #me = me[1:,1:]
        dim1 = me.shape[0]*me.shape[2]
        dim2 = me.shape[1]*me.shape[3]
        value = 1.0 * me
        value = value.reshape((dim1,dim2))
        #print(value)
        #print(me)
        me  = np.delete(me,0,axis=0)
        me  = np.delete(me,0,axis=1)
        me  = np.delete(me,0,axis=2)
        me  = np.delete(me,0,axis=3)
        
        dim1 = me.shape[0]*me.shape[2]
        dim2 = me.shape[1]*me.shape[3]
        value = 1.0 * me
        value = value.reshape((dim1,dim2))
        #print("value")
        #print(value)

    if len(term.active) == 1:
        print("why")
        exit()
        
    return me
# }}}


def fill_diagonal(Hdd,H0d):
    """
    Fill diagonal of the Hdd block 
    """
# {{{
    for k1, v1 in pt_tensor.items():
        for pair_key1, c_1 in v1.items():
            Htemp = Hdd[k1,k1][pair_key1,pair_key1]
            Hdiag = H0d[k1][pair_key1]
            a,b,c,d = Htemp.shape

            Htemp = Htemp.reshape((Htemp.shape[0]*Htemp.shape[1],Htemp.shape[2]*Htemp.shape[3]))
            Hdiag = Hdiag.reshape(Hdiag.shape[0]*Hdiag.shape[1])

            tmp = np.eye(Hdiag.shape[0])
            np.fill_diagonal(tmp,Hdiag)
            Htemp -= tmp
            Htemp =  Htemp.reshape((a,b,c,d))
            Hdd[k1,k1][pair_key1,pair_key1] = Htemp

    return Hdd
# }}}

def hv_2(Hdd,pt_tensor):
    """
    linear Hv 
    """
# {{{
    lcc = OrderedDict()
    for k1, v1 in pt_tensor.items():
        lcc[k1] = OrderedDict()
        for pair_key1, c_1 in v1.items():
            for k2, v2 in pt_tensor.items():
                for pair_key2, c_2 in v2.items():
                    Hikjl = Hdd[k1,k2][pair_key1,pair_key2]
                    #tij = pt_tensor[k1][pair_key1]
                    tkl = pt_tensor[k2][pair_key2]
                    lcc[k1][pair_key1] = np.einsum('ijkl,kl->ij',Hikjl,tkl)
                    print(np.einsum('ijkl,kl->ij',Hikjl,tkl))
    return lcc
# }}}

def flatten_mat(sigma):
    """
    get the vector in numoy array form doubles excited methods
    """
# {{{
    fdim = 0
    for k, v in sigma.items():
        print(k)
        for config, value in v.items():
            dim = value.shape[0]*value.shape[1]
            print(config)
            print(value.shape)
            dim1 = value.shape[0]*value.shape[1]
            dim2 = value.shape[2]*value.shape[3]
            print(value.shape)
            print(dim1,dim2)
            value =value.reshape((dim1,dim2))

            print(value)
            fdim += dim
    print(fdim)

    """
    Ham = np.zeros((fdim,fdim))
    fdim1 = 0
    fdim2 = 0
    for k, v in sigma.items():
        #print(k)
        #print(" fock_space: ",k,end='')
        for config, value in v.items():
            dim1 = value.shape[0]*value.shape[1]
            dim2 = value.shape[2]*value.shape[3]
            tmp = value.swapaxes(1,2)
            tmp.shape = (dim1,dim2)
            Ham[fdim1:fdim1+dim1,fdim2:fdim2+dim2] = tmp
            fdim1 += dim1
            fdim2 += dim2
    return Ham
    """
    return value
# }}}


def build_hamiltonian_diagonal2(clustered_ham,ci_vector):
    """
    Build hamiltonian diagonal in basis in ci_vector
    """
# {{{
    clusters = ci_vector.clusters
    Hd = np.zeros((len(ci_vector)))
    
    shift = 0 
   
    idx = 0
    for fockspace, configs in ci_vector.items():
        for config, coeff in configs.items():
            print("New config")
            print("New config")
            print("New config")
            print("New config")
            delta_fock= tuple([(0,0) for ci in range(len(clusters))])
            terms = clustered_ham.terms[delta_fock]
            for term in terms:
                #print(term.contract_string_matvec)
                print(term.contract_string)
                print(term.active)
                Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
                print(term.matrix_element(fockspace,config,fockspace,config))
            idx += 1

    return Hd

# }}}

def nb22_block_diagonal(term,fock_bra,fock_ket,iprint=0):
    """
    Only for the diagonal coming from 1 active cluster. 
    In DD block, the diagonal has Ha X Ib + Ia X Hb type terms
    """
# {{{
    assert(len(term.active) == 1)

    mats = []
    bra = [0]*len(term.clusters)
    state_sign = 1
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
            d = do[(fock_bra[oi],fock_ket[oi])]  #D(I,J,:,:...)
        except:
            return 0
        mats.append(d)

    me = 0.0
    if len(mats) == 0:
        return 0 
  
    #[print(m.shape) for m in mats]
    #print("Ints")
    #print(term.ints.shape)
    #print()

    if len(mats[0].shape) == 4:
        mats[0] = np.einsum('iijk->ijk', mats[0])
    if len(mats[0].shape) == 6:
        mats[0] = np.einsum('iijklm->ijklm', mats[0])

    me = np.einsum(term.contract_string_matvec,*mats,term.ints) * state_sign
    return me
# }}}

for ri in range(0,20):
    ###     PYSCF INPUT
    r0 = 1.30 + 0.05 * ri
    molecule = '''
    C   {1}  {1}  {1} 
    H   {0}  {0}   0
    H    0   {0}  {0}
    H   {0}   0   {0}
    H    0    0    0
    '''.format(r0,r0/2)
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'ibmo'
    cas = True
    cas_nstart = 1
    cas_nstop =  9
    loc_start = 1
    loc_stop = 9
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

    
    r0 = 2.00 
    molecule = '''
    H   0   0   0
    H   0   0   1
    H   0  1.23  0
    H   0  1.23  1
    #H  02  02   0
    #H  02  02   1
    #H  03  00   0
    #H  03  00   1
    #H   0   2   3
    #H   0   2   4
    #H   0   0   8
    #H   0   0   9
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1],[2,3,4,5],[6,7]]
    init_fspace = ((1, 1), (2, 2), (1, 1))
    blocks = [[0,1],[2,3,4,5]]
    init_fspace = ((1, 1), (2, 2))
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))
    blocks = [[0,1],[2,3],[4,5]]
    init_fspace = ((1, 1), (1, 1),(1, 1))
    blocks = [[0,1],[2,3]]
    init_fspace = ((1, 1), (1, 1))


    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=cas,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

    #cluster using hcore
    idx = e1_order(h,cut_off = 2e-1)
    h,g = reorder_integrals(idx,h,g)
    print(h)
    if do_tci:
        n_blocks = len(blocks)

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
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)
        #clustered_ham.combine_common_terms(iprint=1)


        do_cmf = 1
        if do_cmf:
            # Get CMF reference
            e0,tm = cmf(clustered_ham, ci_vector, h, g, max_iter=20,max_nroots=5,dm_guess=None,thresh=1e-12)

        precompute_cluster_basis_energies(clustered_ham)


        pt_vector = expand_doubles_neutral(ci_vector.copy())
        #emp21,t1 = truncated_pt2(clustered_ham,ci_vector,pt_vector,method = 'mp2')
        #emp2 =  pt_doubles_IJ(clustered_ham,ci_vector)
        #print("MP2:     %16.8f"%emp21)
        #print("MP2:     %16.8f"%emp2)

        pt_tensor = hv2_neutral(clustered_ham,ci_vector)
        pt_tensor = build_block_hamiltonian_2(clustered_ham,ci_vector,pt_tensor)
        print_vec(pt_tensor)
        print("DDDDDDDDDDDDDDD")
        Hdd = build_dd_block(clustered_ham,ci_vector,pt_tensor)
        Hdiag = build_h0_diag2(clustered_ham,ci_vector,pt_tensor)

        print("mat before changin diag")
        ham = flatten_mat(Hdd)
        Hdd = fill_diagonal(Hdd,Hdiag)
        print("mat after chanign diag")
        ham = flatten_mat(Hdd)
        print("is the H0 correct?")
        emp22,t2 = truncated_pt2(clustered_ham,ci_vector,pt_vector,method = 'mplcc2',pt_order=30)


        #denom  v1  e2
        denom = OrderedDict()
        v1 = OrderedDict()
        emp2 = 0
        for k, v in  Hdiag.items():
            denom[k] = OrderedDict()
            v1[k] = OrderedDict()
            for t, c in v.items():
                print(t)
                denom[k][t] = 1/(e0-Hdiag[k][t])
                v1[k][t] = np.multiply(pt_tensor[k][t],denom[k][t])
                print("%16.8f"%np.einsum('ij,ij',v1[k][t],pt_tensor[k][t]))
                emp2 += np.einsum('ij,ij',v1[k][t],pt_tensor[k][t])

        #print(emp21)
        print(emp2)
        print("t2")
        t2.print_configs()
        print_vec(v1)


        pt_order = 30
        E_mpn = np.zeros((pt_order+1))         #PT energy
        E_mpn[0] = emp2
        E_corr = E_mpn[0] 

        vnew = v1
        Eold = E_corr
        print(" %6i  %16.8f  %16.8f "%(2,E_mpn[0],E_corr))
        for it in range(1,pt_order-1):
            h1 = hv_2(Hdd,v1)
            #print_vec(v1)
            #flatten_mat(Hdd)
            #print_vec(h1)
            for k, v in  Hdiag.items():
                for t, c in v.items():
                    v1[k][t] = np.multiply(h1[k][t],denom[k][t])
                    #print("%16.8f"%np.einsum('ij,ij',v1[k][t],pt_tensor[k][t]))
                    E_mpn[it] += np.einsum('ij,ij',v1[k][t],h1[k][t])
                #print(emp2)
            E_corr += E_mpn[it]
            print(" %6i  %16.8f  %16.8f "%(it+2,E_mpn[it],E_corr))
            #pt_tensor = v1
            #print_vec(pt_tensor)
            if abs(E_corr - Eold) < 1e-10:
                print("LCC:%16.8f "%E_corr)
                break
            else:
                Eold = E_corr
        #cepa(clustered_ham,ci_vector,pt_vector,'cepa0')
        exit()

        emp21,t2 = truncated_pt2(clustered_ham,ci_vector,pt_vector,method = 'mplcc2')
        t2.print_configs()

        ham = flatten_mat(Hdd)


    exit()


