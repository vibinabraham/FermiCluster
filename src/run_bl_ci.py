import numpy as np
import scipy
import itertools as it
import time
from math import factorial
import copy as cp

from hubbard_fn import *
from ci_fn import *

from Cluster import *

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

ttt = time.time()

n_orb = 8
U = 0
beta = 1.
n_a = n_orb//2
n_b = n_orb//2
nel = n_a

h, g = get_hubbard_params(n_orb,beta,U)

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,nel)

#blocks = [[0,1,2,3],[4,5,6,7]]
#blocks = [[0,1,2,3,4,5]]
#blocks = [[0,1,2,3]]
blocks = [[0,1,2,3,4]]
#blocks = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
n_blocks = len(blocks)

def get_cluster_eri(bl,h,g):
# {{{
    size_bl = len(bl)
    ha = np.zeros((size_bl,size_bl))
    ga = np.zeros((size_bl,size_bl,size_bl,size_bl))

    #AAAA
    for i,a in enumerate(bl):
        for j,b in enumerate(bl):
            ha[i,j] = h[a,b]
            for k,c in enumerate(bl):
                for l,d in enumerate(bl):
                    ga[i,j,k,l] = g[a,b,c,d]

    return ha,ga
# }}}

def get_block_eri(block,Cluster,g_spin,a,b,c,d):
# {{{
    """
    Gives the two electron integral living in respective blocks <AB|CD>
    """
    g_temp = np.zeros((Cluster[a].n_so_orb, Cluster[b].n_so_orb, Cluster[c].n_so_orb, Cluster[d].n_so_orb))

    #print("       a ")
    #print(Cluster[a].so_orb_list)
    #print("       b ")
    #print(Cluster[b].so_orb_list)
    #print("       c ")
    #print(Cluster[c].so_orb_list)
    #print("       d ")
    #print(Cluster[d].so_orb_list)

    for i,I in enumerate(Cluster[a].so_orb_list):
        for j,J in enumerate(Cluster[b].so_orb_list):
            for k,K in enumerate(Cluster[c].so_orb_list):
                for l,L in enumerate(Cluster[d].so_orb_list):
                    g_temp[i,j,k,l] = g_spin[I,J,K,L]

    return g_temp
# }}}

cluster = {}
#initialize the cluster class
for a in range(n_blocks):

    ha, ga = get_cluster_eri(blocks[a],h,g)  #Form integrals within a cluster
    n_orb = ha.shape[0]

    cluster[a] = Cluster(blocks[a])

    cluster[a].init(blocks[a])

    print("One electron integral for Cluster:%4d"%a)
    print(ha)

    # Have to form symmetry block of each cluster corresponding to N_a = 0, N_b = 0 
    # to N_a = n_orb, N_b = n_orb
    for n_a in range(n_orb+1):
        for n_b in range(n_orb+1):
            HH = run_fci(n_orb,n_a,n_b,ha,ga)
            efci,evec = np.linalg.eigh(HH)
            for i in range(0,efci.shape[0]):
                print(" E :%16.10f  Dim:%4d  Na:%4d Nb:%4d" %(efci[i],efci.shape[0],n_a,n_b))

            #cluster[a].read_block_states(efci,evec,n_a,n_b)
            cluster[a].read_block_states(efci,np.eye(efci.shape[0]),n_a,n_b)
            print()


if 0:
    from pyscf import gto, scf, ao2mo, tools, ci, fci
    print(fci.cistring.gen_cre_str_index([0,1,2,3],3))
    print(fci.cistring.gen_des_str_index([0,1,2,3],4))
    #print(fci.cistring.addr2str(4, 2, [0,1,2]))


# TODO: TDM matrices to compute
# singles:  b, B
# doubles:  AA, Aa, aa
#           AB, Ab, aB, ab
#           BB, Bb, bb
# triples:  AAa, AAb,  Aaa, Baa 
#           ABa, ABb,  Aab, Bab
#           BBa, BBb,  Abb, Bbb

def braJ_a_ketI(ci_vec, cj_vec, n_orb, n_a, n_b):
# {{{

    n_a_j = n_a - 1
    n_b_j = n_b

    dna =  n_a_j - n_a
    dnb =  n_b_j - n_b
    assert(dna == -1)
    assert(dnb == +0)
    #####FULL CI

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_a_j = nCr(n_orb,n_a_j)

    dim_fci = dim_a * dim_b
    dim_I = dim_fci
    dim_J = dim_a_j * dim_b

    des = np.zeros((n_orb,dim_I,dim_J))

    assert(cj_vec.shape[0] == (dim_J))

    #print("Number of Orbitals    :   %d" %n_orb)
    #print("Number of a electrons :   %d" %n_a)
    #print("Number of b electrons :   %d" %n_b)
    #print("Full CI dimension     :   %d" %dim_fci)

    #####STORED ALL THE BINOMIAL INDEX 
    nCr_a = np.zeros((n_orb, n_orb))
    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_a[i, j] = int(nCr(i, j))

    nCr_b = np.zeros((n_orb, n_orb))
    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_b[i, j] = int(nCr(i, j))


    ###     start the loop in |I> space

    a_index = [i for i in range(n_a)]

    for aa in range(0,dim_a):

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)

        b_index = [i for i in range(n_b)]

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)

            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in a_index:

                #antisymmetry
                sym = a2index.index(ii)
                Sphase = (-1)** sym

                #annihilate ii orbital
                a2index.remove(ii)

                #get index of the new space cj
                Ijaa = get_index(nCr_a, a2index, n_orb, n_a + dna)

                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        des[ii,CI,CJ] += Sphase * ci_vec[Ikaa * dim_b + Ikbb,CI] * cj_vec[Ijaa * dim_b + Ikbb,CJ] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    return des
# }}}

def braJ_A_ketI(ci_vec, cj_vec, n_orb, n_a, n_b):
# {{{
    n_a_j = n_a + 1
    n_b_j = n_b

    dna =  n_a_j - n_a
    dnb =  n_b_j - n_b
    assert(dna == +1)
    assert(dnb == +0)
    #####FULL CI

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_a_j = nCr(n_orb,n_a_j)

    dim_fci = dim_a * dim_b

    dim_I = dim_fci
    dim_J = dim_a_j * dim_b

    cre = np.zeros((n_orb,dim_I,dim_J))

    assert(cj_vec.shape[0] == (dim_J))

    #print("Number of Orbitals    :   %d" %n_orb)
    #print("Number of a electrons :   %d" %n_a)
    #print("Number of b electrons :   %d" %n_b)
    #print("Full CI dimension     :   %d" %dim_fci)

    #####STORED ALL THE BINOMIAL INDEX 
    nCr_a = np.zeros((n_orb, n_orb))
    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_a[i, j] = int(nCr(i, j))

    nCr_b = np.zeros((n_orb, n_orb))
    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_b[i, j] = int(nCr(i, j))

    ###     start the loop in |I> space

    a_index = [i for i in range(n_a)]

    for aa in range(0,dim_a):

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)

        b_index = [i for i in range(n_b)]

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)

        for bb in range(0,dim_b):

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)

            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in eaindex:

                #annihilate ii orbital
                a2index.append(ii)
                a2index = sorted(a2index, reverse=False)
                #print(a2index)

                #antisymmetry
                sym = a2index.index(ii)
                Sphase = (-1)** sym


                #get index of the new CI space cj
                Ijaa = get_index(nCr_a, a2index, n_orb, n_a + dna)

                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        cre[ii,CI,CJ] += Sphase * ci_vec[Ikaa * dim_b + Ikbb,CI] * cj_vec[Ijaa * dim_b + Ikbb,CJ] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    return cre
# }}}


des_a = {}
cre_a = {}
for n_a in range(1,n_orb+1):
    for n_b in range(0,n_orb+1):
        print("destroy  a:%4d %4d"%(n_a,n_b))
        des_a[n_a,n_b] = braJ_a_ketI(cluster[0].block_states[n_a,n_b], cluster[0].block_states[n_a-1,n_b], cluster[0].n_orb, n_a, n_b)
for n_a in range(0,n_orb):
    for n_b in range(0,n_orb+1):
        print("create   a:%4d %4d"%(n_a,n_b))
        cre_a[n_a,n_b] = braJ_A_ketI(cluster[0].block_states[n_a,n_b], cluster[0].block_states[n_a+1,n_b], cluster[0].n_orb, n_a, n_b)


print("Run Time %10.6f"%(time.time() - ttt))


def get_energy_tight_binding(state_ind,n_a,n_b):
# {{{
    EE = 0
    #confirm equal to FCI
    for p in range(0,n_orb):
        EE += h[p,p] *  np.dot(des_a[n_a,0][p,state_ind,:],des_a[n_a,0][p,state_ind,:])
        for q in range(p+1,n_orb):
            EE += h[p,q] *  np.dot(des_a[n_a,0][p,state_ind,:],des_a[n_a,0][q,state_ind,:])
            EE += h[q,p] *  np.dot(des_a[n_a,0][q,state_ind,:],des_a[n_a,0][p,state_ind,:])
    print("new    FCI: %16.10f na:%4d nb:%4d state_ind:%4d"%(EE,n_a,n_b,state_ind))
    return EE
# }}}

n_a = 3
n_b = 0
state_ind = 0

EE = get_energy_tight_binding(state_ind,n_a,n_b)

HH = run_fci(n_orb,n_a,n_b,h,g)
efci,evec = np.linalg.eigh(HH)
print("actual FCI: %16.10f na:%4d nb:%4d state_ind:%4d"%(efci[state_ind],n_a,n_b,state_ind))


def get_energy_tight_binding(state_ind1,state_ind2,n_a,n_b):
# {{{
    EE = 0
    #confirm equal to FCI
    for p in range(0,n_orb):
        EE += h[p,p] *  np.dot(des_a[n_a,0][p,state_ind1,:],des_a[n_a,0][p,state_ind2,:])
        for q in range(p+1,n_orb):
            EE += h[p,q] *  np.dot(des_a[n_a,0][p,state_ind1,:],des_a[n_a,0][q,state_ind2,:])
            EE += h[q,p] *  np.dot(des_a[n_a,0][q,state_ind1,:],des_a[n_a,0][p,state_ind2,:])
    print("new    FCI: %16.10f na:%4d nb:%4d state_ind:%4d"%(EE,n_a,n_b,state_ind))
    return EE
# }}}
Hnew = np.zeros((efci.shape[0],efci.shape[0]))
for state_ind1 in range(0,efci.shape[0]):
    for state_ind2 in range(0,efci.shape[0]):
        Hnew[state_ind1,state_ind2] = get_energy_tight_binding(state_ind1,state_ind2,n_a,n_b)
print(Hnew)
print(HH)
print(efci)
        
        
