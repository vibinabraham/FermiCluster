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

n_orb = 5
U = 0
beta = 1.
n_a = n_orb//2
n_b = n_orb//2
nel = n_a

h, g = get_hubbard_params(n_orb,beta,U)

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,nel)

blocks = [[0,1,2,3],[4,5,6,7]]
blocks = [[0,1,2,3,4,5]]
blocks = [[0,1,2,3]]
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
cluster_ci_states = {}
#initialize the cluster class
for a in range(n_blocks):

    ha, ga = get_cluster_eri(blocks[a],h,g)  #Form integrals within a cluster
    n_orb = ha.shape[0]

    cluster[a] = Cluster(blocks[a],ha,ga)

    Cluster.init(cluster[a],blocks[a])

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
                cluster[a].read_block_states(efci[i],evec[:,i],n_a,n_b,i)
            cluster[a].read_block_states_2(efci,evec,n_a,n_b)
            print()


            #cluster_ci_states[a,n_a,n_b] = evec


if 0:
    from pyscf import gto, scf, ao2mo, tools, ci, fci
    print(fci.cistring.gen_cre_str_index([0,1,2,3],3))
    print(fci.cistring.gen_des_str_index([0,1,2,3],4))
    #print(fci.cistring.addr2str(4, 2, [0,1,2]))




def aI(n_orb,n_a,n_b,dna, dnb , cl_vec):
# {{{
    #####FULL CI
    #n_a = n_a + dna
    #n_b = n_b + dnb

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b


    dim_a_new = nCr(n_orb,n_a+dna)

    aket = np.zeros((1,dim_a_new * dim_b))

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

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_site,b_site)
            
            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in a_index:
                if dna != -1:
                    raise AssertionError("only for annihilation coded for now")

                #print("HAVE TO ADD ANTISYMMETRY")
                a2index.remove(ii)
                #print(a2index)

                Ijaa = get_index(nCr_a, a2index, n_orb, n_a + dna)
                #print(Ijaa)
                #print(Ijaa * dim_b + Ikbb )
                aket[0,Ijaa * dim_b + Ikbb] = cl_vec[Ikaa * dim_b + Ikbb] 

                a2index = cp.deepcopy(a_index)


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp

    print(cl_vec)
    print(aket)
    return aket
# }}}

#aI(4,3,3,cluster_ci_states[0,4,3])
#aI(4,4,4, -1, 0,cluster_ci_states[0,4,3][:,2])


def braJ_a_ketI(ci_vec, cj_vec, n_orb, n_a_i, n_b_i, n_a_j, n_b_j):
# {{{
    dna =  n_a_j - n_a_i
    dnb =  n_b_j - n_b_i
    assert(dna == -1)
    assert(dnb == +0)
    #####FULL CI
    n_a = n_a_i
    n_b = n_b_i

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b


    dim_a_new = nCr(n_orb,n_a_j)

    aket = np.zeros((1,dim_a_new * dim_b))

    des = np.zeros(n_orb)



    assert(cj_vec.shape[0] == (dim_a_new * dim_b))

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

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_site,b_site)

            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in a_index:

                #antisymmetry
                sym = a2index.index(ii)
                Sphase = (-1)** sym

                #annihilate ii orbital
                a2index.remove(ii)
                #print(a2index)

                #get index of the new CI space cj
                Ijaa = get_index(nCr_a, a2index, n_orb, n_a + dna)

                ##loop through elements of cj_vec and put elements
                #for ijaa in range(0,cj_vec.shape[0]):
                #    if Ijaa * dim_b + Ikbb == ijaa:
                #        des[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[ijaa] 

                des[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[Ijaa * dim_b + Ikbb] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    #print(ci_vec)
    #print(cj_vec)
    print("des")
    print(des)
    return des
# }}}

def braJ_A_ketI(ci_vec, cj_vec, n_orb, n_a_i, n_b_i, n_a_j, n_b_j):
# {{{
    dna =  n_a_j - n_a_i
    dnb =  n_b_j - n_b_i
    assert(dna == +1)
    assert(dnb == +0)
    #####FULL CI
    n_a = n_a_i
    n_b = n_b_i

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b


    dim_a_new = nCr(n_orb,n_a_j)

    aket = np.zeros((1,dim_a_new * dim_b))

    cre = np.zeros(n_orb)



    assert(cj_vec.shape[0] == (dim_a_new * dim_b))

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

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_site,b_site)

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

                ##loop through elements of cj_vec and put elements
                #for ijaa in range(0,cj_vec.shape[0]):
                #    if Ijaa * dim_b + Ikbb == ijaa:
                #        des[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[ijaa] 
                cre[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[Ijaa * dim_b + Ikbb] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    print("cre")
    print(cre)
    return cre
# }}}


n_a_i = 4
n_b_i = 3

n_a_j = 3
n_b_j = 3

braJ_a_ketI(cluster[0].block_states[n_a_i,n_b_i,1], cluster[0].block_states[n_a_j,n_b_j,1], cluster[0].n_orb, n_a_i, n_b_i, n_a_j, n_b_j)


n_a_i = 3
n_b_i = 3

n_a_j = 4
n_b_j = 3

braJ_A_ketI(cluster[0].block_states[n_a_i,n_b_i,1], cluster[0].block_states[n_a_j,n_b_j,1], cluster[0].n_orb, n_a_i, n_b_i, n_a_j, n_b_j)


# TODO: TDM matrices to compute
# singles:  b, B
# doubles:  AA, Aa, aa
#           AB, Ab, aB, ab
#           BB, Bb, bb
# triples:  AAa, AAb,  Aaa, Baa 
#           ABa, ABb,  Aab, Bab
#           BBa, BBb,  Abb, Bbb

def braJ_a_ketI(ci_vec, cj_vec, n_orb, n_a_i, n_b_i, n_a_j, n_b_j):
# {{{
    dna =  n_a_j - n_a_i
    dnb =  n_b_j - n_b_i
    assert(dna == -1)
    assert(dnb == +0)
    #####FULL CI
    n_a = n_a_i
    n_b = n_b_i

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_a_new = nCr(n_orb,n_a_j)

    dim_fci = dim_a * dim_b
    dim_I = dim_fci
    dim_J = dim_a_new * dim_b



    aket = np.zeros((1,dim_a_new * dim_b))

    des = np.zeros((n_orb,dim_I,dim_J))

    print(ci_vec.shape)
    print(cj_vec.shape)


    assert(cj_vec.shape[0] == (dim_a_new * dim_b))

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

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_site,b_site)

            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in a_index:

                #antisymmetry
                sym = a2index.index(ii)
                Sphase = (-1)** sym

                #annihilate ii orbital
                a2index.remove(ii)
                #print(a2index)

                #get index of the new CI space cj
                Ijaa = get_index(nCr_a, a2index, n_orb, n_a + dna)

                ##loop through elements of cj_vec and put elements
                #for ijaa in range(0,cj_vec.shape[0]):
                #    if Ijaa * dim_b + Ikbb == ijaa:
                #        des[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[ijaa] 

                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        des[ii,CI,CJ] += Sphase * ci_vec[Ikaa * dim_b + Ikbb,CI] * cj_vec[Ijaa * dim_b + Ikbb,CJ] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    #print(ci_vec)
    #print(cj_vec)
    #print("des")
    #print(des)
    return des
# }}}

def braJ_A_ketI(ci_vec, cj_vec, n_orb, n_a_i, n_b_i, n_a_j, n_b_j):
# {{{
    dna =  n_a_j - n_a_i
    dnb =  n_b_j - n_b_i
    assert(dna == +1)
    assert(dnb == +0)
    #####FULL CI
    n_a = n_a_i
    n_b = n_b_i

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_a_new = nCr(n_orb,n_a_j)

    dim_fci = dim_a * dim_b

    dim_I = dim_fci
    dim_J = dim_a_new * dim_b


    aket = np.zeros((1,dim_a_new * dim_b))

    cre = np.zeros((n_orb,dim_I,dim_J))



    assert(cj_vec.shape[0] == (dim_a_new * dim_b))

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

    Det = np.zeros((dim_fci, dim_fci))

    a_index = [i for i in range(n_a)]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1
    for aa in range(0,dim_a):

        for i in range(0,n_orb):
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)


        b_index = [i for i in range(n_b)]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_b):

            for i in range(0,n_orb):
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_site,b_site)

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

                ##loop through elements of cj_vec and put elements
                #for ijaa in range(0,cj_vec.shape[0]):
                #    if Ijaa * dim_b + Ikbb == ijaa:
                #        des[ii] += Sphase * ci_vec[Ikaa * dim_b + Ikbb] * cj_vec[ijaa] 
                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        cre[ii,CI,CJ] += Sphase * ci_vec[Ikaa * dim_b + Ikbb,CI] * cj_vec[Ijaa * dim_b + Ikbb,CJ] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp

        next_index(a_index, n_orb, n_a) #imp

    return cre
# }}}

n_a_i = 4
n_b_i = 3

n_a_j = 3
n_b_j = 3

#des = braJ_a_ketI(cluster[0].block_states_2[n_a_i,n_b_i], cluster[0].block_states_2[n_a_j,n_b_j], cluster[0].n_orb, n_a_i, n_b_i, n_a_j, n_b_j)
#print(des[:,1,1])


des_a = {}
cre_a = {}
for n_a in range(1,n_orb+1):
    for n_b in range(0,n_orb+1):
        print("destroy a",n_a,n_b)
        des_a[n_a,n_b] = braJ_a_ketI(cluster[0].block_states_2[n_a,n_b], cluster[0].block_states_2[n_a-1,n_b], cluster[0].n_orb, n_a, n_b, n_a-1, n_b)
        print(des_a[n_a,n_b][:,0,0])
for n_a in range(0,n_orb):
    for n_b in range(0,n_orb+1):
        print("create a",n_a,n_b)
        cre_a[n_a,n_b] = braJ_A_ketI(cluster[0].block_states_2[n_a,n_b], cluster[0].block_states_2[n_a+1,n_b], cluster[0].n_orb, n_a, n_b, n_a+1, n_b)


print(time.time() - ttt)



EE = 0
state = 9
n_a = 2
n_b = 0

#confirm equal to FCI
for p in range(0,n_orb):
    EE += h[p,p] *  np.dot(des_a[n_a,0][p,state,:],des_a[n_a,0][p,state,:])
    for q in range(p+1,n_orb):
        EE += h[p,q] *  np.dot(des_a[n_a,0][p,state,:],des_a[n_a,0][q,state,:])
        EE += h[q,p] *  np.dot(des_a[n_a,0][q,state,:],des_a[n_a,0][p,state,:])

print(EE)

HH = run_fci(n_orb,n_a,n_b,h,g)
efci,evec = np.linalg.eigh(HH)
print(efci)
print(efci[state])

EE = 0
state = 9
n_a = 3
for p in range(0,n_orb):
    EE += h[p,p] *  np.dot(des_a[n_a,0][p,state,:],des_a[n_a,0][p,state,:])
    for q in range(p+1,n_orb):
        EE += h[p,q] *  np.dot(des_a[n_a,0][p,state,:],des_a[n_a,0][q,state,:])
        EE += h[q,p] *  np.dot(des_a[n_a,0][q,state,:],des_a[n_a,0][p,state,:])
print(EE)
HH = run_fci(n_orb,n_a,n_b,h,g)
efci,evec = np.linalg.eigh(HH)
print(efci)
print(efci[state])


#print(evec)
#print(cluster[0].block_states_2[2,0])

#for p in range(0,n_orb):
#    for q in range(p+1,n_orb):
#        for r in range(q+1,n_orb):
#            for s in range(r+1,n_orb):
#                EE += g[p,q,s,r] *  np.dot(des_a[n_a,0][p,state,:],des_a[n_a,0][q,state,:])
