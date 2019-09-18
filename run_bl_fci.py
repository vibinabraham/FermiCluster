import numpy as np
import scipy
import itertools as it
import time
from math import factorial
import copy as cp

from hubbard_fn import *
from ci_fn import *

from Cluster import *

np.set_printoptions(suppress=True, precision=6, linewidth=1500)

ttt = time.time()

n_orb = 4
U = 1
beta = 1.
n_a = n_orb//2
n_b = n_orb//2
nel = n_a
np.random.seed(4)

h, g = get_hubbard_params(n_orb,beta,U)
if n_orb == 6:
    #h[2,3] = 0.8*h[2,3]
    #h[3,2] = 0.8*h[3,2]
    #h[0,5] = 0.5*h[0,5]
    #h[5,0] = 0.5*h[5,0]
    blocks = [[0,1,2],[3,4,5]]
    #h[0,3] = -beta
    #h[3,0] = -beta
    t = np.random.random((h.shape[0],h.shape[1])) * 0.3
    print(t)
    h += t 
    h += t.T 

if n_orb == 4:
    #h[0,3] = 0.8*h[0,3]
    #h[3,0] = 0.8*h[3,0]
    #h[1,2] = 0.5*h[1,2]
    #h[2,1] = 0.5*h[2,1]
    blocks = [[0,1],[2,3]]
    t = np.random.random((h.shape[0],h.shape[1])) * 0.3
    print(t)
    print(t[1,3])
    h += t 
    h += t.T 

print(h)

if 0:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,nel)

if n_orb == 8:
    blocks = [[0,1,2,3],[4,5,6,7]]
    t = np.random.random((h.shape[0],h.shape[1])) * 0.3
    print(t)
    h += t 
    h += t.T 
#blocks = [[0,1,2,3,4,5]]
#blocks = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
n_blocks = len(blocks)
assert(n_blocks ==2)

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

cluster = {}
#initialize the cluster class
for a in range(n_blocks):

    ha, ga = get_cluster_eri(blocks[a],h,g)  #Form integrals within a cluster
    bn_orb = ha.shape[0]

    cluster[a] = Cluster(blocks[a])

    cluster[a].init(blocks[a])

    print("One electron integral for Cluster:%4d"%a)
    print(ha)

    # Have to form symmetry block of each cluster corresponding to N_a = 0, N_b = 0 
    # to N_a = bn_orb, N_b = bn_orb
    for n_a in range(bn_orb+1):
        for n_b in range(bn_orb+1):
            HH = run_fci(bn_orb,n_a,n_b,ha,ga)
            efci,evec = np.linalg.eigh(HH)
            for i in range(0,efci.shape[0]):
                print(" E :%16.10f  Dim:%4d  Na:%4d Nb:%4d" %(efci[i],efci.shape[0],n_a,n_b))

            cluster[a].read_block_states(efci,evec,n_a,n_b)
            #cluster[a].read_block_states(efci,np.eye(efci.shape[0]),n_a,n_b)
            print()

def braJ_a_ketI(n_orb,cj_vec, ci_vec, naj, nbj, nai, nbi):
# {{{

    dna =  naj - nai
    dnb =  nbj - nbi
    assert(dna == -1)
    assert(dnb == +0)
    #####FULL CI

    dim_ai = nCr(n_orb,nai)
    dim_bi = nCr(n_orb,nbi)

    dim_aj = nCr(n_orb,naj)
    dim_bj = nCr(n_orb,nbj)

    dim_I = dim_ai * dim_bi
    dim_J = dim_aj * dim_bj
    assert(dim_bi == dim_bj)

    des = np.zeros((n_orb,dim_J,dim_I))

    assert(ci_vec.shape[0] == (dim_I))
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

    a_index = [i for i in range(nai)]

    for aa in range(0,dim_ai):

        Ikaa = get_index(nCr_a, a_index, n_orb, nai)

        b_index = [i for i in range(nbi)]

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_bi):

            Ikbb = get_index(nCr_b, b_index, n_orb, nbi)

            ### TYPE1: Form annihilation of alpha electron 
            a2index = cp.deepcopy(a_index)
            for ii in a_index:

                #antisymmetry
                sym = a2index.index(ii)
                Sphase = (-1)** sym

                #annihilate ii orbital
                a2index.remove(ii)

                #get index of the new space cj
                Ijaa = get_index(nCr_a, a2index, n_orb, nai + dna)

                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        des[ii,CJ,CI] += Sphase * ci_vec[Ikaa * dim_bi + Ikbb,CI] * cj_vec[Ijaa * dim_bi + Ikbb,CJ] 

                a2index = cp.deepcopy(a_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, nbi) #imp

        next_index(a_index, n_orb, nai) #imp

    return des
# }}}

def braJ_b_ketI(n_orb,cj_vec, ci_vec, naj, nbj, nai, nbi):
# {{{

    dna =  naj - nai
    dnb =  nbj - nbi
    assert(dna == +0)
    assert(dnb == -1)
    #####FULL CI

    dim_ai = nCr(n_orb,nai)
    dim_bi = nCr(n_orb,nbi)

    dim_aj = nCr(n_orb,naj)
    dim_bj = nCr(n_orb,nbj)

    dim_I = dim_ai * dim_bi
    dim_J = dim_aj * dim_bj
    assert(dim_ai == dim_aj)

    des = np.zeros((n_orb,dim_J,dim_I))

    assert(ci_vec.shape[0] == (dim_I))
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

    a_index = [i for i in range(nai)]

    for aa in range(0,dim_ai):

        Ikaa = get_index(nCr_a, a_index, n_orb, nai)

        b_index = [i for i in range(nbi)]

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)


        for bb in range(0,dim_bi):

            Ikbb = get_index(nCr_b, b_index, n_orb, nbi)

            ### TYPE1: Form annihilation of alpha electron 
            b2index = cp.deepcopy(b_index)
            for ii in b_index:

                #antisymmetry
                sym = b2index.index(ii)
                Sphase = (-1)** sym

                #annihilate ii orbital
                b2index.remove(ii)

                #get index of the new space cj
                Ijbb = get_index(nCr_b, b2index, n_orb, nbi + dnb)

                for CI in range(dim_I):
                    for CJ in range(dim_J):
                        des[ii,CJ,CI] += Sphase * ci_vec[Ikaa * dim_bi + Ikbb,CI] * cj_vec[Ikaa * dim_bj + Ijbb,CJ]  #ikaa*dimJ dimension of J 

                b2index = cp.deepcopy(b_index)

            #print(a_site,b_site)
            next_index(b_index, n_orb, nbi) #imp

        next_index(a_index, n_orb, nai) #imp

    return des
# }}}

for a in range(n_blocks):
    des_a = {}
    cre_a = {}
    bn_orb = cluster[a].n_orb
    for n_a in range(1,bn_orb+1):
        for n_b in range(0,bn_orb+1):
            print("destroy  a:%4d %4d"%(n_a,n_b))
            mata = braJ_a_ketI(cluster[a].n_orb, cluster[a].block_states[n_a-1,n_b], cluster[a].block_states[n_a,n_b], n_a-1, n_b, n_a, n_b)
            #cluster[a].read_tdms(mat,"a",n_a,n_b)
            cluster[a].read_ops(mata,"a", n_a-1, n_b, n_a, n_b)
            matA = np.transpose(mata, (0, 2, 1))
            cluster[a].read_ops(matA,"A",n_a,n_b,n_a-1,n_b)

    for n_a in range(0,bn_orb+1):
        for n_b in range(1,bn_orb+1):
            print("destroy  b:%4d %4d"%(n_a,n_b))
            matb = braJ_b_ketI(cluster[a].n_orb, cluster[a].block_states[n_a,n_b-1], cluster[a].block_states[n_a,n_b], n_a, n_b-1, n_a, n_b)
            cluster[a].read_ops(matb,"b", n_a, n_b-1, n_a, n_b)
            matB = np.transpose(matb, (0, 2, 1))
            cluster[a].read_ops(matB,"B",n_a,n_b,n_a-1,n_b)
    #for n_a in range(0,bn_orb):
    #    for n_b in range(0,bn_orb+1):
    #        print("create   a:%4d %4d"%(n_a,n_b))
    #        #cre_a[n_a,n_b] = braJ_A_ketI(cluster[a].block_states[n_a,n_b], cluster[a].block_states[n_a+1,n_b], cluster[a].n_orb, n_a, n_b)
    #        mat = braJ_A_ketI(cluster[a].block_states[n_a,n_b], cluster[a].block_states[n_a+1,n_b], bn_orb, n_a, n_b)
    #        cluster[a].read_tdms(mat,"A",n_a,n_b)


print("Run Time %10.6f"%(time.time() - ttt))


def get_energy_tight_binding(h,state_ind1,state_ind2,n_a,n_b):
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

def ketHbra(h,des_a,n_a,n_b):
# {{{
    n_orb = h.shape[0]
    dim = nCr(h.shape[0],n_a) * nCr(h.shape[0],n_b) 
    Hnew = np.zeros((dim,dim))
    for state_ind1 in range(0,dim):
        for state_ind2 in range(0,dim):
            EE = 0
            for p in range(0,n_orb):
                EE += h[p,p] *  np.dot(des_a["a",n_a,0][p,state_ind1,:],des_a["a",n_a,0][p,state_ind2,:])
                for q in range(p+1,n_orb):
                    EE += h[p,q] *  np.dot(des_a["a",n_a,0][p,state_ind1,:],des_a["a",n_a,0][q,state_ind2,:])
                    EE += h[q,p] *  np.dot(des_a["a",n_a,0][q,state_ind1,:],des_a["a",n_a,0][p,state_ind2,:])
            Hnew[state_ind1,state_ind2] = EE
    return Hnew
# }}}


if 1:
    #CAUTION: A and B are cluster index and not spin a b
    print("-----------------------------------------------------------")
    print("                 Cluster FCI")
    print("-----------------------------------------------------------")
    nel = 2
    HH = run_fci(n_orb,nel,0,h,g)
    efci,evec = np.linalg.eigh(HH)
    print("THE FCI Matrix:")
    print(HH)
    
    dim_fci = nCr(n_orb,nel)

    print("dim",dim_fci)
    Hfci = np.zeros((dim_fci,dim_fci))

    for a in range(n_blocks):
        for b in range(a+1,n_blocks):
            curr = 0
            for nA in range(0,nel+1):
                nB = nel - nA

                dim = nCr(bn_orb,nA) * nCr(bn_orb,nB)
                strt = curr
                stop = curr + dim


    for a in range(n_blocks):
        for b in range(a+1,n_blocks):
            ##CASE 1: 
            print("CASE1: H_a x I")
            curr = 0
            for nA in range(0,nel+1):
                nB = nel - nA
                dim = nCr(bn_orb,nA) * nCr(bn_orb,nB)

                ha, ga = get_cluster_eri(blocks[a],h,g)  #Form integrals within a cluster
                HA = run_fci(cluster[a].n_orb,nA,a,ha,ga)
                #print("HA\n",HA)
                HB = np.eye(nCr(cluster[b].n_orb,nB))
                #print("HB\n",HB)
                #print("Cluster states\n",cluster[a].block_states[nA,0])
                AHA = cluster[a].block_states[nA,0].T @ HA @ cluster[a].block_states[nA,0]
                BHB = cluster[b].block_states[nB,0].T @ HB @ cluster[b].block_states[nB,0]
                H0 = np.kron(AHA,BHB)
                print(curr,dim+curr)
                #print(H0.shape)

                Hfci[curr:dim+curr,curr:dim+curr] += H0
                curr += dim
                print(Hfci)
                

            ##CASE 2: 
            print("CASE2: I x H_b")
            curr = 0
            for nA in range(0,nel+1):
                nB = nel - nA
                dim = nCr(bn_orb,nA) * nCr(bn_orb,nB)

                hb, gb = get_cluster_eri(blocks[b],h,g)  #Form integrals within a cluster
                HB = run_fci(cluster[b].n_orb,nB,a,hb,gb)
                #print("HA\n",HA)
                HA = np.eye(nCr(cluster[a].n_orb,nA))
                #print("HB\n",HB)
                #print("Cluster staete\n",cluster[a].block_states[nA,0])
                AHA = cluster[a].block_states[nA,0].T @ HA @ cluster[a].block_states[nA,0]
                BHB = cluster[b].block_states[nB,0].T @ HB @ cluster[b].block_states[nB,0]
                H0 = np.kron(AHA,BHB)
                print(curr,dim+curr)
                #print(H0.shape)

                Hfci[curr:dim+curr,curr:dim+curr] += H0
                curr += dim

                print(Hfci)

            print("CASE3: H_ab")

            FTEMP = {}

            curr = 0
            for nA in range(0,nel+1):
                nB = nel - nA
                temp1 = 0
                temp2 = 0

                ket_dim = nCr(bn_orb,nA) * nCr(bn_orb,nB)
                ket_start = curr
                ket_stop = curr + ket_dim


                #print("ks", ket_start,ket_stop)

                #temp = np.kron(cluster[a].tdm_a["A",nA,0][1,0,:], cluster[b].tdm_a["a",nB,0][1,0,:])
                #Sigma1 = (-1)**(nA+nB-1)
                #Sigma2 = (-1)**(nA+nB+1)
                #Sigma1 = 1
                #Sigma2 = 1
                Sigma1 = (-1)**(nA)
                Sigma2 = (-1)**(nA)
                print(Sigma1,Sigma2)
                for p in range(cluster[a].n_orb):
                    for q in range(cluster[b].n_orb):
                        if nA != nel:
                            temp1 += Sigma1 * h[p,bn_orb+q] * np.kron(cluster[a].ops["A", nA+1, 0, nA, 0][p,:,:], cluster[b].ops["a", nB-1, 0, nB, 0][q,:,:])
                            #temp1 += Sigma1 * h[p,bn_orb+q] * np.kron(cluster[b].ops["a", nB-1, 0, nB, 0][q,:,:], cluster[a].ops["A", nA+1, 0, nA, 0][p,:,:])

                        if nA != 0:
                            temp2 -= Sigma2 * h[p,bn_orb+q] * np.kron(cluster[a].ops["a", nA-1, 0, nA, 0][p,:,:], cluster[b].ops["A", nB+1, 0, nB, 0][q,:,:])
                            #temp2 -= Sigma2 * h[p,bn_orb+q] * np.kron(cluster[b].ops["A", nB+1, 0, nB, 0][q,:,:], cluster[a].ops["a", nA-1, 0, nA, 0][p,:,:])

                FTEMP[(nA+1,nB-1,nA,nB)] = temp1
                FTEMP[(nA-1,nB+1,nA,nB)] = temp2
                print(tuple([nA+1,nB-1,nA,nB]))
                print(temp1)
                print(tuple([nA-1,nB+1,nA,nB]))
                print(temp2)

                curr += ket_dim

            print(FTEMP.keys())
                
            start0 = 0
            start1 = nCr(bn_orb,2) 
            start2 = start1 + nCr(bn_orb,1)  * nCr(bn_orb,1)
            start3 = start2 + nCr(bn_orb,2) 

            print(Hfci)
            Hfci[start0:start1,start1:start2] = FTEMP[(0,2,1,1)]
            Hfci[start1:start2,start0:start1] = FTEMP[(1,1,0,2)]
            Hfci[start1:start2,start2:start3] = FTEMP[(1,1,2,0)]
            Hfci[start2:start3,start1:start2] = FTEMP[(2,0,1,1)]


            print(Hfci)
            print(Hfci - Hfci.T)
            print(np.linalg.eigvalsh(Hfci))
            print(efci)
            print(HH)

    #print(np.linalg.eigvalsh(Hfci))
    #print(np.linalg.eigvalsh(HH))
    #print(h)

