import numpy as np
import scipy
import itertools as it
import time
from math import factorial
import copy as cp

class Determinant(object):
    def __init__(self,n_orb,Index,conf):
        self.Ind   = Index
        self.a_conf = conf[0][:]
        self.b_conf = conf[1][:]
        self.n_orb  = n_orb
        self.n_a    = len(self.a_conf)
        self.n_b    = len(self.b_conf)
        self.dim_a  = nCr(self.n_orb,self.n_a) 
        self.dim_b  = nCr(self.n_orb,self.n_b) 
        self.Ikaa = self.Ind// self.dim_b 
        self.Ikbb = self.Ind % self.dim_b
    def get_a_b(self):
        return self.Ikaa,self.Ikbb

def nCr(n, r):
    #{{{
    if n<r:
        return 0
    else:
        return factorial(n) // factorial(r) // factorial(n-r)
    #}}}

def get_index(M, a_index, n, k):
    #{{{
    ind = 0
    if n == k :
        return ind
    for i in range(0, k):
        ind += M[int(a_index[i]), i + 1]
    return int(ind)
    #}}}

def next_index(a_index, n, k):
    #{{{
    if len(a_index) == 0:
        return False
    if a_index[0] == n - k:
        return False

    pivot = 0

    for i in range(0, k - 1):
        if a_index[i + 1] - a_index[i] >= 2:
            a_index[i] += 1
            pivot += 1
            for j in range(0, i):
                a_index[j] = j
            break

    if pivot == 0:
        a_index[k - 1] += 1
        for j in range(0, k - 1):
            a_index[j] = j

    return True
    #}}}

def site_2_index(n_orb,site):
# {{{
    index = []
    for l in range(0,n_orb):
        if site[l] == 1:
            index.append(l)
    return index
# }}}

def index_2_site(n_orb,index):
# {{{
    site = np.zeros(n_orb)
    for i in range(0,n_orb):
        if i in index:
            site[i] = 1
        else:
            site[i] = 0
    return site
# }}}

def index_gen(i,n,k):
# {{{
    """
    returns the i-th combination of k numbers chosen from 0,1,2,...,n-1
    """
    quite = True
    c = []
    pos = i

    if not quite:
        print("\nBegininning  :   ",pos)

    for s in range(0,k):
        pivot = k - s

        #top = k - 2
        top = pivot - 1 

        if not quite:
            print("\nBegininning  :   ",pos)
            print("Pos  ",pos," top ",top,"   nel ",pivot)
        while pos - nCr(top,pivot) >= 0:
            #pos = pos - nCr(top,pivot)
            #print("pos: ",pos,"  ",top,"C",pivot," : ",nCr(top,pivot))
            top += 1
            if not quite:
                print("\nBegininning  :   ",pos)
                print("pos: ",pos,"  ",top,"C",pivot," : ",nCr(top,pivot))
        top = top - 1
        pos = pos - nCr(top,pivot)
        if not quite:
            print("\nBegininning  :   ",pos)
            print("selected: ",top,"   current pos: ",pos, "    substracted: ",nCr(top,pivot))
            print("\n")
        c.append(top)

    return c
# }}}

def run_fci(n_orb,n_a,n_b,h,g):
# {{{
    #####FULL CI

    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b

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

            ebindex = []
            for i in range(0,n_orb):
                if i not in b_index:
                    ebindex.append(i)

            #print(a_index,b_index,Ikaa * dim_b + Ikbb)
            #TYPE: A  
            #Diagonal Terms (equations from Deadwood paper) Eqn 3.3 

            #alpha alpha string
            for i in a_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h[i,i]
                for j in a_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g[i,i,j,j] -  g[i,j,i,j])

            #beta beta string
            for i in b_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h[i,i]
                for j in b_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g[i,i,j,j] -  g[i,j,i,j])

            #alpha beta string
            for i in a_index:
                for j in b_index:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += g[j,j,i,i]

            #print(a_site,b_site)


            #TYPE: B  
            #single alpha (equations from Deadwood paper) Eqn 3.8 

            for j in range(0,n_orb):
                for k in range(0, j):

                    if a_site[j] != a_site[k]:

                        asite2[j], asite2[k] = asite2[k], asite2[j]

                        aindex2 = []
                        for l in range(0,n_orb):
                            if asite2[l] == 1:
                                aindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if a_site[l] == 1:
                                sym += 1
                                
                        Sphase = (-1)**sym

                        Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)


                        mel =  h[j,k]

                        for i in a_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g[i,i,j,k] - g[i,j,i,k]) 

                        for i in b_index:
                            mel +=  g[i,i,j,k] 

                        Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  mel * Sphase
                        #Det[Ijaa * dim_b + Ikbb, Ikaa * dim_b + Ikbb] += 0.5 * mel * Sphase

                        
                        #print(a_site,b_site,asite2)
                        #print(Sphase)
                        asite2 = cp.deepcopy(a_site) #imp

            #TYPE: C  
            #single beta (equations from Deadwood paper) Eqn 3.9 

            for j in range(0,n_orb):
                for k in range(0, j):

                    if b_site[j] != b_site[k]:

                        bsite2[j], bsite2[k] = bsite2[k], bsite2[j]

                        bindex2 = []
                        for l in range(0,n_orb):
                            if bsite2[l] == 1:
                                bindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if b_site[l] == 1:
                                sym += 1
                                
                        Sphase = (-1)**sym

                        Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)


                        mel =  h[j,k]

                        for i in b_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g[i,i,j,k] - g[i,j,i,k]) 

                        for i in a_index:
                            mel +=  g[i,i,j,k] 

                        Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  mel * Sphase
                        #Det[Ikaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] += 0.5 * mel * Sphase

                            
                        #print(a_site,b_site,bsite2)
                        bsite2 = cp.deepcopy(b_site) #imp


            #TYPE: D
            #Double excitation in alpha string Eqn 3.15
            for j in a_index:
                for k in a_index:
                    for l in eaindex:
                        for m in eaindex:
                            if j > k and l > m :
                                asite2[j], asite2[l] = asite2[l], asite2[j]

                                ###Fermionic anti-symmetry
                                sym1 = 0
                                for i in range(min(j,l)+1, max(j,l)):
                                    if asite2[i] == 1:
                                        sym1 += 1
                                Sphase1 = (-1)**sym1


                                asite2[k], asite2[m] = asite2[m], asite2[k]

                                ###Fermionic anti-symmetry
                                sym2 = 0
                                for i in range(min(k,m)+1, max(k,m)):
                                    if asite2[i] == 1:
                                        sym2 += 1

                                Sphase2 = (-1)**sym2


                                aindex2 = []
                                for i in range(0,n_orb):
                                    if asite2[i] == 1:
                                        aindex2.append(i)


                                Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)

                                Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  (g[j,l,k,m] - g[j,m,k,l]) * Sphase1 * Sphase2

                                #print("a",j,k,l,m,a_site,asite2,sym1,sym2)
                                asite2 = cp.deepcopy(a_site) #imp

            #TYPE: E
            #Double excitation in beta string Eqn 3.15
            for j in b_index:
                for k in b_index:
                    for l in ebindex:
                        for m in ebindex:
                            if j < k and l < m :
                                bsite2[j], bsite2[l] = bsite2[l], bsite2[j]

                                ###Fermionic anti-symmetry
                                sym1 = 0
                                for i in range(min(j,l)+1, max(j,l)):
                                    if bsite2[i] == 1:
                                        sym1 += 1
                                Sphase1 = (-1)**sym1

                                bsite2[k], bsite2[m] = bsite2[m], bsite2[k]
                                ###Fermionic anti-symmetry
                                sym2 = 0
                                for i in range(min(k,m)+1, max(k,m)):
                                    if bsite2[i] == 1:
                                        sym2 += 1

                                Sphase2 = (-1)**sym2



                                bindex2 = []
                                for i in range(0,n_orb):
                                    if bsite2[i] == 1:
                                        bindex2.append(i)

                                Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)

                                Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  (g[j,l,k,m] - g[j,m,k,l]) * Sphase1 * Sphase2

                                #print("b",b_site,bsite2)
                                bsite2 = cp.deepcopy(b_site) #imp

            #TYPE: F
            #Single alpha Single beta Eqn 3.19

            for j in range(0,n_orb):
                for k in range(0, j):
                    if a_site[j] != a_site[k]:

                        asite2[j], asite2[k] = asite2[k], asite2[j]

                        aindex2 = []
                        for l in range(0,n_orb):
                            if asite2[l] == 1:
                                aindex2.append(l)

                        ###Fermionic anti-symmetry
                        sym = 0
                        for l in range(k+1,j):
                            if a_site[l] == 1:
                                sym += 1
                                
                        aSphase = (-1)**sym

                        Ijaa = get_index(nCr_a, aindex2, n_orb, n_a)


                        for l in range(0,n_orb):
                            for m in range(0, l):
                                if b_site[l] != b_site[m]:

                                    bsite2[l], bsite2[m] = bsite2[m], bsite2[l]


                                    bindex2 = []
                                    for n in range(0,n_orb):
                                        if bsite2[n] == 1:
                                            bindex2.append(n)

                                    ###Fermionic anti-symmetry
                                    sym = 0
                                    for n in range(m+1,l):
                                        if b_site[n] == 1:
                                            sym += 1
                                            
                                    bSphase = (-1)**sym

                                    Ijbb = get_index(nCr_b, bindex2, n_orb, n_b)


                                    Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=   g[j,k,l,m] * aSphase * bSphase
                                    #Det[Ijaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] +=  0.5 * g[j,k,l,m] * aSphase * bSphase
                                    bsite2 = cp.deepcopy(b_site) #imp

                        #print(a_site,asite2)
                        asite2 = cp.deepcopy(a_site) #imp


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp

    return Det
# }}}

