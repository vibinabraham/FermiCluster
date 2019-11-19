
import numpy as np
import scipy
from math import factorial
import copy as cp

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

def run_casci(const, eff, h_active, g_active, n_orb, n_a, n_b):
    # {{{
    #####HAVE to add ta to get the effecinve ham
    h_active = h_active + eff


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

            #TYPE: A
            #Diagonal Terms (equations from Deadwood paper) Eqn 3.3


            #alpha alpha string
            for i in a_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in a_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #beta beta string
            for i in b_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in b_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #alpha beta string
            for i in a_index:
                for j in b_index:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += g_active[j,j,i,i]

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


                        mel =  h_active[j,k]

                        for i in a_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in b_index:
                            mel +=  g_active[i,i,j,k]

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


                        mel =  h_active[j,k]

                        for i in b_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in a_index:
                            mel +=  g_active[i,i,j,k]

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

                                Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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

                                Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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


                                    Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=   g_active[j,k,l,m] * aSphase * bSphase
                                    #Det[Ijaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] +=  0.5 * g[j,k,l,m] * aSphase * bSphase
                                    bsite2 = cp.deepcopy(b_site) #imp

                        #print(a_site,asite2)
                        asite2 = cp.deepcopy(a_site) #imp


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp


    return Det
    # }}}

def form_S2(n_orb, n_a, n_b):
    #{{{
    #####STORED ALL THE BINOMIAL INDEX
    nCr_a = np.zeros((n_orb, n_orb))

    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_a[i, j] = int(nCr(i, j))

    nCr_b = np.zeros((n_orb, n_orb))

    for i in range(0, n_orb):
        for j in range(0, n_orb):
            nCr_b[i, j] = int(nCr(i, j))
    """
    S2
    """

    dim_a = nCr(n_orb, n_a)
    dim_b = nCr(n_orb, n_b)

    nrange = range(n_orb)
    arange = range(n_a)
    brange = range(n_b)

    dim_tot = dim_a * dim_b

    S2 = np.zeros((dim_tot, dim_tot))
    a_index = [i for i in arange]

    #first config alpha string
    a_site = np.zeros(n_orb)
    for i in range(0, n_orb):
        if i in a_index:
            a_site[i] = 1


    for aa in range(0, dim_a):

        for i in nrange:
            if i in a_index:
                a_site[i] = 1
            else:
                a_site[i] = 0

        Ikaa = get_index(nCr_a, a_index, n_orb, n_a)
        asite2 = cp.deepcopy(a_site)

        b_index = [i for i in brange]

        #first config beta string
        b_site = np.zeros(n_orb)
        for i in range(0, n_orb):
            if i in b_index:
                b_site[i] = 1

        eaindex = []
        for i in range(0,n_orb):
            if i not in a_index:
                eaindex.append(i)

        for bb in range(0, dim_b):

            for i in nrange:
                if i in b_index:
                    b_site[i] = 1
                else:
                    b_site[i] = 0

            Ikbb = get_index(nCr_b, b_index, n_orb, n_b)
            bsite2 = cp.deepcopy(b_site)

            #print(a_index,b_index)

            ebindex = []
            for i in range(0,n_orb):
                if i not in b_index:
                    ebindex.append(i)


            #Diagonal Terms
            for i in a_index:
                if i in b_index:
                    S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.00
                else:
                    S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.75

            for i in b_index:
                if i in a_index:
                    S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.00
                else:
                    S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.75


            #SzSz
            for i in a_index:
                for j in a_index:
                    if i != j:
                        S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.25

            for i in b_index:
                for j in b_index:
                    if i != j:
                        S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] +=  0.25

            for i in a_index:
                for j in b_index:
                    if i !=j:
                        S2[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] -=  0.5



            #SpSm + SmSp
            for a in a_index:
                for b in b_index:
                    if a not in b_index:
                        if b not in a_index:

                            asite2[a] = 0
                            bsite2[a] = 1

                            asite2[b] = 1
                            bsite2[b] = 0

                            b_index2 = []
                            for i in range(0,n_orb):
                                if bsite2[i] == 1:
                                    b_index2.append(i)

                            a_index2 = []
                            for i in range(0,n_orb):
                                if asite2[i] == 1:
                                    a_index2.append(i)

                            #print(a_index,a_site,asite2)
                            assert(np.sum(asite2)==n_a)
                            assert(np.sum(bsite2)==n_b)

                            Ijaa = get_index(nCr_a, a_index2, n_orb, n_a)
                            Ijbb = get_index(nCr_b, b_index2, n_orb, n_b)


                            sym1 = 0
                            for i in range(a+1, n_orb):
                                if a_site[i] == 1:
                                    sym1 += 1
                            for i in range(0, a):
                                if b_site[i] == 1:
                                    sym1 += 1

                            for i in range(b+1, n_orb):
                                if asite2[i] == 1:
                                    sym1 += 1
                            for i in range(0, b):
                                if bsite2[i] == 1:
                                    sym1 += 1

                            Sphase = (-1)**(sym1)


                            S2[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=  Sphase
                            #print(a_site,b_site," -",a,b,sym1, "-->  ",asite2,bsite2)

                    asite2 = cp.deepcopy(a_site)
                    bsite2 = cp.deepcopy(b_site)


            next_index(b_index, n_orb, n_b)

        next_index(a_index, n_orb, n_a)
    return S2
    #}}}

def casci(h_active,g_active,h_full,g_full, n_orb, n_core, n_a, n_b):
# {{{

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

            #TYPE: A
            #Diagonal Terms (equations from Deadwood paper) Eqn 3.3

            a_core_active_index = []
            b_core_active_index = []

            for i in range(0,n_core):
                a_core_active_index.append(i)
                b_core_active_index.append(i)
            for i in a_index:
                a_core_active_index.append(n_core+i)
            for i in b_index:
                b_core_active_index.append(n_core+i)

            #print(a_core_active_index)
            #print(a_index)

            #alpha alpha string
            for i in a_core_active_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_full[i,i]
                for j in a_core_active_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_full[i,i,j,j] -  g_full[i,j,i,j])

            #beta beta string
            for i in b_core_active_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_full[i,i]
                for j in b_core_active_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_full[i,i,j,j] -  g_full[i,j,i,j])

            #alpha beta string
            for i in a_core_active_index:
                for j in b_core_active_index:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += g_full[j,j,i,i]

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


                        mel =  h_active[j,k]


                        J = j + n_core
                        K = k + n_core

                        for i in a_core_active_index:
                            if i != J: # and a_site[j] == 1:
                                mel +=  (g_full[i,i,J,K] - g_full[i,J,i,K])

                        for i in b_core_active_index:
                            mel +=  g_full[i,i,J,K]

                        """
                        for i in a_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in b_index:
                            mel +=  g_active[i,i,j,k]
                        """

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


                        mel =  h_active[j,k]

                        J = j + n_core
                        K = k + n_core

                        for i in b_core_active_index:
                            if i != J: # and a_site[j] == 1:
                                mel +=  (g_full[i,i,J,K] - g_full[i,J,i,K])

                        for i in a_core_active_index:
                            mel +=  g_full[i,i,J,K]

                        """
                        for i in b_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in a_index:
                            mel +=  g_active[i,i,j,k]
                        """

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

                                Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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

                                Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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


                                    Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=   g_active[j,k,l,m] * aSphase * bSphase
                                    #Det[Ijaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] +=  0.5 * g[j,k,l,m] * aSphase * bSphase
                                    bsite2 = cp.deepcopy(b_site) #imp

                        #print(a_site,asite2)
                        asite2 = cp.deepcopy(a_site) #imp


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp


    return Det
    # }}}

def run_fci(h_active,g_active,n_orb, n_a, n_b):
    # {{{
    dim_a = nCr(n_orb,n_a)
    dim_b = nCr(n_orb,n_b)

    dim_fci = dim_a * dim_b

    print("Number of Orbitals    :   %d" %n_orb)
    print("Number of a electrons :   %d" %n_a)
    print("Number of b electrons :   %d" %n_b)
    print("Full CI dimension     :   %d" %dim_fci)

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

            #TYPE: A
            #Diagonal Terms (equations from Deadwood paper) Eqn 3.3


            #alpha alpha string
            for i in a_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in a_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #beta beta string
            for i in b_index:
                Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += h_active[i,i]
                for j in b_index:
                    #if j < i:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += 0.5 * (g_active[i,i,j,j] -  g_active[i,j,i,j])

            #alpha beta string
            for i in a_index:
                for j in b_index:
                    Det[Ikaa*dim_b+Ikbb,Ikaa*dim_b+Ikbb] += g_active[j,j,i,i]


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


                        mel =  h_active[j,k]

                        for i in a_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in b_index:
                            mel +=  g_active[i,i,j,k]

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


                        mel =  h_active[j,k]

                        for i in b_index:
                            if i != j: # and a_site[j] == 1:
                                mel +=  (g_active[i,i,j,k] - g_active[i,j,i,k])

                        for i in a_index:
                            mel +=  g_active[i,i,j,k]

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

                                Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ikbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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

                                Det[Ikaa * dim_b + Ikbb, Ikaa * dim_b + Ijbb] +=  (g_active[j,l,k,m] - g_active[j,m,k,l]) * Sphase1 * Sphase2

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


                                    Det[Ikaa * dim_b + Ikbb, Ijaa * dim_b + Ijbb] +=   g_active[j,k,l,m] * aSphase * bSphase
                                    #Det[Ijaa * dim_b + Ijbb, Ikaa * dim_b + Ikbb] +=  0.5 * g[j,k,l,m] * aSphase * bSphase
                                    bsite2 = cp.deepcopy(b_site) #imp

                        #print(a_site,asite2)
                        asite2 = cp.deepcopy(a_site) #imp


            #print(a_site,b_site)
            next_index(b_index, n_orb, n_b) #imp


        next_index(a_index, n_orb, n_a) #imp


    return Det
    # }}}

def pt_infinity_full(H0,H1,E,l,no):
# {{{
    #   Form the full effective hamiltonian. This becomes non hermitian after second order

    #   H0 : zero order H in full nb dimension
    #   H1 : perturbation H in full nb dimension
    #   E  : The energy of the state
    #   l  : Dimension of P space
    #   no : max order of pt

    matsize = H0.shape[0]

    H0aa = cp.deepcopy(H0[0:l,0:l])
    H1aa = cp.deepcopy(H1[0:l,0:l])

    H1ab = cp.deepcopy(H1[0:l,l:matsize])
    H1ba = cp.deepcopy(H1[l:matsize,0:l])

    H0bb = cp.deepcopy(H0[l:matsize,l:matsize])
    H1bb = cp.deepcopy(H1[l:matsize,l:matsize])

    II  = np.eye(matsize-l,matsize-l)

    res = np.linalg.inv((E)*II - H0bb)

    I = np.eye(l)

    Heff0  = cp.deepcopy(H0aa)
    Heff1  = cp.deepcopy(H1aa)

    print("H0aa")
    print(Heff0)
    print()
    print("H1aa")
    print(Heff1)
    print()
    print("residual")
    print(res)
    print()

    WO = np.zeros((no,matsize - l,l))

    Heff = np.zeros((no+1,l,l))

    #WO0  zero order wave operator is P space
    WO1 = res @ H1ba


    WO[0,:,:] = cp.deepcopy(WO1)


    Heff[0,:,:] = cp.deepcopy(Heff1)

    Heff[1,:,:] = H1ab @ WO1

    Htest = cp.deepcopy(Heff0)
    Htest += Heff1
    Htest += Heff[1,:,:]
    e1 = np.linalg.eigvals(Htest)
    print("First Order Energy for Perturbation")
    idx = e1.argsort()[::+1]
    e1 = e1[idx]
    for m in range(0,l):
        print("%4d      %16.12f   %16.12f"  %(m,e1[m],e1[m]-e1[0]))
    print("")

    for i in range(1,no):
        WO[i,:,:] = H1bb @ WO[i-1,:,:]

        for k in range(0,i):
            WO[i,:,:] -= WO[i-k-1,:,:] @ Heff[k,:,:]

        WO[i,:,:] = res @ WO[i,:,:]

        Heff[i+1,:,:] = H1ab @ WO[i,:,:]

        Htest += Heff[i+1,:,:]

        e_val = np.linalg.eigvals(Htest)

        print("Order   %4d   " %(i))
        for m in range(0,l):
            #print(" %4d     %16.12f     " %(m,e_val[m]))
            print("%4d      %16.12f "  %(m,e_val[m]))
        print("")
    print("Final Energy for Perturbation")
    idx = e_val.argsort()[::+1]
    e_val = e_val[idx]
    for m in range(0,l):
        print("%4d      %16.12f   %16.12f"  %(m,e_val[m],e_val[m]-e_val[0]))
    print("")

# }}}

