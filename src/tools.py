import numpy as np
import scipy
import itertools as it
import copy as cp
from helpers import *

from ClusteredOperator import *
from ClusteredState import *

def matvec1(h,v):
    print(" Compute matrix vector product:")
    clusters = h.clusters
    sigma = ClusteredState(clusters) 
    for fock_ri, fock_r in enumerate(v.fblocks()):

        for terms in h.terms:
            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue
            
            print(fock_l, "<--", fock_r)

            configs_l = OrderedDict()

            for term in h.terms[terms]:
                print(" term: ", term)
                for conf_ri, conf_r in enumerate(v[fock_r]):
                    print("  ", conf_r)
                    
                    nonzeros = []
                    sig = np.array([1])
                    nnz = 0
                    opii = -1
                    mats = []
                    good = True
                    for opi,op in enumerate(term.ops):
                        if op == "":
                            continue
                        opii += 1
                        #print(opi,term.active)
                        ci = clusters[term.active[opii]]
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
                    print('mats:', end='')
                    [print(m.shape,end='') for m in mats]
                    print()
                    print('ints:', term.ints.shape)
                    print("contract_string       :", term.contract_string)
                    print("contract_string_matvec:", term.contract_string_matvec)
                    tmp = 0
                    if len(mats) == 1:
                        tmp = np.einsum(term.contract_string_matvec, mats[0], term.ints)
                    elif len(mats) == 2:
                        tmp = np.einsum(term.contract_string_matvec, mats[0], mats[1], term.ints)
                    elif len(mats) == 3:
                        tmp = np.einsum(term.contract_string_matvec, mats[0], mats[1], mats[2], term.ints)
                    elif len(mats) == 4:
                        tmp = np.einsum(term.contract_string_matvec, mats[0], mats[1], mats[2], mats[2], term.ints)
                    elif len(mats) == 0:
                        print(mats)
                        print('wtf?')
                        exit()
                    print("output:", tmp.shape)
                    print()
                   
                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(it.product(*new_configs)):
                        print(" New config: ", spi)
#                            nonzeros_curr = []
#                            for i,ii in enumerate(oi[:,ket[ci_idx]]):
#                                if ii*ii > thresh_transition:
#                                    nonzeros_curr.append(i)
#                                    nnz += 1
#                            nonzeros.append(nonzeros_curr)
#                            if len(nonzeros_curr) > 0:
#                                sig = np.kron(sig,oi[nonzeros_curr,ket[ci_idx]])
#                            else:
#                                continue

