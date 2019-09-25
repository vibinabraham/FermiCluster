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
                            continue 
                    
                    print('mats:')
                    [print(m.shape) for m in mats]
                    print(term.ints.shape)
                    if len(mats) == 1:
                        print(" Do TDM(:,pqrs)J Ints(pqrs) v(...J...,t)")
                        mi = 1
                        #old_shape = cp.deepcopy(m[0].shape)
                        for i in range(1,len(mats[0].shape)):
                            mi *= mats[0].shape[i]
                        mats[0].shape = (mats[0].shape[0],mi)
                    elif len(mats) == 0:
                        print(mats)
                        print('wtf?')
                        exit()
                    
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

