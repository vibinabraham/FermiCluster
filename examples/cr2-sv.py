import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

import pickle

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues



###     PYSCF INPUT
r0 = 1.50
molecule = '''
Cr
Cr   1   {}
'''.format(r0)
charge = 0
spin  = 0
#basis_set = 'def2-svp'

###     TPSCI BASIS INPUT
orb_basis = 'scf'
cas = True
cas_nstart = 12
cas_nstop =  42
cas_nel = 24

def ordering_diatomics_cr(mol,C):
# {{{
    ##DZ basis diatomics reordering with frozen 1s

    orb_type = ['s','pz','dz','px','dxz','py','dyz','dx2-y2','dxy']
    ref = np.zeros(C.shape[1])

    ## Find dimension of each space
    dim_orb = []
    for orb in orb_type:
        print("Orb type",orb)
        idx = 0
        for label in mol.ao_labels():
            if orb in label:
                #print(label)
                idx += 1

        ##frozen 1s orbitals
        if orb == 's':
            idx -= 6
        elif orb == 'px':
            idx -=2
        elif orb == 'py':
            idx -=2
        elif orb == 'pz':
            idx -=2
        dim_orb.append(idx)
        print(idx)


    new_idx = []
    ## Find orbitals corresponding to each orb space
    for i,orb in enumerate(orb_type):
        print("Orbital type:",orb)
        from pyscf import mo_mapping
        s_pop = mo_mapping.mo_comps(orb, mol, C)
        print(s_pop)
        ref += s_pop
        cas_list = s_pop.argsort()[-dim_orb[i]:]
        print('cas_list', np.array(cas_list))
        new_idx.extend(cas_list)
        #print(orb,' population for active space orbitals', s_pop[cas_list])

    ao_labels = mol.ao_labels()
    #idx = mol.search_ao_label(['N.*s'])
    #for i in idx:
    #    print(i, ao_labels[i])
    print(ref)
    print(new_idx)
    for label in mol.ao_labels():
        print(label)

    return new_idx
# }}}

# basis is SVP read comments by alex thom paper DOI:10.1021/acs.jctc.9b01023
from pyscf import gto
basis_set={'Cr': gto.parse('''
BASIS "ao basis" PRINT
#BASIS SET: (14s,8p,5d) -> [5s,2p,2d]
Cr    S
  51528.086349               0.14405823106E-02
   7737.2103487              0.11036202287E-01
   1760.3748470              0.54676651806E-01
    496.87706544             0.18965038103
    161.46520598             0.38295412850
     55.466352268            0.29090050668
Cr    S
    107.54732999            -0.10932281100
     12.408671897            0.64472599471
      5.0423628826           0.46262712560
Cr    S
      8.5461640165          -0.22711013286
      1.3900441221           0.73301527591
      0.56066602876          0.44225565433
Cr    S
      0.71483705972E-01      1.0000000000
Cr    S
      0.28250687604E-01      1.0000000000
Cr    P
    640.48536096             0.96126715203E-02
    150.69711194             0.70889834655E-01
     47.503755296            0.27065258990
     16.934120165            0.52437343414
      6.2409680590           0.34107994714
Cr    P
      3.0885463206           0.33973986903
      1.1791047769           0.57272062927
      0.43369774432          0.24582728206
Cr    D
     27.559479426            0.30612488044E-01
      7.4687020327           0.15593270944
      2.4345903574           0.36984421276
      0.78244754808          0.47071118077
Cr    D
      0.21995774311          0.33941649889
END''')}


###     TPSCI CLUSTER INPUT
#init_fspace = ((1, 1),(3, 3),(3, 3),(3, 3), (1, 1), (1, 1))
#blocks = [range(0,4),range(4,10),range(10,16),range(16,22),range(22,26),range(26,30)]

init_fspace = ((1, 1),(2, 2),(1, 1),(2, 2),(1, 1),(2, 2),(1, 1), (1, 1), (1, 1))
blocks = [list(range(0,4)),list(range(4,6)),list(range(6,10)),list(range(10,12)),list(range(12,16)),list(range(16,18)),list(range(18,22)),list(range(22,26)),list(range(26,30))]

nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel

# Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True)

h = pmol.h
g = pmol.g
ecore = pmol.ecore
print("Ecore:%16.8f"%ecore)
C = pmol.C
K = pmol.K
mol = pmol.mol
mo_energy = pmol.mf.mo_energy[cas_nstart:cas_nstop]
dm_aa = pmol.dm_aa
dm_bb = pmol.dm_bb

do_tci = 1


#cluster using hcore

idx = ordering_diatomics_cr(mol,C)
h,g = reorder_integrals(idx,h,g)
C = C[:,idx]
mo_energy = mo_energy[idx]
dm_aa = dm_aa[:,idx]
dm_aa = dm_aa[idx,:]
dm_bb = dm_bb[:,idx]
dm_bb = dm_bb[idx,:]

print(dm_aa)


from pyscf import molden
molden.from_mo(pmol.mol, 'h8.molden', C)
print(h)
mol = pmol.mol
if mol.symmetry == True:
    from pyscf import symm
    mo = symm.symmetrize_orb(mol, C)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))


#################################
#            MBE                #
#################################


n_blocks = len(blocks)
clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]

print(" Ecore   :%16.8f"%ecore)
print(" Clusters:")
[print(ci) for ci in clusters]

clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
print(" Add 1-body terms")
clustered_ham.add_local_terms()
clustered_ham.add_1b_terms(h)
print(" Add 2-body terms")
clustered_ham.add_2b_terms(g)

ci_vector = ClusteredState()
ci_vector.init(clusters, init_fspace)

ecmf, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, diis = True,dm_guess    = (dm_aa,dm_bb))
print(ecmf+ecore)

def join_blocks(ind_to_join,blocks,fspace):
    """
    ind_to_join:    indices of clusters to join
    blocks:         cluster block input
    fspace:         fock space of each clusters 
    """
# {{{
    fb = ind_to_join[0] #first block
    nb = ind_to_join[1:]#rest of the blocks

    # blocks
    print(nb)
    print(blocks)
    nblocks = len(blocks)
    newblocks = cp.deepcopy(blocks)
    for ai,a in enumerate(nb):
        newblocks[fb].extend(blocks[a])
    for ai in range(nblocks,-1,-1):
        if ai in nb:
            del newblocks[ai]
    print(newblocks)

    # fock space
    flist = [list(x) for x in fspace]
    flist = np.array(flist)
    actflist = [flist[a]  for a in ind_to_join]
    for fi,fl in enumerate(actflist):
        if fi >0:
            flist[fb] += fl
    flist = list(flist)
    for ai in range(nblocks,-1,-1):
        if ai in nb:
            del flist[ai]

    fspace = tuple([tuple(x) for x in flist])
    print(fspace)
    return newblocks,fspace
# }}}

ind_to_join = [5,8]
blocks,init_fspace = join_blocks(ind_to_join,blocks,init_fspace)
print(blocks,init_fspace)
n_blocks = len(blocks)
clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]

print(" Ecore   :%16.8f"%ecore)
print(" Clusters:")
[print(ci) for ci in clusters]

clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
print(" Add 1-body terms")
clustered_ham.add_local_terms()
clustered_ham.add_1b_terms(h)
print(" Add 2-body terms")
clustered_ham.add_2b_terms(g)

ci_vector = ClusteredState()
ci_vector.init(clusters, init_fspace)
ecmf, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, diis = True,dm_guess    = (dm_aa,dm_bb))

print(ecmf+ecore)
