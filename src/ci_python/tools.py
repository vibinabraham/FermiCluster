import psi4
import numpy as np
from Hamiltonian import *


class Options:
# {{{
    def __init__(self):
        self.opts = {}
        self.set('n_occ_frzn',0)
        self.set('n_vir_frzn',0)
        self.set('n_roots',1)
        self.set('localize','boys')
        self.set('molden_filename','molden.out')

        self.set('fragmentation','atomic')
        #   atomic = mulliken population to associate with atoms requires 'atom_fragments' to be set
        #   spectral_clustering = use spectral clustering of orbitals to determine clusters

    def __str__(self):
        out = "--------------Options:\n"
        for key, val in self.opts.items():
            out += " %20s:  " %(key)
            out += str(val)
            out += "\n"
        return out
    
    def exists(self,key):
        return key in self.opts.keys() 
    
    def get(self,key):
        return self.opts[key]
    
    def set(self,key,val):
        self.opts[key] = val
    # }}}

#
# Computes the J and K (Coulomb and exchange) integrals in the MO basis.
#
def compute_jk( wfn, C ):
  # {{{
    # construct + initialize JK object
    jk = psi4.core.JK.build(wfn.basisset())
    jk.initialize()

    #
    # clear and pack MO coefficients
    Cmat = psi4.core.Matrix.from_array(C)
    jk.C_clear()
    jk.C_add(Cmat)

    #
    # compute JK, then finalize/clear memory
    jk.compute()
    jk.finalize()
    j = psi4.core.Matrix.to_array(jk.J()[0])
    k = psi4.core.Matrix.to_array(jk.K()[0])

    #rotate these AO quantities into the full MO basis
    #Cfull = psi4.core.Matrix.to_array(wfn.Ca(), copy=True)
    #j = Cfull.T.dot(j).dot(Cfull)
    #k = Cfull.T.dot(k).dot(Cfull)
    return (j, k)
# }}}


#
# Computes energy of single configuration
#
def compute_determinant_energy(H,str_a,str_b):
# {{{
    """
    For a given alpha and beta string, compute and return the energy
    """
    E = 0 
    #E = H.e_nuc + H.e_core
   
    """
    e1 = 0
    ej = 0
    ek = 0
    J = np.zeros((H.nmo(),H.nmo()))
    for i in range(H.nmo()):
        for j in range(H.nmo()):
            J[i,j] = H.V[i,i,j,j]
    """
    for i in str_a:
        E += H.t[i,i]
        #e1 += H.t[i,i]
        for j in str_a:
            if i<j:
                E += H.V[i,i,j,j] - H.V[i,j,j,i] 
                #ej += H.V[i,i,j,j] 
                #ek += H.V[i,j,j,i] 
        for j in str_b:
            E += H.V[i,i,j,j] 
            #ej += H.V[i,i,j,j] 
    for i in str_b:
        E += H.t[i,i]
        #e1 += H.t[i,i]
        for j in str_b:
            if i<j:
                E += H.V[i,i,j,j] - H.V[i,j,j,i] 
                #ej += H.V[i,i,j,j] 
                #ek += H.V[i,j,j,i] 
    #print(" e1, ej, ek: %12.8f %12.8f %12.8f " % (e1, ej, ek))

    #printm(J)
    return E
# }}}


#
# Computes the ERI integrals in the MO basis.
#
def compute_eri( wfn, C1, C2, C3, C4):
# {{{
    C1_matrix = psi4.core.Matrix.from_array(C1, name="Ca")
    C2_matrix = psi4.core.Matrix.from_array(C2, name="Ca")
    C3_matrix = psi4.core.Matrix.from_array(C3, name="Ca")
    C4_matrix = psi4.core.Matrix.from_array(C4, name="Ca")
    # getting ERI
    mints = psi4.core.MintsHelper(wfn)
    eri = mints.mo_eri(C1_matrix,C2_matrix,C3_matrix,C4_matrix)
    eri = psi4.core.Matrix.to_array(eri)
    return eri
# }}}

def printm(m):
    # {{{
    """ print matrix """
    print("[")
    for r in m:
        print("[",)
        for ri in r:
            print("%10.3e" %ri,)
        print("]")
    print("]")
    # }}}
    
    
    


    
    
#
#   Add some psi4 dependent stuff to Hamiltonian
def build_psi4(self,C_core, C,ref_wfn):
# {{{
    """
    Build Hamiltonian with a specified set of frozen (active) MOs: C_core (C), and a PSI4 ref_wfn_object

        if frozen core orbitals are involved, these need to be defined before running this function
    """
    self.e_core = 0
    n_bf        = ref_wfn.basisset().nbf()

    self.C = cp.deepcopy(C)
    self.t = psi4.core.Matrix.to_array(ref_wfn.H(), copy=True)
    self.t = C.T.dot(self.t).dot(C)
    self.V = compute_eri(ref_wfn, C, C, C, C)
    self.S = psi4.core.Matrix.to_array(ref_wfn.S(), copy=True)
    
    self.e_nuc  = ref_wfn.molecule().nuclear_repulsion_energy()
    
    
    # get h and JK, which comes from a closed shell frozen core
    if len(C_core.shape) > 1:
        if C_core.shape[1] > 0:
            # do core
            J, K = compute_jk(ref_wfn,C_core)
            t_ao = 2.0*J - K
            self.t += C.T.dot(t_ao).dot(C)
            pass
    else:
        self.C_core = np.zeros((n_bf,0))
   
    printm(self.t)
    self.C_core = cp.deepcopy(C)
# }}}


#
#   Now bind to class
Hamiltonian.build_psi4 = build_psi4




#   
#   Compute TDMs
def compute_tdm_ca_aa(ci1,ci2):
    """# {{{
    Compute Pij(v1,v2) = <v1|a_i' a_j|v2>

    v1 and v2 correspond to the current vectors in ci1 and ci2. 

    We use multiple ci_string objects becauase want to compute transition densities between 
    states with different numbers of electrons.

    """

    H = ci1.H   # either ci1.H or ci2.H could be used as they must be the same
    
    #   Create local references to ci_strings
    bra_a = ci1.bra_a
    bra_b = ci1.bra_b
    ket_a = ci2.ket_a
    ket_b = ci2.ket_b
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 

    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_ket_a_no = range(ket_a.no)
    range_ket_b_no = range(ket_b.no)
  
    _abs = abs
   
    v1 = ci1.results_v
    v2 = ci2.results_v

    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    tdm_ca_aa = np.zeros((nv1,nv2,H.nmo(),H.nmo()))
    
    
    ket_b.reset()
    for Kb in range(ket_b_max): 
        
        ket_a.reset()
        for Ka in range(ket_a_max): 
    
            K = Ka + Kb * ket_a_max
            
            #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
            for r in range_ket_a_no:
                for p in range_ket_a_no:
                    La = ket_a._ca_lookup[Ka][p+r*ket_a.no]
                    if La == 0:
                        continue
                    sign_a = 1
                    if La < 0:
                        sign_a = -1
                        La = -La
                    La = La - 1
                                
                    L = La + Kb * bra_a_max  # Lb doesn't change
                
                    tmp_KL = np.kron(v1[K,:].T,v2[L,:])
                    tmp_KL.shape = (v1.shape[1],v2.shape[1])
                    tdm_ca_aa[:,:,p,r] += tmp_KL * sign_a
   
    #for ni in range(nv1):
        #for nj in range(nv2):
            #print(" Trace Paa(%4i, %4i): %12.8f" %( ni, nj, np.trace(tdm_ca_aa[ni,nj,:,:])))
    return tdm_ca_aa
# }}}


