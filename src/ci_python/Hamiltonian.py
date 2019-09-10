import numpy as np
import copy as cp

class Hamiltonian:
# {{{
    def __init__(self):
        self.t          = np.array([])  # 1e electrons (with frozen core contributions
        self.V          = np.array([])  # 2e electrons
        self.e_nuc      = 0
        self.e_core     = 0
        self.C          = np.array([])  # MO coeffs defining the basis for H 
        self.S          = np.array([])  # Overlap integrals defining the basis for H 
        self.C_core     = np.array([])  # MO coeffs defining the frozen core 

    def transform_orbs(self,U):
        """
        Rotate orbitals by a unitary transformation matrix, U
        U(old_basis, new_basis)
        """
        assert(len(U.shape)==2)
        self.C = self.C.dot(U)
        self.t = U.T.dot(self.t).dot(U)
        
        self.V = np.tensordot(self.V, U, axes=(0,0))
        self.V = np.tensordot(self.V, U, axes=(0,0))
        self.V = np.tensordot(self.V, U, axes=(0,0))
        self.V = np.tensordot(self.V, U, axes=(0,0))

        print(" todo: Check that this function leave FCI energy invariant")
        return

    def transform_to_new_mos(self,C_new):
        """
        Transform the hamiltonian to the new mo basis, C_new
        """
        assert(C_new.shape[0] <= self.C.shape[0])
        assert(C_new.shape[1] <= self.C.shape[1])
       
        # Get transformation matrix
        U = self.C.T.dot(self.S).dot(C_new)
        self.transform_orbs(U)
    
        return U

    def extract_Hamiltonian(self,orb_subset):
        """
        Project Hamiltonian onto a given orbital subset and return Hamiltonian object

        i.e., 
        orb_subset = [4,6,8,9]
        """
        assert(len(orb_subset) <= self.t.shape[0])
        
        H = Hamiltonian()
        H.e_nuc  = cp.deepcopy(self.e_nuc)
        H.e_core = cp.deepcopy(self.e_core)
        H.S      = cp.deepcopy(self.S)
        H.C_core = cp.deepcopy(self.C_core)
        
        H.C = self.C[:,orb_subset]
        H.t = self.t[orb_subset,:][:,orb_subset]
        #H.V = self.V[:,:,:,orb_subset][:,:,orb_subset,:][:,orb_subset,:,:][orb_subset,:,:,:]
        H.V = self.V[orb_subset,:,:,:][:,orb_subset,:,:][:,:,orb_subset,:][:,:,:,orb_subset]
    
        return H

    def reorder_orbitals(self,orb_order):
        """
        Reorder Hamiltonian         
        """
        assert(len(orb_order) <= self.t.shape[0])
        
        self.C = self.C[:,orb_order]
        self.t = self.t[orb_order,:][:,orb_order]
        #self.V = self.V[:,:,:,orb_order][:,:,orb_order,:][:,orb_order,:,:][orb_order,:,:,:]
        self.V = self.V[orb_order,:,:,:][:,orb_order,:,:][:,:,orb_order,:][:,:,:,orb_order]
        return 

    def get_C(self):
        if(self.C.shape[0] != self.C_core.shape[0]):
            print("ERROR: self.C.shape[0] != self.C_core.shape[0])")
            exit(-1)
        return np.hstack((self.C_core,self.C))
    
    def nbf(self):
        return self.C.shape[0]
    def nmo(self):
        return self.C.shape[1]
   
    def get_eri_1122(self,ob1,ob2):
        """
        Return 2e integrals corresponding to (pq|rs)
        where pq in ss1
        and   rs in ss2
        """
        ss1 = ob1.orbs
        ss2 = ob2.orbs
        return self.V[ss1,:,:,:][:,ss1,:,:][:,:,ss2,:][:,:,:,ss2]

    def get_eri(self,ob1,ob2,ob3,ob4):
        """
        Return 2e integrals corresponding to (pq|rs)
        where pq in ss1
        and   rs in ss2
        """
        ss1 = ob1.orbs
        ss2 = ob2.orbs
        ss3 = ob3.orbs
        ss4 = ob4.orbs
        return self.V[ss1,:,:,:][:,ss2,:,:][:,:,ss3,:][:,:,:,ss4]


# }}}


class Orbital_Block: 
# {{{
    def __init__(self):
        self.H              = Hamiltonian()  
        self.index          = 0
        self.orbs           = []
        self.vecs           = np.array([])  # local eigenvector matrix for block [P|Q]
        self.ss_dims        = []            # number of vectors in each subspace [P,Q,...]
        self.n_ss           = 0             # number of subspaces, usually 2 or 3
        self.full_dim       = 1             # dimension of full CI space in block  
    
        self.tdms           = {}            # dictionary of transition density matrices 
                                            # i.e.,     self.tdms["ca_aa"] = <I|p^t q|J> = Gamma^{IJ}_pq where both 
                                            #                                               p and q are alpha
                                            #           self.tdms["cca_aab"] = <I|p^t q^t r|J> = Gamma_cca_aab^{IJ}_pqr where both 
                                            #                                               p and q are alpha and r is
                                            #                                               beta - only non-zero if I
                                            #                                               and J have different numbers
                                            #                                               of alpha beta electrons

        # NYI:

        #
        # still to convert to ab initio ....
        self.Spi = {}                # matrix_rep of i'th S^+ in local basis
        self.Smi = {}                # matrix_rep of i'th S^- in local basis
        self.Szi = {}                # matrix_rep of i'th S^z in local basis

        #   in tucker basis
        self.Ham   = np.array([])    # Hamiltonian on block sublattice
        self.S2    = np.array([])    # S2 on block sublattice
        self.Sz    = np.array([])    # Sz on block sublattice

        #   in configuration basis
        self.full_H     = np.array([])    # Hamiltonian on block sublattice
        self.full_S2    = np.array([])    # S2 on block sublattice
        self.full_Sz    = np.array([])    # Sz on block sublattice

        self.diis_vecs = np.array([]) 
    
    def init(self,_index,_orbs,_ss):
        """
        _index = index of block
        _sites = list of lattice sites contained in block
        _ss    = list of dimensions of vectors per subspace, -1 indicates all remaining states
        """
        self.index = _index
        self.orbs = _orbs
        for si in range(0,self.n_orbs()):
            self.full_dim *= 4
        
        vec_count = 0
        for ss in _ss:
            # if we have asked for orthog compliment
            if ss == -1:
                self.ss_dims.append(self.full_dim - vec_count)
                vec_count = self.full_dim
            else:
                self.ss_dims.append(ss)
                vec_count += ss
        if (self.full_dim-vec_count) < 0:
            print("Problem setting block dimensions", self)
            exit(-1)
        self.ss_dims.append(self.full_dim-vec_count)
        return 
    
    def n_orbs(self):
        return len(self.orbs)

    def fill_H(self,Hfull):
        self.H = Hfull.extract_Hamiltonian(self.orbs)
    
    def __str__(self):
        out = " Block %-4i:" %(self.index)
        for si in range(0,self.n_orbs()):
            if si < self.n_orbs()-1:
                out += "%5i," %(self.orbs[si])
            else:
                out += "%5i" %(self.orbs[si])
        out += " : " + str(self.ss_dims)
        return out

# }}}


class Molecule:
# {{{
    def __init__(self):
        self.na   = 0
        self.nb   = 0
        self.n_mo       = 0
        self.n_bf       = 0
        self.ftc        = []    # function_to_center
        self.n_atom     = 0

    def function_to_center(self,i):
        """
        The atomic center for the i'th function
        """
        assert(len(self.ftc) == self.n_bf)
        return self.ftc[i]

# }}}


