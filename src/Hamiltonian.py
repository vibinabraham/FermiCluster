import numpy as np
import copy as cp

class Hamiltonian:

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





