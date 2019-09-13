import numpy as np
import scipy 
import scipy.sparse
import copy as cp
from Hamiltonian import *
from davidson import *


class ci_string:
# {{{
    """
    Class to organize all the configuration string stuff
    """
    def __init__(self,o,e):
        self.no                 = int(o)
        self.ne                 = int(e) 
        self._sign              = 1
        self._max               = calc_nchk(o,e)
        self._lin_index         = 0
        self._config            = list(range(self.ne) )
        self._ca_lookup         = [] # single excitation lookup table 

    def dcopy(self,other):
        """
        Create a copy of this CI_string
        """
        self.no         = int(other.no)
        self.ne         = int(other.ne)
        self.ne         = int(other.ne)
        self._max       = int(other._max)
        self._sign      = int(other._sign)
        self._lin_index = int(other._lin_index)
        #self._config    = cp.deepcopy(other._config)
        self._config    = [x for x in other._config]
        return


    def __len__(self):
        """
        return number of strings
        """
        return self._max

    def __str__(self):
        return "%12i %2i " %(self._lin_index, self._sign)+ str(self._config)
    
    
    def fill_ca_lookup(self):
        """
        Create an index table relating each string with all ia substitutions

        i.e., ca_lookup[Ka][c(p) + a(p)*n_p] = La
        """
        ket = ci_string(self.no,self.ne)
        bra = ci_string(self.no,self.ne)
       
        for K in range(ket.max()):
            Kv = []
            for p in range(ket.n_orbs()):
                for q in range(ket.n_orbs()):
                    bra.dcopy(ket)
                    bra.a(p)
                    bra.c(q)

                    if bra.sign() == 0:
                        Kv.append(0)
                    else:
                        Kv.append(bra.sign() * (bra.linear_index()+1))
                    # end q
                # end p
            self._ca_lookup.append(Kv)
            ket.incr()
            # end K 
                
        return


    def calc_linear_index(self):
        """
        Return linear index for lexically ordered __config string
        """
        self._lin_index = 0
        v_prev = -1

        for i in range(len(self._config)):
            v = self._config[i]
            M = self.no - v
            N = len(self._config) - i - 1
            w = v - v_prev - 1
            #todo: change mchn from function call to data lookup!
            for j in range(0,w):
                self._lin_index += calc_nchk(M+j,N)
            v_prev = v
        
        return 

    def a(self,orb_index):
        """
        apply annihilation operator a_i to current string
        where orb_index is i
        """
        assert(orb_index < self.no)
        #assert(i in self._config)
        if self._sign == 0:
            # This state is already destroyed
            return
        
        found = -1
        for i in range(len(self._config)):
            if self._config[i] == orb_index:
                found = i
                break
        if found == -1:
            self.destroy_config()
            return
        if found % 2 != 0:
            self._sign *= -1
        
        del self._config[found]

        self.ne         -= 1

        #unset data that need to be recomputed
        self._max        = -1
        self._lin_index  = -1
      
    def c(self,orb_index):
        """
        apply creation operator a_i to current string
        where orb_index is i
        """
        assert(orb_index < self.no)
        if self._sign == 0:
            # This state is already destroyed
            return
        
        insert_here = 0
        for i in range(len(self._config)):
            if self._config[i] > orb_index:
                insert_here = i
                break
            elif self._config[i] == orb_index:
                self.destroy_config()
                return 
            else:
                insert_here += 1

        if insert_here % 2 != 0:
            self._sign *= -1

        self._config.insert(insert_here,orb_index)
        self.ne         += 1

        #unset data that need to be recomputed
        self._max        = -1
        self._lin_index  = -1


    def destroy_config(self):
        self._sign      = 0
        self._max       = -1
        self._lin_index = -1
        self.ne         = 0
        del self._config[:]
        return
    
    def reset(self):
        self._config            = list(range(self.ne))
        self._sign              = 1
        self._lin_index         = 0
        return

    def incr(self):
        if self.linear_index() == (self.max()-1):
            #self.destroy_config()
            return
        self._lin_index += 1
        incr_comb(self._config, self.no)
        return   

    #
    #   Access Data
    #
    def max(self):
        if self._max == -1:
            self._max = calc_nchk(self.no,self.ne)
        return self._max
    def linear_index(self):
        if self._lin_index == -1:
            self.calc_linear_index()
        return self._lin_index
    def config(self):
        return self._config
    def sign(self):
        return self._sign
    def occ(self):
        return self._config
    def vir(self):
        return list(set(range(self.no))-set(self._config))
    def n_orbs(self):
        return self.no 
# }}}


class ci_solver:
    """
    Class to handle the solution of a full CI problem

    Input: Hamiltonian and Number of electrons alpha and beta
    """
    def __init__(self):
        self.algorithm  = 'direct'    #  options: direct/davidson 
        self.nea        = 0
        self.neb        = 0
        self.H          = Hamiltonian() 
        self.no         = 0 
        self.full_dim   = 0
        self.n_roots    = 1
        self.status     = "uninitialized"
        self.Hdiag_s    = [np.array(()),np.array(())]    #   where we will store the single-spin Hamiltonian matrices

    def __str__(self):
        msg = " CI solver:: Dim: %-8i NOrb: %-4i NAlpha: %-4i NBeta: %-4i "%(self.full_dim,self.no,self.nea,self.neb)
        msg += " Status: "+self.status
        return msg

    def init(self,H,ea,eb,n_roots):
        self.nea        = ea
        self.neb        = eb
        self.H          = H 
        self.no         = H.C.shape[1] 
        self.full_dim   = calc_nchk(self.no, self.nea)*calc_nchk(self.no, self.neb)
        self.n_roots    = n_roots 
        self.status     = "initialized"

    def compute_ab_terms_sigma(self,sig,vec):
# {{{

        #   Create local references to ci_strings
        ket_a = self.ket_a
        ket_b = self.ket_b
        bra_a = self.bra_a
        bra_b = self.bra_b

        
        # avoid python function call overhead
        ket_a_max = ket_a.max()
        ket_b_max = ket_b.max()
        bra_a_max = bra_a.max()
        bra_b_max = bra_b.max()
    
        range_ket_a_no = range(ket_a.no)
        range_ket_b_no = range(ket_b.no)
  
        range_s = range(sig.shape[1])
        
        _abs = abs
        
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

                        
                        for s in range_ket_b_no:
                            for q in range_ket_b_no:
                                Lb = ket_b._ca_lookup[Kb][q+s*ket_b.no]
                                if Lb == 0:
                                    continue
                                sign_b = 1
                                if Lb < 0:
                                    sign_b = -1
                                    Lb = -Lb 
                                Lb = Lb - 1
                               
                                L = La + Lb * bra_a_max 
        
                                
                                Iprqs = self.H.V[p,r,q,s]

                                for si in range_s:
                                    sig[K,si] += Iprqs * sign_a * sign_b * vec[L,si]
                ket_a.incr()
                #   end Ka 

            ket_b.incr()
            #   end Kb 
# }}}
    
    def compute_ab_terms_direct(self,H):
# {{{

        #   Create local references to ci_strings
        ket_a = self.ket_a
        ket_b = self.ket_b
        bra_a = self.bra_a
        bra_b = self.bra_b

        
        # avoid python function call overhead
        ket_a_max = ket_a.max()
        ket_b_max = ket_b.max()
        bra_a_max = bra_a.max()
        bra_b_max = bra_b.max()
    
        range_ket_a_no = list(range(ket_a.no))
        range_ket_b_no = list(range(ket_b.no))
   
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

                        
                        for s in range_ket_b_no:
                            for q in range_ket_b_no:
                                Lb = ket_b._ca_lookup[Kb][q+s*ket_b.no]
                                if Lb == 0:
                                    continue
                                sign_b = 1
                                if Lb < 0:
                                    sign_b = -1
                                    Lb = -Lb 
                                Lb = Lb - 1
                               
                                L = La + Lb * bra_a_max 
        
                                
                                Iprqs = self.H.V[p,r,q,s]

                                H[K,L] += Iprqs * sign_a * sign_b 
                ket_a.incr()
                #   end Ka 

            ket_b.incr()
            #   end Kb 
# }}}

    def run(self):
        #print(" Compute spin diagonals")
        self.Hdiag_s[0] = self.precompute_spin_diagonal_block(self.nea)
        self.Hdiag_s[1] = self.precompute_spin_diagonal_block(self.neb)
        
        self.ket_a = ci_string(self.no,self.nea)
        self.ket_b = ci_string(self.no,self.neb)
        self.bra_a = ci_string(self.no,self.nea)
        self.bra_b = ci_string(self.no,self.neb)
        
        ket_a = self.ket_a
        ket_b = self.ket_b
        bra_a = self.bra_a
        bra_b = self.bra_b

        ket_a.fill_ca_lookup()
        ket_b.fill_ca_lookup()

        if self.algorithm == "direct":
            self.run_direct()
        elif self.algorithm == "davidson":
            self.run_davidson()
        else:
            print(" Wrong option for algorithm")
            exit(-1)
       

    def run_direct(self):
# {{{
        #print(" self.H.e_core: %12.8f" %self.H.e_core)
        #print(" self.H.e_nuc : %12.8f" %self.H.e_nuc)
        Hci = np.zeros((self.full_dim,self.full_dim))
        #Hci = np.eye(self.full_dim)* (self.H.e_core + self.H.e_nuc)

        ket_a = self.ket_a
        ket_b = self.ket_b
       
        #   
        #   Add spin diagonal components
        #print(" Add spin diagonals")
        Hci += np.kron(np.eye(ket_b.max()), self.Hdiag_s[0])
        Hci += np.kron(self.Hdiag_s[1],np.eye(ket_a.max()))

        #print(" Do alpha/beta terms")
        self.compute_ab_terms_direct(Hci)
   
        #print(" Hamiltonian Matrix:")
        #tools.printm(Hci)
        
        print(" Diagonalize Matrix for %i roots" %self.n_roots)
        l,C = scipy.sparse.linalg.eigsh(Hci,self.n_roots,which='SA')
        #print(" Diagonalize Matrix")
        #l,C = np.linalg.eigh(Hci)
        #l = l[0:self.n_roots]
        #C = C[:,0:self.n_roots]
       
        print(" Eigenvalues of CI matrix:")
        for i,li in enumerate(l):
            print(" State: %4i     %12.8f"%(i,l[i]))
        
        self.results_e = l 
        self.results_v = C 
        
        """ 
        self.results_e = []
        self.results_v = []
        
        for s in range(self.n_roots):
            self.results_e.append(l[s])
            self.results_v.append(C[:,s])

        """
        return
        #print(" E(nuc) + E(core) = %16.10f" %(self.H.e_nuc+self.H.e_core))
# }}}
    
    def run_davidson(self):
    # {{{
        print(" Diagonalize Matrix")
        #e,C = scipy.sparse.linalg.eigsh(Hci,self.n_roots)
        #e,C = np.linalg.eigh(Hci)
        #e += self.H.e_nuc + self.H.e_core
       
        ket_a = self.ket_a
        ket_b = self.ket_b
        bra_a = self.bra_a
        bra_b = self.bra_b

        dav = Davidson(self.full_dim, self.n_roots)
        dav.thresh      = 1e-5 
        dav.max_vecs    = 100 
        dav.max_iter    = 100 
        dav.form_rand_guess()
        dav.sig_curr = np.zeros((self.full_dim, self.n_roots))

        #np.fill_diagonal(Hci, Hci.diagonal() + self.H.e_nuc+self.H.e_core)

        for dit in range(0,dav.max_iter):
           
            dav.sig_curr = np.zeros(dav.vec_curr.shape) 
            #dav.sig_curr = dav.vec_curr * (self.H.e_nuc + self.H.e_core)

            self.compute_ab_terms_sigma(dav.sig_curr, dav.vec_curr)
       
            for s in range(dav.n_roots):
                sig_curr = cp.deepcopy(dav.sig_curr[:,s])
                vec_curr = cp.deepcopy(dav.vec_curr[:,s])
                sig_curr.shape = (ket_a.max(), ket_b.max())
                vec_curr.shape = (ket_a.max(), ket_b.max())
                       
                # (I x A)c = vec(AC)
                sig_curr += self.Hdiag_s[0].dot(vec_curr);
                # (B x I)c = vec(CB)
                sig_curr += vec_curr.dot(self.Hdiag_s[1]);
                
                sig_curr.shape = (self.full_dim)
                vec_curr.shape = (self.full_dim)
                
                dav.sig_curr[:,s] = cp.deepcopy(sig_curr)
                dav.vec_curr[:,s] = cp.deepcopy(vec_curr)

            dav.update()
            dav.print_iteration()
            if dav.converged():
                break
        if dav.converged():
            print(" Davidson Converged")
        else:
            print(" Davidson Not Converged")
        print("") 
       
        self.results_e = dav.eigenvalues()
        self.results_v = dav.eigenvectors()
        
        print(" Eigenvalues of CI matrix:")
        for i in range(min(30,len(self.results_e))):
            print(" State: %4i     %12.8f"%(i,self.results_e[i]))
    
        #print(" E(nuc) + E(core) = %16.10f" %(self.H.e_nuc+self.H.e_core))
       
        """
        self.results_e = []
        self.results_v = []
        
        for s in range(self.n_roots):
            self.results_e.append(l[s])
            self.results_v.append(v[:,s])
            """

# }}}

    def precompute_spin_diagonal_block(self,e):
# {{{
        """
        Precompute the Hamiltonian matrix representation of the operator terms which exist in only a single spin
        sub-block. i.e., p'q'rs where pqrs are all either alpha or beta

        Input, e is number of electrons, either alpha or beta
        """
        ket = ci_string(self.no,e)
        bra = ci_string(self.no,e)
        H = np.zeros((ket.max(),ket.max()))
        
        # avoid python function call overhead
        ket_max = ket.max()
        bra_max = bra.max()
        range_ket_n_orbs  = list(range(ket.n_orbs()))
        range_ket_max = list(range(ket.max()))
    
        for K in range_ket_max:
            bra.dcopy(ket)

            #  hpq p'q 
            for p in range_ket_n_orbs:
                for q in range_ket_n_orbs:
                    bra.dcopy(ket)
                    bra.a(q)
                    if bra.sign() == 0:
                        continue
                    bra.c(p)
                    if bra.sign() == 0:
                        continue
                    
                    L = bra.linear_index()

                    #print(str(ket),q,' -> ',p,str(bra))
                
                    term = self.H.t[q,p]
                    sign = bra.sign() 
                    
                    H[K,L] += term * sign

            #  <pq|rs> p'q'sr -> (pr|qs) 
            for r in range(0,ket.n_orbs()):
                for s in range(r+1,ket.n_orbs()):
                    for p in range(0,ket.n_orbs()):
                        for q in range(p+1,ket.n_orbs()):
                        
                            bra.dcopy(ket)
                            
                            bra.a(r) 
                            if bra.sign() == 0:
                                continue
                            bra.a(s) 
                            if bra.sign() == 0:
                                continue
                            bra.c(q) 
                            if bra.sign() == 0:
                                continue
                            bra.c(p) 
                            if bra.sign() == 0:
                                continue
                            L = bra.linear_index()
                            Ipqrs = self.H.V[p,r,q,s]-self.H.V[p,s,q,r]
                            sign = bra.sign()
                            H[K,L] += Ipqrs*sign
            ket.incr()

        return H
# }}}

    



################################################################################33
#   Tools
################################################################################33

def calc_nchk(n,k):
    """
    Calculate n choose k
    """
    accum = 1
    for i in range(1,k+1):
        accum = accum * (n-k+i)/i
    return int(accum)

def incr_comb(comb, Mend):
    """
    For a given combination, form the next combination
    """
    N = len(comb)
    for i in reversed(range(N)):
        if comb[i] < Mend-N+i:
            comb[i] += 1
            for j in range(i+1,N):
                comb[j]=comb[j-1]+1
            return
    return



#   
#   Compute TDMs

def build_annihilation(no,bra_space,ket_space,basis):
    """# {{{
    Compute a(v1,v2,j) = <v1|a_j|v2>

    no = n_orbs
    bra_space = (n_alpha,n_beta) for bra
    ket_space = (n_alpha,n_beta) for ket

    basis = dict of basis vectors
    """
    
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
   
    spin_case = ""
    if (bra_space[0] == ket_space[0]-1) and (bra_space[1] == ket_space[1]):
        spin_case = "alpha"
    elif (bra_space[0] == ket_space[0]) and (bra_space[1] == ket_space[1]-1):
        spin_case = "beta"
    else:
        print(" Incompatible transition")
        assert(1==0) 
    
    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_no = range(no)
  
    _abs = abs
   
    v1 = basis[bra_space] 
    v2 = basis[ket_space] 

    assert(v1.shape[0] == len(bra_a)*len(bra_b))
    assert(v2.shape[0] == len(ket_a)*len(ket_b))
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    
    #alpha term 
    if spin_case == "alpha":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "beta":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            bra.dcopy(ket)
            bra.a(p)
            if bra.sign() == 0:
                continue
            L = bra.linear_index()
            sign = bra.sign()
            tdm_1spin[L,K,p] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
    
    if spin_case == "alpha":
        # v(IJs) <IJ|p|KL> v(KLt)   = v(IJs) tdm(IKp) v(KJt) = A(stp)
        tmp = np.einsum('ikp,kjt->ipjt',tdm_1spin,v2)
        tdm = np.einsum('ijs,ipjt->stp',v1,tmp)
    elif spin_case == "beta":
        # v(IJs) <IJ|p|KL> v(KLt)   = v(IJs) tdm(JLp) v(ILt) = A(stp)
        tmp = np.einsum('jlp,ilt->jpit',tdm_1spin,v2)
        tdm = np.einsum('ijs,jpit->stp',v1,tmp) * (-1)**ket_a.ne

    
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}




