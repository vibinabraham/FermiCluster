import numpy as np
import scipy 
import scipy.sparse
import copy as cp
from Hamiltonian import *
from davidson import *

import helpers
import opt_einsum as oe

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
        self.thresh     = 1e-5 
        self.max_iter   = 300 
        self.full_dim   = 0
        self.n_roots    = 1
        self.status     = "uninitialized"
        self.Hdiag_s    = [np.array(()),np.array(())]    #   where we will store the single-spin Hamiltonian matrices

    def __str__(self):
        msg = " CI solver:: Dim: %-8i NOrb: %-4i NAlpha: %-4i NBeta: %-4i NRoots: %-4i"%(self.full_dim,self.no,self.nea,self.neb, self.n_roots)
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

        """
        Better way to do this:

            sigma(Ia,Ib,s) += <pq|rs> p'q'sr C(Ia,Jb,s)
                            = <a(Ia)|p'r|a(Ja)> <b(Ib)|q's|b(Jb)> <pq|rs> C(Ja,Jb,s)
                            = sum_pq Apr(Ia,Ja) Bpr(Ib,Jb) C(Ja,Jb,s)
        
            Form these quantities separately
        """

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
                #ket_a.incr()
                #   end Ka 

            #ket_b.incr()
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
   
        ket_a.reset()
        for Ka in range(ket_a_max): 
            
            ket_b.reset()
            for Kb in range(ket_b_max): 
    
                K = Kb + Ka * ket_b_max
                
                #  <pq|rs> p'q'sr  --> (pr|qs) (a,b)
                for r in range_ket_b_no:
                    for p in range_ket_b_no:
                        Lb = ket_b._ca_lookup[Kb][p+r*ket_b.no]
                        if Lb == 0:
                            continue
                        sign_b = 1
                        if Lb < 0:
                            sign_b = -1
                            Lb = -Lb
                        Lb = Lb - 1

                        
                        for s in range_ket_a_no:
                            for q in range_ket_a_no:
                                La = ket_a._ca_lookup[Ka][q+s*ket_a.no]
                                if La == 0:
                                    continue
                                sign_a = 1
                                if La < 0:
                                    sign_a = -1
                                    La = -La 
                                La = La - 1
                               
                                L = Lb + La * bra_b_max 
        
                                
                                Iprqs = self.H.V[p,r,q,s]

                                H[K,L] += Iprqs * sign_a * sign_b 
                ket_b.incr()
                #   end Ka 

            ket_a.incr()
            #   end Kb 
# }}}

    def build_H_matrix(self, basis):

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

        ket_a = self.ket_a
        ket_b = self.ket_b
            
        if self.algorithm=='direct':
            
            Hci = np.zeros((self.full_dim,self.full_dim))
           
            #   
            #   Add spin diagonal components
            #print(" Add spin diagonals")
            Hci += np.kron(np.eye(ket_a.max()), self.Hdiag_s[1])
            Hci += np.kron(self.Hdiag_s[0],np.eye(ket_b.max()))

            #print(" Do alpha/beta terms")
            self.compute_ab_terms_direct(Hci)

            Hci = .5*(Hci+Hci.T)

            return basis.T @ Hci @ basis

        elif self.algorithm == 'davidson':


            self.Apr = self.build_spin_tdms(bra_a,ket_a)
            self.Bpr = self.build_spin_tdms(bra_b,ket_b)
            self.Bpr = np.einsum('KLqs,prqs->KLpr',self.Bpr,self.H.V)

            vec = 1.0 * basis
            nvecs = vec.shape[1]
            c = vec*1

            c.shape = (ket_a.max(), ket_b.max(), nvecs)
            sigma = np.zeros((ket_a.max(),ket_b.max(),nvecs))
            sigma += np.einsum('IJpr,KLpr,JLt->IKt',self.Apr,self.Bpr,c,optimize=True)
            sigma.shape = (ket_a.max()*ket_b.max(),nvecs)

                
            for s in range(nvecs):
                sig_curr = cp.deepcopy(sigma[:,s])
                vec_curr = cp.deepcopy(vec[:,s])
                sig_curr.shape = (ket_a.max(), ket_b.max())
                vec_curr.shape = (ket_a.max(), ket_b.max())
                #sig_curr.shape = (ket_b.max(), ket_a.max())
                #vec_curr.shape = (ket_b.max(), ket_a.max())
                       
                # (I x A)c = vec(AC)
                sig_curr += self.Hdiag_s[0].dot(vec_curr);
                # (B x I)c = vec(CB)
                sig_curr += vec_curr.dot(self.Hdiag_s[1]);
                
                sig_curr.shape = (self.full_dim)
                vec_curr.shape = (self.full_dim)
                
                sigma[:,s] = cp.deepcopy(sig_curr)
                vec[:,s] = cp.deepcopy(vec_curr)
        
            return basis.T @ sigma
       

    def run(self,s2=False,iprint=0):
        if iprint>0:
            print(" Compute spin diagonals",flush=True)
        self.Hdiag_s[0] = self.precompute_spin_diagonal_block(self.nea)
        self.Hdiag_s[1] = self.precompute_spin_diagonal_block(self.neb)
        if iprint>0:
            print(" done",flush=True)
        
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
            Hci = self.run_direct(s2)
            return Hci
        elif self.algorithm == "davidson":
            #faster (still crazy slow) 
            print(" Form TDMs for FCI",flush=True)
            self.Apr = self.build_spin_tdms(bra_a,ket_a)
            self.Bpr = self.build_spin_tdms(bra_b,ket_b)
            self.Bpr = np.einsum('KLqs,prqs->KLpr',self.Bpr,self.H.V)
            print(" done.",flush=True)
            sigma = self.run_davidson()
            return sigma
        elif self.algorithm == "davidson2":
            #slowest
            self.run_davidson()
            return
        else:
            print(" Wrong option for algorithm")
            exit(-1)
        return
       
    def build_spin_tdms(self,bra,ket):
    # {{{
        A = np.zeros((bra.max(), ket.max(), ket.no, ket.no))
        ket.reset()
        bra.reset()
        for K in range(ket.max()):
            bra.dcopy(ket)
        
            for p in range(ket.no):
                for q in range(ket.no):
                    bra.dcopy(ket)
                    bra.a(q)
                    if bra.sign() == 0:
                        continue
                    bra.c(p)
                    if bra.sign() == 0:
                        continue
                    
                    L = bra.linear_index()
        
                    sign = bra.sign() 
                    
                    A[K,L,p,q] += sign
            ket.incr()
        return A
    # }}}

    def run_direct(self,s2=False, iprint=0):
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
        Hci += np.kron(np.eye(ket_a.max()), self.Hdiag_s[1])
        Hci += np.kron(self.Hdiag_s[0],np.eye(ket_b.max()))

        if iprint>0:
            print(" Do alpha/beta terms")
        self.compute_ab_terms_direct(Hci)
   
        #print(" Hamiltonian Matrix:")
        #tools.printm(Hci)
       
        #helpers.print_mat(Hci)
        if iprint>0:
            print(" Diagonalize Matrix for %i roots" %self.n_roots)
        
        if s2:
            S2 = self.build_S2_matrix()
            Hci = .5*(Hci+Hci.T)
            if Hci.shape[0] > 1 and Hci.shape[0] > self.n_roots:
                l,C = scipy.sparse.linalg.eigsh(Hci+0.089*S2,self.n_roots,which='SA')
                l = np.diag(C.T @ Hci @ C)
                sort_ind = np.argsort(l)
                l = l[sort_ind]
                C = C[:,sort_ind]
            elif Hci.shape[0] > 1 and Hci.shape[0] <= self.n_roots:
                l,C = np.linalg.eigh(Hci+0.089*S2)
                l = np.diag(C.T @ Hci @ C)
                sort_ind = np.argsort(l)
                l = l[sort_ind]
                C = C[:,sort_ind]
            elif Hci.shape[0] == 1:
                l = [Hci[0,0]]
                C = np.array([[1.0]])
            else:
                print(" Problem with Hci dimension")
                exit()
        else:
            Hci = .5*(Hci+Hci.T)
            if Hci.shape[0] > 1 and Hci.shape[0] > self.n_roots:
                l,C = scipy.sparse.linalg.eigsh(Hci,self.n_roots,which='SA')
                sort_ind = np.argsort(l)
                l = l[sort_ind]
                C = C[:,sort_ind]
            elif Hci.shape[0] > 1 and Hci.shape[0] <= self.n_roots:
                l,C = np.linalg.eigh(Hci)
                sort_ind = np.argsort(l)
                l = l[sort_ind]
                C = C[:,sort_ind]
            elif Hci.shape[0] == 1:
                l = [Hci[0,0]]
                C = np.array([[1.0]])
            else:
                print(" Problem with Hci dimension")
                exit()
        #print(" Diagonalize Matrix")
        #l,C = np.linalg.eigh(Hci)
        #sort_ind = np.argsort(l)
        #l = l[sort_ind]
        #C = C[:,sort_ind]
        #l = l[0:self.n_roots]
        #C = C[:,0:self.n_roots]
      
        if iprint>0:
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
        return Hci
        #print(" E(nuc) + E(core) = %16.10f" %(self.H.e_nuc+self.H.e_core))
# }}}
    
    def run_davidson(self):
    # {{{
        #print(" Diagonalize Matrix")
        #e,C = scipy.sparse.linalg.eigsh(Hci,self.n_roots)
        #e,C = np.linalg.eigh(Hci)
        #e += self.H.e_nuc + self.H.e_core
       
        ket_a = self.ket_a
        ket_b = self.ket_b
        bra_a = self.bra_a
        bra_b = self.bra_b

        dav = Davidson(self.full_dim, self.n_roots)
        dav.thresh      = self.thresh 
        dav.max_vecs    = 150 
        dav.max_iter    = self.max_iter 
        dav.form_rand_guess()
        dav.sig_curr = np.zeros((self.full_dim, self.n_roots))

        #np.fill_diagonal(Hci, Hci.diagonal() + self.H.e_nuc+self.H.e_core)

        for dit in range(0,dav.max_iter):
           
            dav.sig_curr = np.zeros(dav.vec_curr.shape) 
            #dav.sig_curr = dav.vec_curr * (self.H.e_nuc + self.H.e_core)

            if self.algorithm == 'davidson2':
                self.compute_ab_terms_sigma(dav.sig_curr, dav.vec_curr)
            elif self.algorithm == 'davidson':
                #print(self.Bpr.shape)
                #print(self.H.V.shape)
                #print(dav.vec_curr.shape)
                nvecs = dav.vec_curr.shape[1]
                c = dav.vec_curr*1
                #print(c.shape)
                c.shape = (ket_a.max(), ket_b.max(), nvecs)
                dav.sig_curr.shape = (ket_a.max(),ket_b.max(),nvecs)
                dav.sig_curr += np.einsum('IJpr,KLpr,JLt->IKt',self.Apr,self.Bpr,c,optimize=True)
                dav.sig_curr.shape = (ket_a.max()*ket_b.max(),nvecs)

                    
            for s in range(dav.n_roots):
                sig_curr = cp.deepcopy(dav.sig_curr[:,s])
                vec_curr = cp.deepcopy(dav.vec_curr[:,s])
                sig_curr.shape = (ket_a.max(), ket_b.max())
                vec_curr.shape = (ket_a.max(), ket_b.max())
                #sig_curr.shape = (ket_b.max(), ket_a.max())
                #vec_curr.shape = (ket_b.max(), ket_a.max())
                       
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
        return dav.sig_curr

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

    
    def svd_state(self,norbs1,norbs2, max_dim=None, thresh=-1, both=False):
        """
        Do an SVD of the FCI vector partitioned into clusters with (norbs1 | norbs2)
        where the orbitals are assumed to be ordered for cluster 1| cluster 2 haveing norbs1 and 
        norbs2, respectively.
        """
        # {{{
        
        print(" In svd_state",flush=True)
        assert(norbs1+norbs2==self.no)
        from collections import OrderedDict
        
        vector = OrderedDict()
        ket_a = ci_string(self.no,self.nea)
        ket_b = ci_string(self.no,self.neb)
        v = self.results_v
        v.shape = (ket_a.max(), ket_b.max())
        from bisect import bisect

        fock_labels_a = [[None,None] for i in range(ket_a.max())]
        fock_labels_b = [[None,None] for i in range(ket_b.max())]

        for I in range(ket_a.max()):
            fock_labels_a[I] = bisect(ket_a.config(),norbs1-1)
            ket_a.incr()
        for I in range(ket_b.max()):
            fock_labels_b[I] = bisect(ket_b.config(),norbs1-1)
            ket_b.incr()
       
        for I in range(v.shape[0]):
            for J in range(v.shape[1]):
                try:
                    vector[fock_labels_a[I],fock_labels_b[J]].append(v[I,J])
                except KeyError:
                    vector[fock_labels_a[I],fock_labels_b[J]] = [v[I,J]]

        schmidt_basis = {}
        schmidt_basis_r = {}
        norm = 0
        for fock in vector:
            print()
            print(" Prepare fock space: ", fock)
            vector[fock] = np.array(vector[fock])
        
            ket_a1 = ci_string(norbs1,fock[0])
            ket_b1 = ci_string(norbs1,fock[1])
            ket_a2 = ci_string(norbs2,self.nea-fock[0])
            ket_b2 = ci_string(norbs2,self.neb-fock[1])
          
            # when swapping alpha2 and beta1 do we flip sign?
            sign = 1
            if (self.nea-fock[0]%2)==1 and (fock[1]%2)==1:
                sign = -1
            print(" Dimensions: %5i x %-5i" %(ket_a1.max()*ket_b1.max(), ket_a2.max()*ket_b2.max()))
            norm_curr = vector[fock].T @ vector[fock]
            print(" Norm: %12.8f"%(np.sqrt(norm_curr)))
            vector[fock].shape = (ket_a1.max(), ket_a2.max(), ket_b1.max(), ket_b2.max())
            #vector[fock] = np.ascontiguousarray(np.swapaxes(vector[fock],1,2))
            vector[fock] = sign*np.ascontiguousarray(np.swapaxes(vector[fock],1,2))
            vector[fock].shape = (ket_a1.max()*ket_b1.max(), ket_a2.max()*ket_b2.max())
            norm += norm_curr

            #rdm = vector[fock] @ vector[fock].T
            #print(" Diagonalize RDM of size:",rdm.shape)
            #print(" SVD current block of FCI vector with shape:",vector[fock].shape)
            U,n,V = np.linalg.svd(vector[fock])
            #sort_ind = np.argsort(n)[::-1]
            #n = n[sort_ind]
            #C = C[:,sort_ind]
            print("   %5s:    %12s"%('State','Population'), flush=True)
            nkeep = 0
            for ni_idx,ni in enumerate(n):
                if ni > thresh:
                    nkeep += 1
                #if abs(ni/norm) > 1e-18:
                    print("   %5i:    %12.8f"%(ni_idx,ni), flush=True)
                else:
                    print("   %5i:    %12.8f*"%(ni_idx,ni), flush=True)
            
            if nkeep > 0:
                schmidt_basis[fock] = U[:,:nkeep]
                if both:
                    fock_r = (self.nea-fock[0],self.neb-fock[1])
                    schmidt_basis_r[fock_r] = V[:nkeep,:].T
            
        norm = np.sqrt(norm)
        assert(abs(norm - 1) < 1e-14)
        if both:
            return schmidt_basis, schmidt_basis_r 
        return schmidt_basis 
    # }}}

    def build_S2_matrix(self):
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

        dim = ket_a_max * ket_b_max
        assert(bra_a_max* bra_b_max == dim)
        S2 = np.zeros((dim,dim)) 

        range_ket_a_no = list(range(ket_a.no))
        range_ket_b_no = list(range(ket_b.no))

        ket_a.reset()
        for Ka in range(ket_a_max): 
            
            ket_b.reset()
            for Kb in range(ket_b_max): 
                

                K = Kb + Ka * ket_b_max

                # Sz.Sz
                for ai in ket_a._config:
                    for aj in ket_a._config:
                        if ai != aj:
                            S2[K,K] += 0.25

                for bi in ket_b._config:
                    for bj in ket_b._config:
                        if bi != bj:
                            S2[K,K] += 0.25

                for ai in ket_a._config:
                    for bj in ket_b._config:
                        if ai != bj:
                            S2[K,K] -= 0.5
                
                #Sp.Sm + Sm.Sp 
                for ai in ket_a._config:
                    if ai in ket_b._config:
                        a = 10
                    else:
                        S2[K,K] += 0.75

                for bi in ket_b._config:
                    if bi in ket_a._config:
                        a = 10
                    else:
                        S2[K,K] += 0.75

                #ket_a2 = self.ket_a # dont use
                #ket_b2 = self.ket_b

                ket_a2 = ci_string(ket_a.no, ket_a.ne)
                ket_b2 = ci_string(ket_b.no, ket_b.ne)


                for ai in ket_a._config:
                    for bj in ket_b._config:
                        if ai not in ket_b._config:
                            if bj not in ket_a._config:

                                ket_a2.dcopy(ket_a)
                                ket_b2.dcopy(ket_b)


                                #print("b",ket_a2._config,ket_b2._config)
                                ket_a2.a(ai)  

                                if ket_a2.sign() == 0:
                                    continue

                                ket_b2.c(ai)  
                                if ket_b2.sign() == 0:
                                    continue

                                ket_a2.c(bj)  
                                if ket_a2.sign() == 0:
                                    continue
                                ket_b2.a(bj)  
                                if ket_b2.sign() == 0:
                                    continue

                                sign_a = ket_a2.sign()
                                sign_b = ket_b2.sign()

                                La =ket_a2.linear_index()
                                Lb = ket_b2.linear_index()
                                #print("a",ket_a2._config,ket_b2._config)
                                
                                L = Lb + La * ket_b_max
                               
                                S2[K,L] += 1 * sign_a * sign_b 

                ket_b.incr()
                #   end Ka 

            ket_a.incr()
            #   end Kb 
        self.S2 = S2
        return S2
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
    """
    Compute a(v1,v2,j) = <v1|a_j|v2>

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
    """
    # {{{
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

def build_ca_ss_tdm(no, bra_space, ket_space, ket_basis, bra_basis, spin_case):
    """
    Compute a(v1,v2,j) = <v1|a_j|v2>

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
   
    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_no = range(no)
  
    _abs = abs
   
    v1 = bra_basis[bra_space] 
    v2 = ket_basis[ket_space] 

    assert(v1.shape[0] == len(bra_a)*len(bra_b))
    assert(v2.shape[0] == len(ket_a)*len(ket_b))
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    
    #alpha term 
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                bra.dcopy(ket)
                bra.a(q)
                if bra.sign() == 0:
                    continue
                bra.c(p)
                if bra.sign() == 0:
                    continue
                L = bra.linear_index()
                sign = bra.sign()
                tdm_1spin[L,K,p,q] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)

    
    if spin_case == "a":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(IKpq) v(KJt) = A(stpq)
        tmp = np.einsum('ikpq,kjt->ipqjt',tdm_1spin,v2)
        tdm = np.einsum('ijs,ipqjt->stpq',v1,tmp)
    elif spin_case == "b":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(JLpq) v(ILt) = A(stpq)
        tmp = np.einsum('jlpq,ilt->jpqit',tdm_1spin,v2)
        tdm = np.einsum('ijs,jpqit->stpq',v1,tmp) 

    
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_ca_ss(no,bra_space,ket_space,basis,spin_case):
    """
    Compute a(v1,v2,j) = <v1|a_j|v2>

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
   
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
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                bra.dcopy(ket)
                bra.a(q)
                if bra.sign() == 0:
                    continue
                bra.c(p)
                if bra.sign() == 0:
                    continue
                L = bra.linear_index()
                sign = bra.sign()
                tdm_1spin[L,K,p,q] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)

    
    if spin_case == "a":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(IKpq) v(KJt) = A(stpq)
        tmp = np.einsum('ikpq,kjt->ipqjt',tdm_1spin,v2)
        tdm = np.einsum('ijs,ipqjt->stpq',v1,tmp)
    elif spin_case == "b":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(JLpq) v(ILt) = A(stpq)
        tmp = np.einsum('jlpq,ilt->jpqit',tdm_1spin,v2)
        tdm = np.einsum('ijs,jpqit->stpq',v1,tmp) 

    
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}


def build_ccaa_ss(no,bra_space,ket_space,basis,spin_case):
    """
    Compute a(v1,v2,pqrs) = <v1|p'q'rs|v2>
    
    if spin_case a
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'rs|I'J'> v(I',J',t)
                             v(I,J,s) <I|p'q'rs|I'> v(I',J',t) 
    if spin_case b
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'rs|I'J'> v(I',J',t)
                             v(I,J,s) <J|p'q'rs|J'> v(I',J',t) 

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
    assert(bra_a.ne == ket_a.ne) 
    assert(bra_b.ne == ket_b.ne) 
   
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
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no,no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                for r in range_no:
                    for s in range_no:
                        bra.dcopy(ket)

                        bra.a(s)
                        if bra.sign() == 0:
                            continue
                        
                        bra.a(r)
                        if bra.sign() == 0:
                            continue
                        
                        bra.c(q)
                        if bra.sign() == 0:
                            continue
                        
                        bra.c(p)
                        if bra.sign() == 0:
                            continue
                        
                        L = bra.linear_index()
                        sign = bra.sign()
                        tdm_1spin[L,K,p,q,r,s] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
    
    if spin_case == "a":
        # v(IJt) <IJ|pqrs|KL> v(KLu)  = v(IJt) <I|pqrs|K> v(KJu) = A(tupqrs)
        tdm = oe.contract('ijt,ikpqrs,kju->tupqrs',v1,tdm_1spin,v2)
    elif spin_case == "b":
        # v(IJt) <IJ|pqrs|KL> v(KLu)   = v(IJt) tdm(JLpqrs) v(ILu) = A(tupqrs)
        tdm = oe.contract('ijt,jlpqrs,ilu->tupqrs',v1,tdm_1spin,v2)

 
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}


def build_ca_os(no,bra_space,ket_space,basis,spin_case):
    """
    Compute D(v1,v2,pq) = <v1|p'q|v2> where p,q have different spins

    if spin_case ab
    D(v1,v2,pq) =          v(I,J,s) <IJ|p'q|I'J'> v(I',J',t)
                           v(I,J,s) <I|p'|I'><J|q|J'> v(I',J',t) (-1)^N(I')
    
    if spin_case ba
    D(v1,v2,pq) =          v(I,J,s) <IJ|p'q|I'J'> v(I',J',t)
                          -v(I,J,s) <I|q|I'><J|p'|J'> v(I',J',t) (-1)^N(I')

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'ab' or 'ba'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'ab' or spin_case == 'ba')
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no)
    if spin_case == "ab":
        assert(bra_a.ne == ket_a.ne+1) 
        assert(bra_b.ne == ket_b.ne-1) 
    elif spin_case == "ba":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne+1) 
   
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
    assert(spin_case == "ba" or spin_case=="ab")
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    NAK = ket_space[0]
    Da = np.zeros((bra_a.max(),ket_a.max(),no))
    Db = np.zeros((bra_b.max(),ket_b.max(),no))
    
    #alpha term 
    ket_a.reset()
    bra_a.reset()
    for K in range(ket_a.max()): 
        for p in range_no:
            bra_a.dcopy(ket_a)
            if spin_case == 'ab':
                bra_a.c(p)
            else:
                bra_a.a(p)

            if bra_a.sign() == 0:
                continue
            L = bra_a.linear_index()
            sign = bra_a.sign()
            Da[L,K,p] += sign
        ket_a.incr()


    #beta term 
    ket_b.reset()
    bra_b.reset()
    for K in range(ket_b.max()): 
        for q in range_no:
            bra_b.dcopy(ket_b)
            if spin_case == 'ab':
                bra_b.a(q)
            else:
                bra_b.c(q)
            if bra_b.sign() == 0:
                continue
            L = bra_b.linear_index()
            sign = bra_b.sign()
            Db[L,K,q] += sign

        ket_b.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
   

    # v(IJs) Da(IKp) Db(JLq) v(KLt)   
    #                               = v(IJs) Da(IKp) Dbv(JKqt)
    #                               = vDa(JsKp) Dbv(JqKt)
    #                               = D(stpq)
    if spin_case == 'ab':
        tdm = (-1)**NAK * oe.contract('ijs,ikp,jlq,klt->stpq',v1,Da,Db,v2)
    if spin_case == 'ba':
        tdm = (-1)**(NAK+1) * oe.contract('ijs,ikp,jlq,klt->stqp',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}


def build_ccaa_os(no,bra_space,ket_space,basis,spin_case):
    """
    Compute D(v1,v2,pqrs) = <v1|p'q'r s|v2> where p,q have different spins

    if spin_case abba
    D(v1,v2,p,q,r,s) =          v(I,J,m) <IJ|p'q'r s|I'J'> v(I',J',n)
                           v(I,J,m) <I|p's|I'><J|q'r|J'> v(I',J',n) * sign 
                           sign = (-1)^2*(N(I')+1)
    
    if spin_case ba
    D(v1,v2,p,q,r,s) =          v(I,J,m) <IJ|p'q'r s|I'J'> v(I',J',n)
                          v(I,J,m) <I|q'r|I'><J|p's|J'> v(I',J',n) * sign
                          sign = (-1)^2*(N(I'))+2

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'abba' or 'baab'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'abba' or spin_case == 'baab')
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no)
    assert(bra_a.ne == ket_a.ne) 
    assert(bra_b.ne == ket_b.ne) 
   
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

    NAK = ket_space[0]
    
    #alpha term 
    Da = np.zeros((bra_a.max(),ket_a.max(),no,no))

    ket_a.reset()
    bra_a.reset()
    for K in range(ket_a.max()): 
        for p in range_no:
            for q in range_no:
                bra_a.dcopy(ket_a)
                bra_a.a(q)
                if bra_a.sign() == 0:
                    continue
                bra_a.c(p)
                if bra_a.sign() == 0:
                    continue
                L = bra_a.linear_index()
                sign = bra_a.sign()
                Da[L,K,p,q] += sign

        ket_a.incr()


    #beta term 
    Db = np.zeros((bra_b.max(),ket_b.max(),no,no))

    ket_b.reset()
    bra_b.reset()
    for K in range(ket_b.max()): 
        for p in range_no:
            for q in range_no:
                bra_b.dcopy(ket_b)
                bra_b.a(q)
                if bra_b.sign() == 0:
                    continue
                bra_b.c(p)
                if bra_b.sign() == 0:
                    continue

                L = bra_b.linear_index()
                sign = bra_b.sign()
                #print(ket_b._config,bra_b._config,sign)
                Db[L,K,p,q] += sign
        ket_b.incr()

    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
   

    # v(IJm) Da(IKps) Db(JLqr) v(KLn) = D(mnpqrs)
    if spin_case == 'abba':
        sig = (-1)**(2*(NAK+1))
        tdm = sig*  oe.contract('ijm,ikps,jlqr,kln->mnpqrs',v1,Da,Db,v2)

    if spin_case == 'baab':
        sig = (-1)**(2*(NAK+1))
        tdm = sig * oe.contract('ijm,ikqr,jlps,kln->mnpqrs',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}



def build_aa_ss(no,bra_space,ket_space,basis,spin_case):
    """
    Compute a(v1,v2,p,q) = <v1|pq|v2>
    spin case: a
    D(m,n,p,q) = v(I,J,m) <IJ|pq|I'J'> v(I'J'n)
                 v(I,J,m) <I|pq|I'><J|J'> v(I'J'n)
    
    no sign in 'a' or 'b'

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 

    if spin_case == 'a':
        assert(ket_space[0] == bra_space[0]+2)
    if spin_case == 'b':
        assert(ket_space[1] == bra_space[1]+2)
   
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
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                bra.dcopy(ket)
                bra.a(q)
                if bra.sign() == 0:
                    continue
                bra.a(p)
                if bra.sign() == 0:
                    continue
                L = bra.linear_index()
                sign = bra.sign()
                tdm_1spin[L,K,p,q] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)

    
    if spin_case == "a":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(IKpq) v(KJt) = A(stpq)
        #tmp = np.einsum('ikpq,kjt->ipqjt',tdm_1spin,v2)
        #tdm = np.einsum('ijs,ipqjt->stpq',v1,tmp)
        #tdm = np.einsum('IJm,IKpq,KJn->mnpq',v1,tdm_1spin,v2)
        tdm = oe.contract('IJm,IKpq,KJn->mnpq',v1,tdm_1spin,v2)
        
    elif spin_case == "b":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(JLpq) v(ILt) = A(stpq)
        #tmp = np.einsum('jlpq,ilt->jpqit',tdm_1spin,v2)
        #tdm = np.einsum('ijs,jpqit->stpq',v1,tmp) 
        #tdm = np.einsum('IJm,JLpq,ILn->mnpq',v1,tdm_1spin,v2)
        tdm = oe.contract('IJm,JLpq,ILn->mnpq',v1,tdm_1spin,v2)

    
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_aa_os(no,bra_space,ket_space,basis,spin_case):
    """
    Compute D(v1,v2,pq) = <v1|pq|v2> where p,q have different spins

    if spin_case ab
    D(v1,v2,pq) =          v(I,J,s) <IJ|pq|I'J'> v(I',J',t)
                           v(I,J,s) <I|p|I'><J|q|J'> v(I',J',t) (-1)^N(I')
    
    if spin_case ba
    D(v1,v2,pq) =          v(I,J,s) <IJ|p'q|I'J'> v(I',J',t)
                          -v(I,J,s) <I|q|I'><J|p'|J'> v(I',J',t) (-1)^N(I')

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'ab' or 'ba'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'ab' or spin_case == 'ba')
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no)
    if spin_case == "ab":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne-1) 
    elif spin_case == "ba":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne-1) 
   
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
    assert(spin_case == "ba" or spin_case=="ab")
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    NAK = ket_space[0]
    Da = np.zeros((bra_a.max(),ket_a.max(),no))
    Db = np.zeros((bra_b.max(),ket_b.max(),no))
    
    #alpha term 
    ket_a.reset()
    bra_a.reset()
    for K in range(ket_a.max()): 
        for p in range_no:
            bra_a.dcopy(ket_a)
            if spin_case == 'ab':
                bra_a.a(p)
            else:
                bra_a.a(p)

            if bra_a.sign() == 0:
                continue
            L = bra_a.linear_index()
            sign = bra_a.sign()
            Da[L,K,p] += sign
        ket_a.incr()


    #beta term 
    ket_b.reset()
    bra_b.reset()
    for K in range(ket_b.max()): 
        for q in range_no:
            bra_b.dcopy(ket_b)
            if spin_case == 'ab':
                bra_b.a(q)
            else:
                bra_b.a(q)
            if bra_b.sign() == 0:
                continue
            L = bra_b.linear_index()
            sign = bra_b.sign()
            Db[L,K,q] += sign

        ket_b.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
   

    # v(IJs) Da(IKp) Db(JLq) v(KLt)   
    #                               = v(IJs) Da(IKp) Dbv(JKqt)
    #                               = vDa(JsKp) Dbv(JqKt)
    #                               = D(stpq)
    if spin_case == 'ab':
        tdm = (-1)**NAK * oe.contract('ijs,ikp,jlq,klt->stpq',v1,Da,Db,v2)
    if spin_case == 'ba':
        tdm = (-1)**(NAK+1) * oe.contract('ijs,ikp,jlq,klt->stqp',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_cca_ss(no,bra_space,ket_space,basis,spin_case):
    """
    Compute a(v1,v2,pqr) = <v1|p'q'r|v2>
    
    if spin_case a
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'r|I'J'> v(I',J',t)
                             v(I,J,s) <I|p'q'r|I'> v(I',J',t) 
    if spin_case b
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'r|I'J'> v(I',J',t)
                             v(I,J,s) <J|p'q'r|J'> v(I',J',t) 

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    if spin_case == "a":
        assert(bra_a.ne == ket_a.ne+1) 
        assert(bra_b.ne == ket_b.ne) 
    elif spin_case == "b":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne+1) 
   
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
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                for r in range_no:
                    bra.dcopy(ket)

                    bra.a(r)
                    if bra.sign() == 0:
                        continue
                    
                    bra.c(q)
                    if bra.sign() == 0:
                        continue
                    
                    bra.c(p)
                    if bra.sign() == 0:
                        continue
                    
                    L = bra.linear_index()
                    sign = bra.sign()
                    tdm_1spin[L,K,p,q,r] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
    
    if spin_case == "a":
        # v(IJt) <IJ|pqrs|KL> v(KLu)  = v(IJt) <I|pqrs|K> v(KJu) = A(tupqrs)
        tdm = oe.contract('ijm,ikpqr,kjn->mnpqr',v1,tdm_1spin,v2)
        #tdm = np.einsum('ijm,ikpqr,kjn->mnpqr',v1,tdm_1spin,v2)
    elif spin_case == "b":
        # v(IJt) <IJ|pqrs|KL> v(KLu)   = v(IJt) tdm(JLpqrs) v(ILu) = A(tupqrs)
        sign = (-1)**ket_a.ne
        tdm = oe.contract('ijm,jlpqr,iln->mnpqr',v1,tdm_1spin,v2) * sign

 
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_caa_ss(no,bra_space,ket_space,basis,spin_case):
    """
    Compute a(v1,v2,pqr) = <v1|p'q'r|v2>
    
    if spin_case a
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'r|I'J'> v(I',J',t)
                             v(I,J,s) <I|p'q'r|I'> v(I',J',t) 
    if spin_case b
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'r|I'J'> v(I',J',t)
                             v(I,J,s) <J|p'q'r|J'> v(I',J',t) 

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    if spin_case == "a":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne) 
    elif spin_case == "b":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne-1) 
   
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
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                for r in range_no:
                    bra.dcopy(ket)

                    bra.a(r)
                    if bra.sign() == 0:
                        continue
                    
                    bra.a(q)
                    if bra.sign() == 0:
                        continue
                    
                    bra.c(p)
                    if bra.sign() == 0:
                        continue
                    
                    L = bra.linear_index()
                    sign = bra.sign()
                    tdm_1spin[L,K,p,q,r] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
    
    if spin_case == "a":
        # v(IJt) <IJ|pqrs|KL> v(KLu)  = v(IJt) <I|pqrs|K> v(KJu) = A(tupqrs)
        tdm = oe.contract('ijm,ikpqr,kjn->mnpqr',v1,tdm_1spin,v2)
        #tdm = np.einsum('ijm,ikpqr,kjn->mnpqr',v1,tdm_1spin,v2)
    elif spin_case == "b":
        # v(IJt) <IJ|pqrs|KL> v(KLu)   = v(IJt) tdm(JLpqrs) v(ILu) = A(tupqrs)
        tdm = oe.contract('ijm,jlpqr,iln->mnpqr',v1,tdm_1spin,v2) * (-1)**ket_a.ne

 
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_cca_os(no,bra_space,ket_space,basis,spin_case):
    """
    Compute D(v1,v2,pqr) = <v1|p'q'r|v2> where p,q have different spins

    if spin_case abb
    D(v1,v2,p,q,r) =          v(I,J,m) <IJ|p'q'r|I'J'> v(I',J',n)
                          v(I,J,m) <I|p'|I'><J|q'r|J'> v(I',J',n) * sign 
                          sign = (-1)^2*N(I')
    
    if spin_case baa
    D(v1,v2,p,q,r) =          v(I,J,m) <IJ|p'q'r|I'J'> v(I',J',n)
                          v(I,J,m) <I|q'r|I'><J|p'|J'> v(I',J',n) * sign
                          sign = (-1)^*N(I')

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'abb' or 'baa'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'abb' or spin_case == 'baa' or spin_case =='aba' or spin_case == 'bab')
    if spin_case == "abb":
        assert(bra_a.ne == ket_a.ne+1) 
        assert(bra_b.ne == ket_b.ne) 
    elif spin_case == "baa":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne+1) 
    elif spin_case == "aba":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne+1) 
    if spin_case == "bab":
        assert(bra_a.ne == ket_a.ne+1) 
        assert(bra_b.ne == ket_b.ne) 

   
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

    NAK = ket_space[0]
        
    if spin_case == 'abb' or spin_case == 'bab':
        #alpha term 
        Da = np.zeros((bra_a.max(),ket_a.max(),no))

        ket_a.reset()
        bra_a.reset()
        for K in range(ket_a.max()): 
            for p in range_no:
                bra_a.dcopy(ket_a)
                bra_a.c(p)
                if bra_a.sign() == 0:
                    continue
                L = bra_a.linear_index()
                sign = bra_a.sign()
                Da[L,K,p] += sign

            ket_a.incr()


        #beta term 
        Db = np.zeros((bra_b.max(),ket_b.max(),no,no))

        ket_b.reset()
        bra_b.reset()
        for K in range(ket_b.max()): 
            for p in range_no:
                for q in range_no:
                    bra_b.dcopy(ket_b)
                    bra_b.a(q)
                    if bra_b.sign() == 0:
                        continue
                    bra_b.c(p)
                    if bra_b.sign() == 0:
                        continue

                    L = bra_b.linear_index()
                    sign = bra_b.sign()
                    #print(ket_b._config,bra_b._config,sign)
                    Db[L,K,p,q] += sign
            ket_b.incr()

    elif spin_case == 'baa' or spin_case == 'aba':
        #alpha term 
        Da = np.zeros((bra_a.max(),ket_a.max(),no,no))

        ket_a.reset()
        bra_a.reset()
        for K in range(ket_a.max()): 
            for p in range_no:
                for q in range_no:
                    bra_a.dcopy(ket_a)
                    bra_a.a(q)
                    if bra_a.sign() == 0:
                        continue
                    bra_a.c(p)
                    if bra_a.sign() == 0:
                        continue
                    L = bra_a.linear_index()
                    sign = bra_a.sign()
                    Da[L,K,p,q] += sign

            ket_a.incr()


        #beta term 
        Db = np.zeros((bra_b.max(),ket_b.max(),no))

        ket_b.reset()
        bra_b.reset()
        for K in range(ket_b.max()): 
            for p in range_no:
                bra_b.dcopy(ket_b)
                bra_b.c(p)
                if bra_b.sign() == 0:
                    continue

                L = bra_b.linear_index()
                sign = bra_b.sign()
                #print(ket_b._config,bra_b._config,sign)
                Db[L,K,p] += sign
            ket_b.incr()

    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)


    # v(IJm) Da(IKps) Db(JLqr) v(KLn) = D(mnpqrs)
    if spin_case == 'abb':
        sig = (-1)**(2*(NAK))
        tdm = sig*  oe.contract('ijm,ikp,jlqr,kln->mnpqr',v1,Da,Db,v2)

    if spin_case == 'baa':
        sig = (-1)**(NAK)
        tdm = sig * oe.contract('ijm,ikqr,jlp,kln->mnpqr',v1,Da,Db,v2)

    if spin_case == 'bab':
        sig = (-1)**(2*(NAK)+1)
        tdm = sig*  oe.contract('ijm,ikq,jlpr,kln->mnpqr',v1,Da,Db,v2)

    if spin_case == 'aba':
        sig = (-1)**(NAK-1)
        tdm = sig * oe.contract('ijm,ikpr,jlq,kln->mnpqr',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}

def build_caa_os(no,bra_space,ket_space,basis,spin_case):
    """
    Compute D(v1,v2,qrs) = <v1|q'r s|v2> where p,q have different spins

    if spin_case aab
    D(v1,v2,q,r,s) =          v(I,J,m) <IJ|q'r s|I'J'> v(I',J',n)
                          v(I,J,m) <I|s|I'><J|q'r|J'> v(I',J',n) * sign 
                          sign = (-1)^*(N(I')
    
    if spin_case bba
    D(v1,v2,q,r,s) =          v(I,J,m) <IJ|q'r s|I'J'> v(I',J',n)
                          v(I,J,m) <I|q'r|I'><J|s|J'> v(I',J',n) * sign
                          sign = (-1)^2*(N(I')+1)

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'bba' or 'aab'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'bba' or spin_case == 'aab' or spin_case == 'bab' or spin_case == 'aba')

    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no)
    if spin_case == "aab":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne-1) 
    elif spin_case == "bba":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne) 
    elif spin_case == "aba":
        assert(bra_a.ne == ket_a.ne) 
        assert(bra_b.ne == ket_b.ne-1) 
    elif spin_case == "bab":
        assert(bra_a.ne == ket_a.ne-1) 
        assert(bra_b.ne == ket_b.ne) 
   
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

    NAK = ket_space[0]
    
    if spin_case == 'aab' or spin_case == 'aba':
        #alpha term 
        Da = np.zeros((bra_a.max(),ket_a.max(),no,no))

        ket_a.reset()
        bra_a.reset()
        for K in range(ket_a.max()): 
            for p in range_no:
                for q in range_no:
                    bra_a.dcopy(ket_a)
                    bra_a.a(q)
                    if bra_a.sign() == 0:
                        continue
                    bra_a.c(p)
                    if bra_a.sign() == 0:
                        continue
                    L = bra_a.linear_index()
                    sign = bra_a.sign()
                    Da[L,K,p,q] += sign

            ket_a.incr()


        #beta term 
        Db = np.zeros((bra_b.max(),ket_b.max(),no))

        ket_b.reset()
        bra_b.reset()
        for K in range(ket_b.max()): 
            for q in range_no:
                bra_b.dcopy(ket_b)
                bra_b.a(q)
                if bra_b.sign() == 0:
                    continue

                L = bra_b.linear_index()
                sign = bra_b.sign()
                #print(ket_b._config,bra_b._config,sign)
                Db[L,K,q] += sign
            ket_b.incr()

    if spin_case == 'bba' or spin_case == 'bab':
        #alpha term 
        Da = np.zeros((bra_a.max(),ket_a.max(),no))

        ket_a.reset()
        bra_a.reset()
        for K in range(ket_a.max()): 
            for q in range_no:
                bra_a.dcopy(ket_a)
                bra_a.a(q)
                if bra_a.sign() == 0:
                    continue
                L = bra_a.linear_index()
                sign = bra_a.sign()
                Da[L,K,q] += sign

            ket_a.incr()


        #beta term 
        Db = np.zeros((bra_b.max(),ket_b.max(),no,no))

        ket_b.reset()
        bra_b.reset()
        for K in range(ket_b.max()): 
            for p in range_no:
                for q in range_no:
                    bra_b.dcopy(ket_b)
                    bra_b.a(q)
                    if bra_b.sign() == 0:
                        continue
                    bra_b.c(p)
                    if bra_b.sign() == 0:
                        continue

                    L = bra_b.linear_index()
                    sign = bra_b.sign()
                    #print(ket_b._config,bra_b._config,sign)
                    Db[L,K,p,q] += sign
            ket_b.incr()

    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
   

    # v(IJm) Da(IKps) Db(JLqr) v(KLn) = D(mnpqrs)
    if spin_case == 'bba':
        sig = (-1)**(2*(NAK+1))
        tdm = sig*  oe.contract('ijm,iks,jlqr,kln->mnqrs',v1,Da,Db,v2)

    if spin_case == 'aab':
        sig = (-1)**(NAK)
        tdm = sig * oe.contract('ijm,ikqr,jls,kln->mnqrs',v1,Da,Db,v2)

    if spin_case == 'bab':
        sig = (-1)**(2*NAK-1)
        tdm = sig*  oe.contract('ijm,ikr,jlqs,kln->mnqrs',v1,Da,Db,v2)

    if spin_case == 'aba':
        sig = (-1)**(NAK-1)
        tdm = sig * oe.contract('ijm,ikqs,jlr,kln->mnqrs',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    return tdm
# }}}


def build_1tdm_costly(no,bra_space,ket_space,basis,spin_case):
    """
    Copy of ca_ss

    Compute a(v1,v2,j) = <v1|a_j|v2>

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
   
    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_no = range(no)
  
    _abs = abs
   
    v1 = basis
    v2 = basis

    assert(v1.shape[0] == len(bra_a)*len(bra_b))
    assert(v2.shape[0] == len(ket_a)*len(ket_b))
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    
    #alpha term 
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                bra.dcopy(ket)
                bra.a(q)
                if bra.sign() == 0:
                    continue
                bra.c(p)
                if bra.sign() == 0:
                    continue
                L = bra.linear_index()
                sign = bra.sign()
                tdm_1spin[L,K,p,q] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)

    
    if spin_case == "a":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(IKpq) v(KJt) = A(stpq)
        tmp = np.einsum('ikpq,kjt->ipqjt',tdm_1spin,v2)
        tdm = np.einsum('ijs,ipqjt->stpq',v1,tmp)
    elif spin_case == "b":
        # v(IJs) <IJ|pq|KL> v(KLt)   = v(IJs) tdm(JLpq) v(ILt) = A(stpq)
        tmp = np.einsum('jlpq,ilt->jpqit',tdm_1spin,v2)
        tdm = np.einsum('ijs,jpqit->stpq',v1,tmp) 

    
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    tdm = tdm.reshape(tdm.shape[2],tdm.shape[3])

    print(tdm)
    print(tdm.shape)

    return tdm
# }}}

def build_2tdmss_costly(no,bra_space,ket_space,basis,spin_case):
    """
    Copy of ccaa_ss

    Compute a(v1,v2,pqrs) = <v1|p'q'rs|v2>
    
    if spin_case a
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'rs|I'J'> v(I',J',t)
                             v(I,J,s) <I|p'q'rs|I'> v(I',J',t) 
    if spin_case b
    D(v1,v2,pqrs) =          v(I,J,s) <IJ|p'q'rs|I'J'> v(I',J',t)
                             v(I,J,s) <J|p'q'rs|J'> v(I',J',t) 

    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        spin_case: 'a' or 'b'
        basis = dict of basis vectors
    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
   
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no) 
    assert(bra_a.ne == ket_a.ne) 
    assert(bra_b.ne == ket_b.ne) 
   
    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_no = range(no)
  
    _abs = abs
   
    v1 = basis
    v2 = basis

    assert(v1.shape[0] == len(bra_a)*len(bra_b))
    assert(v2.shape[0] == len(ket_a)*len(ket_b))
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    
    #alpha term 
    if spin_case == "a":
        ket = cp.deepcopy(ket_a)
        bra = cp.deepcopy(bra_a)
    elif spin_case == "b":
        ket = cp.deepcopy(ket_b)
        bra = cp.deepcopy(bra_b)
    
    tdm_1spin = np.zeros((bra.max(),ket.max(),no,no,no,no))
    ket.reset()
    bra.reset()
    for K in range(ket.max()): 
        for p in range_no:
            for q in range_no:
                for r in range_no:
                    for s in range_no:
                        bra.dcopy(ket)

                        bra.a(s)
                        if bra.sign() == 0:
                            continue
                        
                        bra.a(r)
                        if bra.sign() == 0:
                            continue
                        
                        bra.c(q)
                        if bra.sign() == 0:
                            continue
                        
                        bra.c(p)
                        if bra.sign() == 0:
                            continue
                        
                        L = bra.linear_index()
                        sign = bra.sign()
                        tdm_1spin[L,K,p,q,r,s] += sign

        ket.incr()
  
    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
    
    if spin_case == "a":
        # v(IJt) <IJ|pqrs|KL> v(KLu)  = v(IJt) <I|pqrs|K> v(KJu) = A(tupqrs)
        tdm = oe.contract('ijt,ikpqrs,kju->tupqrs',v1,tdm_1spin,v2)
    elif spin_case == "b":
        # v(IJt) <IJ|pqrs|KL> v(KLu)   = v(IJt) tdm(JLpqrs) v(ILu) = A(tupqrs)
        tdm = oe.contract('ijt,jlpqrs,ilu->tupqrs',v1,tdm_1spin,v2)

 
    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    tdm = tdm.reshape(tdm.shape[2],tdm.shape[3],tdm.shape[4],tdm.shape[5])
    print(tdm.shape)
    return tdm
# }}}

def build_2tdmos_costly(no,bra_space,ket_space,basis,spin_case):
    """
    Copy of ccaa_os

    Compute D(v1,v2,pqrs) = <v1|p'q'r s|v2> where p,q have different spins

    if spin_case abba
    D(v1,v2,p,q,r,s) =          v(I,J,m) <IJ|p'q'r s|I'J'> v(I',J',n)
                           v(I,J,m) <I|p's|I'><J|q'r|J'> v(I',J',n) * sign 
                           sign = (-1)^2*(N(I')+1)
    
    if spin_case ba
    D(v1,v2,p,q,r,s) =          v(I,J,m) <IJ|p'q'r s|I'J'> v(I',J',n)
                          v(I,J,m) <I|q'r|I'><J|p's|J'> v(I',J',n) * sign
                          sign = (-1)^2*(N(I'))+2

    where N(I') is Number of Alpha electrons in the ket_space
        
    Input: 
        no = n_orbs
        bra_space = (n_alpha,n_beta) for bra
        ket_space = (n_alpha,n_beta) for ket
        basis = dict of basis vectors
        spin_case: 'abba' or 'baab'

    """
    # {{{ 
    bra_a = ci_string(no, bra_space[0])
    bra_b = ci_string(no, bra_space[1])
    ket_a = ci_string(no, ket_space[0])
    ket_b = ci_string(no, ket_space[1])
  

    assert(spin_case == 'abba' or spin_case == 'baab')
    assert(ket_a.no == ket_b.no) 
    assert(bra_a.no == ket_a.no)
    assert(bra_a.ne == ket_a.ne) 
    assert(bra_b.ne == ket_b.ne) 
   
    # avoid python function call overhead
    ket_a_max = ket_a.max()
    ket_b_max = ket_b.max()
    bra_a_max = bra_a.max()
    bra_b_max = bra_b.max()
    
    range_no = range(no)
  
    _abs = abs
   
    v1 = basis
    v2 = basis

    assert(v1.shape[0] == len(bra_a)*len(bra_b))
    assert(v2.shape[0] == len(ket_a)*len(ket_b))
    nv1 = v1.shape[1]
    nv2 = v2.shape[1]

    NAK = ket_space[0]
    
    #alpha term 
    Da = np.zeros((bra_a.max(),ket_a.max(),no,no))

    ket_a.reset()
    bra_a.reset()
    for K in range(ket_a.max()): 
        for p in range_no:
            for q in range_no:
                bra_a.dcopy(ket_a)
                bra_a.a(q)
                if bra_a.sign() == 0:
                    continue
                bra_a.c(p)
                if bra_a.sign() == 0:
                    continue
                L = bra_a.linear_index()
                sign = bra_a.sign()
                Da[L,K,p,q] += sign

        ket_a.incr()


    #beta term 
    Db = np.zeros((bra_b.max(),ket_b.max(),no,no))

    ket_b.reset()
    bra_b.reset()
    for K in range(ket_b.max()): 
        for p in range_no:
            for q in range_no:
                bra_b.dcopy(ket_b)
                bra_b.a(q)
                if bra_b.sign() == 0:
                    continue
                bra_b.c(p)
                if bra_b.sign() == 0:
                    continue

                L = bra_b.linear_index()
                sign = bra_b.sign()
                #print(ket_b._config,bra_b._config,sign)
                Db[L,K,p,q] += sign
        ket_b.incr()

    v2.shape = (ket_a_max,ket_b_max,nv2)
    v1.shape = (bra_a_max,bra_b_max,nv1)
   

    # v(IJm) Da(IKps) Db(JLqr) v(KLn) = D(mnpqrs)
    if spin_case == 'abba':
        sig = (-1)**(2*(NAK+1))
        tdm = sig*  oe.contract('ijm,ikps,jlqr,kln->mnpqrs',v1,Da,Db,v2)

    if spin_case == 'baab':
        sig = (-1)**(2*(NAK+1))
        tdm = sig * oe.contract('ijm,ikqr,jlps,kln->mnpqrs',v1,Da,Db,v2)

    v2.shape = (ket_a_max*ket_b_max,nv2)
    v1.shape = (bra_a_max*bra_b_max,nv1)
   
    tdm = tdm.reshape(tdm.shape[2],tdm.shape[3],tdm.shape[4],tdm.shape[5])
    print(tdm.shape)
    return tdm
# }}}
