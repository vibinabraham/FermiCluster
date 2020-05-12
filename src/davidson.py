import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import argparse
import scipy.sparse
import scipy.sparse.linalg

class Davidson:
    def __init__(self,dim,n_roots):
        self.iter           = 0     # Current iteration
        self.max_iter       = 20    # Max iterations
        self.do_precondition= 0     # Use preconditioning?
        self.n_vecs         = 0     # Current number of vectors in subspace
        self.max_vecs       = 10    # Max number of vectors in subspace
        self.thresh         = 1e-5  # Convergence threshold
        self.n_roots        = n_roots  # Number of roots to optimize
        self.dim            = dim      # Dimension of matrix to diagonalize

        self.res_vals       = []    # Residual values
        self.ritz_vals      = []    # Ritz values
 
        self.sig_curr           = np.empty((self.dim, 0))  # Current sigma vectors
        self.sig_prev           = np.empty((self.dim, 0))  # Previous sigma vectors
        self.vec_curr           = np.empty((self.dim, 0))  # Current guess vectors
        self.vec_prev           = np.empty((self.dim, 0))  # Previous guess vectors
        self.precond_diag       = np.array(())  # Previous guess vectors

        self.ritz_vecs          = np.array(())

    def form_sigma(self):
        pass

    def restart(self):
        self.vec_prev           = self.vec_prev.dot(self.ritz_vecs[:,0:self.n_roots])
        self.sig_prev           = self.sig_prev.dot(self.ritz_vecs[:,0:self.n_roots])
        self.ritz_vecs          = np.eye(self.n_roots) 
        #self.ritz_vecs          = self.ritz_vecs[:,0:self.n_roots] 
        self.res_vals           = self.vec_prev.T.dot(self.sig_prev).diagonal().tolist() 
        
            
    def vec(self):
        return np.hstack((self.vec_prev,self.vec_curr))

    def sig(self):
        return np.hstack((self.sig_prev,self.sig_curr))

    def form_rand_guess(self):
        self.vec_curr, tmp = np.linalg.qr(np.random.rand(self.dim, self.n_roots))
        #self.vec_curr = np.vstack((np.eye(self.n_roots), np.zeros((max(0,self.dim-self.n_roots), self.n_roots)) ))+np.random.rand(self.dim, self.n_roots)
        #self.vec_curr, tmp = np.linalg.qr(self.vec_curr)
    
    def form_p_guess(self):
        #self.vec_curr, tmp = np.linalg.qr(np.random.rand(self.dim, self.n_roots))
        if self.dim > self.n_roots:
            self.vec_curr = np.vstack((np.eye(self.n_roots), np.zeros((max(0,self.dim-self.n_roots), self.n_roots)) ))
        else:
            self.vec_curr = np.eye(self.dim)
   
    """
    def form_p_guess(self):
        #self.vec_curr, tmp = np.linalg.qr(np.random.rand(self.dim, self.n_roots))
        if self.dim > self.n_roots:
            self.vec_curr = np.vstack((np.eye(self.n_roots), np.zeros((max(0,self.dim-self.n_roots), self.n_roots)) ))
            self.vec_curr += np.random.rand(self.dim, self.n_roots)*1e-4
            self.vec_curr, tmp = np.linalg.qr(self.vec_curr)
        else:
            self.vec_curr = np.eye(self.dim)
            self.vec_curr += np.random.rand(self.dim, self.dim)*1e-4
            self.vec_curr, tmp = np.linalg.qr(self.vec_curr)
    """
    
    def eigenvectors(self):
        return self.vec_prev.dot(self.ritz_vecs[:,0:self.n_roots])
    
    def eigenvalues(self):
        return self.ritz_vals

    def set_preconditioner(self,d):
        self.precond_diag = d
        self.precond_diag.shape = (self.dim,1)
        self.do_precondition = 1

    def update(self):
        if self.n_vecs > self.max_vecs:
            self.restart()

        vec = self.vec()
        sig = self.sig()
        T = vec.T.dot(sig)
        #print self.AV_j.shape
        #print self.sigma.shape
        T = .5*(T+T.T)
        l,v = np.linalg.eigh(T)
        
        sort_ind = np.argsort(l)
        l = l[sort_ind]
        v = v[:,sort_ind]
        
        res_vals = []
        ritz_vals = []
        v_new = np.empty((self.dim, 0))
        for n in range(0,self.n_roots):
            l_n = l[n]
            ritz_vals.append(l_n)
            v_n = v[:,n]
            r_n = (sig - l_n*vec).dot(v_n);
            b_n = np.linalg.norm(r_n)
            if b_n > 1e-15:
                r_n = r_n/b_n
            res_vals.append(b_n)
            
            r_n.shape = (r_n.shape[0],1)

            if self.do_precondition:
                r_n = self.precondition(l_n,r_n)
                #for i in range(0,self.dim):
                #    print "%12.8f" %r_n[i]


            if b_n > self.thresh:
                r_n = r_n - vec.dot(np.dot(vec.T,r_n))
                
                if (v_new.shape[1] > 0):
                    r_n = r_n - v_new.dot(np.dot(v_new.T,r_n))
                b_n_p = np.linalg.norm(r_n)
                if (b_n / b_n_p > 1e-15):
                    r_n = r_n / b_n_p
                    tmp = np.hstack((vec,v_new))
                    r_n = r_n - tmp.dot(np.dot(tmp.T,r_n))
                    r_n = r_n/np.linalg.norm(r_n)
                    v_new = np.hstack((v_new, r_n))

        self.ritz_vecs = v
        self.ritz_vals = ritz_vals
        self.res_vals  = res_vals

        self.sig_prev = np.hstack((self.sig_prev, self.sig_curr))
        self.vec_prev = np.hstack((self.vec_prev, self.vec_curr))


        self.vec_curr = v_new
        self.n_vecs = self.vec().shape[1]
        self.iter += 1

    def print_iteration(self):
        print("  Davidson Iter %4i " %self.iter,"|"," Vecs:%4li : "% self.n_vecs ,"|",end='')
        for r in range(0,self.n_roots):
            print(" %16.8f "%self.ritz_vals[r],end='')
        print("|",end='')
        for r in range(0,self.n_roots):
            print(" %6.1e " % self.res_vals[r],end='') 
        print()

    def converged(self):
        """
        Check for convergence
        """
        done = 1
        for k in range(0,len(self.res_vals)):
            if abs(self.res_vals[k] > self.thresh):
                done = 0
        return done

    def precondition(self,l_n,r_n):
        if len(self.precond_diag.shape) == 0:
            print(" No diagonal found!")
            exit(-1)
        lin_dep_thresh = 1e-12
        m = self.precond_diag - l_n + lin_dep_thresh  
        """
        for i in range(0,self.dim): 
            if abs(m[i]) < lin_dep_thresh:
                m[i] = lin_dep_thresh;
                """
        r_n = r_n/m
        """
        for i in range(0,self.dim): 
            if abs(m[i]) < lin_dep_thresh:
                m[i] = 1.0/lin_dep_thresh;
            else:
                m[i] = 1.0/m[i];
                r_n[i] = m[i]*r_n[i]
        """
        return r_n

