from Cluster import *

class DetCluster(Cluster):
    """
    DetCluster defines a cluster of orbitals whose basis remains as a set of determinants
    The purpose of this class, is to allow for a mixture of cluster states, and determinant states.
    """
    def build_op_matrices(self):
        """
        build all operators needed
        """
        self.ops['A'] = {}
        self.ops['a'] = {}
        self.ops['B'] = {}
        self.ops['b'] = {}
        self.ops['Aa'] = {}
        self.ops['Bb'] = {}
        self.ops['Ab'] = {}
        self.ops['Ba'] = {}
        self.ops['AAaa'] = {}
        self.ops['BBbb'] = {}
        self.ops['ABba'] = {}
        self.ops['BAab'] = {}
        self.ops['AA'] = {}
        self.ops['BB'] = {}
        self.ops['AB'] = {}
        self.ops['BA'] = {}
        self.ops['aa'] = {}
        self.ops['bb'] = {}
        self.ops['ba'] = {}
        self.ops['ab'] = {}
        self.ops['AAa'] = {}
        self.ops['BBb'] = {}
        self.ops['ABb'] = {}
        self.ops['BAa'] = {}
        self.ops['BAb'] = {}
        self.ops['ABa'] = {}
        self.ops['Aaa'] = {}
        self.ops['Bbb'] = {}
        self.ops['Aab'] = {}
        self.ops['Bba'] = {}
        self.ops['Aba'] = {}
        self.ops['Bab'] = {}

        self.strings_bra = {}
        for na in range(0,self.n_orb+1):
            for nb in range(0,self.n_orb+1):
                a = ci_string(self.n_orb, na)
                b = ci_string(self.n_orb, nb)
                self.strings_bra[(na,nb)] = [a,b]
        self.strings_ket = cp.deepcopy(self.strings_bra)
    
    def get_ops(self):
        return self.ops

    def get_op_mel(self,opstr,fI,fJ,I,J):
        out = np.array([])
        print(opstr)
        I+=6
        if opstr == 'Aa':
            out = np.zeros((self.n_orb,self.n_orb))
            assert(fI[0] == fJ[0])
            assert(fI[1] == fJ[1])
            bra_a = ci_string(self.n_orb, fI[0])
            bra_b = ci_string(self.n_orb, fI[1])
            ket_a = ci_string(self.n_orb, fJ[0])
            ket_b = ci_string(self.n_orb, fJ[1])
            
            Ib = I % bra_b.max()
            Ia = int(I/bra_b.max())
            Jb = J % ket_b.max()
            Ja = int(J/ket_b.max())
   
            print(Ia,Ib,Ja,Jb)
            if Ib != Jb:
                return 
            if Ia == Ja:
                return np.eye(self.n_orb)
            else:
                bra_a.set_to_index(Ia)
                ket_a.set_to_index(Ja)
                #bra_b.set_to_index(Ib)
                #ket_b.set_to_index(Jb)
            print(bra_a, ket_a)
            occIa = np.array([0]*self.n_orb,dtype=int)
            occJa = np.array([0]*self.n_orb,dtype=int)
            for i in bra_a.config():
                occIa[i] = 1
            for i in ket_a.config():
                occJa[i] = 1
            print(occIa-occJa)
            #print(bra_a,bra_a.linear_index())
            #print(bra_b,bra_b.linear_index())
        

        elif opstr == 'Bb':
            out = np.zeros((self.n_orb,self.n_orb))
            assert(fI[0] == fJ[0])
            assert(fI[1] == fJ[1])
            bra_a = ci_string(self.n_orb, fI[0])
            bra_b = ci_string(self.n_orb, fI[1])
            ket_a = ci_string(self.n_orb, fJ[0])
            ket_b = ci_string(self.n_orb, fJ[1])
            
            Ib = I % bra_b.max()
            Ia = int(I/bra_b.max())
            Jb = J % ket_b.max()
            Ja = int(J/ket_b.max())
    
            if Ia != Ja:
                return 
            if Ib == Jb:
                return np.eye(self.n_orb)
            else:
                bra_a.set_to_index(Ia)
                ket_a.set_to_index(Ja)
                #bra_b.set_to_index(Ib)
                #ket_b.set_to_index(Jb)
            print(bra_a, ket_a)
            #print(bra_a,bra_a.linear_index())
            #print(bra_b,bra_b.linear_index())
        return 
            

