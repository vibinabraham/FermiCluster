import numpy as np
import scipy 
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt

thresh = 1-1e-7
################################################################################
H = np.load('H_cmf.npy')
e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
idx = e.argsort()
e = e[idx]
v = v[:,idx]
v = v[:,0]
e = e[0]
print(" Min of diagonal: CMF    ", min(H.diagonal()))
print(" State Energy            ", e)
plt.plot(sorted(v*v,reverse=True), marker='', label="4x4 CMF Basis")
val = 0
count = 0
pop = 0
for i in sorted(v*v,reverse=True):
    pop += i
    count += 1
    val = i
    if pop >= thresh:
        break
print(count, val)
plt.plot([count], [val], marker='+', markersize=20, color="#1f77b4")

################################################################################
H = np.load('H_tucker.npy')
e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
idx = e.argsort()
e = e[idx]
v = v[:,idx]
v = v[:,0]
e = e[0]
print(" Min of diagonal: Tucker ", min(H.diagonal()))
print(" State Energy            ", e)
plt.plot(sorted(v*v,reverse=True), marker='', label="4x4 Tucker Basis")
val = 0
count = 0
pop = 0
for i in sorted(v*v,reverse=True):
    pop += i
    count += 1
    val = i
    if pop >= thresh:
        break
print(count, val)
plt.plot([count], [val], marker='+',  markersize=20, color="#ff7f0e")



################################################################################
print(" Doing HF")
H = np.load('H_hf.npy')
e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
idx = e.argsort()
e = e[idx]
v = v[:,idx]
v = v[:,0]
e = e[0]
print(" Min of diagonal         ", min(H.diagonal()))
print(" State Energy            ", e)
plt.plot(sorted(v*v,reverse=True), marker='', label="HF Determinant Basis")
val = 0
count = 0
pop = 0
for i in sorted(v*v,reverse=True):
    pop += i
    count += 1
    val = i
    if pop >= thresh:
        break
print(count, val)
plt.plot([count], [val], marker='+', markersize=20, color="#2ca02c")



################################################################################
H = np.load('H_no.npy')
e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
#e,v = np.linalg.eig(H)
idx = e.argsort()
e = e[idx]
v = v[:,idx]
v = v[:,0]
e = e[0]
print(" Min of diagonal         ", min(H.diagonal()))
print(" State Energy            ", e)
plt.plot(sorted(v*v,reverse=True), marker='', label="NO Determinant Basis")
val = 0
count = 0
pop = 0
for i in sorted(v*v,reverse=True):
    pop += i
    count += 1
    val = i
    if pop >= thresh:
        break
print(count, val)
plt.plot([count], [val], marker='+', markersize=20, color="#d62728")
#plt.plot([count], [val], marker='+', markersize=20, color="#d62728", label="%e of FCI "%thresh)



plt.yscale('log')
#plt.xlim(-100,1e3)
plt.ylim(1e-16,1)
plt.xlabel('Basis vector index (Sorted)')
plt.ylabel('$|c_x|^2$,  weight of basis vector in FCI state')

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])


#plt.show()
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig("plot_weight.pdf", bbox_inches='tight')
plt.close()


plt.grid(True)
plt.plot(sorted(H.diagonal()), '--', label="4x4 Basis")
plt.plot(sorted(H.diagonal()), '--', label="Determinant Basis")
plt.xlabel('Basis Vector (Sorted)')
plt.ylabel('Diagonal of Hamiltonian')
#plt.show()
plt.legend(loc='lower right')
plt.savefig("plot_hdiag.pdf", bbox_inches='tight')
plt.close()
