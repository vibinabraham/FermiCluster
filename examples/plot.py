import numpy as np
import scipy 
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt

H = np.load('H_cmf.npy')
print(H.shape)
print(" Min of diagonal", min(H.diagonal()))
e,v = scipy.sparse.linalg.eigsh(H,1,which='SA')
idx = e.argsort()
e = e[idx]
v = v[:,idx]
v = v[:,0]
e = e[0]
print(e)
#plt.matshow(np.abs(H))
#plt.colorbar()
#plt.plot(sorted(v*v,reverse=True)/max(v*v))



plt.plot(sorted(v*v,reverse=True), marker='', label="4x4 Basis")
#plt.plot(sorted(v*v))

H2 = np.load('H_hf.npy')
print(" Min of diagonal", min(H2.diagonal()))
e2,v2 = scipy.sparse.linalg.eigsh(H2,10,which='SA')
idx = e2.argsort()
e2 = e2[idx]
v2 = v2[:,idx]
v2 = v2[:,0]
e2 = e2[0]
#e = e[0]
#v2 = v2[:,0]
print(e)
#plt.matshow(np.abs(H))
#plt.colorbar()
#plt.plot(v2*v2/)
#plt.plot(sorted(v2*v2))
plt.plot(sorted(v2*v2,reverse=True), marker='', label="Determinant Basis")
#plt.plot(sorted(v2*v2,reverse=True)/max(v2*v2))
#plt.plot(sorted(abs(H2.diagonal())))
plt.yscale('log')
plt.xlim(-100,1e3)
plt.ylim(1e-9,1)
plt.xlabel('Basis Vector (Sorted)')
plt.ylabel('$|c_x|^2$')

print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
thresh = 1-1e-5
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
plt.plot([count], [val], marker='+', markersize=20, color="black")
#plt.plot([count], [val], marker='|', markersize=20, color="#1f77b4")


val = 0
count = 0
pop = 0
for i in sorted(v2*v2,reverse=True):
    pop += i
    count += 1
    val = i
    if pop >= thresh:
        break
print(count, val)
plt.plot([count], [val], marker='+', markersize=20, color="black")
#plt.plot([count], [val], marker='|', markersize=20, color="#ff7f02")


#plt.show()
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig("plot_weight.pdf", bbox_inches='tight')
plt.close()


plt.grid(True)
plt.plot(sorted(H.diagonal()), '--', label="4x4 Basis")
plt.plot(sorted(H2.diagonal()), '--', label="Determinant Basis")
plt.xlabel('Basis Vector (Sorted)')
plt.ylabel('Diagonal of Hamiltonian')
#plt.show()
plt.legend(loc='lower right')
plt.savefig("plot_hdiag.pdf", bbox_inches='tight')
plt.close()
