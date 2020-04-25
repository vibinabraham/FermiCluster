import sklearn.cluster as cluster
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt 
import time
import seaborn as sns

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


from tools import *

scf = -2.16024391299511
ecore = 2.29310125
correlation = np.zeros((8,8)) + scf - ecore
correlation[1, 2] =  -4.46594490
correlation[0, 2] =  -4.45915068
correlation[1, 3] =  -4.45557827
correlation[1, 5] =  -4.45525483
correlation[0, 7] =  -4.45488078
correlation[0, 3] =  -4.45450627
correlation[0, 4] =  -4.45429425
correlation[1, 4] =  -4.45428814
correlation[0, 6] =  -4.45425764
correlation[1, 6] =  -4.45425276
correlation[1, 7] =  -4.45397744
correlation[0, 5] =  -4.45370727

correlation += ecore - scf


correlation += correlation.T

#data = np.exp(abs(correlation))
data = abs(correlation)
#data = abs(correlation + .001*np.random.rand(8,8))
data += data.T
data = data / np.max(np.max(data))
for di in range(data.shape[0]):
    data[di,di] = 0
print_mat(data)

A = data
D = sum(A)
L = np.diag(D) - A

#Normalized
if 0:
    Dinvsqrt = np.sqrt(D)
    for i in range(Dinvsqrt.shape[0]):
        Dinvsqrt[i] = 1/Dinvsqrt[i]
    L = np.eye(L.shape[0]) - np.diag(Dinvsqrt) @ L @ np.diag(Dinvsqrt)
    print_mat(L)

Li,Lv = np.linalg.eigh(L)
idx = Li.argsort()
Lv = Lv[:,idx]
Li = Li[idx]


for ei in range(Li.shape[0]):
    if ei==0:
        print(" %4i Eval = %12.8f" %(ei+1,Li[ei]))
    else:
        print(" %4i Eval = %12.8f Gap = %12.8f" %(ei+1,Li[ei],Li[ei]-Li[ei-1]))

print(" Fiedler vector")
for i in range(Li.shape[0]):
    print(" %4i %12.8f" %(i,Lv[i,1]))

if 0:
    plt.matshow(data);
    plt.colorbar()
    plt.show()

print(" Now do kmeans clustering")
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(data)
print(kmeans.labels_)

print(" Now do spectral clustering")
clustering = SpectralClustering(n_clusters=2,affinity='precomputed').fit(A)
#clustering = SpectralClustering(n_clusters=2,random_state=0, affinity='precomputed').fit(A)
print(clustering.labels_)

exit()
import networkx as nx
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    plt.show()

G=nx.from_numpy_matrix(A)
draw_graph(G)


exit()
plt.matshow(data);
plt.colorbar()
plt.show()

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    print(labels)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    #frame.axes.get_xaxis().set_visible(False)
    #frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.show()

plot_clusters(data, cluster.KMeans, (), {'n_clusters':2})
