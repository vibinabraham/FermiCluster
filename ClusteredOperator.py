import numpy as np
import scipy
import itertools as it


class ClusteredOperator:
    """
    Defines a fermionic operator which can act on multiple clusters

    data:
    self.ints = dict of operators: transition list -> operator list -> integral tensor
                e.g., self.ints[["","","Aab","",B]] = ndarray(p,q,r,s)
                with p,q,r on cluster 2, and s on cluster 4
    """
    def __init__(self):
        self.n_clusters = 0
        self.clusters = []
        self.ints = {}
