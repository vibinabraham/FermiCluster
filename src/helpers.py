import numpy as np
import copy as cp
import scipy

def print_cmat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print("%8.5f +%8.5f  " %(mat[i,j].real, mat[i,j].imag),end='')
        print()


def print_mat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print("%16.12f  " %(mat[i,j]),end='')
        print()

def print_row(row):
    [print("%8.5f  " %(i),end='') for i in row]
    print()

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


