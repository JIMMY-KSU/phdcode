import numpy as np
import scipy.linalg as la


def dynamics_to_adjacency_matrix(A_in, threshold=1e-3):
    n = A_in.shape[0]

    A = np.zeros(A_in.shape)
    A[np.where(np.abs(A_in) > threshold)] = 1.
    # A[1:n,1:n] *= 2.

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i,j] *= 2.
            elif A[j,i] == 1.:
                A[i,j] = 1.

    return A


def laplacian_matrix(A):
    D = np.eye(A.shape[0])*np.sum(A,axis=0)
    return D - A
