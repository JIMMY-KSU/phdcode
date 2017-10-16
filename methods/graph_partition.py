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


def spectral_bisection(A, cluster_size=None):
    L = laplacian_matrix(A)
    evals, evecs = la.eig(L)

    idxs = np.argsort(evals)
    lambda2 = evals[idxs[1]]
    v2 = evecs[idxs[1]]

    if cluster_size is None:
        s = np.ones(v2.size)
        s[np.where(v2 < 0)] = -1

        return s
    else:
        partition_idxs = np.argsort(v2)

        # first partition
        s1 = np.ones(v2.size)
        s1[partition_idxs[0:cluster_size]] = -1
        R1 = cut_size(s1, A)

        # second partition
        s2 = np.ones(v2.size)
        s2[partition_idxs[-cluster_size:]] = -1
        R2 = cut_size(s2, A)

        if R1 < R2:
            return s1
        else:
            return s2


def cut_size(s, A):
    L = laplacian_matrix(A)
    return 0.25*np.dot(s.T, np.dot(L, s))
