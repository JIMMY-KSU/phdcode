import numpy as np
import scipy.linalg as la


def dynamics_to_adjacency_matrix(A_in, threshold=1e-3):
    n = A_in.shape[0]

    A = np.zeros(A_in.shape)
    A[np.where(np.abs(A_in) > threshold)] = 1.

    for i in range(n):
        for j in range(n):
            if A[j,i] == 1.:
                A[i,j] = 1.

    return A


def laplacian_matrix(A):
    D = np.diag(np.sum(A,axis=0))
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


def modularity(A, c, undirected=False):
    n = A.shape[0]

    k_in = np.sum(A, axis=0)
    k_out = np.sum(A, axis=1)
    m = np.sum(A)

    Q = 0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                if undirected:
                    Q += (A[i,j] - k_in[i]*k_out[j]/(2.*m))
                else:
                    Q += (A[i,j] - k_in[i]*k_out[j]/m)

    if undirected:
        Q /= 2.*m
    else:
        Q /= m

    return Q


def modularity_signed(A, c, undirected=False):
    n = A.shape[0]

    A_plus = np.maximum(A, 0.)
    A_minus = np.maximum(-A, 0.)

    k_in_plus = np.sum(A_plus, axis=0)
    k_out_plus = np.sum(A_plus, axis=1)
    m_plus = np.sum(A_plus)
    k_in_minus = np.sum(A_minus, axis=0)
    k_out_minus = np.sum(A_minus, axis=1)
    m_minus = np.sum(A_minus)

    Q_plus = 0
    Q_minus = 0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                if undirected:
                    Q_plus += (A_plus[i,j] - k_in_plus[i]*k_out_plus[j]/(2.*m_plus))/(2.*m_plus)
                    Q_minus += (A_minus[i,j] - k_in_minus[i]*k_out_minus[j]/(2.*m_minus))/(2.*m_minus)
                else:
                    Q_plus += (A_plus[i,j] - k_in_plus[i]*k_out_plus[j]/m_plus)/m_plus
                    Q_minus += (A_minus[i,j] - k_in_minus[i]*k_out_minus[j]/m_minus)/m_minus

    Q = (m_plus*Q_plus - m_minus*Q_minus)/(m_plus + m_minus)

    return Q