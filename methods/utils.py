import numpy as np
import scipy.linalg as la


def unit_circle(spacing=.01):
    return np.vstack((np.cos(np.arange(0,2*np.pi,spacing)), np.sin(np.arange(0,2*np.pi,spacing))))


def hankel_matrix(Xin, delay):
    n_inputs, n_samples = Xin.shape

    X = np.zeros((n_inputs*(delay+1), n_samples - delay - 1))
    for i in range(delay+1):
        X[i + (delay+1)*np.arange(n_inputs)] = Xin[:,i:i+n_samples-delay-1]
    return X


def jordan(A, real=False, threshold=1e-5):
    J = np.zeros(A.shape, dtype=np.complex)
    evals, evecs = la.eig(A)

    evals_idx = np.arange(evals.size)
    evals_copy = evals.copy()
    i = 0
    while i < J.shape[0]:
        comparison = np.abs(evals_copy[0] - evals_copy) < threshold
        if np.sum(comparison) > 1:
            algebraic_multiplicity = np.sum(comparison)
            geometric_multiplicity = np.linalg.matrix_rank(evecs[evals_idx[comparison]])
            J[i:i+algebraic_multiplicity,i:i+algebraic_multiplicity] = np.eye(algebraic_multiplicity)*evals_copy[comparison]
            if geometric_multiplicity < algebraic_multiplicity:
                J[i:i+algebraic_multiplicity,i:i+algebraic_multiplicity] += np.eye(algebraic_multiplicity,1)
            evals_idx = evals_idx[~comparison]
            evals_copy = evals_copy[~comparison]
            i += algebraic_multiplicity
        else:
            J[i,i] = evals_copy[0]
            evals_idx = evals_idx[1:]
            evals_copy = evals_copy[1:]
            i += 1

    return J
