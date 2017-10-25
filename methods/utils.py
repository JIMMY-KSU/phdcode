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


def jordan(A, real=False, threshold=1e-8, return_transition_matrix=False):
    if real:
        J = np.zeros(A.shape)
        if return_transition_matrix:
            P = np.zeros(A.shape)
    else:
        J = np.zeros(A.shape, dtype=np.complex)
        if return_transition_matrix:
            P = np.zeros(A.shape, dtype=np.complex)
    evals, evecs = la.eig(A)

    evals_idx = np.arange(evals.size)
    evals_copy = evals.copy()
    i = 0
    while i < J.shape[0]:
        comparison = np.abs(evals_copy[0] - evals_copy) < threshold
        if np.sum(comparison) > 1:
            if real and np.abs(np.imag(evals_copy[0])) > threshold:
                raise NotImplementedError('not implemented for duplicate complex eigenvalues')
            algebraic_multiplicity = np.sum(comparison)
            geometric_multiplicity = np.linalg.matrix_rank(evecs[evals_idx[comparison]])
            J[i:i+algebraic_multiplicity,i:i+algebraic_multiplicity] = np.eye(algebraic_multiplicity)*evals_copy[comparison]
            if geometric_multiplicity < algebraic_multiplicity:
                J[i:i+algebraic_multiplicity,i:i+algebraic_multiplicity] += np.eye(algebraic_multiplicity,1)

            if return_transition_matrix:
                raise NotImplementedError('calculating transition matrix not implemented for repeat eigenvalues')
                # M = A - evals_copy[0]*np.eye(A.shape[0])
                # u,s,v = la.svd(M, full_matrices=False)
                # P[:,i:i+geometric_multiplicity] = v[-geometric_multiplicity:].conj()
                # for j in range(algebraic_multiplicity - geometric_multiplicity):
                #     P[:,i+geometric_multiplicity+j] = la.solve(M, P[:,i+geometric_multiplicity+j-1])
                # P[:,i:i+algebraic_multiplicity] = evecs[:,comparison]

            evals_idx = evals_idx[~comparison]
            evals_copy = evals_copy[~comparison]
            i += algebraic_multiplicity
        elif real and np.abs(np.imag(evals_copy[0])) > threshold:
            conj_idx = np.argsort(np.abs(np.conj(evals_copy[0]) - evals_copy))[0]
            J[i:i+2,i:i+2] = np.eye(2)*np.real(evals_copy[0])
            J[i,i+1] = np.abs(np.imag(evals_copy[0]))
            J[i+1,i] = -np.abs(np.imag(evals_copy[0]))

            mask = np.ones(evals_idx.size, dtype=bool)
            mask[[0,conj_idx]] = False

            if return_transition_matrix:
                P[:,i] = np.real(evecs[:,evals_idx[0]])
                P[:,i+1] = np.imag(evecs[:,evals_idx[0]])

            evals_idx = evals_idx[mask]
            evals_copy = evals_copy[mask]
            i += 2
        else:
            if real:
                J[i,i] = np.real(evals_copy[0])
            else:
                J[i,i] = evals_copy[0]

            if return_transition_matrix:
                if real:
                    P[:,i] = np.real(evecs[:,evals_idx[0]])
                else:
                    P[:,i] = evecs[:,evals_idx[0]]

            evals_idx = evals_idx[1:]
            evals_copy = evals_copy[1:]
            i += 1

    if return_transition_matrix:
        return J,P
    else:
        return J
