import numpy as np
import scipy.linalg as la


class DMD:
    def __init__(self, threshold=1e-10):
        self.threshold = threshold

    def fit(self, Xin, dt):

        X = Xin[:,:-1]
        Xp = Xin[:,1:]

        U,s,Vt = la.svd(X, full_matrices=False)
        r = np.where(s > self.threshold)[0].size
        U = U[:,:r]
        s = s[:r]
        V = Vt[:r].T

        tmp = np.dot(Xp, V/s)
        A_tilde = np.dot(U.T, tmp)
        evals, evecs = la.eig(A_tilde)

        self.Phi = np.dot(tmp, evecs)
        self.omega = np.log(evals)/dt
        self.b = la.lstsq(self.Phi, X[:,0])[0]

        self.Atilde = A_tilde

    def reconstruct(self, t):
        K = self.b.size
        return np.real(np.dot(self.Phi*self.b, np.exp(np.outer(self.omega,t))))
