import numpy as np
import scipy.linalg as la


class DMD:
    def __init__(self, dynamics_rank=None):
        self.dynamics_rank = dynamics_rank

    def fit(self, Xin, dt):
        if self.dynamics_rank is None:
            self.dynamics_rank = Xin.shape[0]

        X = Xin[:,:-1]
        Xp = Xin[:,1:]

        U,s,Vt = la.svd(X, full_matrices=False)
        U = U[:,:self.dynamics_rank]
        s = s[:self.dynamics_rank]
        V = Vt[:self.dynamics_rank].T

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
