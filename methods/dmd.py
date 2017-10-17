import numpy as np
import scipy.linalg as la
from .utils import hankel_matrix


class DMD:
    def __init__(self, threshold=1e-10, time_delay=0):
        self.threshold = threshold
        self.time_delay = time_delay

    def fit(self, Xin, dt):
        self.dt = dt

        if self.time_delay > 0:
            H = hankel_matrix(Xin, self.time_delay)
            X = H[:,:-1]
            Xp = H[:,1:]
        else:
            X = Xin[:,:-1]
            Xp = Xin[:,1:]

        U,s,Vt = la.svd(X, full_matrices=False)
        r = np.where(s > self.threshold)[0].size
        self.rank = r
        U = U[:,:r]
        s = s[:r]
        V = Vt[:r].T

        tmp = np.dot(Xp, V/s)
        A_tilde = np.dot(U.T, tmp)
        evals, evecs = la.eig(A_tilde)

        self.Phi = np.dot(tmp, evecs)
        self.omega = np.log(evals)/dt
        self.b = la.lstsq(self.Phi, X[:,0])[0]

        self.A = np.dot(tmp, U.T)
        self.Atilde = A_tilde
        self.P = U

    def reconstruct(self, t):
        K = self.b.size
        return np.real(np.dot(self.Phi*self.b, np.exp(np.outer(self.omega,t))))

    def project(self, Xin, T, reduced=False):
        n_steps = int(T/self.dt)+1
        n_samples = Xin.shape[1]

        if self.time_delay == 0:
            X = np.zeros((Xin.shape[0],n_samples+n_steps))
            Xtilde = np.zeros((self.rank, n_samples+n_steps))
            Xtilde[:,:n_samples-1] = np.dot(np.dot(self.Atilde, self.P.T), Xin[:,:-1])
            # X[:,:n_samples-1] = np.dot(self.P, Xtilde)
            X[:,:n_samples-1] = np.dot(self.P, Xtilde[:,:n_samples-1])
            for i in range(n_steps+1):
                # xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,n_samples-2+i])
                Xtilde[:,n_samples-1+i] = np.dot(np.dot(self.Atilde, self.P.T), X[:,n_samples-2+i])
                X[:,n_samples-1+i] = np.dot(self.P, Xtilde[:,n_samples-1+i])

            if reduced:
                return Xtilde
            return X

        H = hankel_matrix(Xin[:,:-1],self.time_delay)

        X = np.zeros((Xin.shape[0]*(self.time_delay+1), n_samples + n_steps))
        Xtilde = np.dot(np.dot(self.Atilde, self.P.T), H)
        X[:,:n_samples-self.time_delay-2] = np.dot(self.P, Xtilde)
        for i in range(n_steps + self.time_delay + 2):
            idx = n_samples - self.time_delay - 2 + i
            xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,idx-1])
            X[:,idx] = np.dot(self.P, xtilde)

        return X
