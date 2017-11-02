import numpy as np
import scipy.linalg as la
from .utils import hankel_matrix
from ..utils.optimal_svht_coef import optimal_svht_coef


class DMD:
    def __init__(self, threshold=None, time_delay=0):
        self.threshold = threshold
        self.time_delay = time_delay

    def fit(self, Xin, dt, real=None):
        self.dt = dt
        if real is None:
            self.real = (np.where(np.iscomplex(Xin))[0].size < 1)
        else:
            self.real = real

        if self.time_delay > 0:
            H = hankel_matrix(Xin, self.time_delay)
            X = H[:,:-1]
            Xp = H[:,1:]
        else:
            X = Xin[:,:-1]
            Xp = Xin[:,1:]

        U,s,Vt = la.svd(X, full_matrices=False)
        if self.threshold is None:
            beta = X.shape[0]/X.shape[1]
            if beta > 1:
                beta = 1/beta
            omega = optimal_svht_coef(beta,False) * np.median(s)
            r = np.sum(s > omega)
            if r < 1:
                r = 1
        else:
            r = np.where(s > self.threshold)[0].size
        self.rank = r
        U = U[:,:r]
        s = s[:r]
        V = Vt[:r].conj().T

        tmp = np.dot(Xp, V/s)
        A_tilde = np.dot(U.conj().T, tmp)
        evals, evecs = la.eig(A_tilde)

        if not self.real:
            self.Phi = np.dot(tmp, evecs)
            self.omega = np.log(evals)/dt
            self.b = la.lstsq(self.Phi, X[:,0])[0]
        else:
            Phi = np.dot(tmp, evecs)
            omega = np.log(evals)/dt
            b = la.lstsq(Phi, X[:,0])[0]

            Phi_real = np.zeros(Phi.shape)
            omega_realpart = []
            omega_imagpart = []

            omega_tmp = []
            b_tmp = []

            omega_idx = np.arange(omega.size)
            omega_copy = omega.copy()
            i = 0
            while i < omega.size:
                if np.iscomplex(omega_copy[0]):
                    Phi_real[:,i] = 2*np.real(Phi[:,omega_idx[0]])
                    omega_realpart.append(np.real(omega_copy[0]))
                    omega_realpart.append(np.real(omega_copy[0]))
                    b_tmp.append(b[omega_idx[0]])

                    Phi_real[:,i+1] = -2*np.imag(Phi[:,omega_idx[0]])
                    omega_imagpart.append(np.imag(omega_copy[0]))
                    omega_imagpart.append(-np.imag(omega_copy[0]))
                    b_tmp.append(b[omega_idx[0]].conj())

                    # find complex conjugate eval
                    conj_idx = np.argsort(np.abs(np.conj(omega_copy[0]) - omega_copy))[0]

                    # mask out this eigenvalue and its conjugate
                    mask = np.ones(omega_idx.size, dtype=bool)
                    mask[[0,conj_idx]] = False
                    omega_idx = omega_idx[mask]
                    omega_copy = omega_copy[mask]
                    i += 2
                else:
                    omega_realpart.append(np.real(omega_copy[0]))
                    omega_imagpart.append(0.0)
                    omega_tmp.append(omega_copy[0])
                    b_tmp.append(np.real(b[omega_idx[0]]))
                    Phi_real[:,i] = np.real(Phi[:,omega_idx[0]])
                    omega_idx = omega_idx[1:]
                    omega_copy = omega_copy[1:]
                    i += 1

            self.Phi = Phi_real
            self.omega = np.vstack((np.array(omega_realpart), np.array(omega_imagpart)))
            self.b = np.array(b_tmp)

        self.A = np.dot(tmp, U.conj().T)
        self.Atilde = A_tilde
        self.A_continuous = (self.A - np.eye(self.A.shape[0]))/dt
        self.Atilde_continuous = (self.Atilde - np.eye(self.Atilde.shape[0]))/dt
        self.P = U

    def reduced_dynamics(self, t):
        if self.omega.ndim == 2:
            x = np.zeros((self.rank, t.size))
            # for i in range(self.omega.shape[1]):
            #     if self.omega[1,i] > 0:
            #         x[i] = (np.cos(self.omega[1,i]*t)*np.exp(self.omega[0,i]*t))*self.b[i]
            #     else:
            #         x[i] = (np.sin(self.omega[1,i]*t)*np.exp(self.omega[0,i]*t))*self.b[i]
            x = np.zeros((self.rank, t.size))
            i = 0
            while i < self.omega.shape[1]:
                if self.omega[1,i] != 0:
                    x[i] = np.exp(self.omega[0,i]*t)*(np.real(self.b[i])*np.cos(self.omega[1,i]*t)
                                                      - np.imag(self.b[i])*np.sin(self.omega[1,i]*t))
                    x[i+1] = np.exp(self.omega[0,i]*t)*(np.imag(self.b[i])*np.cos(self.omega[1,i]*t)
                                                      + np.real(self.b[i])*np.sin(self.omega[1,i]*t))
                    i += 2
                else:
                    x[i] = np.exp(self.omega[0,i]*t)*np.real(self.b[i])
                    i += 1
            return x
        return (np.exp(np.outer(self.omega,t)).conj().T*self.b).conj().T

    def reconstruct(self, t):
        return np.dot(self.Phi, self.reduced_dynamics(t))

    def project(self, Xin, T, reduced=False):
        n_steps = int(T/self.dt)+1
        n_samples = Xin.shape[1]

        if self.time_delay == 0:
            X = np.zeros((Xin.shape[0],n_samples+n_steps), dtype=np.complex)
            Xtilde = np.zeros((self.rank, n_samples+n_steps), dtype=np.complex)
            Xtilde[:,:n_samples-1] = np.dot(np.dot(self.Atilde, self.P.conj().T), Xin[:,:-1])
            # X[:,:n_samples-1] = np.dot(self.P, Xtilde)
            X[:,:n_samples-1] = np.dot(self.P, Xtilde[:,:n_samples-1])
            for i in range(n_steps+1):
                # xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,n_samples-2+i])
                Xtilde[:,n_samples-1+i] = np.dot(np.dot(self.Atilde, self.P.conj().T), X[:,n_samples-2+i])
                X[:,n_samples-1+i] = np.dot(self.P, Xtilde[:,n_samples-1+i])

            if reduced:
                return Xtilde
            return X

        H = hankel_matrix(Xin[:,:-1],self.time_delay)

        X = np.zeros((Xin.shape[0]*(self.time_delay+1), n_samples + n_steps), dtype=np.complex)
        Xtilde = np.dot(np.dot(self.Atilde, self.P.conj().T), H)
        X[:,:n_samples-self.time_delay-2] = np.dot(self.P, Xtilde)
        for i in range(n_steps + self.time_delay + 2):
            idx = n_samples - self.time_delay - 2 + i
            xtilde = np.dot(np.dot(self.Atilde, self.P.conj().T), X[:,idx-1])
            X[:,idx] = np.dot(self.P, xtilde)

        return X
