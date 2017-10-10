import numpy as np
import scipy.linalg as la


class DMDc:
    def __init__(self, threshold=1e-10, known_control=False):
        self.known_control = known_control
        self.threshold = threshold

    def fit(self, X_in, Ups_in, dt, B=None):
        # if self.input_rank is None:
        #     self.input_rank = X_in.shape[0] + Ups_in.shape[0]
        # if self.output_rank is None:
        #     self.output_rank = X_in.shape[0]

        self.dt = dt

        X = X_in[:, :-1]
        Ups = Ups_in[:, :-1]
        n = X.shape[0]
        l = Ups.shape[0]

        if not self.known_control:
            Xp = X_in[:, 1:]
            Omega = np.concatenate((X,Ups),axis=0)

            U,s,Vt = la.svd(Omega, full_matrices=False)
            p = np.where(s > self.threshold)[0].size
            U = U[:,:p]
            s = s[:p]
            V = Vt[:p].T

            U_hat,s_hat,Vt_hat = la.svd(Xp, full_matrices=False)
            r = np.where(s_hat > self.threshold)[0].size
            U_hat = U_hat[:,:r]
            s_hat = s_hat[:r]
            V_hat = Vt_hat[:r].T

            tmp = np.dot(Xp, V/s)
            #tmp2 = np.dot(U_hat.T, tmp)
            U1U = np.dot(U[:n].T, U_hat)
            A = np.dot(tmp, U[:n].T)
            B = np.dot(tmp, U[n:].T)
            # A_tilde = np.dot(tmp2, U1U)
            # B_tilde = np.dot(tmp2, U[n:].T)
            A_tilde = np.dot(np.dot(U_hat.T, A),U_hat)
            B_tilde = np.dot(U_hat.T, B)

            evals, evecs = la.eig(A_tilde)
            Phi = np.dot(np.dot(tmp, U1U), evecs)

            omega = np.log(evals)/dt
            b = la.lstsq(Phi, X[:,0])[0]

            self.A = A
            self.B = B
            self.Atilde = A_tilde
            self.Btilde = B_tilde
            self.P = U_hat
        else:
            if B is None:
                raise AttributeError('control matrix B must be provided')

            Xp = X_in[:, 1:] - np.dot(B, Ups)
            U,s,Vt = la.svd(X, full_matrices=False)
            r = np.where(s > self.threshold)[0].size
            Ur = U[:, :r]
            sr = s[:r]
            Vr = Vt[:r].T

            tmp = np.dot(Xp,Vr/sr)
            A_tilde = np.dot(Ur.T, tmp)

            evals, evecs = la.eig(A_tilde)
            Phi = np.dot(tmp, evecs)

            omega = np.log(evals)/dt
            b = la.lstsq(Phi, X[:,0])[0]

            self.A = np.dot(tmp, Ur.T)
            self.B = B
            self.Atilde = A_tilde
            self.Btilde = np.dot(Ur.T, B)
            self.P = Ur

        self.Phi = Phi
        self.omega = omega
        self.b = b

    def reconstruct(self, x0, U, t0, T):
        n_timesteps = int((T-t0)/self.dt)+1

        X = np.zeros((x0.size,n_timesteps))
        X[:,0] = x0
        for i,t in enumerate(np.arange(t0,T,self.dt)):
            xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,i]) + np.dot(self.Btilde, U[:,i])
            X[:,i+1] = np.dot(self.P, xtilde)

        return X

    def project(self, Xin, Uin, T):
        n_steps = int(T/self.dt)+1
        n_samples = Xin.shape[1]

        X = np.zeros((Xin.shape[0],n_samples+n_steps))
        Xtilde = np.dot(np.dot(self.Atilde, self.P.T), Xin[:,:-1]) + np.dot(self.Btilde, Uin[:,:n_samples-1])
        X[:,:n_samples-1] = np.dot(self.P, Xtilde)
        for i in range(n_steps+1):
            xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,n_samples-2+i]) + np.dot(self.Btilde, Uin[:,n_samples-2+i])
            X[:,n_samples-1+i] = np.dot(self.P, xtilde)

        return X
