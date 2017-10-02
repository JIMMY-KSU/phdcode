import numpy as np
import scipy.linalg as la
from scipy.integrate import ode


class DMDc:
    def __init__(self, threshold=1e-10, known_control=False):
        self.known_control = known_control
        self.threshold = threshold

    def fit(self, X_in, Ups_in, dt, B=None):
        # if self.input_rank is None:
        #     self.input_rank = X_in.shape[0] + Ups_in.shape[0]
        # if self.output_rank is None:
        #     self.output_rank = X_in.shape[0]

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
            tmp2 = np.dot(U_hat.T, tmp)
            U1U = np.dot(U[:n].T, U_hat.T)
            A_tilde = np.dot(tmp2, U1U)
            B_tilde = np.dot(tmp, U[n:].T)

            evals, evecs = la.eig(A_tilde)
            Phi = np.dot(np.dot(tmp, U1U), evecs)

            omega = np.log(evals)/dt
            b = la.lstsq(Phi, X[:,0])[0]

            self.Btilde = B_tilde
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

            self.Btilde = B

        self.Phi = Phi
        self.omega = omega
        self.b = b
        self.Atilde = A_tilde

    def reconstruct(self, x0, U, t):
        K = self.b.size
        return np.real(np.dot(self.Phi*self.b, np.exp(np.outer(self.omega,t))))

    # def reconstruct(self, x0, U, t0, dt, T):
        # f = lambda t,x: np.dot(self.Atilde, x) + np.dot(self.Btilde, U[:,int((t-t0)/dt)])
        #
        # r = ode(f).set_integrator('zvode', method='bdf')
        # r.set_initial_value(x0, t0)
        #
        # x = [x0]
        # t = [t0]
        # while r.successful() and r.t < T:
        #     r.integrate(r.t + dt)
        #     x.append(np.real(r.y))
        #     t.append(r.t)
        #
        # return np.array(x).T, np.array(t)
