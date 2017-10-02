import numpy as np
import scipy.linalg as la
from scipy.integrate import ode


class DMDc:
    def __init__(self, input_rank=None, output_rank=None, known_control=False):
        self.known_control = known_control
        self.input_rank = input_rank
        self.output_rank = output_rank

    def fit(self, Xin, Uin, dt, B=None):
        if self.input_rank is None:
            self.input_rank = Xin.shape[0] + Uin.shape[0]
        if self.output_rank is None:
            self.output_rank = Xin.shape[0]

        X = Xin[:,:-1]
        U = Uin[:,:-1]
        n = X.shape[0]
        l = U.shape[0]

        if not self.known_control:
            Xp = Xin[:,1:]
            Omega = np.concatenate((X,U),axis=0)

            U,s,Vt = la.svd(Omega, full_matrices=False)
            U = U[:, :self.input_rank]
            s = s[:self.input_rank]
            V = Vt[:self.input_rank].T

            U_hat,s_hat,Vt_hat = la.svd(Xp, full_matrices=False)
            U_hat = U_hat[:, :self.output_rank]
            s_hat = s_hat[:self.output_rank]
            V_hat = Vt_hat[:self.output_rank].T

            tmp = np.dot(Xp, V/s)
            tmp2 = np.dot(U_hat.T, tmp)
            U1U = np.dot(U[:n].T, U_hat)
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

            Xp = Xin[:,1:] - np.dot(B,U)
            U,s,Vt = la.svd(X, full_matrices=False)
            Ur = U[:, :self.output_rank]
            sr = s[:self.output_rank]
            Vr = Vt[:self.output_rank].T

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

    def reconstruct(self, x0, U, t0, dt, T):
        f = lambda t,x: np.dot(self.Atilde, x) + np.dot(self.Btilde, U[:,int((t-t0)/dt)])

        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(x0, t0)

        x = [x0]
        t = [t0]
        while r.successful() and r.t < T:
            r.integrate(r.t + dt)
            x.append(np.real(r.y))
            t.append(r.t)

        return np.array(x).T, np.array(t)
