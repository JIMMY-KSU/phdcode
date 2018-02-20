import numpy as np
import scipy.linalg as la
from .utils import hankel_matrix
from .dmd import compute_dmd_rank, compute_real_dmd_modes


class DMDc:
    def __init__(self, truncation='optimal', threshold=None, time_delay=1, time_delay_spacing=1):
        self.truncation = truncation
        if (self.truncation == 'soft') and (threshold is None):
            self.threshold = 1e-10
        else:
            self.threshold = threshold
        self.time_delay = time_delay
        self.time_delay_spacing = time_delay_spacing

    def fit(self, *args, **kwargs):
        self._fit_exact(*args, **kwargs)

    def _fit_exact(self, X_fit, Upsilon_fit, dt, B=None, real=None, t0=0.0, sample_spacing=1):
        self.dt = dt
        if real is None:
            self.real = (np.where(np.iscomplex(X_fit))[0].size < 1)
        else:
            self.real = real

        if B is None:
            if self.time_delay > 1:
                H = hankel_matrix(X_fit, self.time_delay, spacing=self.time_delay_spacing)
                Ups_subsample_1 = Upsilon_fit[:,:H.shape[1]]
                X = H[:,:-1:sample_spacing]
                Omega = np.concatenate((X, Ups_subsample_1[:,:-1:sample_spacing]), axis=0)
                Xp = H[:,1::sample_spacing]
            else:
                X = X_fit[:, :-1:sample_spacing]
                Omega = np.concatenate((X, Upsilon_fit[:,:-1:sample_spacing]), axis=0)
                Xp = X_fit[:, 1::sample_spacing]

            n = X.shape[0]
            l = Upsilon_fit.shape[0]

            U,s,Vt = la.svd(Omega, full_matrices=False)
            p = compute_dmd_rank(s, self.truncation, shape=X.shape, threshold=self.threshold)
            U = U[:,:p]
            s = s[:p]
            V = Vt[:p].conj().T

            U_hat,s_hat,Vt_hat = la.svd(Xp, full_matrices=False)
            r = compute_dmd_rank(s_hat, self.truncation, shape=X.shape, threshold=self.threshold)
            U_hat = U_hat[:,:r]
            s_hat = s_hat[:r]
            V_hat = Vt_hat[:r].conj().T

            tmp = np.dot(Xp, V/s)
            U1U = np.dot(U[:n].T, U_hat)
            A = np.dot(tmp, U[:n].T)
            B = np.dot(tmp, U[n:].T)
            A_tilde = np.dot(np.dot(U_hat.conj().T, A),U_hat)
            B_tilde = np.dot(U_hat.T, B)

            evals, evecs = la.eig(A_tilde)

            # check for a negative real eigenvalue, which is sometimes an issue with time delay DMD
            if np.any(evals[~np.iscomplex(evals)] <= 0):
                raise ValueError('found negative eigenvalue')

            # get modes and normalize them
            Phi = np.dot(np.dot(tmp, U1U), evecs)
            Phi = Phi / np.sqrt(np.sum(Phi**2, axis=0)) / np.sqrt(r)
            omega = np.log(evals)/dt

            b = la.lstsq(Phi, X[:,0])[0]

            sort_order = np.argsort(np.abs(b))[::-1]
            Phi = Phi[:,sort_order]
            omega = omega[sort_order]
            b = b[sort_order]

            # only take the beginning rows of Phi in the case of time delay
            Phi = Phi[:X_fit.shape[0]]

            if not self.real:
                self.Phi = Phi
                self.omega = omega
                self.b = b
            else:
                self.Phi, self.omega, self.b = compute_real_dmd_modes(Phi, omega, b)

            self.A = A
            self.B = B
            self.Atilde = A_tilde
            self.Btilde = B_tilde
            self.P = U_hat
        else:
            if self.time_delay > 1:
                H = hankel_matrix(X_fit, self.time_delay, spacing=self.time_delay_spacing)
                Upsilon_subsample_1 = Upsilon_fit[:,:H.shape[1]]
                X = H[:,:-1:sample_spacing]
                Xp = H[:,1::sample_spacing] - np.dot(B, Upsilon_subsample_1[:,:-1:sample_spacing])
            else:
                X = X_fit[:, :-1:sample_spacing]
                Xp = X_fit[:, 1::sample_spacing] - np.dot(B, Upsilon_fit[:,:-1:sample_spacing])

            U,s,Vt = la.svd(X, full_matrices=False)
            r = np.where(s > self.threshold)[0].size
            U = U[:, :r]
            s = s[:r]
            V = Vt[:r].conj().T

            tmp = np.dot(Xp, V/s)
            A_tilde = np.dot(U.conj().T, tmp)
            evals, evecs = la.eig(A_tilde)

            # check for a negative real eigenvalue, which is sometimes an issue with time delay DMD
            if np.any(evals[~np.iscomplex(evals)] <= 0):
                raise ValueError('found negative eigenvalue')

            # get modes and normalize them
            Phi = np.dot(tmp, evecs)
            Phi = Phi / np.sqrt(np.sum(Phi**2, axis=0)) / np.sqrt(r)
            omega = np.log(evals)/dt

            b = la.lstsq(Phi, X[:,0])[0]

            sort_order = np.argsort(np.abs(b))[::-1]
            Phi = Phi[:,sort_order]
            omega = omega[sort_order]
            b = b[sort_order]

            # only take the beginning rows of Phi in the case of time delay
            Phi = Phi[:X_fit.shape[0]]

            if not self.real:
                self.Phi = Phi
                self.omega = omega
                self.b = b
            else:
                self.Phi, self.omega, self.b = compute_real_dmd_modes(Phi, omega, b)

            self.A = np.dot(tmp, U.conj().T)
            self.Atilde = A_tilde
            self.A_continuous = (self.A - np.eye(self.A.shape[0]))/dt
            self.Atilde_continuous = (self.Atilde - np.eye(self.Atilde.shape[0]))/dt
            self.B = B
            self.Btilde = np.dot(U.conj().T, B)
            self.P = U

    def reconstruct(self, x0, U, t0, T):
        n_timesteps = int((T-t0)/self.dt)+1

        X = np.zeros((x0.size,n_timesteps))
        X[:,0] = x0
        for i,t in enumerate(np.arange(t0,T,self.dt)):
            xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,i]) + np.dot(self.Btilde, U[:,i])
            X[:,i+1] = np.dot(self.P, xtilde)

        return X

    # def reduced_dynamics(self, t, growth_rate_threshold=0):
    #     if self.omega.ndim == 2:
    #         x = np.zeros((self.rank, t.size))
    #         i = 0
    #         while i < self.omega.shape[1]:
    #             if np.abs(self.omega[0,i]) < growth_rate_threshold:
    #                 growth_rate = 0
    #             else:
    #                 growth_rate = self.omega[0,i]
    #
    #             if self.omega[1,i] != 0:
    #                 x[i] = np.exp(growth_rate*t)*(np.real(self.b[i])*np.cos(self.omega[1,i]*t)
    #                                                   - np.imag(self.b[i])*np.sin(self.omega[1,i]*t))
    #                 x[i+1] = np.exp(growth_rate*t)*(np.imag(self.b[i])*np.cos(self.omega[1,i]*t)
    #                                                   + np.real(self.b[i])*np.sin(self.omega[1,i]*t))
    #                 i += 2
    #             else:
    #                 x[i] = np.exp(growth_rate*t)*np.real(self.b[i])
    #                 i += 1
    #         return x
    #     return (np.exp(np.outer(self.omega,t)).conj().T*self.b).conj().T
    #
    # def reconstruct(self, t, growth_rate_threshold=0):
    #     return np.dot(self.Phi, self.reduced_dynamics(t, growth_rate_threshold=growth_rate_threshold))


# class DMDc:
#     def __init__(self, threshold=1e-10, known_control=False):
#         self.known_control = known_control
#         self.threshold = threshold
#
#     def fit(self, X_in, Ups_in, dt, B=None):
#         # if self.input_rank is None:
#         #     self.input_rank = X_in.shape[0] + Ups_in.shape[0]
#         # if self.output_rank is None:
#         #     self.output_rank = X_in.shape[0]
#
#         self.dt = dt
#
#         X = X_in[:, :-1]
#         Ups = Ups_in[:, :-1]
#         n = X.shape[0]
#         l = Ups.shape[0]
#
#         if not self.known_control:
#             Xp = X_in[:, 1:]
#             Omega = np.concatenate((X,Ups),axis=0)
#
#             U,s,Vt = la.svd(Omega, full_matrices=False)
#             p = np.where(s > self.threshold)[0].size
#             U = U[:,:p]
#             s = s[:p]
#             V = Vt[:p].T
#
#             U_hat,s_hat,Vt_hat = la.svd(Xp, full_matrices=False)
#             r = np.where(s_hat > self.threshold)[0].size
#             U_hat = U_hat[:,:r]
#             s_hat = s_hat[:r]
#             V_hat = Vt_hat[:r].T
#
#             tmp = np.dot(Xp, V/s)
#             #tmp2 = np.dot(U_hat.T, tmp)
#             U1U = np.dot(U[:n].T, U_hat)
#             A = np.dot(tmp, U[:n].T)
#             B = np.dot(tmp, U[n:].T)
#             # A_tilde = np.dot(tmp2, U1U)
#             # B_tilde = np.dot(tmp2, U[n:].T)
#             A_tilde = np.dot(np.dot(U_hat.T, A),U_hat)
#             B_tilde = np.dot(U_hat.T, B)
#
#             evals, evecs = la.eig(A_tilde)
#             Phi = np.dot(np.dot(tmp, U1U), evecs)
#
#             omega = np.log(evals)/dt
#             b = la.lstsq(Phi, X[:,0])[0]
#
#             self.A = A
#             self.B = B
#             self.Atilde = A_tilde
#             self.Btilde = B_tilde
#             self.P = U_hat
#         else:
#             if B is None:
#                 raise AttributeError('control matrix B must be provided')
#
#             Xp = X_in[:, 1:] - np.dot(B, Ups)
#             U,s,Vt = la.svd(X, full_matrices=False)
#             r = np.where(s > self.threshold)[0].size
#             Ur = U[:, :r]
#             sr = s[:r]
#             Vr = Vt[:r].T
#
#             tmp = np.dot(Xp,Vr/sr)
#             A_tilde = np.dot(Ur.T, tmp)
#
#             evals, evecs = la.eig(A_tilde)
#             Phi = np.dot(tmp, evecs)
#
#             omega = np.log(evals)/dt
#             b = la.lstsq(Phi, X[:,0])[0]
#
#             self.A = np.dot(tmp, Ur.T)
#             self.B = B
#             self.Atilde = A_tilde
#             self.Btilde = np.dot(Ur.T, B)
#             self.P = Ur
#
#         self.Phi = Phi
#         self.omega = omega
#         self.b = b
#
#     def reconstruct(self, x0, U, t0, T):
#         n_timesteps = int((T-t0)/self.dt)+1
#
#         X = np.zeros((x0.size,n_timesteps))
#         X[:,0] = x0
#         for i,t in enumerate(np.arange(t0,T,self.dt)):
#             xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,i]) + np.dot(self.Btilde, U[:,i])
#             X[:,i+1] = np.dot(self.P, xtilde)
#
#         return X
#
#     def project(self, Xin, Uin, T):
#         n_steps = int(T/self.dt)+1
#         n_samples = Xin.shape[1]
#
#         X = np.zeros((Xin.shape[0],n_samples+n_steps))
#         Xtilde = np.dot(np.dot(self.Atilde, self.P.T), Xin[:,:-1]) + np.dot(self.Btilde, Uin[:,:n_samples-1])
#         X[:,:n_samples-1] = np.dot(self.P, Xtilde)
#         for i in range(n_steps+1):
#             xtilde = np.dot(np.dot(self.Atilde, self.P.T), X[:,n_samples-2+i]) + np.dot(self.Btilde, Uin[:,n_samples-2+i])
#             X[:,n_samples-1+i] = np.dot(self.P, xtilde)
#
#         return X
