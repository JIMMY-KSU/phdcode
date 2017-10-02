import numpy as np
import scipy.linalg as la


class mrDMD:
    def __init__(self, dynamics_rank=None):
        self.dynamics_rank = dynamics_rank

    def fit(self, Xin, dt, n_levels, max_cycles):
        self.tree = self._fit_recursion(Xin, dt, n_levels, max_cycles)

    def _fit_recursion(self, Xin, dt, n_levels, max_cycles):
        # OUTPUTS:
        # At each level, have
        # Phi is a matrix of the DMD modes
        # omega is a vector of the DMD eigenvalues (continuous time)
        # P is a vector of the magnitudes of the modes of Phi
        # hit is a boolean telling whether this level has any modes
        # rho is the frequency cutoff for this level
        # T is the total time accounted for on this level

        N,n_samples = Xin.shape
        T = n_samples*dt
        rho = max_cycles/float(T)   # frequency cutoff

        if self.dynamics_rank is None:
            self.dynamics_rank = N

        X = Xin[:,:-1]
        Xp = Xin[:,1:]

        U,s,Vt = la.svd(X, full_matrices=False)
        Ur = U[:, :self.dynamics_rank]
        sr = s[:self.dynamics_rank]
        Vr = Vt[:self.dynamics_rank].T
        A_tilde = np.dot(Ur.T, np.dot(Xp,Vr/sr))

        evals, evecs = la.eig(A_tilde)
        Phi = np.dot(np.dot(Xp,Vr/sr), evecs)
        omega = np.log(evals)/dt
        b = la.lstsq(Phi, X[:,0])[0]

        slow_modes = np.where(np.abs(omega/(2.*np.pi)) <= rho)[0]
        fast_modes = [x for x in range(omega.size) if x not in slow_modes]

        t = np.arange(0,T,dt)
    #     Xslow = np.dot(Phi[:,slow_modes]*b[slow_modes], np.exp(np.outer(omega[slow_modes],t)))
    #     Xfast = np.dot(Phi[:,fast_modes]*b[fast_modes], np.exp(np.outer(omega[fast_modes],t)))

        thislevel = {'T': T, 'n_modes': slow_modes.size, 'omega': omega[slow_modes],
                     'Phi': Phi[:,slow_modes], 'b': b[slow_modes], 'rho': rho}

        n_remaining_modes = self.dynamics_rank - thislevel['n_modes']
        if (n_levels > 1) and (n_remaining_modes > 0):
            sep = int(n_samples/2)
            nextlevel1 = self._fit_recursion(Xin[:,:sep], dt, n_levels-1, max_cycles)
            nextlevel2 = self._fit_recursion(Xin[:,sep:], dt, n_levels-1, max_cycles)
        else:
            nextlevel1 = {}
            nextlevel2 = {}

        tree = {'thislevel': thislevel, 'nextlevel1': nextlevel1, 'nextlevel2': nextlevel2}

        return tree

    def reconstruct(self, t):
        return self._reconstruct_recursion(self.tree, t)

    def _reconstruct_recursion(self, tree, t):
        n_samples = t.size
        sep = int(n_samples/2)

        if self.tree['nextlevel1']:
            x1 = self._reconstruct_recursion(self.tree['nextlevel1'], t[:sep]).astype(float)
        else:
            x1 = np.zeros((1,sep))

        if tree['nextlevel2']:
            x2 = self._reconstruct_recursion(tree['nextlevel2'], t[sep:]).astype(float)
        else:
            x2 = np.zeros((1,sep))

        if tree['thislevel']['Phi'].size != 0:
            x = np.dot(tree['thislevel']['Phi']*tree['thislevel']['b'],
                       np.exp(np.outer(tree['thislevel']['omega'],t))).astype(float)
    #         x = np.empty((tree['thislevel']['Phi'].shape[0], n_samples))
    #         for i in range(t.size):
    #             x[:,i] = np.dot(tree['thislevel']['Phi'],
    #                             np.dot(np.eye(K)*np.exp(tree['thislevel']['omega']*t[i]),
    #                                    tree['thislevel']['P']))
        else:
            x = np.zeros((np.maximum(x1.shape[0],x2.shape[0]), n_samples)).astype(float)

        x[:,:sep] += x1
        x[:,sep:] += x2
        return x
