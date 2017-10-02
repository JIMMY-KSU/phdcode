import numpy as np
from scipy.integrate import ode
from scipy.special import binom


def pool_data(Xin, poly_order=2, use_sine=False):
    if (poly_order > 5):
        poly_order = 5

    if Xin.ndim == 1:
        T = 1
        n = Xin.size
    else:
        n,T = Xin.shape
    n_vars = 0
    for k in range(poly_order+1):
        n_vars += int(binom(n+k-1,k))
    if use_sine:
        n_vars += 2*n
    Xout = np.zeros((n_vars,T))

    Xout[0] = np.ones(T)
    index = 1

    for i in range(n):
        Xout[index] = Xin[i]
        index += 1

    if (poly_order >= 2):
        for i in range(n):
            for j in range(i,n):
                Xout[index] = Xin[i]*Xin[j]
                index += 1

    if (poly_order >= 3):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    Xout[index] = Xin[i]*Xin[j]*Xin[k]
                    index += 1

    if (poly_order >= 4):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]
                        index += 1

    if (poly_order >= 5):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        for m in range(l,n):
                            Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]*Xin[m]
                            index += 1

    if use_sine:
        for i in range(n):
            Xout[index] = np.sin(Xin[i])
            index += 1
        for i in range(n):
            Xout[index] = np.cos(Xin[i])
            index += 1

    return Xout


class SINDy:
    def __init__(self, poly_order=2, use_sine=False):
        self.poly_order = poly_order
        self.use_sine = use_sine

    def fit(self, X, Xprime, threshold):
        Theta = pool_data(X, self.poly_order, self.use_sine)

        n,T = Xprime.shape
        Xi = np.linalg.lstsq(Theta.T,Xprime.T)[0]

        for k in range(10):
            small_inds = (np.abs(Xi) < threshold)
            Xi[small_inds] = 0
            for i in range(n):
                big_inds = ~small_inds[:,i]
                Xi[big_inds,i] = np.linalg.lstsq(Theta[big_inds].T, Xprime[i])[0]

        self.Xi = Xi

    def reconstruct(self, x0, t0, dt, T):
        f = lambda t,x: np.dot(self.Xi.T, pool_data(np.real(x), poly_order=self.poly_order, use_sine=self.use_sine))

        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(x0, t0)

        x = [x0]
        t = [t0]
        while r.successful() and r.t < T:
            r.integrate(r.t + dt)
            x.append(np.real(r.y))
            t.append(r.t)

        return np.array(x).T, np.array(t)
