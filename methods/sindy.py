import numpy as np
from scipy.integrate import ode
from scipy.special import binom


def pool_data(Xin, poly_order=2, use_sine=False, varname='x'):
    if poly_order > 5:
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

    labels = []

    Xout[0] = np.ones(T)
    index = 1
    labels.append('1')

    for i in range(n):
        Xout[index] = Xin[i]
        index += 1
        labels.append('%s%d' % (varname,i+1))

    if (poly_order >= 2):
        for i in range(n):
            for j in range(i,n):
                Xout[index] = Xin[i]*Xin[j]
                index += 1
                labels.append('%s%d*%s%d' % (varname,i+1,varname,j+1))

    if (poly_order >= 3):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    Xout[index] = Xin[i]*Xin[j]*Xin[k]
                    index += 1
                    labels.append('%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1))

    if (poly_order >= 4):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]
                        index += 1
                        labels.append('%s%d*%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1,varname,l+1))

    if (poly_order >= 5):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        for m in range(l,n):
                            Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]*Xin[m]
                            index += 1
                            labels.append('%s%d*%s%d*%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1,varname,l+1,varname,m+1))

    if use_sine:
        for i in range(n):
            Xout[index] = np.sin(Xin[i])
            index += 1
            labels.append('sin(%s%d)' % (varname,i+1))
        for i in range(n):
            Xout[index] = np.cos(Xin[i])
            index += 1
            labels.append('cos(%s%d)' % (varname,i+1))

    return Xout,labels


class SINDy:
    def __init__(self, use_sine=False):
        self.use_sine = use_sine

    def fit(self, Xin, poly_order, dt=None, Xprime=None, coefficient_threshold=.01):
        if Xprime is None:
            if dt is None:
                raise ValueError('must provide at least one of derivative or time step')
            Xprime = (Xin[:,2:]-Xin[:,:-2])/(2*dt)
            X = Xin[:,1:-1]
        else:
            X = Xin

        Theta,labels = pool_data(X, poly_order, self.use_sine)

        self.labels = labels

        n,T = Xprime.shape
        Xi = np.linalg.lstsq(Theta.T,Xprime.T)[0]

        for k in range(10):
            small_inds = (np.abs(Xi) < coefficient_threshold)
            Xi[small_inds] = 0
            for i in range(n):
                big_inds = ~small_inds[:,i]
                if np.where(big_inds)[0].size == 0:
                    continue
                Xi[big_inds,i] = np.linalg.lstsq(Theta[big_inds].T, Xprime[i])[0]

        self.poly_order = poly_order
        self.Xi = Xi

    def fit_incremental(self, Xin, dt=None, Xprime=None, coefficient_threshold=.01, error_threshold=1e-3):
        if Xprime is None:
            if dt is None:
                raise ValueError('must provide at least one of derivative or time step')
            Xprime = (Xin[:,2:]-Xin[:,:-2])/(2*dt)
            X = Xin[:,1:-1]
        else:
            X = Xin

        poly_orders = np.arange(1,6)

        for order in poly_orders:
            Theta,labels = pool_data(X, order, self.use_sine)

            self.labels = labels

            n,T = Xprime.shape
            Xi = np.linalg.lstsq(Theta.T,Xprime.T)[0]

            for k in range(10):
                small_inds = (np.abs(Xi) < coefficient_threshold)
                Xi[small_inds] = 0
                for i in range(n):
                    big_inds = ~small_inds[:,i]
                    if np.where(big_inds)[0].size == 0:
                        continue
                    Xi[big_inds,i] = np.linalg.lstsq(Theta[big_inds].T, Xprime[i])[0]

            error = np.sum(np.mean((Xprime - np.dot(Xi.T,Theta))**2,axis=1))
            print("order %d, error %f" % (order, error))
            if error < error_threshold:
                break

        self.poly_order = order
        self.Xi = Xi

    def reconstruct(self, x0, t0, dt, T):
        f = lambda t,x: np.dot(self.Xi.T, pool_data(np.real(x), poly_order=self.poly_order, use_sine=self.use_sine)[0])

        n_timesteps = int((T-t0)/dt) + 1

        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(x0, t0)

        x = [x0]
        t = [t0]
        while r.successful() and len(x) < n_timesteps:
            r.integrate(r.t + dt)
            x.append(np.real(r.y))
            t.append(r.t)

        return np.array(x).T, np.array(t)

    def print(self, threshold=1e-10):
        for j in range(self.Xi.shape[1]):
            eqn = "x%d' =" % (j+1)
            for i,l in enumerate(self.labels):
                if np.abs(self.Xi[i,j]) > threshold:
                    eqn += " (%f)%s +" % (self.Xi[i,j],l)
            print(eqn.strip('+'))
