import numpy as np
from scipy.integrate import ode
from .sindy import pool_data
from .utils import integrate


class SINDyC:
    def __init__(self, use_sine=False, method='derivative'):
        self.use_sine = use_sine
        self.method = method

    def fit(self, Xin, Uin, poly_order, dt=None, Xprime=None, coefficient_threshold=.01):
        if self.method == 'derivative':
            if Xprime is None:
                if dt is None:
                    raise ValueError('must provide at least one of derivative or time step')
                Xprime = (Xin[:,2:]-Xin[:,:-2])/(2*dt)
                X = Xin[:,1:-1]
                U = Uin[:,1:-1]
            else:
                X = Xin
                U = Uin

            LHS = Xprime
            Thetax,labelsx = pool_data(X, poly_order, self.use_sine)
            Thetau,labelsu = pool_data(U, poly_order, self.use_sine, include_constant=False, varname='u')
            RHS = np.concatenate((Thetax,Thetau), axis=0)
        elif self.method == 'integral':
            if dt is None:
                raise ValueError('must provide time step')

            LHS = Xin - np.broadcast_to(Xin[:,0], (Xin.shape[1],Xin.shape[0])).T

            Thetax,labelsx = pool_data(Xin, poly_order, self.use_sine)
            Thetau,labelsu = pool_data(Uin, poly_order, self.use_sine, include_constant=False, varname='u')
            Theta = np.concatenate((Thetax,Thetau), axis=0)
            RHS = integrate(Theta, dt)
        else:
            raise ValueError('invalid fitting method')

        self.labels = labelsx + labelsu

        n,T = LHS.shape
        Xi = np.linalg.lstsq(RHS.T,LHS.T)[0]

        for k in range(10):
            small_inds = (np.abs(Xi) < coefficient_threshold)
            Xi[small_inds] = 0
            for i in range(n):
                big_inds = ~small_inds[:,i]
                if np.where(big_inds)[0].size == 0:
                    continue
                Xi[big_inds,i] = np.linalg.lstsq(RHS[big_inds].T, LHS[i])[0]

        self.poly_order = poly_order
        self.Xi = Xi
        self.error = np.sum(np.mean((LHS - np.dot(Xi.T,RHS))**2,axis=1))

    def fit_incremental(self, Xin, Uin, dt=None, Xprime=None, coefficient_threshold=.01, error_threshold=1e-3):
        if self.method == 'derivative':
            if Xprime is None:
                if dt is None:
                    raise ValueError('must provide at least one of derivative or time step')
                Xprime = (Xin[:,2:]-Xin[:,:-2])/(2*dt)
                X = Xin[:,1:-1]
                U = Uin[:,1:-1]
            else:
                X = Xin
                U = Uin
        elif self.method == 'integral':
            if dt is None:
                raise ValueError('must provide time step')
            X = Xin
        else:
            raise ValueError('invalid fitting method')

        poly_orders = np.arange(1,6)

        for order in poly_orders:
            Thetax,labelsx = pool_data(X, order, self.use_sine)
            Thetau,labelsu = pool_data(U, order, self.use_sine, include_constant=False, varname='u')
            Theta = np.concatenate((Thetax,Thetau), axis=0)
            if self.method == 'derivative':
                RHS = Theta
                LHS = Xprime
            else:
                RHS = integrate(Theta, dt)
                LHS = X - np.broadcast_to(X[:,0], (X.shape[1],X.shape[0])).T

            self.labels = labelsx + labelsu

            n,T = LHS.shape
            Xi = np.linalg.lstsq(RHS.T,LHS.T)[0]

            for k in range(10):
                small_inds = (np.abs(Xi) < coefficient_threshold)
                Xi[small_inds] = 0
                for i in range(n):
                    big_inds = ~small_inds[:,i]
                    if np.where(big_inds)[0].size == 0:
                        continue
                    Xi[big_inds,i] = np.linalg.lstsq(RHS[big_inds].T, LHS[i])[0]

            error = np.sum(np.mean((LHS - np.dot(Xi.T,RHS))**2,axis=1))
            print("order %d, error %f" % (order, error))
            if error < error_threshold:
                break

        self.poly_order = order
        self.Xi = Xi

    def reconstruct(self, x0, U, t0, dt, n_timesteps):
        raise NotImplementedError('reconstruction not implemented yet for SINDYc')

        n = x0.shape
        f = lambda t,x: np.dot(self.Xi.T, np.concatenate((pool_data(np.real(x[:n]), poly_order=self.poly_order,
                                                                    use_sine=self.use_sine)[0],
                                                          pool_data(np.real(x[n:]), poly_order=self.poly_order,
                                                                   use_sine=self.use_sine, include_constant=False)[0]),
                                                         axis=0))

        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(x0, t0)

        x = [x0]
        t = [t0]
        while r.successful() and len(x) < n_timesteps:
            r.integrate(r.t + dt)
            x.append(np.real(r.y[:n]))
            t.append(r.t)

        return np.array(x).T, np.array(t)

    def print(self, threshold=1e-10):
        for j in range(self.Xi.shape[1]):
            eqn = "x%d' =" % (j+1)
            for i,l in enumerate(self.labels):
                if np.abs(self.Xi[i,j]) > threshold:
                    eqn += " (%f)%s +" % (self.Xi[i,j],l)
            print(eqn.strip('+'))
