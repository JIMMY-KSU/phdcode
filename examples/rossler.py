import numpy as np
from scipy.integrate import ode


def rossler(a, b, c, tau=1.):
    f = lambda t,x : [(-x[1] - x[2])/tau, (x[0] + a*x[1])/tau, (b + x[2]*(x[0] - c))/tau]
    jac = lambda t,x : [[0., -1/tau, -1/tau],
                        [1/tau, a/tau, 0.],
                        [x[2]/tau, 0., x[0]/tau]]
    return f,jac


def simulate_rossler(dt, n_timesteps, x0=None, a=0.2, b=0.2, c=5.7, tau=1.):
    if x0 is None:
        x0 = [0, 10, 0]

    f,jac = rossler(a, b, c, tau=tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.0)

    x = [x0]
    t = [0.0]
    xprime = [f(0.0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T
