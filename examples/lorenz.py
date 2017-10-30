import numpy as np
from scipy.integrate import ode


def lorenz(sigma, rho, beta):
    f = lambda t,x : [sigma*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]]
    jac = lambda t,x : [[-sigma, sigma, 0.],
                        [rho - x[2], -1., -x[0]],
                        [x[1], x[0], -beta]]
    return f,jac


def simulate_lorenz(t0, dt, n_timesteps, x0=None, sigma=10., rho=28., beta=8/3):
    if x0 is None:
        x0 = [-8, 7, 27]

    f,jac = lorenz(sigma, rho, beta)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)

    x = [x0]
    t = [t0]
    xprime = [f(t0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_lorenz(sigma1, rho1, beta1, sigma2, rho2, beta2, c1, c2):
    f = lambda t,x : [sigma1*(x[1] - x[0]) + c1*x[3], x[0]*(rho1 - x[2]) - x[1], x[0]*x[1] - beta1*x[2],
                      sigma2*(x[4] - x[3]) + c2*x[0], x[3]*(rho2 - x[5]) - x[4], x[3]*x[4] - beta2*x[5]]
    jac = lambda t,x : [[-sigma1, sigma1, 0., c1, 0., 0.],
                        [rho1 - x[2], -1., -x[0], 0., 0., 0.],
                        [x[1], x[0], -beta1, 0., 0., 0.],
                        [c2, 0., 0., -sigma2, sigma2, 0.],
                        [0., 0., 0., rho2 - x[5], -1., -x[3]],
                        [0., 0., 0., x[4], x[3], -beta2]]
    return f,jac


def simulate_coupled_lorenz(t0, dt, n_timesteps, x0=None, sigma1=10., rho1=28., beta1=8/3, sigma2=10., rho2=50., beta2=8/3, c1=1, c2=1):
    if x0 is None:
        x0 = [-8, 7, 27, 10, -4, 70]

    f,jac = coupled_lorenz(sigma1, rho1, beta1, sigma2, rho2, beta2, c1, c2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)

    x = [x0]
    t = [t0]
    xprime = [f(t0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def lorenz96(N, F):
    def f(t, x):
        xp = []
        for i in range(N-1):
            xp.append((x[i+1] - x[i-2])*x[i-1] - x[i] + F)
        xp.append((x[0] - x[N-3])*x[N-2] - x[N-1] + F)
        return xp

    def jac(t, x):
        jac = np.zeros((N,N))
        for i in range(N-1):
            jac[i,i-2] = -x[i-1]
            jac[i,i-1] = x[i+1] - x[i-2]
            jac[i,i] = -1
            jac[i,i+1] = x[i-1]
        jac[N-1,N-3] = -x[N-2]
        jac[N-1,N-2] = x[0] - x[N-3]
        jac[N-1,N-1] = -1
        jac[N-1,0] = x[N-2]
        return jac.tolist()

    return f,jac


def simulate_lorenz96(t0, dt, n_timesteps, N, x0=None, F=0):
    if x0 is None:
        x0 = np.ones(N)

    f,jac = lorenz96(N, F)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, t0)

    x = [x0]
    t = [t0]
    xprime = [f(t0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T
