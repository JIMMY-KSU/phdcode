import numpy as np
from scipy.integrate import ode


def lorenz(sigma, rho, beta, tau=1.):
    f = lambda t,x : [sigma*(x[1] - x[0])/tau, (x[0]*(rho - x[2]) - x[1])/tau, (x[0]*x[1] - beta*x[2])/tau]
    jac = lambda t,x : [[-sigma/tau, sigma/tau, 0.],
                        [(rho - x[2])/tau, -1./tau, -x[0]/tau],
                        [x[1]/tau, x[0]/tau, -beta/tau]]
    return f,jac


def simulate_lorenz(dt, n_timesteps, x0=None, sigma=10., rho=28., beta=8/3, tau=1.):
    if x0 is None:
        x0 = [-8, 7, 27]

    f,jac = lorenz(sigma, rho, beta, tau=tau)
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


def simulate_coupled_lorenz(dt, n_timesteps, x0=None, sigma1=10., rho1=28., beta1=8/3, sigma2=10., rho2=50., beta2=8/3, c1=1, c2=1):
    if x0 is None:
        x0 = [-8, 7, 27, 10, -4, 70]

    f,jac = coupled_lorenz(sigma1, rho1, beta1, sigma2, rho2, beta2, c1, c2)
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


def lorenz96(N, F, tau=1):
    def f(t, x):
        xp = np.zeros(N, dtype=np.complex)
        for i in range(N-1):
            xp[i] = (x[i+1] - x[i-2])*x[i-1] - x[i] + F
        xp[N-1] = (x[0] - x[N-3])*x[N-2] - x[N-1] + F
        return xp/tau

    def jac(t, x):
        jac = np.zeros((N,N), dtype=np.complex)
        for i in range(N-1):
            jac[i,i-2] = -x[i-1]
            jac[i,i-1] = x[i+1] - x[i-2]
            jac[i,i] = -1
            jac[i,i+1] = x[i-1]
        jac[N-1,N-3] = -x[N-2]
        jac[N-1,N-2] = x[0] - x[N-3]
        jac[N-1,N-1] = -1
        jac[N-1,0] = x[N-2]
        return jac/tau

    return f,jac


def simulate_lorenz96(dt, n_timesteps, N, x0=None, F=0, tau=1):
    if x0 is None:
        x0 = np.ones(N)

    f,jac = lorenz96(N, F, tau=tau)
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


def coupled_lorenz96(N1, N2, F1, F2, C1, C2, tau1=1, tau2=1):
    def f(t, x):
        C1x = np.dot(C1, x[N1:])
        C2x = np.dot(C2, x[:N1])
        xp = np.zeros(N1+N2, dtype=np.complex)
        for i in range(2,N1-1):
            xp[i] = (x[i+1] - x[i-2])*x[i-1] - x[i] + F1 + C1x[i]
        xp[0] = (x[1] - x[N1-2])*x[N1-1] - x[0] + F1 + C1x[0]
        xp[1] = (x[2] - x[N1-1])*x[0] - x[1] + F1 + C1x[1]
        xp[N1-1] = (x[0] - x[N1-3])*x[N1-2] - x[N1-1] + F1 + C1x[N1-1]
        xp[:N1] /= tau1

        for i in range(2,N2-1):
            xp[N1+i] = (x[N1+i+1] - x[N1+i-2])*x[N1+i-1] - x[N1+i] + F2 + C2x[i]
        xp[N1] = (x[N1+1] - x[N1+N2-2])*x[N1+N2-1] - x[N1] + F2 + C2x[0]
        xp[N1+1] = (x[N1+2] - x[N1+N2-1])*x[N1] - x[N1+1] + F2 + C2x[1]
        xp[N1+N2-1] = (x[N1] - x[N1+N2-3])*x[N1+N2-2] - x[N1+N2-1] + F2 + C2x[N2-1]
        xp[N1:] /= tau2
        return xp

    def jac(t, x):
        jac = -np.eye(N1+N2, dtype=np.complex)
        for i in range(2,N1-1):
            jac[i,i-2] = -x[i-1]
            jac[i,i-1] = x[i+1] - x[i-2]
            jac[i,i+1] = x[i-1]
        jac[0,N1-2] = -x[N1-1]
        jac[0,N1-1] = x[1] - x[N1-2]
        jac[0,1] = x[N1-1]
        jac[1,N1-1] = -x[0]
        jac[1,0] = x[2] - x[N1-1]
        jac[1,2] = x[0]
        jac[N1-1,N1-3] = -x[N1-2]
        jac[N1-1,N1-2] = x[0] - x[N1-3]
        jac[N1-1,0] = x[N1-2]
        jac[:N1,:N1] /= tau1

        for i in range(2,N2-1):
            jac[N1+i,N1+i-2] = -x[N1+i-1]
            jac[N1+i,N1+i-1] = x[N1+i+1] - x[N1+i-2]
            jac[N1+i,N1+i+1] = x[N1+i-1]
        jac[N1,N1+N2-2] = -x[N1+N2-1]
        jac[N1,N1+N2-1] = x[N1+1] - x[N1+N2-2]
        jac[N1,N1+1] = x[N1+N2-1]
        jac[N1+1,N1+N2-1] = -x[N1]
        jac[N1+1,N1] = x[N1+2] - x[N1+N2-1]
        jac[N1+1,N1+2] = x[N1]
        jac[N1+N2-1,N1+N2-3] = -x[N1+N2-2]
        jac[N1+N2-1,N1+N2-2] = x[N1] - x[N1+N2-3]
        jac[N1+N2-1,N1] = x[N1+N2-2]
        jac[N1:,N1:] /= tau2
        return jac

    return f,jac


def simulate_coupled_lorenz96(dt, n_timesteps, N1, N2, C1, C2, x0=None, F1=0, F2=0, tau1=1, tau2=1):
    if x0 is None:
        x0 = np.ones(N1+N2)

    f,jac = coupled_lorenz96(N1, N2, F1, F2, C1, C2, tau1=tau1, tau2=tau2)
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
