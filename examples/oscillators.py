import numpy as np
import scipy.linalg as la
from scipy.integrate import ode


def vanderpol_oscillator(mu, tau=1):
    f = lambda t,x : [x[1]/tau, (mu*(1-x[0]**2)*x[1] - x[0])/tau]
    jac = lambda t,x : [[0., 1./tau], [(-2.*mu*x[0]*x[1] - 1.)/tau, -mu*x[0]**2/tau]]
    return f,jac


def simulate_vanderpol_oscillator(dt, n_timesteps, x0=None, mu=10., tau=1):
    if x0 is None:
        x0 = [2.,0.]

    f,jac = vanderpol_oscillator(mu,tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def duffing_oscillator(alpha, beta, gamma, delta, omega):
    f = lambda t,x : [x[1], -delta*x[1] - alpha*x[0] - beta*x[0]**3 + gamma*np.cos(x[2]), omega]
    jac = lambda t,x : [[0., 1., 0.], [-alpha - 3*beta*x[0]**2, -delta, -gamma*np.sin(x[2])], [0., 0., 0.]]
    return f,jac


def simulate_duffing_oscillator(dt, n_timesteps, x0=None, alpha=1., beta=1, gamma=0., delta=0., omega=1.):
    if x0 is None:
        x0 = [1.,0.]

    f,jac = duffing_oscillator(alpha,beta,gamma,delta,omega)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(np.real(f(r.t,r.y)))
        t.append(r.t)

    return np.array(x).T,np.array(t),np.array(xprime).T


def coupled_vdp_duffing(mu, alpha, beta, gamma, delta, omega, c1, c2):
    f = lambda t,x : [x[1] + c1*x[2], mu*(1-x[0]**2)*x[1] - x[0],
                  x[3] + c2*x[0], -delta*x[3] - alpha*x[2] - beta*x[2]**3 + gamma*np.cos(x[4]), omega]
    jac = lambda t,x : [[0., 1., c1, 0., 0.],
                        [-2.*mu*x[0]*x[1] - 1., -mu*x[0]**2, 0., 0., 0.],
                        [c2, 0., 0., 1., 0.],
                        [0., 0., -alpha - 3*beta*x[0]**2, -delta, -gamma*np.sin(x[2])],
                        [0., 0., 0., 0., 0.]]
    return f,jac


def simulate_coupled_vdp_duffing(dt, n_timesteps, x0=None, mu=1., alpha=1., beta=2., gamma=0., delta=0., omega=1., c1=1., c2=1.):
    if x0 is None:
        x0 = [2.,0.,0.,1.,0.]

    f,jac = coupled_vdp_duffing(mu,alpha,beta,gamma,delta,omega,c1,c2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_linear(k1, k2, c1, c2, tau1=1., tau2=1.):
    f = lambda t,x : [(x[1] + c1*x[2])/tau1, -k1*x[0]/tau1,
                      (x[3] + c2*x[0])/tau2, -k2*x[2]/tau2]
    jac = lambda t,x : [[0., 1./tau1, c1/tau1, 0.],
                        [-k1/tau1, 0., 0., 0.],
                        [c2/tau2, 0., 0., 1./tau2],
                        [0., 0., -k2/tau2, 0.]]
    return f,jac


def simulate_coupled_linear(dt, n_timesteps, x0=None, k1=1.1, k2=.5, c1=-.5, c2=-1., tau1=1., tau2=1.):
    if x0 is None:
        x0 = [1.,0.,0.,1.]

    f,jac = coupled_linear(k1, k2, c1, c2, tau1=tau1, tau2=tau2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,r.y))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


# def simulate_coupled_linear(dt, T, k1=1.1, k2=.5, c1=-.5, c2=-1.):
#     K = np.array([[-k1,c1],[c2,-k2]])
#
#     t = np.arange(0,T+dt,dt)
#
#     evals,evecs = la.eig(-K)
#     if np.any(np.imag(evals) != 0):
#         print(evals)
#     omega1 = np.real(np.sqrt(evals[0]))
#     omega2 = np.real(np.sqrt(evals[1]))
#     omegas = np.array([omega1,omega2])
#
#     xa = np.outer(evecs[:,0],np.cos(omega1*t))
#     xb = np.outer(evecs[:,1],np.cos(omega2*t))
#     x_soln = xa + xb
#
#     # for dynamical system view - return solutions for y as well
#     At = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [-k1, c1, 0., 0.], [c2, -k2, 0., 0.]])
#     ya = -omega1*np.outer(evecs[:,0], np.sin(omega1*t))
#     yb = -omega2*np.outer(evecs[:,1], np.sin(omega2*t))
#     y_soln = ya + yb
#
#     #evals,evecs = la.eig(At)
#     modes = np.empty((4,4),dtype=complex)
#     evals = np.empty(4,dtype=complex)
#     for i in range(4):
#         idx = int(i/2)
#         if (i % 2) == 0:
#             modes[:,i] = np.kron(np.array([1,1j*omegas[idx]]),evecs[:,idx])
#             evals[i] = 1j*omegas[idx]
#         else:
#             modes[:,i] = np.kron(np.array([1,-1j*omegas[idx]]),evecs[:,idx])
#             evals[i] = -1j*omegas[idx]
#     coeffs = np.array([.5,.5,.5,.5]) # not returning right now
#
#     return np.vstack((x_soln,y_soln)), modes, evals, coeffs


def coupled_linear_duffing(k, alpha, beta, gamma, delta, omega, c1, c2):
    f = lambda t,x: [x[1] + c1*x[2], -k*x[0],
                     x[3] + c2*x[0], -delta*x[3] - alpha*x[2] - beta*x[2]**3 + gamma*np.cos(x[4]), omega]
    jac = lambda t,x : [[0., 1., c1, 0., 0.],
                        [-k, 0., 0., 0., 0.],
                        [c2, 0., 0., 1., 0.],
                        [0., 0., -alpha - 3*beta*x[2]**2, -delta, -gamma*np.sin(x[4])],
                        [0., 0., 0., 0., 0.]]
    return f,jac


def simulate_coupled_linear_duffing(dt, n_timesteps, x0=None, k=1., alpha=1., beta=1., gamma=0., delta=0., omega=1., c1=1., c2=1.):
    if x0 is None:
        x0 = [1.,0.,1.,0.,0.]

    f,jac = coupled_linear_duffing(k,alpha,beta,gamma,delta,omega,c1,c2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_linear_vdp(k, mu, c1, c2):
    f = lambda t,x: [x[1] + c1*x[2], -k*x[0],
                     x[3] + c2*x[0], mu*(1-x[2]**2)*x[3] - x[2]]
    jac = lambda t,x : [[0., 1., c1, 0.],
                        [-k, 0., 0., 0.],
                        [c2, 0., 0., 1.],
                        [0., 0., -2.*mu*x[2]*x[3] - 1., -mu*x[2]**2]]
    return f,jac


def simulate_coupled_linear_vdp(dt, n_timesteps, x0=None, k=1., mu=1., c1=1., c2=1.):
    if x0 is None:
        x0 = [1.,0.,2.,0.]

    f,jac = coupled_linear_vdp(k,mu,c1,c2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def duffing_map(a, b):
    f = lambda x: [x[1], -b*x[0] + a*x[1] - x[1]**3]
    return f


def simulate_duffing_map(n_steps, x0=None, a=2.75, b=0.2):
    if x0 is None:
        x0 = [0.,1.]

    x = np.zeros((2,n_steps))
    f = duffing_map(a,b)

    x[:,0] = np.array(x0)
    for i in range(1,n_steps):
        x[:,i] = np.array(f(x[:,i-1]))
    return x


def linear_oscillator(k):
    f = lambda t,x: [x[1],-k*x[0]]
    jac = lambda t,x: [[0.,1.], [-k,0.]]
    return f,jac


def simulate_linear_oscillator(dt, n_timesteps, x0=None, k=1.):
    if x0 is None:
        x0 = [1.,0.]

    f,jac = linear_oscillator(k)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_duffing(alpha1, beta1, gamma1, delta1, omega1, alpha2, beta2, gamma2, delta2, omega2, c1, c2):
    f = lambda t,x: [x[1] + c1*x[3], -delta1*x[1] - alpha1*x[0] - beta1*x[0]**3 + gamma1*np.cos(x[2]), omega1,
                     x[4] + c2*x[0], -delta2*x[4] - alpha2*x[3] - beta2*x[3]**3 + gamma2*np.cos(x[5]), omega2]
    jac = lambda t,x : [[0., 1., 0., c1, 0., 0.],
                        [-alpha1 - 3*beta1*x[0]**2, -delta1, -gamma1*np.sin(x[2]), 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [c2, 0., 0., 0., 1., 0.],
                        [0., 0., 0., -alpha2 - 3*beta2*x[3]**2, -delta2, -gamma2*np.sin(x[5])],
                        [0., 0., 0., 0., 0., 0.]]
    return f,jac


def simulate_coupled_duffing(dt, n_timesteps, x0=None, alpha1=1., beta1=1., gamma1=0., delta1=0., omega1=1., alpha2=1.,
                             beta2=1., gamma2=0., delta2=0., omega2=1., c1=1., c2=1.):
    if x0 is None:
        x0 = [1.,0.,0.,1.,0.,0.]

    f,jac = coupled_duffing(alpha1, beta1, gamma1, delta1, omega1, alpha2, beta2, gamma2, delta2, omega2, c1, c2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_vdp(mu1, mu2, c1, c2, tau1=1, tau2=1):
    f = lambda t,x: [(x[1] + c1*x[2])/tau1, (mu1*(1-x[0]**2)*x[1] - x[0])/tau1,
                     (x[3] + c2*x[0])/tau2, (mu2*(1-x[2]**2)*x[3] - x[2])/tau2]
    jac = lambda t,x : [[0., 1./tau1, c1/tau1, 0.],
                        [(-2.*mu1*x[0]*x[1] - 1.)/tau1, (-mu1*x[0]**2)/tau1, 0., 0.],
                        [c2/tau2, 0., 0., 1./tau2],
                        [0., 0., (-2.*mu2*x[2]*x[3] - 1.)/tau2, (-mu2*x[2]**2)/tau2]]
    return f,jac


def simulate_coupled_vdp(dt, n_timesteps, x0=None, mu1=1., mu2=1., c1=1., c2=1., tau1=1, tau2=1):
    if x0 is None:
        x0 = [2.,0.,0.,2.]

    f,jac = coupled_vdp(mu1,mu2,c1,c2, tau1=1, tau2=1)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_vdp_lorenz(mu, sigma, rho, beta, c1, c2, tau1=1, tau2=1):
    f = lambda t,x: [(x[1] + c1*x[2])/tau1, (mu*(1-x[0]**2)*x[1] - x[0])/tau1,
                     (sigma*(x[3] - x[2]) + c2*x[0])/tau2, (x[2]*(rho - x[4]) - x[3])/tau2, (x[2]*x[3] - beta*x[4])/tau2]
    jac = lambda t,x : [[0., 1./tau1, c1/tau1, 0., 0.],
                        [(-2.*mu*x[0]*x[1] - 1.)/tau1, (-mu*x[0]**2)/tau1, 0., 0., 0.],
                        [c2/tau2, 0., -sigma/tau2, sigma/tau2, 0.],
                        [0., 0., (rho - x[4])/tau2, -1./tau2, -x[2]/tau2],
                        [0., 0., x[3]/tau2, x[2]/tau2, -beta/tau2]]
    return f,jac


def simulate_coupled_vdp_lorenz(dt, n_timesteps, x0=None, mu=1., sigma=10., rho=28., beta=8/3, c1=1., c2=1., tau1=1, tau2=1):
    if x0 is None:
        x0 = [2., 0., -8., 7., 27.]

    f,jac = coupled_vdp_lorenz(mu,sigma,rho,beta,c1,c2,tau1=tau1,tau2=tau2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T
