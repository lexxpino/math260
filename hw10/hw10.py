import matplotlib.pyplot as plt
import numpy as np


def rk4(func, interval, y0, h, params):
    """ RK4 method, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)
    t = interval[0]
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < interval[1] - 1e-12:
        f0 = func(t, y, params)
        f1 = func(t + 0.5*h, y + 0.5*h*f0, params)
        f2 = func(t + 0.5*h, y + 0.5*h*f1, params)
        f3 = func(t + h, y + h*f2, params)
        y += (1.0/6)*h*(f0 + 2*f1 + 2*f2 + f3)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)

    return tvals, yvals


def read_sir_data(fname):
    """ Reads the SIR data for the HW problem.
        Inputs:
            fname - the filename (should be "sir_data.txt")
        Returns:
            t, x - the data (t[k], I[k]), where t=times, I= # infected
            pop - the initial susceptible population (S(0))
    """
    with open(fname, 'r') as fp:
        parts = fp.readline().split()
        pop = float(parts[0])
        npts = int(float(parts[1]))
        t = np.zeros(npts)
        x = np.zeros(npts)

        for k in range(npts):
            parts = fp.readline().split()
            t[k] = float(parts[0])
            x[k] = float(parts[1])

    return t, x, pop


def example_odef(t, x, params):
    """ example ODE (not the SIR one),
        u' = a*u + v
        v' = u + b*v
    """
    du = params[0]*x[0] + x[1]
    dv = x[0] + params[1]*x[1]
    return np.array((du, dv))

def sir(t,x,params):
    alpha = params[0]
    beta = params[1]
    S = x[0]
    I = x[1]
    R = x[2]
    P = 100
    ds = (-1 * alpha * S * I)/P
    di = ((alpha * S * I)/P) - (beta * I)
    dr = beta * I
    return np.array((ds, di, dr))

def err_func(params):
    tvals, data, pop = read_sir_data('hw10/sir_data.txt')
    h = 7/8
    t,x = rk4(sir, (0,140), [100 - 4.8512558909454855, 4.8512558909454855, 0], h, params)

    return least_squares(data, x[1])

def least_squares(ivals, results):
    sum = 0
    for i in range(len(ivals)):
        sum += (results[8 * i] - ivals[i])**2
    return sum

def err_grad(params):
    alpha = params[0]
    beta = params[1]
    delta = 0.01
    da = (err_func([alpha + delta, beta]) - err_func([alpha - delta, beta]))/(2 * delta)
    dB = (err_func([alpha, beta + delta]) - err_func([alpha, beta - delta]))/(2 * delta)
    return np.array((da, dB))

def minimize(f, x0, tol=1e-4, maxsteps=100, verb=False):
    """ Gradient descent with a *very simple* line search
        Inputs:
            f - the function f(x)
            df - gradient of f(x)
            x0 - initial point
    """
    x = np.array(x0)
    err = 100
    it = 0
    pts = [np.array(x)]
    while it < maxsteps and err > tol:
        fx = f(x)
        v = err_grad(x)
        v /= sum(np.abs(v))
        alpha = 1
        while (x[0] - alpha * v[0]) < 0 or (x[1] - alpha * v[1]) < 0:
            alpha /= 2
        while alpha > 1e-10 and f(x - alpha*v) >= fx:
            alpha /= 2
        if verb:
            print(f"it={it}, x[0]={x[0]:.8f}, f={fx:.8f}")
        x -= alpha*v
        pts.append(np.array(x))
        err = max(np.abs(alpha*v))
        it += 1

    return x, it, pts

if __name__ == '__main__':
    x0 = np.array((0.5,0.5))

    actual = minimize(err_func,x0)
    ret = actual[0]
    print('Infection Rate:',actual[0][0])
    print("Recovery Rate:", actual[0][1])

    tvals, data, pop = read_sir_data('hw10/sir_data.txt')
    t, x = rk4(sir, (0,140), (100 - 4.8512558909454855, 4.8512558909454855, 0), 7/8, ret)

    plt.figure(1)
    plt.plot(tvals, data, '.k', markersize=12)
    plt.plot(t, x[1], 'green')
    plt.legend(['Real Data', 'Model Solution'])
    plt.xlabel('t')
    plt.ylabel('Infected')
    plt.show()
    '''
    plt.figure(2)
    plt.plot(t, x[0], '-k', t, x[1], '-r')
    plt.xlabel('t')
    plt.legend(['u(t)', 'v(t)'])
    '''
    



