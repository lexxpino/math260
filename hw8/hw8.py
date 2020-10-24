import numpy as np
from numpy.random import uniform, rand
from scipy import sparse
import matplotlib.pyplot as plt


def stationary(pt, alpha=0.9):
    """Power method to find stationary distribution.
    """
    x = rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    pt = alpha * pt
    for it in range(10000):
        x += (1 - alpha) / pt.shape[0]
        x = pt.dot(x)
        x /= sum(x)
    return x

def read_file(fname):
    """Returns two dictionaries representing the graph in the text"""
    names = {}
    adj = {}
    with open(fname, 'r') as fp:
        for k, line in enumerate(fp):
            line = line.strip('\n').split(' ')
            if len(line) == 1: # deals with EOF
                continue
            vert_name = int(line[1])
            if line[0] == 'n':
                names[vert_name] = line[2]
            elif line[0] == 'e':
                if vert_name not in adj:
                    adj[vert_name] = []
                adj[vert_name].append(int(line[2]))
    return names, adj

def build_sparse(adj, size):
    """Loops through adjacency matrix to build sparse matrix"""
    r = []
    c = []
    data = []
    for node in adj:
        neighbors = adj[node]
        prob = 1 / len(neighbors)
        for n in neighbors:
            r.append(node)
            c.append(n)
            data.append(prob)
    mat = sparse.coo_matrix((data, (r, c)), shape = (size, size))
    return mat.transpose()

def pagerank_test(txt_file, alpha=0.9):
    """returns top 10 stationary probabilities from the graph/text file"""
    names, adj = read_file(txt_file)
    mat = build_sparse(adj, len(names))
    probs = stationary(mat, alpha)
    indices = np.argsort(probs)
    probs = np.sort(probs)
    l = len(indices)
    for i in range(l - 1, l - 11, -1):
        idx = indices[i]
        print(f'{names[idx]}: {probs[i]}')


pagerank_test("hw8/california.txt")






#Q2
def dydx(t, y): 
    '''test case from slides'''
    return (2*t*y) 

def rk4(t0, y0, t, h):
    '''implementation of rk4 method of approximation'''
    n = int((t - t0)/h)
    y = y0
    for i in range(n):
        k1 = dydx(t0, y) 
        k2 = dydx(t0 + 0.5 * h, y + 0.5 * h*k1) 
        k3 = dydx(t0 + 0.5 * h, y + 0.5 * h*k2) 
        k4 = dydx(t0 + h, y + h*k3)  

        y = y + h*(k1 + 2*k2 + 2*k3 + k4)/(6.0)

        t0 = t0 + h
    return y

def rk4_err():
    '''Solve y' = 2ty, y(0) = 1 using Euler's method. Use the true solution to compute the max error, and show that it is O(h^4)'''
    hvals = [(0.1)*2**(-k) for k in range(11)]
    ref = [hvals[i]**4 for i in range(len(hvals))]
    t0 = 0
    y0 = 1
    t = 2
    exact = np.exp(2 ** 2)

    err = [0]*len(hvals)
    for k in range(len(hvals)):  # err[k] is max error at h = hvals[k]
        approximate = rk4(t0, y0, t, hvals[k])
        err[k] = abs(approximate - exact)
    
    plt.figure()
    plt.loglog(hvals, ref, '--r')
    plt.loglog(hvals, err, '.--k')
    plt.legend(['slope 4', 'max. err.'])
    plt.xlabel('$h$') 
    plt.show()


#Q3

def fwd_euler_sys(f, a, b, y0, h):
    """ Forward euler, to solve the system y' = f(t,y) of m equations
        Inputs:
            f - a function returning an np.array of length m
         a, b - the interval to integrate over
           y0 - initial condition (a list or np.array), length m
           h  - step size to use
        Returns:
            t - a list of the times for the computed solution
            y - computed solution; list of m components (len(y[k]) == len(t))
    """
    y = np.array(y0)  # copy!
    t = a
    tvals = [t]
    yvals = [[v] for v in y]  # y[j][k] is j-th component of solution at t[k]
    while t < b - 1e-12:
        y += h*f(t, y)
        t += h
        for k in range(len(y)):
            yvals[k].append(y[k])
        tvals.append(t)
        
    return tvals, yvals

def trap_sys(a,b,y0,h):
    t = a
    tvals = [t] # tvals = [0, ...]
    yvals = [y0] # yvals = [{0, 1}, ...]
    t = t + h
    while t < b - 1e-12:
        xprev = yvals[-1][0]
        yprev = yvals[-1][1]
        x_next = (xprev*(1 - (0.25 * (h**2))) + yprev*h)/(1 + (0.25 * (h**2)))
        y_next = (yprev*(1 - (0.25 * (h**2))) - xprev*h)/(1 + (0.25 * (h**2)))
        yvals.append([x_next, y_next])
        tvals.append(t)
        t += h
    
    return tvals, yvals


def ode_func(t, v):
    return np.array([v[1], -v[0]])

def trap_ex(h):
    v_init = [0, 1.0]
    t, v = trap_sys(0, 2 * np.pi, v_init, h)
    # v = [(x0, y0), (x1, y1), ...]
    x = [v[i][0] for i in range(len(v))]
    y = [v[i][1] for i in range(len(v))]

    plt.figure()
    plt.plot(t, x, '-k', t, y, '-r')
    plt.legend(["$\\sin(t)$", "$\\cos(t)$"])
    plt.xlabel("$t$")
    plt.show()



def euler_test(h):
    v_init = [1.0, 0]
    t, v = fwd_euler_sys(ode_func, 0, 2*np.pi, v_init, h)
    x = v[0]
    y = v[1]

    plt.figure()
    plt.plot(t, x, '-k', t, y, '-r')
    plt.legend(["$\\theta(t)$", "$\\theta'(t)$"])
    plt.xlabel("$t$")
    plt.show()





if __name__ == "__main__":
    #Q1
    pagerank_test("hw8/california.txt")

    #Q2
    # RK4 is a fourth order method, so we test to verify that error is fourth order
    # the plot shows that the line has slope 4
    rk4_err()

    #Q3 (a)
    '''
    x_n+1 = x_n + h/2(y_n + y_n+1)
    y_n+1 = y_n - h/2(x_n + x_n+1)
    '''
    # we use substitution to solve for the following equations in (b)
    #Q3 (b)
    '''
    x_n+1 = (x_n(1 - .25h^2) + h*y_n)/(1 + .25h^2)
    y_n+1 = (y_n(1 - .25h^2) + h*x_n)/(1 + .25h^2)
    '''
    # the trapezoidal method produces an accurate graph faster than the euler method
    # relative to h as h approaches 0
    # the difference in accuracy is shown from the following plots
    trap_ex(.2)
    trap_ex(.001)
    euler_test(.2)
    euler_test(.001)
