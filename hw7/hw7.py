import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#Q1
def power_method(a, steps,v1):
    """ Calculates error between an estimate, x, and v1 for a given number of steps
    """
    v1 = np.asarray(v1)
    v1 = v1/np.sqrt(v1.dot(v1))
    n = a.shape[0]
    x = random.rand(n)
    i = 0
    error1 = [0]*steps
    error2 = [0]*steps
    while i < steps:  # other stopping conditions would go here
        q = np.dot(a, x)  # compute a*x
        x = q/np.sqrt(q.dot(q))  # normalize x to a unit vector
        error1[i] = np.max(np.absolute(np.subtract(x,v1)))
        error2[i] = np.max(np.absolute(np.subtract(x,-1*v1)))
        i += 1
    return x, error1, error2

def err_plot(a,steps, v1):
    x_vals = [i for i in range(steps)]
    x, errs1, errs2 = power_method(a,steps,v1)
    plt.plot(x_vals, errs1, label = 'pos error')
    plt.plot(x_vals, errs2, label = 'neg error')
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


#Q2
def stationary(pt, max_steps=100, tol=1e-3):
    """Power method to find stationary distribution.
       Given the largest eigenvalue is 1, finds the eigenvector.
    """
    x = random.rand(pt.shape[0])  # random initial vector
    x /= sum(x)
    for i in range(max_steps):
        x_last = x
        x = np.dot(pt, x)
        if np.linalg.norm(np.subtract(x,x_last)) <= tol:
            break
    return x

def computeM(P, alpha):
    n = len(P)
    P= np.asarray(P)
    P_t = np.transpose(P)
    E = np.ones((n,n))
    return alpha*P_t + ((1-alpha) / n)*E

#Q3
def read_graph_data(fname):
    adj = {}
    names = {}

    with open(fname,'r') as fp:
        for k, line in enumerate(fp):
            line = line.strip('\n').split(' ')
            if line[0] == 'n':
                names[line[1]] = line[2]
            elif line[0] == 'e':
                if line[1] not in adj:
                    adj[line[1]] = []
                adj[line[1]].append(line[2])
    return adj, names


if __name__ == "__main__":
    #Q1
    a = np.asarray([[0,1,0],[0,0,1],[6,-11,6]])
    v1 = [1,3,9]
    err_plot(a,20, v1)
    '''1(b) The error decreases at a rate that depends on the difference in magnitude of
    the dominant eigenvalue and the second largest. If the dominant eigenvalue is much
    larger than the next largest, error will converge to 0 faster than if they are similar
    in magnitude.'''

    #Q2
    P = [[0,1/3,1/3,1/3,0], [1/2,0,0,0,1/2], [1/2,1/2,0,0,0], [1/2,1/2,0,0,0], [0,1,0,0,0]]
    M = computeM(P, 0.95)
    stationary_dist = stationary(M)
    #print(stationary_dist)
    '''2(b) Around 10 iterations are necessary for a reasonable solution'''
    """2(c) The highest ranked page is the 1st node with a stationary
    probability of 0.357. When checking results with small alphas (0.0000001),
    the 1st node/page remains the highest ranked page. """

    #Q3
    #print(read_graph_data('hw7/large_graph.txt')[1])