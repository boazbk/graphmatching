import numpy as np
from graphs import randomGraph


def degreeSequence(g):
    n = g.shape[0]
    degree = np.sum(g, axis=1)
    degree1 = np.empty([n, n])
    for i in range(n):
        neighbor = degree[g[i, :] >= 1]
        degree1[i, :] = np.histogram(neighbor, np.arange(n+1))[0]
    g = g @ g
    degree2 = np.empty([n, n])
    for i in range(n):
        neighbor = degree[g[i, :] >= 1]
        degree2[i, :] = np.histogram(neighbor, np.arange(n+1))[0]
    return np.concatenate([degree[:, np.newaxis], degree1, degree2], axis=1)

g = randomGraph(100, 0.3)
print(degreeSequence(g))

