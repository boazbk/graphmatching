import numpy as np
from graphs import randomGraph

def moduleGraph(g1, g2):
    n1 = g1.shape[0]
    n2 = g2.shape[0]
    g = np.fromfunction(
        lambda i, j: 1 * (g1[i // n2, j // n2] == g2[i % n2, j % n2]),
        [n1 * n2, n1 * n2],
        dtype=int)
    return g.astype(int)

def maxClique(s, g):
    if not s:
        return 0
    v = s.pop()
    neighbor = np.argwhere(g[v, :] == 1)
    neighbor = neighbor.reshape(-1)
    neighbor = set(neighbor)
    return max(maxClique(set(s), g), maxClique(s & neighbor, g) + 1)

g1 = randomGraph(3, 1)
g2 = randomGraph(3, 1)
print(g1, g2)
g = moduleGraph(g1, g2)
print(g)
print(maxClique(set(np.arange(9)), g))
