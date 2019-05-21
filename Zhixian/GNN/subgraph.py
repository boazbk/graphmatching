import numpy as np
from scipy.optimize import linear_sum_assignment

from graphs import randomGraph
from graphs import randomSample

from threshold import findThreshold
from threshold import thresholdError


def countSubgraph(g, l):
    n = g.shape[0]
    counts = np.empty([n, l])
    a = g
    for i in range(l):
        a = a @ g
        counts[:, i] = a.diagonal()
    return counts

def maximumMatching(v0, v1):
    n = v0.shape[0]
    cost = np.empty([n, n])
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum(np.abs(v0[i, :] - v1[j, :]))
    rowind, colind = linear_sum_assignment(cost)
    # print(cost)
    return sum(rowind == colind)

def subgraphPrecision(n, p, rs, repeat, length):
    precision = []
    for r in np.linspace(1, 0.1, rs):
        seq1 = np.empty(repeat)
        seq2 = np.empty(repeat)
        for i in range(repeat):
            g = randomGraph(n, p)
            g1 = randomSample(g, r)
            g2 = randomSample(g, r)
            g10 = randomGraph(n, p * r)
            g20 = randomGraph(n, p * r)
            hist1 = countSubgraph(g1, length)
            hist2 = countSubgraph(g2, length)
            # print(hist1, hist2)
            hist1 = hist1 / (np.sum(hist1, axis=0) + 1)
            hist2 = hist2 / (np.sum(hist2, axis=0) + 1)
            hist1 = np.sum(hist1, axis=1)
            hist2 = np.sum(hist2, axis=1)
            hist1 = np.sort(hist1)
            hist2 = np.sort(hist2)
            # print(hist1, hist2)
            hist10 = countSubgraph(g10, length)
            hist20 = countSubgraph(g20, length)
            hist10 = hist10 / (np.sum(hist10, axis=0) + 1)
            hist20 = hist20 / (np.sum(hist20, axis=0) + 1)
            hist10 = np.sum(hist10, axis=1)
            hist20 = np.sum(hist20, axis=1)
            hist10 = np.sort(hist10)
            hist20 = np.sort(hist20)
            # print(hist10, hist20)
            seq1[i] = np.sum(np.absolute(hist1 - hist2))
            seq2[i] = np.sum(np.absolute(hist10 - hist20))
        t = findThreshold(seq1[:repeat // 2], seq2[:repeat // 2])
        error = thresholdError(seq1[repeat // 2:], seq2[repeat // 2:], t)
        precision.append(1 - error / repeat)
    return np.array(precision)

def countsMatching(n, p, rs, repeat, length):
    precision = []
    for r in np.linspace(1, 0.1, rs):
        correct = 0
        for i in range(repeat):
            g = randomGraph(n, p)
            g1 = randomSample(g, r)
            g2 = randomSample(g, r)
            hist1 = countSubgraph(g1, length)
            hist2 = countSubgraph(g2, length)
            hist1 = hist1 / (np.sum(hist1, axis=0) + 1)
            hist2 = hist2 / (np.sum(hist2, axis=0) + 1)
            correct += maximumMatching(hist1, hist2)
        correct = correct / repeat
        precision.append(correct)
    return np.array(precision)


if __name__ == "__main__":
    g = randomGraph(100, 0.5)
    print(countSubgraph(g, 3))
