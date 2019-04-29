import numpy as np

from graphs import randomGraph
from graphs import randomSample

from threshold import findThreshold
from threshold import thresholdError

def pathHistogram(g, l):
    n = g.shape[0]
    res = np.array([])
    a = g
    for i in range(l):
        degree = np.sum(a, axis=1)
        degreehist, bins = np.histogram(degree)
        res = np.append(res, degreehist)
        a = a @ g
    return res

def histogramPrecision(n, p, rs, repeat, length):
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
            hist1 = pathHistogram(g1, length)
            hist2 = pathHistogram(g2, length)
            hist10 = pathHistogram(g10, length)
            hist20 = pathHistogram(g20, length)
            seq1[i] = np.sum(np.absolute(hist1 - hist2))
            seq2[i] = np.sum(np.absolute(hist10 - hist20))
        t = findThreshold(seq1[:repeat // 2], seq2[:repeat // 2])
        error = thresholdError(seq1[repeat // 2:], seq2[repeat // 2:], t)
        precision.append(1 - error / repeat)
    return np.array(precision)
    

if __name__ == "__main__":
    g = randomGraph(100, 0.5)
    print(pathHistogram(g, 4))
