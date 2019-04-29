import numpy as np

def randomGraph(n, p):
    m = np.random.rand(n, n)
    m = 1 * (m < p)
    mu = np.triu(m, 1)
    m = mu + mu.T
    return m.astype(int)

def randomSample(g, r):
    n = g.shape[0]
    m = np.random.rand(n, n)
    m = 1 * (m < r)
    mu = np.triu(m, 1)
    m = mu + mu.T
    return (m * g).astype(int)

if __name__ == "__main__":
    data1 = np.empty([10, 400, 2, 50, 50])
    data0 = np.empty([10, 400, 2, 50, 50])
    for i in range(10):
        for repeat in range(400):
            g = randomGraph(50, 0.3)
            g1 = randomSample(g, 1 - i * 0.1)
            g2 = randomSample(g, 1 - i * 0.1)
            data1[i, repeat, 0, :, :] = g1
            data1[i, repeat, 1, :, :] = g2
            g01 = randomGraph(50, 0.3 * (1 - i * 0.1))
            g02 = randomGraph(50, 0.3 * (1 - i * 0.1))
            data0[i, repeat, 0, :, :] = g01
            data0[i, repeat, 1, :, :] = g02

    np.save("C:\\Users\\Zhixian Lei\\Downloads\\data1", data1)
    np.save("C:\\Users\\Zhixian Lei\\Downloads\\data0", data0)
        

        
