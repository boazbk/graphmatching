import numpy as np

def findThreshold(seq1, seq2):
    n1 = seq1.shape[0]
    n2 = seq2.shape[0]
    seq1 = np.concatenate((seq1[:, np.newaxis], np.zeros([n1, 1])), axis = 1)
    seq2 = np.concatenate((seq2[:, np.newaxis], np.ones([n2, 1])), axis = 1)
    seq = np.concatenate([seq1, seq2], axis=0)
    ind = np.argsort(seq[:, 0])
    # print(seq.shape)
    seq = seq[ind, :]
    # print(seq)
    minerror = n1
    error = n1
    threshold = seq[0, 0]
    for i in range(n1 + n2):
        if seq[i, 1] == 0:
            error -= 1
            if error < minerror:
                threshold = seq[i, 0]
                minerror = error
        else:
            error += 1
    return threshold

def thresholdError(seq1, seq2, threshold):
    n = seq1.shape[0]
    return np.sum(seq1 > threshold) + np.sum(seq2 < threshold)

if __name__ == "__main__":
    print(findThreshold(np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6])))
    print(thresholdError(np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6]), 4.5))
