import matplotlib.pyplot as plt
import numpy as np

def plotResult(null,struct,gamma):

    null_avg = np.zeros(gamma.size);
    struct_avg = np.zeros(gamma.size);
    null_std = np.zeros(gamma.size);
    struct_std = np.zeros(gamma.size);

    for g in range(gamma.size):
        null_avg[g] = np.average(null[g,:])
        struct_avg[g] = np.average(struct[g,:])
        null_std[g] = np.std(null[g,:])
        struct_std[g] = np.std(struct[g,:])

    plt.plot(gamma, null_avg, 'rs', gamma, null_avg-null_std, 'r--', gamma, null_avg+null_std, 'r--')
    plt.plot(gamma, struct_avg, 'bs', gamma, struct_avg-struct_std, 'b--', gamma, struct_avg+struct_std, 'b--')


