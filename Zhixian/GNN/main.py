from gnn import GNNTrainer
import matplotlib.pyplot as plt
from subgraph import subgraphPrecision
import numpy as np

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    p = 0.3
##    dnn = SimGNNTrainer(100, 0, 5, 10000)
##    result = []
##    for r in range(10):
##        dnn.train(p, r / 10)
##        ac = 1 - dnn.test(p, r/10)
##        result.append(ac)
##        print(r, ac)
##    print(result)
##    plt.plot(result, color="red")
    
    dnn = GNNTrainer(100, 5, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(np.linspace(0, 0.9, 10), result, color="green", label="GCN with 5 features")
    dnn = GNNTrainer(100, 10, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(np.linspace(0, 0.9, 10), result, color="red", label="GCN with 10 features")
    dnn = GNNTrainer(100, 50, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(np.linspace(0, 0.9, 10), result, color="blue", label="GCN with 50 features")
##    dnn = SimGNNTrainer(100, 5, 0, 1000)
##    result = []
##    for r in range(10):
##        dnn.train(p, r / 10)
##        ac = 1 - dnn.test(p, r/10)
##        result.append(ac)
##        print(r, ac)
##    print(result)
##    plt.plot(result, color="blue")
    plt.plot(np.linspace(0, 0.9, 10), subgraphPrecision(100, 0.3, 10, 1000, 5), color="black", label="Subgraph counts with 5 features")
    plt.legend()
    plt.xlabel("noise")
    plt.ylabel("accuracy")
    plt.savefig("gcn.pdf")
    
if __name__ == "__main__":
    main()
