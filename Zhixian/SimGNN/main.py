from simgnn import SimGNNTrainer
import matplotlib.pyplot as plt
from subgraph import subgraphPrecision

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
    plt.plot(subgraphPrecision(100, 0.3, 10, 1000, 5), color="black", label="Subgraph counts")
    dnn = SimGNNTrainer(100, 5, 0, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(result, color="green", label="Simgnn with 5 features")
    dnn = SimGNNTrainer(100, 10, 0, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(result, color="red", label="Simgnn with 10 features")
    dnn = SimGNNTrainer(100, 50, 0, 1000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        ac = 1 - dnn.test(p, r/10)
        result.append(ac)
        print(r, ac)
    print(result)
    plt.plot(result, color="blue", label="Simgnn with 50 features")
##    dnn = SimGNNTrainer(100, 5, 0, 1000)
##    result = []
##    for r in range(10):
##        dnn.train(p, r / 10)
##        ac = 1 - dnn.test(p, r/10)
##        result.append(ac)
##        print(r, ac)
##    print(result)
##    plt.plot(result, color="blue")
    plt.legend()
    plt.savefig("simgnn.pdf")
    
if __name__ == "__main__":
    main()
