from simgnn import SimGNNTrainer
import matplotlib.pyplot as plt

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    p = 0.3
    dnn = SimGNNTrainer(100, 0, 5, 10000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        result.append(1 - dnn.test(p, r / 10))
        print(r)
    print(result)
    plt.plot(result, color="red")
    dnn = SimGNNTrainer(100, 50, 0, 10000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        result.append(1 - dnn.test(p, r / 10))
        print(r)
    print(result)
    plt.plot(result, color="green")
    dnn = SimGNNTrainer(100, 50, 5, 10000)
    result = []
    for r in range(10):
        dnn.train(p, r / 10)
        result.append(1 - dnn.test(p, r / 10))
        print(r)
    print(result)
    plt.plot(result, color="blue")
    plt.savefig("simgnn.pdf")
    
if __name__ == "__main__":
    main()
