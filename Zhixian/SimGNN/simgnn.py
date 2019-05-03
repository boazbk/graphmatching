import torch
import matplotlib
from layers import ConvolutionModule, AttentionModule, TensorNetworkModule
import matplotlib.pyplot as plt

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, graph_size, feature_size, count_size):
        """
        :param args: Arguments object.
        :param graph_size: Size of the graph
        :param count_feature: bool for adding subgraph counts
        """
        super().__init__()
        self.n = graph_size
        self.f = feature_size
        self.cf = count_size
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.conv1 = ConvolutionModule(self.n, self.f)
        self.conv2 = ConvolutionModule(self.n, self.f)
        self.conv3 = ConvolutionModule(self.n, self.f)
        self.attention = AttentionModule(self.n, self.f)
        self.tensor_network = TensorNetworkModule(self.f + self.cf)

    def reset_parameters(self):
        self.conv1.init_parameters()
        self.conv2.init_parameters()
        self.conv3.init_parameters()
        self.attention.init_parameters()
        self.tensor_network.init_parameters()

    def count_subgraph(self, g):
        a = g
        f = torch.zeros([self.cf, 1])
        for i in range(self.cf):
            a = torch.matmul(a, g)
            f[i, 0] = torch.trace(a)
        return f
        
    def forward(self, g1, g2):
        features_1 = self.conv3(self.conv2(self.conv1(g1)))
        features_2 = self.conv3(self.conv2(self.conv1(g2)))
        pooled_features_1 = self.attention(features_1)
        pooled_features_2 = self.attention(features_2)
        if self.cf > 0:
            pooled_features_1 = torch.cat((pooled_features_1, self.count_subgraph(g1)), dim=0)
            pooled_features_2 = torch.cat((pooled_features_2, self.count_subgraph(g2)), dim=0) 
        score = self.tensor_network(pooled_features_1, pooled_features_2)
        # print(pooled_features_1, pooled_features_2, score)
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, n, feature_size, count_size, repeat):
        """
        :param args: Arguments object.
        """
        self.n = n
        self.f = feature_size
        self.cf = count_size
        self.r = repeat
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.n, self.f, self.cf)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
    def train(self, prob, noise):
        """
        Training a model.
        """
        def symmetrize(m):
            mu = torch.triu(m, diagonal=1)
            return mu + torch.t(mu)

        self.model.reset_parameters()
        for repeat in range(self.r):
            g = (torch.rand(self.n, self.n) < prob)
            m = (torch.rand(self.n, self.n) < (1 - noise))
            g1 = (g * m)
            m = (torch.rand(self.n, self.n) < (1 - noise))
            g2 = (g * m)
            g1 = symmetrize(g1)
            g2 = symmetrize(g2)
            g10 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g20 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g10 = symmetrize(g10)
            g20 = symmetrize(g20)
            self.optimizer.zero_grad()
            # print(g1, g2, g10, g20)
            output1 = self.model(g1.float(), g2.float())
            loss1 = self.criterion(output1.view(1, -1), torch.tensor([1]))
            output0 = self.model(g10.float(), g20.float())
            loss0 = self.criterion(output0.view(1, -1), torch.tensor([0]))
            loss = loss0 + loss1
            loss.backward()
            self.optimizer.step()
            # print(output1, output0)

    def test(self, prob, noise):
        
        def symmetrize(m):
            mu = torch.triu(m, diagonal=1)
            return mu + torch.t(mu)

        error = 0
        for repeat in range(100):
            g = (torch.rand(self.n, self.n) < prob)
            m = (torch.rand(self.n, self.n) < (1 - noise))
            g1 = g * m
            m = (torch.rand(self.n, self.n) < (1 - noise))
            g2 = g * m
            g1 = symmetrize(g1)
            g2 = symmetrize(g2)
            g10 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g20 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g10 = symmetrize(g10)
            g20 = symmetrize(g20)
            output1 = self.model(g1.float(), g2.float()).view(-1)
            error += 1 if output1[1] < output1[0] else 0
            output0 = self.model(g10.float(), g20.float()).view(-1)
            error += 1 if output0[0] < output0[1] else 0
            # print(output1, output0)
        return error / 200


if __name__ == "__main__":
    dnn = SimGNNTrainer(10, 0, 5, 1000)
    result = []
    for r in range(10):
        dnn.train(0.3, r / 10)
        result.append(1 - dnn.test(0.3, r / 10))
        print(r)
    print(result)
    #plt.plot(result)
    #plt.show()
    
