import torch
from layers import AggregateModule
import matplotlib.pyplot as plt

class GNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, graph_size, feature_size):
        """
        :param args: Arguments object.
        :param graph_size: Size of the graph
        :param count_feature: bool for adding subgraph counts
        """
        super().__init__()
        self.n = graph_size
        self.f = feature_size
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.embedding0 = torch.nn.Parameter(torch.empty([self.f, self.n]))
        self.aggr = AggregateModule(self.n, self.f)
        self.linear = torch.nn.Linear(2 * self.f, 2)

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding0, std=1)
        self.aggr.init_parameters()
        torch.nn.init.normal_(self.linear.weight, std=1)
        torch.nn.init.normal_(self.linear.bias, std=1)
        
    def forward(self, g1, g2):
        feature_1 = self.embedding0
        feature_2 = self.embedding0
        for i in range(5):
            feature_1 = self.aggr(feature_1, g1)
            feature_2 = self.aggr(feature_2, g2)
        feature_1 = torch.mean(feature_1, dim=1)
        feature_2 = torch.mean(feature_2, dim=1)
        combine = torch.cat([feature_1, feature_2], dim=0)
        combine = combine.view(1, -1)
        score = self.linear(combine)
        return score
        
class GNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, n, feature_size, repeat):
        """
        :param args: Arguments object.
        """
        self.n = n
        self.f = feature_size
        self.r = repeat
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = GNN(self.n, self.f)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
    def train(self, prob, noise):
        """
        Training a model.
        """
        # print("train")
        def symmetrize(m):
            mu = torch.triu(m, diagonal=1)
            return mu + torch.t(mu)

        self.model.reset_parameters()
        for repeat in range(self.r // 10):
            # print(repeat)
            self.optimizer.zero_grad()
            loss = 0
            for i in range(10):
                # print("hello")
                g = (torch.rand(self.n, self.n) < prob)
                m = (torch.rand(self.n, self.n) < (1 - noise))
                g1 = (g * m)
                m = (torch.rand(self.n, self.n) < (1 - noise))
                g2 = (g * m)
                g1 = symmetrize(g1)
                g2 = symmetrize(g2)
                r = torch.randperm(self.n)
                g2 = g2[r][:, r]
                g10 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
                g20 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
                g10 = symmetrize(g10)
                g20 = symmetrize(g20)
                output1 = self.model(g1.float(), g2.float())
                # print(g1)
                # print(g2)
                # print(1, output1)
                loss1 = self.criterion(output1, torch.tensor([1]))
                output0 = self.model(g10.float(), g20.float())
                # print(g10)
                # print(g20)
                # print(0, output0)
                loss0 = self.criterion(output0, torch.tensor([0]))
                loss += loss0 + loss1
            loss.backward()
            self.optimizer.step()

    def test(self, prob, noise):
        # print("test")
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
            r = torch.randperm(self.n)
            g2 = g2[r][:, r]
            g10 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g20 = (torch.rand(self.n, self.n) < (1 - noise) * prob)
            g10 = symmetrize(g10)
            g20 = symmetrize(g20)
            output1 = self.model(g1.float(), g2.float()).view(-1)
            # print(1, output1)
            error += 1 if output1[1] < output1[0] else 0
            output0 = self.model(g10.float(), g20.float()).view(-1)
            # print(0, output0)
            error += 1 if output0[0] < output0[1] else 0
            # print(output1, output0)
        return error / 200
