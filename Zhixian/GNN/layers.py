import torch

class AggregateModule(torch.nn.Module):

    def __init__(self, n, f):
        super().__init__()
        self.n = n
        self.f = f
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight = torch.nn.Parameter(torch.empty([self.f, self.f]))

    def init_parameters(self):
        torch.nn.init.normal_(self.weight, std=1)

    def forward(self, embedding, graph):
        deg = torch.sum(graph, dim=0)
        embedding = torch.matmul(embedding, graph)
        embedding = embedding / (deg.view(1, -1) + 1)
        embedding = torch.matmul(self.weight, embedding)
        embedding = torch.nn.functional.relu(embedding)
        return embedding
