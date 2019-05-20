import torch

class ConvolutionModule(torch.nn.Module):

    def __init__(self, n, f):
        super().__init__()
        self.n = n
        self.f = f
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.empty([self.f, self.n, self.n]))
        self.bias = torch.nn.Parameter(torch.empty([self.f, 1, 1]))

    def init_parameters(self):
        torch.nn.init.normal_(self.weight_matrix, std=1)
        torch.nn.init.normal_(self.bias, std=1)

    def forward(self, embedding):
        context = torch.matmul(embedding, self.weight_matrix)
        context += self.bias
        representation = torch.sigmoid(context)
        return representation

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, n, f):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.n = n
        self.f = f
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.empty([self.f, self.n, self.n]))
        self.bias = torch.nn.Parameter(torch.empty([self.f, 1]))
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.normal_(self.weight_matrix, std=1)
        torch.nn.init.normal_(self.bias, std=1)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """
        # print(embedding)
        # print(self.weight_matrix)
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=2)
        sigmoid_scores = torch.sigmoid(global_context + self.bias)
        representation = torch.matmul(embedding,sigmoid_scores.view(self.f, self.n, 1))
        representation = torch.matmul(sigmoid_scores.view(self.f, 1, self.n), representation.view(self.f, self.n, 1))
        return representation.view(-1, 1)

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, f):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.f = f
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.empty([2, 2 * self.f, 2 * self.f]))
        self.bias = torch.nn.Parameter(torch.empty([1, 2]))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.normal_(self.weight_matrix, std=1)
        torch.nn.init.normal_(self.bias, std=1)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        embedding = torch.cat((embedding_1, embedding_2), dim = 0)
        scoring = torch.matmul(torch.t(embedding), self.weight_matrix)
        scoring = torch.matmul(scoring, embedding)
        scores = scoring.view(1, -1) + self.bias
        return scores

if __name__ == "__main__":
    g = torch.rand(100, 100)
    g1 = torch.rand(100, 100)
    model = AttentionModule(100, 2)
    model1 = TensorNetworkModule(2)
    print(model1(model(g), model(g1)))
