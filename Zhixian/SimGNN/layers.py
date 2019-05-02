import torch

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
        self.weight_matrix = torch.nn.Parameter(torch.zeros([self.f, self.n, self.n])) 
        
    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding,transformed_global.view(self.f, -1, 1)))
        representation = torch.matmul(torch.t(embedding),sigmoid_scores)
        representation = torch.mean(representation.view(self.f, -1), dim=1)
        return representation.view(-1, 1)

class TenorNetworkModule(torch.nn.Module):
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
        self.weight_matrix = torch.nn.Parameter(torch.zeros([2, self.f, self.f]))
        self.bias = torch.nn.Parameter(torch.zeros([2, 1]))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.matmul(torch.t(embedding_1), self.weight_matrix)
        scoring = torch.matmul(scoring, embedding_2)
        scores = scoring.view(-1, 1) + self.bias
        return scores.view
