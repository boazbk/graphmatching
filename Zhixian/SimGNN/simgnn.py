import torch
from layers import AttentionModule, TenorNetworkModule

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
        self.attention = AttentionModule(self.n, self.f)
        self.tensor_network = TenorNetworkModule(self.f + self.cf)

    def count_subgraph(g, self):
        a = g
        f = torch.zeros([self.cf, 1])
        for i in range(self.cf):
            a = torch.matmul(a, g)
            f[i, 0] = torch.trace(a)
        return f
        
    def forward(self, g1, g0):
        
        pooled_features_1 =  self.attention(g1)
        pooled_features_2 = self.attention(g2)
        if count_size > 0:
            pooled_features_1 = torch.cat((pooled_features_1, self.count_subgraph(g1)), dim=0)
            pooled_features_2 = torch.cat((pooled_features_2, self.count_subgraph(g2)), dim=0) 
        score = self.tensor_network(pooled_features_1, pooled_features_2)
        return score

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels)
        
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)
         
    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = [self.training_graphs[graph:graph+self.args.batch_size] for graph in range(0, len(self.training_graphs), self.args.batch_size)]
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        edges_1 = torch.from_numpy(np.array(data["graph_1"], dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(data["graph_2"], dtype=np.int64).T).type(torch.long)

        features_1 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in data["labels_1"]]))
        features_2 = torch.FloatTensor(np.array([[ 1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()] for node in data["labels_2"]]))
        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2
        normalized_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        new_data["target"] =  torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch. 
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
        losses.backward(retain_graph = True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc = "Epoch")
        main_index = 0
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score
            loss = self.loss_sum/main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            self.scores.append(calculate_loss(prediction, target))
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error= np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error,5))+".")
        print("\nModel test error: " +str(round(model_error,5))+".")