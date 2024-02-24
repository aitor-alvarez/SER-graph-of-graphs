from torch import nn
from torch_geometric.nn import GraphConv, global_mean_pool, GCNConv
from torch.functional import F


class GraphEmbedding(nn.Module):
    def __init__(self, embedding_size, hidden_channels, num_classes):
        channels = 128
        super(GraphEmbedding, self).__init__()
        self.gconv1 = GraphConv(embedding_size, hidden_channels)
        self.gconv2 = GraphConv(hidden_channels, channels)
        self.linear = nn.Linear(channels, num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, batch):
        print(x_embeddings.shape)
        print(edge_index.shape)
        x = self.gconv1(x_embeddings, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Mean pooling
        x = global_mean_pool(x, batch)
        # Graph classification
        x = F.dropout(x, p=0.2, training=self.training)
        out = self.linear(x)
        return out


class NodePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.gconv1 = GCNConv(128, 128)
        self.gconv2 = GCNConv(128, 4)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index, weights):
        x = self.gconv1(x_embeddings, edge_index, weights)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index, weights)
        out = F.softmax(x, dim=1)
        return out
