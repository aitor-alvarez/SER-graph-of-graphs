from torch import nn
from torch_geometric.nn import GraphConv, GCNConv
from torch.functional import F


class GraphEmbedding(nn.Module):
    def __init__(self, embedding_size, hidden_channels, num_classes):
        channels = 1024
        super(GraphEmbedding, self).__init__()
        self.gconv1 = GraphConv(embedding_size, hidden_channels)
        self.gconv2 = GraphConv(hidden_channels, channels)
        self.gconv3 = GraphConv(channels, int(channels/2))
        self.linear = nn.Linear(int(channels/2), num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x_embeddings, edge_index):

        x = self.gconv1(x_embeddings, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.linear(x)
        return out

