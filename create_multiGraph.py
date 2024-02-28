from torch_geometric.nn import knn
from torch_geometric.loader import DataLoader, GraphSAINTSampler
import torch
from models.GraphEmbedding import GraphEmbedding, MultiGraphAttention
from utils.loader import load_graphs
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
import networkx as nx

# Path to the speech encoder, in this case Resnet, Whisper, or wav2vec.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODEL_PATH = '../trained/local_graph_embedding.pt'
MULTIGRAPH_PATH = '../trained/multigraph.pt'


# Global graph
class MultiGraph:
    def __init__(self, graph_test_path, graph_train_path, num_class, emb_size, batch_size):
        self.graph_train_path = graph_train_path
        self.graph_test_path = graph_test_path
        self.num_class = num_class
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.percent_labels = 1.0
        self.classes = num_class
        self.data = None
        self.no_label_data = None
        self.local_graph_created = True
        self.is_local_trained = True


    def train_local_graphs(self):
        self.data = load_graphs(self.graph_train_path)
        self.data, self.no_label_data = train_test_split(self.data, train_size=self.percent_labels, shuffle=True)
        loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        model = GraphEmbedding(embedding_size=self.emb_size, hidden_channels=128, num_classes=self.classes)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        epochs_stop = 3
        no_improve = 0
        num_epochs = 40
        epoch_min_loss = None
        start_epoch = 1
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = []
            epoch_acc = []
            i = 0
            for graph in loader:
                optimizer.zero_grad()
                out = model(graph.x, graph.edge_index)
                loss = criterion(out, graph.y)
                total = graph.y.size(0)
                _, predicted = torch.max(out.data, 1)
                correct = (predicted == graph.y).sum().item()
                epoch_acc.append(correct / total)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss)

                ### Epoch check ###
            e_loss = sum(epoch_loss) / len(epoch_loss)
            print(e_loss)
            print(sum(epoch_acc) / len(epoch_acc))
            if epoch_min_loss == None:
                epoch_min_loss = e_loss
            elif e_loss < epoch_min_loss:
                epoch_min_loss = e_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve == epochs_stop:
                torch.save(model, GRAPH_MODEL_PATH)
                print("Model saved to {}".format(GRAPH_MODEL_PATH))
                break

    def find_knn(self, n):
        k = 3
        edges_pos = []
        edges_neg = []
        ind = int(abs(len(n) / 2))
        n1 = n[:ind]
        n2 = n[ind:]
        xn1 = torch.squeeze(torch.stack([n[1]['x'] for n in n[:ind]]))
        xn2 = torch.squeeze(torch.stack([n[1]['x'] for n in n[ind:]]))
        kn = knn(xn1, xn2, k)
        for i in range(ind):
            for j in range(k):
                if j != 2:
                    edges_pos.append((n1[int(kn[0][i+j])][0], n2[int(kn[1][i+j])][0]))
                elif j == 2:
                    edges_neg.append((n1[int(kn[0][i+j])][0], n2[int(kn[1][i+j])][0]))
        return edges_pos, edges_neg


    def generate_edges(self, graph):
        nodes_1 = [n for n in graph.nodes(data=True) if int(n[1]['y']) == 0]
        nodes_2 = [n for n in graph.nodes(data=True) if int(n[1]['y']) == 1]
        nodes_3 = [n for n in graph.nodes(data=True) if int(n[1]['y']) == 2]
        nodes_4 = [n for n in graph.nodes(data=True) if int(n[1]['y']) == 3]

        edges_1_pos, edges_1_neg = self.find_knn(nodes_1)
        edges_2_pos, edges_2_neg = self.find_knn(nodes_2)
        edges_3_pos, edges_3_neg = self.find_knn(nodes_3)
        edges_4_pos, edges_4_neg = self.find_knn(nodes_4)
        epos = edges_1_pos + edges_2_pos + edges_3_pos + edges_4_pos
        eneg = edges_1_neg + edges_2_neg + edges_3_neg + edges_4_neg
        graph.add_edges_from(epos, weight=1)
        graph.add_edges_from(eneg, weight=-1)
        return graph

    def generate_multigraph(self):
        model = GraphEmbedding(embedding_size=self.emb_size, hidden_channels=1024 * 2, num_classes=self.num_class)
        model.load_state_dict(torch.load(GRAPH_MODEL_PATH))
        model.linear = torch.nn.Identity()
        model.eval()
        graph = nx.Graph()
        with torch.no_grad():
            for d in self.data:
                gemb = model(d.x, d.edge_index, d.batch)
                graph.add_node(str(d.id[0][0]).split('-')[0], x=gemb, y=d.y)

        multi_graph = self.generate_edges(graph)
        output = from_networkx(multi_graph)
        output.y = torch.swapaxes(output.y, 0, 1)
        torch.save(output, MULTIGRAPH_PATH)
        print("Multigraph created successfully at {}".format(MULTIGRAPH_PATH))
        return None

    def train_multigraph(self):
        data = torch.load(MULTIGRAPH_PATH)
        loader = GraphSAINTSampler(data, int(round(data.num_nodes/5)))
        model = MultiGraphAttention()
