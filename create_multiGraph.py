from torch_geometric.nn import knn
from torch_geometric.loader import DataLoader
import torch
from models.GraphEmbedding import GraphEmbedding
from utils.loader import graph_loader
from utils.intonation_patterns import get_speech_representations
from sklearn.model_selection import train_test_split
import os
from torch_geometric.utils import from_networkx
import networkx as nx
import torchaudio

# Path to the speech encoder, in this case Resnet, Whisper, or wav2vec.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODEL_PATH = 'trained/local_graph_embedding.pt'
MULTIGRAPH_PATH = 'trained/multigraph.pt'


# Global graph
class MultiGraph:
    def __init__(self, graph_test_path, graph_train_path, num_class, emb_size, is_trained):
        self.graph_train_path = graph_train_path
        self.graph_test_path = graph_test_path
        self.num_class = num_class
        self.emb_size = emb_size
        self.batch_size = 32
        self.percent_labels = 0.8
        self.classes = 4
        self.data = None
        self.no_label_data = None
        self.is_trained = is_trained
        self.local_graph_created = True

    def get_dataset(self, dir):
        dataset = []
        max_len = self.get_audio_max_len(dir)
        for root, dirs, files in os.walk(dir):
            for f in files:
                fp = root + '/' + f
                if 'patterns' in root and fp.endswith('.pt'):
                    graph = torch.load(fp)
                    graph.x = get_speech_representations(graph.id, root, max_len)
                    torch.save(graph, fp)
                    dataset.append(fp)
        return dataset

    # Function to get the maximum length of the patterns for padding.
    def get_audio_max_len(self, dir):
        max_len = 0
        for root, dirs, files in os.walk(dir):
            for f in files:
                fp = root + '/' + f
                if 'patterns' in root and fp.endswith('.wav'):
                    audio_len = torchaudio.load(fp)[0][0].shape
                    if audio_len.numel() > max_len:
                        max_len = audio_len.numel()
        return max_len

    def train_local_graphs(self):
        if not self.local_graph_created:
            self.data = self.get_dataset(self.graph_train_path)
        self.data = graph_loader(self.data)
        self.data, self.no_label_data = train_test_split(self.data, train_size=self.percent_labels, shuffle=True)
        train_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        model = GraphEmbedding(embedding_size=self.emb_size, hidden_channels=128, num_classes=self.classes)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        epochs_stop = 3
        no_improve = 0
        acc_list = []
        num_epochs = 40
        epoch_min_loss = None
        start_epoch = 1
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = []
            for graph in train_loader:
                optimizer.zero_grad()
                out = model(graph.x, graph.edge_index, graph.batch)
                loss = criterion(out, graph.y)
                total = graph.y.size(0)
                _, predicted = torch.max(out.data, 1)
                correct = (predicted == graph.y).sum().item()
                acc_list.append(correct / total)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss)

            ### Epoch check ###
            e_loss = sum(epoch_loss) / len(epoch_loss)
            print(e_loss)
            print(correct / total)
            if epoch_min_loss == None:
                epoch_min_loss = e_loss
            elif e_loss < epoch_min_loss:
                epoch_min_loss = e_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve == epochs_stop:
                torch.save(model, GRAPH_MODEL_PATH)
                break

    def find_knn(self, n):
        edges_pos = []
        edges_neg = []
        n1 = n[:len(n) / 2]
        n2 = n[len(n) / 2:]
        xn1 = [n.x for n in n1]
        xn2 = [n.x for n in n2]
        kn = knn(xn1, xn2, len(n1) - 1)
        k1, k2 = train_test_split(kn, train_size=0.8, shuffle=False)
        for i in k1:
            edges_pos.append((n1[int(i[0])], n2[int(i[1])]))
        for j in k2:
            edges_neg.append((n1[int(j[0])], n2[int(j[1])]))
        return edges_pos, edges_neg

    def generate_pseudo_labels(self, graph):
        nodes = [n for n in graph.nodes(data=True) if n['y'] is not None]
        nodesx = [n.x for n in nodes]
        nodes_no = [n for n in graph.nodes(data=True) if n['y'] is None]
        nodes_no_x = [n.x for n in nodes_no]
        kn = knn(nodesx, nodes_no_x, len(nodes_no_x) - 1)
        k1, k2 = train_test_split(kn, train_size=0.3, shuffle=False)
        for i in k1:
            graph.add_edge((nodes[int(i[0])], nodes_no[int(i[1])]))
            graph.nodes[nodes_no[int(i[1])]].y = nodes[int(i[0])].y
        for j in k2:
            graph.add_edge((nodes[int(j[0])], nodes_no[int(j[1])]))
            graph.nodes[nodes_no[int(j[1])]].y = nodes[int(j[0])].y
        return graph

    def generate_edges(self, graph):
        nodes_1 = [n for n in graph.nodes(data=True) if n['y'] == 1]
        nodes_2 = [n for n in graph.nodes(data=True) if n['y'] == 2]
        nodes_3 = [n for n in graph.nodes(data=True) if n['y'] == 3]
        nodes_4 = [n for n in graph.nodes(data=True) if n['y'] == 4]
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
        model = torch.load(GRAPH_MODEL_PATH)
        model.eval()
        graph = nx.Graph()
        for d in self.data:
            gemb = model(d.x, d.edge_index, d.batch)
            graph.add_node(d.id, x=gemb, y=d.y)
        for l in self.no_label_data:
            gemb = model(l.x, l.edge_index, l.batch)
            graph.add_node(l.id, x=gemb, y=None, z=l.y)
        multi_graph = self.generate_edges(graph)
        multi_graph = self.generate_pseudo_labels(multi_graph)
        output = from_networkx(multi_graph)
        torch.save(output, MULTIGRAPH_PATH)
        return None
