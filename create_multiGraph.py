from torch_geometric.nn import knn
from torch_geometric.loader import DataLoader, LinkNeighborLoader
import torch
from models.GraphEmbedding import GraphEmbedding, MultiGraphAttention
from utils.loader import load_graphs
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
import networkx as nx
import evaluate
import numpy as np

# Path to the speech encoder, in this case Resnet oe HuBERT.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODEL_PATH = 'trained/local_graph_embedding.pt'
MULTIGRAPH_PATH = 'trained/multigraph.pt'
MULTIGRAPH_TEST_PATH = 'trained/multigraph_test.pt'

recall = evaluate.load('recall')
accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')

# Global graph
class MultiGraph:
    def __init__(self, graph_train_path, num_class, emb_size, batch_size, is_local_trained=True):
        self.graph_train_path = graph_train_path
        self.num_class = num_class
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.classes = num_class
        self.is_local_trained = is_local_trained
        self.data = None
        self.no_label_data = None


    def train_local_graphs(self):
        self.data = load_graphs(self.graph_train_path)
        self.data, self.no_label_data = train_test_split(self.data, train_size=0.8, shuffle=True)
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
        output.x = torch.squeeze(output.x)
        output.y = torch.squeeze(torch.swapaxes(output.y, 0, 1))
        torch.save(output, MULTIGRAPH_PATH)
        print("Multigraph created successfully at {}".format(MULTIGRAPH_PATH))
        return None

    def compute_metrics(self, predictions, label_ids):
        rec_w = recall.compute(predictions=predictions, references=label_ids, average='weighted')
        rec_u = recall.compute(predictions=predictions, references=label_ids, average=None)
        rec_u = np.mean(rec_u['recall'])
        f1_score = f1.compute(predictions=predictions, references=label_ids, average='micro')
        acc = accuracy.compute(predictions=predictions, references=label_ids, average='weighted')
        return rec_w, rec_u, f1_score, acc

    def train_multigraph(self):
        model = MultiGraphAttention(embedding_size=512)
        data = torch.load(MULTIGRAPH_PATH)
        data.x = data.x.to(torch.float)
        data.edge_index = data.edge_index.to(torch.int64)
        data.weight = data.weight.to(torch.float)
        epochs = 100
        epochs_stop = 3
        no_improve = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        min_loss = None
        for epoch in range(epochs):
            out = model(data.x, data.edge_index, data.weight)
            optimizer.zero_grad()
            loss = criterion(out, data.y)
            _, predicted = torch.max(out, 1)
            recall_w, recall_u, f1_score, acc = self.compute_metrics(predicted, data.y)
            print("Epoch: {}, Loss: {:.4f}".format(epoch, loss))
            print("Weighted Recall: ", recall_w)
            print("Unweighted Recall: ", recall_u)
            print("F1 micro: ", f1_score)
            print("Acc.: ", acc)
            loss.backward()
            optimizer.step()
            if min_loss == None:
                min_loss = loss
            elif loss < min_loss:
                min_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve == epochs_stop:
                torch.save(model, 'trained/multigraph_gnn_model.pt')
                print("Model trained completed")
                break
        return model

    def test_multigraph(self, model, test_graph):
        model.eval()
        with torch.no_grad():
            out = model(test_graph)
            _, predicted = torch.max(out.data, 1)
            recall_w, recall_u, f1_micro, acc = self.compute_metrics(predicted, test_graph.y)
            print("Weighted Recall: ", recall_w)
            print("Unweighted Recall: ", recall_u)
            print("f1_micro: ", f1_micro)
            print("Acc.: ", acc)
            print("Test completed")

    def run(self):
        if self.is_local_trained:
            self.generate_multigraph()
            model = self.train_multigraph()
            self.test_multigraph(model, test_graph=MULTIGRAPH_TEST_PATH)

        else:
            self.train_local_graphs()
            self.generate_multigraph()
            model = self.train_multigraph()
            self.test_multigraph(model, test_graph=MULTIGRAPH_TEST_PATH)
