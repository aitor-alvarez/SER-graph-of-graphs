from torch_geometric.nn import knn
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from models.resnet import Resnet, Bottleneck
from models.GraphEmbedding import GraphEmbedding
from transformers import AutoFeatureExtractor
from utils.loader import graph_loader
from sklearn.model_selection import train_test_split
import os

#Path to the speech encoder, in this case Resnet, Whisper, or wav2vec.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODEL_PATH = 'trained/local_graph_embedding.pt'

#Global graph
class MultiGraph:
	def __int__(self, speech_model_path, graph_test_path, graph_train_path, num_class, emb_size, is_trained=False):
		self.speech_encoder = speech_model_path
		self.graph_train_path = graph_train_path
		self.graph_test_path = graph_test_path
		self.classes = num_class
		self.emb_size = emb_size
		self.batch_size = 32
		self.percent_labels = 0.8
		self.data = None
		self.no_label_data = None
		self.is_trained = is_trained

	def get_dataset(self, dir):
		dataset=[]
		for root, dirs, files in os.walk(dir):
			for f in files:
				fp = root + '/' + f
				if 'patterns' in root and fp.endswith('.pt'):
					dataset.append(fp)
		return dataset

	def train_local_graphs(self):
		self.data = self.get_dataset()
		self.data = graph_loader(self.data)
		self.data, self.no_label_data = train_test_split(self.data, train_size=self.percent_labels, shuffle=True)
		train_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
		train_loader.to(device)
		model = GraphEmbedding(embedding_size=self.emb_size, hidden_channels=128, num_classes=self.classes).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
		criterion = torch.nn.CrossEntropyLoss()
		model.train()
		epochs_stop = 3
		no_improve = 0
		acc_list = []
		epoch_min_loss = None
		start_epoch = 1
		for epoch in range(start_epoch, num_epochs):
			epoch_loss = []
			for batch in train_loader:
				optimizer.zero_grad()
				out = model(batch.x, batch.edge_index, batch.batch)
				loss = criterion(out, batch.y)
				total = batch.y.size(0)
				_, predicted = torch.max(out.data, 1)
				correct = (predicted == batch.y).sum().item()
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

	def find_knn(self, nodes):
		n1 = n[:len(n) // 2]
		n2 = n[len(n) // 2:]
		xn1 = [n.x for n in n1]
		xn2 = [n.x for n in n2]
		kn = knn(xn1, xn2, len(n1-1))
		return knn


	def generate_edges(self, graph):
		nodes_1 = [n for n in graph.nodes(data=True) if g['y'] == 1]
		nodes_2 = [n for n in graph.nodes(data=True) if g['y'] == 2]
		nodes_3 = [n for n in graph.nodes(data=True) if g['y'] == 3]
		nodes_4 = [n for n in graph.nodes(data=True) if g['y'] == 4]
		edges_1 = self.find_knn(nodes_1)
		edges_2 = self.find_knn(nodes_2)
		edges_3 = self.find_knn(nodes_3)
		edges_4 = self.find_knn(nodes_4)

	def generate_multigraph(self):
		model = torch.load(GRAPH_MODEL_PATH)
		model.eval()
		graph = nx.Graph()
		for d in self.data:
			gemb = model(d.x, d.edge_index, d.batch)
			graph.add_node(d.id, x=gemb, y=d.y)
		for l in self.no_label_data:
			gemb = model(d.x, d.edge_index, d.batch)
			graph.add_node(d.id, x=gemb, y=None, z=d.y)
		multi_graph = self.generate_edges(graph)