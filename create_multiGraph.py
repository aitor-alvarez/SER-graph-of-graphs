from torch_geometric.nn import knn
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from models.resnet import Resnet, Bottleneck
from models.GraphEmbedding import GraphEmbedding
from transformers import AutoFeatureExtractor
import os

#Path to the speech encoder, in this case Resnet, Whisper, or wav2vec.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODEL_PATH = 'trained/local_graph_embedding.pt'

#Global graph
class MultiGraph:
	def __int__(self, speech_model_path, graph_test, graph_train, num_class, emb_size):
		self.speech_encoder = speech_model_path
		self.graph_train = graph_train
		self.graph_test = graph_test
		self.classes = num_class
		self.emb_size = emb_size
		self.batch_size = 32
		self.percent_labels = 80

	def get_dataset(self):
		dataset=[]
		for root, dirs, files in os.walk(self.graph_dir):
			for f in files:
				fp = root + '/' + f
				if 'patterns' in root and fp.endswith('.pt'):
					dataset.append(fp)
		return dataset

	def train_local_graphs(self):
		data = self.get_dataset()
		train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True).to(device)
		model = GraphEmbedding(embedding_size=self.emb_size, hidden_channels=128, num_classes=self.classes).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
		criterion = torch.nn.CrossEntropyLoss()
		model.train()
		epochs_stop = 3
		min_loss = None
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






