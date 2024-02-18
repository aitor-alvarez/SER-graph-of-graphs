from torch_geometric.nn import knn
import torch
from tqdm import tqdm
from models.resnet import Resnet, Bottleneck
from transformers import AutoFeatureExtractor
import os

#Path to the speech encoder, in this case Resnet, Whisper, or wav2vec.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global graph
class MultiGraph:
	def __int__(self, speech_model_path, graph_dir):
		self.speech_encoder = speech_model_path
		self.graph_dir = graph_dir

	def get_dataset(self):
		for root, dirs, files in os.walk(data_path):
			for f in files:
				fp = root + '/' + f
				if 'patterns' in root and fp.endswith('.pt'):
					print(fp)



