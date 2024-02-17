from torch_geometric.nn import knn
import torch
from tqdm import tqdm
from models.resnet import Resnet, Bottleneck
from transformers import AutoFeatureExtractor

#Path to the speech encoder, in this case Resnet, Whisper, Wav2vec.
SPEECH_MODEL_PATH='.'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def speech_encoder(SPEECH_MODEL_PATH, data):
	emb=[]
	if 'resblstm' in SPEECH_MODEL_PATH:
		model = Resnet(Bottleneck, [3, 6, 3])
		model.to(device)
		model.load_state_dict(SPEECH_MODEL_PATH)
		model.linear = torch.nn.Identity()
		model.eval()
		with torch.no_grad():
			for b in data:
				emb.append(model(b))
	elif 'wav2vec' or 'hubert' in SPEECH_MODEL_PATH:
		feature_extractor = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL_PATH)
	for b in data:
		emb.app(feature_extractor(b["audio"]["array"], return_tensors="pt").input_values)
	return emb
