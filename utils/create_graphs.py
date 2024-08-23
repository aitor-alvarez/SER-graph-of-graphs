import os
from utils.segmentation import segment_utterance
from torch_geometric.utils import from_networkx
from transformers import (AutoConfig, Wav2Vec2FeatureExtractor,
                          TrainingArguments, Trainer, AutoFeatureExtractor)
from models.transformer_speech import HubertEmotion, Wav2VecEmotion
import networkx as nx
import torch
import torchaudio

label2id = {'ang':1, 'hap':2, 'neu':3, 'sad':4}

id2label = {1: 'ang', 2: 'hap', 3: 'neu', 4: 'sad'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEECH_MODEL_PATH = 'data/wav2vec'

def get_acoustic_feat(audio_file):
    audio_tensor = torchaudio.load(audio_file)
    num_labels = len(label2id)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=SPEECH_MODEL_PATH,
                                        num_labels=num_labels, label2id=label2id, id2label=id2label)
    if 'hubert' in SPEECH_MODEL_PATH:
        model = HubertEmotion.from_pretrained(SPEECH_MODEL_PATH, config=config)
    elif 'wav2vec' in SPEECH_MODEL_PATH:
        model = Wav2VecEmotion.from_pretrained(SPEECH_MODEL_PATH, config=config)
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL_PATH)
    feat = feature_extractor(audio_tensor, sampling_rate = feature_extractor.sampling_rate, max_length = 16000, padding = True,
							  truncation = True, return_tensors="pt")
    with torch.no_grad():
        output = model(feat.input_values)
    emb = output.last_hidden_state
    return emb

def generate_graphs(audio_dir):
    for emo in label2id.keys():
        files = os.listdir(audio_dir+emo+'/')
        for f in files:
            segment_utterance(audio_dir+emo+'/'+f)
            G = nx.Graph()
            segs = os.listdir('tmp/')
            for s in segs:
                G.add_node(s.split('.mp3')[0], id = s.split('.mp3')[0], y=torch.tensor(label2id[emo]))
            if G.number_of_nodes()>0:
                path_graph = nx.path_graph(G)
                graph = from_networkx(path_graph)
                graph.x = get_acoustic_feat()
                graph.id = [g[1]['id'] for g in G.nodes.data()]
                graph.y = torch.tensor([g[1]['y'] for g in G.nodes.data()])
                graph_file = audio_dir + f.replace('.wav', '') + '.pt'
                if not os.path.exists(graph_file):
                    torch.save(graph, graph_file)
    print("Graphs creation process completed")
    return None
