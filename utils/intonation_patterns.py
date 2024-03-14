import os
import torch
import numpy as np
import parselmouth
from pydub import AudioSegment
from utils.gapbide import Gapbide
from utils.process_file import create_dictionary, create_nodes_dictionary
import uuid
import networkx as nx
from torch_geometric.utils import from_networkx
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from models.resnet import Resnet, Bottleneck
import torchaudio

SPEECH_MODEL_PATH = 'data/HuBERT'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {'ang':1, 'hap':2, 'neu':3, 'sad':4}

def generate_initial_graph(audio_dir):
	for emo in label2id.keys():
		filename = emo
		contours, files, pitches, inds= create_contours(audio_dir+emo+'/')
		pattern_length = 4
		Gapbide(contours, 4, 0, 0, pattern_length, audio_dir+emo+'/'+filename).run()
		dictionary = create_dictionary(audio_dir+emo+'/'+filename+'_intervals.txt')
		#create_patterns_audio_dataset(dictionary, contours, audio_dir+emo+'/', files)
		create_graph_of_audio_samples(dictionary, contours, files, pitches, inds, audio_dir+filename+'/', audio_dir+filename+'/patterns/', emo)
	print("Graph generation completed")
	return None


def create_contours(audio_dir):
	fqs, files, pitches = get_f0_praat(audio_dir)
	contours, inds = get_interval_contour(fqs)
	return contours, files, pitches, inds


def slice_audio(slice_from, slice_to, path, audio_file, path_out):
	audio = AudioSegment.from_wav(path)
	try:
		seg = audio[slice_from * 1000:slice_to * 1000]
		seg.set_channels(2)
		seg.export(path_out+audio_file, format="wav")
	except:
		print(f"ERROR PROCESSING AUDIO FILE: {path}")


#extract f0 from Parselmouth Praat function
def get_f0_praat(audio_dir):
	files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(audio_dir+f).to_pitch(time_step=0.025, pitch_floor=50.0, pitch_ceiling=650.0) for f in files]
	fqs = [pitch.kill_octave_jumps().selected_array['frequency'] for pitch in pitches]
	return fqs, files, pitches

#return a list of intervallic distances between F0 points expressed in cents
def get_interval_contour(fqs):
	contours = []
	inds= []

	for f in fqs:
		carry = 0
		direction = None
		contour = []
		ind = []
		for i in range(len(f)-1):
			if i < len(f):
				if f[i] == 0 and f[i+1] == 0:
					continue
				elif f[i] == 0 and f[i+1] != 0:
					continue
				elif f[i] != 0 and f[i+1] == 0:
					continue
				else:
					if f[i]-f[i+1]<0: new_dir = '-'
					if f[i]-f[i+1]>=0: new_dir = '+'
					dist_cents = 1200 * np.log2(f[i+1]/f[i])
					if carry > 0 : dist_cents +=carry
					dist = get_interval(dist_cents)
					if dist == '0':
						if direction == new_dir:
							carry += dist_cents
						else:
							direction = None
							carry = 0
					elif dist !='0':
						carry = 0
						if direction == new_dir:
							direction = direction
						else:
							direction = None
					contour.append(dist)
					ind.append((i, i+1))
		contours.append(contour)
		inds.append(ind)
	return contours, inds


def find_sublist(s,l):
	result=[]
	sll=len(s)
	for ind in (i for i,e in enumerate(l) if e==s[0]):
		if l[ind:ind+sll]==s:
			result.append((ind,ind+sll-1))
	return result


def get_interval(dist):
	i = abs(dist)
	if i < 50:
		if dist < 0:
			return '-1'
		elif dist == 0:
			return '0'
		else:
			return '1'
	elif i >= 50 and i < 100:
		if dist < 0:
			return '-2'
		else:
			return '2'
	elif i >= 100 and i < 150:
		if dist < 0:
			return '-3'
		else:
			return '3'
	elif i >= 150 and i < 200:
		if dist < 0:
			return '-4'
		else:
			return '4'
	elif i >= 200 and i < 250:
		if dist < 0:
			return '-5'
		else:
			return '5'
	elif i >= 250 and i < 300:
		if dist < 0:
			return '-6'
		else:
			return '6'
	elif i >= 300 and i < 350:
		if dist < 0:
			return '-7'
		else:
			return '7'
	elif i >= 350 and i < 400:
		if dist < 0:
			return '-8'
		else:
			return '8'
	elif i >= 400 and i < 450:
		if dist < 0:
			return '-9'
		else:
			return '9'
	elif i >= 450 and i < 500:
		if dist < 0:
			return '-10'
		else:
			return '10'
	elif i >= 500 and i < 550:
		if dist < 0:
			return '-11'
		else:
			return '11'
	else:
		if dist < 0:
			return '-12'
		else:
			return '12'


def create_graph(G, type='path'):
	if type == 'cycle':
		e = nx.cycle_graph(G)
	elif type == 'path':
		G = nx.path_graph(G)
	return G


def create_patterns_audio_dataset(dictionary, contours, path, files):
	for i, c in enumerate(contours):
		time = [t * 0.025 for t in range(1, len(c) + 1)]
		if not os.path.isdir(path+'patterns/'):
			os.makedirs(path+'patterns/')
		for d in dictionary:
			if len(d) > len(c):
				continue
			else:
				sub = find_sublist(d, c)
			if sub:
				for s in sub:
					name = files[i].replace('.wav', '_')+str(uuid.uuid4())+'.wav'
					slice_audio(time[s[0]], time[s[1]], path+files[i], name, path+'patterns/')
	print("Patterns generated")
	return None


def slice_audio(slice_from, slice_to, path, audio_file, path_out):
	audio = AudioSegment.from_wav(path)
	try:
		#we add 100 ms extra at the beginning and at the end.
		seg = audio[slice_from * 900:slice_to * 1100]
		seg.set_channels(2)
		seg.export(path_out + audio_file, format="wav")
	except:
		print(f"ERROR PROCESSING AUDIO FILE: {path}")


#Takes as the input a dictionary of (intonation) patterns and contours and slices audio files based on the patterns
# contained in the dictionary. At the same time it saves the co-occurrences of patterns in an adjacency list to
#create a graph.
def create_graph_of_audio_samples(dictionary, contours, files, pitches, inds, path, path_out_audio, emo):
	if not os.path.exists(path_out_audio):
		os.mkdir(path_out_audio)
	for i, c in enumerate(contours):
		G = nx.Graph()
		path2 = path+files[i]
		for d in dictionary:
			if len(d) > len(c):
				continue
			else:
				sub = find_sublist(d, c)
			if sub:
				for s in sub:
					name = files[i].replace('.wav', '_')+str(uuid.uuid4())+'.wav'
					ini = inds[i][s[0]][0]+1
					end = inds[i][s[1]][0]+1
					slice_audio(pitches[i].get_time_from_frame_number(ini), pitches[i].get_time_from_frame_number(end), path2, name, path_out_audio)
					G.add_node(name, id = name, labels=label2id[emo], node_audio=path_out_audio + name)
		if G.number_of_nodes()>0:
			path_graph = nx.path_graph(G)
			graph = from_networkx(path_graph)
			graph.id = [g[1]['id'] for g in G.nodes.data()]
			graph.y = torch.tensor([g[1]['labels'] for g in G.nodes.data()])
			graph_file = path_out_audio +files[i].replace('.wav', '')+ '.pt'
			if not os.path.exists(graph_file):
				graph.x = get_speech_representations(graph.id, path_out_audio)
				torch.save(graph, graph_file)
	return None

def get_acoustic_feat(audio_tensor):
	if 'resblstm' in SPEECH_MODEL_PATH:
		model = Resnet(Bottleneck, [3, 6, 3])
		model.to(device)
		model.load_state_dict(torch.load(SPEECH_MODEL_PATH, map_location=torch.device(device)))
		model.linear = torch.nn.Identity()
		model.eval()
		with torch.no_grad():
			emb = model(audio_tensor)
	elif 'wav2vec' in SPEECH_MODEL_PATH:
		model = Wav2Vec2Model.from_pretrained(SPEECH_MODEL_PATH).to(device)
		model.eval()
		feature_extractor = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL_PATH)
		feat = feature_extractor(audio_tensor, sampling_rate = feature_extractor.sampling_rate, max_length = 16000, padding = True,
							  truncation = True, return_tensors="pt")
		with torch.no_grad():
			output = model(feat.input_values)
		emb = output.last_hidden_state
	return emb

def get_speech_representations(data, path, max_len=84608):
	embeddings=[]
	data = [torchaudio.load(path+'/'+d)[0] for d in data]
	data = padding_tensor(data, max_len)
	for audio in data:
		outputs = get_acoustic_feat(audio)
		embeddings.append(outputs)
	out = torch.cat(embeddings)
	return out

def padding_tensor(sequences, max_len):
	"""
	input=list of tensors
	"""
	num = len(sequences)
	out_dims = (num, max_len)
	out_tensor = sequences[0].data.new(*out_dims).fill_(0)
	for i, tensor in enumerate(sequences):
		length = tensor.size(1)
		out_tensor[i, :length] = tensor
	return out_tensor

def get_audio_max_len(dir):
	max_len = 0
	for root, dirs, files in os.walk(dir):
		for f in files:
			fp = root + '/' + f
			if 'patterns' in root and fp.endswith('.wav'):
				audio_len = torchaudio.load(fp)[0].shape[1]
				if audio_len > max_len:
					max_len = audio_len
	return max_len