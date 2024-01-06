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


def generate_dataset(audio_dir, emo='neutral'):
	filename = emo
	contours, files, pitches, inds= create_contours(audio_dir)
	pattern_length = 8
	Gapbide(contours, 12, 0, 0, pattern_length, filename).run()
	dictionary = create_dictionary(filename+'_intervals.txt')
	create_patterns_audio_dataset(dictionary, contours, audio_dir, files)
	print("Dataset generation completed")
	return None


def create_contours(audio_dir):
	fqs, files, pitches = get_f0_praat(audio_dir)
	contours, inds = get_interval_contour(fqs)
	return contours, files, pitches, inds

###Creates a graph based on the prosodic similarity of the speech utterances.
def generate_graph(contours, files):
	dictionary = create_nodes_dictionary('patterns/train/')
	G = nx.Graph()
	node_list=[]
	for d in dictionary:
		nodes = []
		for i, c in enumerate(contours):
			nodename = files[i]
			if len(d) > len(c):
				continue
			else:
				sub = find_sublist(d, c)
			if sub:
				nodes.append(nodename)
				G.add_node(nodename, node_id=nodename, y=nodename[nodename.rfind('/')-3:nodename.rfind('/')])
		g = nx.Graph()
		g.add_nodes_from(nodes)
		sg= create_graph(g)
		G.add_edges_from(sg.edges, weight=1.00)
		node_list.append(nodes)
	graph = add_edge_attributes(G, node_list)
	gp= from_networkx(graph)
	torch.save(gp, 'patterns/graph.pt')


def add_edge_attributes(G, nodes):
	for e in G.edges:
		for n in nodes:
			if e[0] and e[1] in n:
				if 'weight' in G[e[0]][e[1]]:
					G[e[0]][e[1]]['weight'] +=1
	return G


def slice_audio(slice_from, slice_to, path, audio_file, path_out):
	audio = AudioSegment.from_wav(path)
	try:
		seg = audio[slice_from * 1000:slice_to * 1000]
		seg.set_channels(2)
		seg.export(path_out+audio_file, format="mp3")
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
	carry =0
	for f in fqs:
		contour = []
		ind = []
		for i in range(len(f)-1):
			if i < len(f):
				if f[i] == 0 and f[i+1] == 0:
					continue
				else:
					dist = 1200 * np.log2(f[i+1]/f[i])
					dist = get_interval(dist)
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


def create_graph(G, type='cycle'):
	if type == 'cycle':
		e = nx.cycle_graph(G)
	G.add_edges_from(e.edges)
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
					name = files[i].replace('.wav', '_')+str(uuid.uuid4())+'.mp3'
					slice_audio(time[s[0]], time[s[1]], path+files[i], name, path+'patterns/')
	print("Patterns generated")
	return None


def slice_audio(slice_from, slice_to, path, audio_file, path_out):
	audio = AudioSegment.from_wav(path)
	try:
		#we add 100 ms extra at the beginning and at the end.
		seg = audio[slice_from * 900:slice_to * 1100]
		seg.set_channels(2)
		seg.export(path_out + audio_file, format="mp3")
	except:
		print(f"ERROR PROCESSING AUDIO FILE: {path}")