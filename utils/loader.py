import torch
import os

#Cleaning out the mess. Labels should start at index 0.
classes = {1:0, 2:1, 3:2, 4:3}
def load_graphs(dir):
    graphs=[]
    for root, dirs, files in os.walk(dir):
        for f in files:
            y = []
            emb=[]
            fp = root + '/' + f
            if 'patterns' in root and fp.endswith('.pt'):
                g = torch.load(fp)
                for k in range(g.num_nodes):
                    emb.append(g.x[k].view(g.x[k].shape[0] * g.x[k].shape[1]))
                    y.append(classes[int(g.y[k])])
                g.x = torch.stack(emb)
                g.y = torch.tensor(y)
                graphs.append(g)
    return graphs