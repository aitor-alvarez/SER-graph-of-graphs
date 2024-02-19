import os
import torch

def graph_loader(graph_path):
    data = []
    for p in graph_path:
        g = torch.load(p)
        if g.num_nodes > 1:
            x = torch.stack(g.x)
            g.x = x.squeeze(1)
            g.y = g.y[0]
            g.id = f.replace('.pt', '')
            data.append(g)
    return data