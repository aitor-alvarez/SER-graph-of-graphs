import torch

def graph_loader(graph_path):
    data_graph = []
    for p in graph_path:
        g = torch.load(p)
        if g.num_nodes > 1:
            x = torch.stack([g.x])
            g.x = x.squeeze(1)
            g.y = g.y[0]
            g.id = p.split('/')[-1].replace('.pt', '')
            data_graph.append(g)
    return data_graph
