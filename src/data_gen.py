import torch
import networkx as nx
from torch_geometric.data import Data
import random

def generate_social_graph(num_nodes: int = 100, num_features: int = 16):
    """
    Generates a synthetic scale-free social network (Barabasi-Albert model).
    Returns:
        data: PyG Data object
        G_nx: NetworkX graph object
        names: List of fake user IDs
    """
    # 1. Structure: Scale-free graph (common in social networks)
    G_nx = nx.barabasi_albert_graph(num_nodes, 3) 
    
    # 2. Features: Random latent vectors (representing age, location, interests)
    x = torch.randn((num_nodes, num_features))
    
    # 3. Connectivity: Convert to PyTorch tensor
    edge_index = torch.tensor(list(G_nx.edges)).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
    
    data = Data(x=x, edge_index=edge_index)
    
    # 4. Metadata
    names = [f"User_{i}" for i in range(num_nodes)]
    
    return data, G_nx, names