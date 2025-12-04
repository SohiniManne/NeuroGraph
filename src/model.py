import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for inductive link prediction.
    Encodes node features into embeddings and decodes them into link probabilities.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # Layer 1: Aggregation + ReLU + Dropout
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        # Layer 2: Aggregation
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Dot product similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

def train_model(data, hidden_channels: int, epochs: int, update_callback=None):
    """
    Training pipeline with callback for UI progress updates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(data.num_features, hidden_channels, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simple train/val split logic (85/15)
    train_edge_index = data.edge_index[:, :int(0.85 * data.edge_index.size(1))].to(device)
    data = data.to(device)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, train_edge_index)
        
        # Positive & Negative Sampling
        pos_edge_index = train_edge_index
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1), method='sparse')

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            torch.ones(pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ], dim=0).to(device)

        out = model.decode(z, edge_label_index).view(-1)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update UI if callback provided
        if update_callback and epoch % 5 == 0:
            update_callback(epoch, epochs, loss.item())

    return model, losses