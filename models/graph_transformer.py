import torch
import torch.nn as nn
import torch.nn.functional as F

#class GraphTransformerLayer(nn.Module):
#    def __init__(self, input_dim, hidden_dim):
#        super(GraphTransformerLayer, self).__init__()
#        self.linear = nn.Linear(input_dim, hidden_dim)
#        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
#        self.norm = nn.LayerNorm(hidden_dim)
#
#    def forward(self, x, edge_index):
#        x = self.linear(x)
#        x = torch.transpose(x, 0, 1)
#        x = self.norm(x)
#        x, _ = self.attention(x, x, x, key_padding_mask=None, attn_mask=None, need_weights=False)
#        x = torch.transpose(x, 0, 1)
#        return x
#
#class GraphTransformer(nn.Module):
#    def __init__(self, input_dim, hidden_dim, num_layers):
#        super(GraphTransformer, self).__init__()
#        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
#        self.output_linear = nn.Linear(hidden_dim, input_dim)
#
#    def forward(self, x, edge_index):
#        for layer in self.layers:
#            x = layer(x, edge_index)
#        x = self.output_linear(x)
#        return x
##
#input_dim = 1024
#hidden_dim = 256
#num_layers = 3

#model = GraphTransformer(input_dim, hidden_dim, num_layers)
#input_features = torch.randn(10, 100, input_dim)  # shape£º[batch_size, num_nodes, input_dim]
#edge_index = torch.tensor(...)  # edge of graph
#output_features = model(input_features, edge_index)  #


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_heads, hidden_dim):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.permute(1, 0, 2)  
        outputs = self.transformer_encoder(embedded)
        outputs = outputs.permute(1, 0, 2)
        outputs = self.fc(outputs)
        return outputs

##
#model = GraphTransformer(input_dim=1024, output_dim=1024, num_layers=2, num_heads=4, hidden_dim=512)
#
##
#inputs = torch.randn(10, 100, 1024)
#
## 
#outputs = model(inputs)
#
##
#print(outputs.shape)  # Êä³ö: torch.Size([10, 100, 1024])