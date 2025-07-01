
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.attention.performer import generalized_kernel
from torch_geometric.nn import GATv2Conv

# ---------------------------------------------------------------------------------------------------------------------------


class GAT_model(nn.Module):
    """Graph Attention Network (GAT) model with positional encoding and multiple layers.
    This model uses GATv2Conv layers with pre-normalization and a feed-forward network.
    Args:
        in_channels (int): Number of input features per node.
        pe_input_dim (int): Dimension of the positional encoding input.
        pe_dim (int): Dimension of the positional encoding output.
        hidden_channels (int): Number of hidden channels in the GAT layers.
        heads (int): Number of attention heads in GAT layers.
        num_layers (int): Number of GAT layers to stack.
        dropout (float): Dropout rate applied in GAT layers and feed-forward network.
    """
    def __init__(
        self,
        in_channels: int,
        pe_input_dim: int,
        pe_dim: int,
        hidden_channels: int,
        heads: int,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        # Input embedding
        self.node_emb = nn.Linear(in_channels, hidden_channels - pe_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_input_dim)
        
        self.gat_norm = nn.LayerNorm(hidden_channels)
        # GAT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Pre-normalization
                'norm': nn.LayerNorm(hidden_channels),
                
                # GAT component
                'gat': GATv2Conv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    concat=True, 
                    dropout=dropout
                ),

                #Normalization
                'gat_norm': nn.LayerNorm(hidden_channels*heads),
                
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(hidden_channels*heads, hidden_channels*2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_channels*2, hidden_channels)
                )
            })
            self.layers.append(layer)
        
        # Output layer for prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, pe, edge_index):
        # Initial embedding
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        
        # Process through GAT layers
        for layer in self.layers:
            # Pre-normalization
            x_norm = layer['norm'](x)
            x_gat = layer['gat'](x_norm, edge_index)
            x_gat = layer['gat_norm'](x_gat)
            x_ff = layer['ffn'](x_gat)
            
            # Feed-forward network with residual
            x = x + x_ff
        
        # Final prediction
        return self.output_layer(x)
    
    def get_attention_weights(self, x, pe, edge_index, layer_idx=0):
        """Extract the attention weights from GATv2Conv for a specific layer.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            layer_idx: Index of the layer to extract attention from (default: 0)
            
        Returns:
            Tuple of (edge_index, attention_weights) where attention_weights has
            shape [num_edges, heads]
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            # Process through previous layers
            for i in range(layer_idx):
                layer = self.layers[i]
                x_norm = layer['norm'](x)
                x_gat = layer['gat'](x_norm, edge_index)
                x_gat = layer['gat_norm'](x_gat)
                x_ff = layer['ffn'](x_gat)
                x = x + x_ff
            
            # Get the target layer
            layer = self.layers[layer_idx]
            x_norm = layer['norm'](x)
            
            # Return the edge_index and attention weights
            # Note: For GATv2Conv, we need to run a forward pass and 
            # access the _alpha attribute to get attention weights
            _, (att_edge_ind, attention_weights)= layer['gat'](x_norm, edge_index, return_attention_weights=True)
            
            # The attention weights will have shape [num_edges, heads]
            return att_edge_ind, attention_weights


# ---------------------------------------------------------------------------------------------------------------------------


class PerformerModel(nn.Module):
    """Performer model with positional encoding and multiple layers.
    This model uses PerformerAttention layers with pre-normalization and a feed-forward network.
    Args:        
        in_channels (int): Number of input features per node.
        pe_input_dim (int): Dimension of the positional encoding input.
        pe_dim (int): Dimension of the positional encoding output.
        hidden_channels (int): Number of hidden channels in the Performer layers.       
        heads (int): Number of attention heads in Performer layers.
        head_channels (int): Number of channels per attention head.
        num_layers (int): Number of Performer layers to stack.
        dropout (float): Dropout rate applied in Performer layers and feed-forward network.
    """

    def __init__(
        self,
        in_channels: int,
        pe_input_dim: int,
        pe_dim: int,
        hidden_channels: int,
        heads: int,
        head_channels: int,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        # Input embedding
        self.node_emb = nn.Linear(in_channels, hidden_channels - pe_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_input_dim)
        
        self.trans_norm = nn.LayerNorm(hidden_channels)
        # Performer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Pre-normalization
                'norm': nn.LayerNorm(hidden_channels),
                
                # Performer attention component only (no GAT)
                'attention': PerformerAttention(
                    channels=hidden_channels,
                    heads=heads,
                    head_channels=head_channels,
                    kernel=torch.nn.ReLU(),
                    dropout=dropout
                ),
                # Normalization after attention
                
                'trans_norm': nn.LayerNorm(hidden_channels),
                
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels*2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_channels*2, hidden_channels)
                )
            })
            self.layers.append(layer)
        
        # Output layer for prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, pe, edge_index=None, batch=None):
        # Initial embedding
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        
        # Process through Performer layers
        for layer in self.layers:
            # Pre-normalization
            x_norm = layer['norm'](x)
            
            # Performer attention
            x_perf = x_norm.unsqueeze(0)
            x_perf = layer['attention'](x_perf).squeeze(0)
            x_perf = layer['trans_norm'](x_perf)
            
            # Residual connection (simpler now - no need to combine GAT outputs)
            x_combined = x_perf + x
            
            # Feed-forward network with residual
            x = x_combined + layer['ffn'](x_combined)
        
        # Final prediction
        return self.output_layer(x)
    
    def get_performer_attention(self, x, pe, edge_index=None, layer_idx=0, mask=None):
        """Extract the attention approximation used by Performer for a specific layer."""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            # Process through previous layers
            for i in range(layer_idx):
                layer = self.layers[i]
                x_norm = layer['norm'](x)
                x_trans = x_norm.unsqueeze(0)
                x_trans = layer['attention'](x_trans).squeeze(0)
                x_combined = x_trans + x
                x = x_combined + layer['ffn'](x_combined)
            
            # Get the target layer
            layer = self.layers[layer_idx]
            x_norm = layer['norm'](x)
            x = x_norm.unsqueeze(0)
            
            # Get Q, K, V projections
            attention = layer['attention']
            B, N, C = x.shape
            q, k, v = attention.q(x), attention.k(x), attention.v(x)
            
            # Reshape to multi-head format
            q, k, v = map(
                lambda t: t.reshape(B, N, attention.heads, attention.head_channels).permute(
                    0, 2, 1, 3), (q, k, v))
            
            # Apply the kernel approximation to q and k
            fast_attn = attention.fast_attn
            q_prime = generalized_kernel(q, fast_attn.projection_matrix, fast_attn.kernel)
            k_prime = generalized_kernel(k, fast_attn.projection_matrix, fast_attn.kernel)
            
            # Compute approximate attention matrix
            k_prime_sum = k_prime.sum(dim=-2, keepdim=True)
            
            # For explicit attention weights
            attention_weights = []
            for h in range(attention.heads):
                head_q = q_prime[0, h]
                head_k = k_prime[0, h]
                
                attn_mat = torch.zeros(N, N, device=q.device)
                for i in range(N):
                    q_i = head_q[i:i+1]
                    attn_i = (q_i @ head_k.transpose(-1, -2)) / (q_i @ k_prime_sum[0, h].transpose(-1, -2))
                    attn_mat[i] = attn_i
                
                attention_weights.append(attn_mat)
            
            return torch.stack(attention_weights)  # [heads, N, N]

    def redraw_projection_matrix(self):
        """Redraw projection matrices in all attention layers."""
        for layer in self.layers:
            layer['attention'].redraw_projection_matrix()

# ---------------------------------------------------------------------------------------------------------------------------


class HybridModelv2(nn.Module):
    """Hybrid model combining Performer and GATv2Conv layers with positional encoding.
    This model uses PerformerAttention for global context and GATv2Conv for local structure.
    Args:
        in_channels (int): Number of input features per node.
        pe_input_dim (int): Dimension of the positional encoding input.
        pe_dim (int): Dimension of the positional encoding output.
        hidden_channels (int): Number of hidden channels in the Performer and GAT layers.
        heads (int): Number of attention heads in Performer and GAT layers.
        head_channels (int): Number of channels per attention head in Performer.
        num_layers (int): Number of layers to stack for both Performer and GAT.
        dropout (float): Dropout rate applied in Performer and GAT layers and feed-forward network.
        combine_method (str): Method to combine outputs from Performer and GAT layers ('sum' or 'concat').
    """

    def __init__(
        self,
        in_channels: int,
        pe_input_dim: int,
        pe_dim: int,
        hidden_channels: int,
        heads: int,
        head_channels: int,
        num_layers: int = 3,
        dropout: float = 0.4,
        combine_method: str = 'sum',  # 'sum', 'concat'
    ):
        super().__init__()
        
        # Input embedding
        self.node_emb = nn.Linear(in_channels, hidden_channels - pe_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_input_dim)

        # GPS layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Pre-normalization
                'norm': nn.LayerNorm(hidden_channels),
                
                # Transformer component
                'attention': PerformerAttention(
                    channels=hidden_channels,
                    heads=heads,
                    head_channels=head_channels,
                    kernel=torch.nn.ReLU(),
                    dropout=dropout
                ),
                
                # GAT component
                'gat': GATv2Conv(hidden_channels, hidden_channels // heads, 
                               heads=heads, concat=True, dropout=dropout),

                # Normalization
                'trans_norm': nn.LayerNorm(hidden_channels),
                'gat_norm': nn.LayerNorm(hidden_channels),
                
                # Combination method
                'combine': self._make_combiner(combine_method, hidden_channels),
                
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels*2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_channels*2, hidden_channels)
                )
            })
            self.layers.append(layer)
        
        # Output layer for edge prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def _make_combiner(self, method, hidden_channels):
        if method == 'sum':
            return nn.Identity()
        elif method == 'concat':
            return nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LeakyReLU()
            )
        
    
    def forward(self, x, pe, edge_index, batch=None):
        # Initial embedding
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        
        # Process through GPS layers
        for layer in self.layers:
            # Pre-normalization
            x_norm = layer['norm'](x)
            
            # Transformer component
            x_trans = x_norm.unsqueeze(0)
            x_trans = layer['attention'](x_trans).squeeze(0)
            x_trans = layer['trans_norm'](x_trans)
            
            
            # GAT component
            x_gat = layer['gat'](x_norm, edge_index)
            x_gat = layer['gat_norm'](x_gat)
            
            
            # Combine the outputs with residual connection
            if isinstance(layer['combine'], nn.Identity):  # Sum
                x_combined = x_trans + x_gat + x  # Residual
            elif isinstance(layer['combine'], nn.Sequential):  # Concat
                x_combined = layer['combine'](torch.cat([x_trans, x_gat], dim=1)) + x
            
            # Feed-forward network with residual
            x = x_combined + layer['ffn'](x_combined)

        
        return self.output_layer(x)
    
       

    def get_performer_attention(self, x, pe, edge_index, layer_idx=0, mask=None):
        """Extract the attention approximation used by Performer for a specific layer.
        
        Makes more sense to use this method for interpretability once the model is trained.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            layer_idx: Index of the layer to extract attention from (default: 0)
            mask: Optional mask
            
        Returns:
            Tensor of shape [heads, num_nodes, num_nodes] representing attention weights
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            # Process through previous layers
            for i in range(layer_idx):
                layer = self.layers[i]
                # Pre-normalization
                x_norm = layer['norm'](x)
                
                # Transformer component
                x_trans = x_norm.unsqueeze(0)
                x_trans = layer['attention'](x_trans).squeeze(0)
                x_trans = layer['trans_norm'](x_trans)
                
                # GAT component
                x_gat = layer['gat'](x_norm, edge_index)
                x_gat = layer['gat_norm'](x_gat)
                
                # Combine the outputs with residual connection
                if isinstance(layer['combine'], nn.Identity):  # Sum
                    x_combined = x_trans + x_gat + x  # Residual
                elif isinstance(layer['combine'], nn.Sequential):  # Concat
                    x_combined = layer['combine'](torch.cat([x_trans, x_gat], dim=1)) + x
                
                # Feed-forward network with residual
                x = x_combined + layer['ffn'](x_combined)
            
            # Now x contains the input to the target layer
            # Get the target layer
            layer = self.layers[layer_idx]
            
            # Apply layer norm
            x_norm = layer['norm'](x)
            
            # Add batch dimension for attention
            x = x_norm.unsqueeze(0)
            
            # Get Q, K, V projections
            attention = layer['attention']
            B, N, C = x.shape
            q, k, v = attention.q(x), attention.k(x), attention.v(x)
            
            # Reshape to multi-head format
            q, k, v = map(
                lambda t: t.reshape(B, N, attention.heads, attention.head_channels).permute(
                    0, 2, 1, 3), (q, k, v))
            
            # Apply the kernel approximation to q and k
            fast_attn = attention.fast_attn
            q_prime = generalized_kernel(q, fast_attn.projection_matrix, fast_attn.kernel)
            k_prime = generalized_kernel(k, fast_attn.projection_matrix, fast_attn.kernel)
            
            # Compute approximate attention matrix
            # This is analogous to softmax(QK^T) in standard attention
            k_prime_sum = k_prime.sum(dim=-2, keepdim=True)  # Sum over nodes
            
            # For explicit attention weights (inefficient but interpretable)
            attention_weights = []
            for h in range(attention.heads):
                head_q = q_prime[0, h]  # [N, features]
                head_k = k_prime[0, h]  # [N, features]
                
                # Compute normalized attention (shape: [N, N])
                attn_mat = torch.zeros(N, N, device=q.device)
                for i in range(N):
                    q_i = head_q[i:i+1]  # [1, features]
                    # Normalized kernel dot product for each node pair
                    attn_i = (q_i @ head_k.transpose(-1, -2)) / (q_i @ k_prime_sum[0, h].transpose(-1, -2))
                    attn_mat[i] = attn_i
                
                attention_weights.append(attn_mat)
            
            return torch.stack(attention_weights)  # [heads, N, N]

    def get_attention_weights(self, x, pe, edge_index, layer_idx=0, mask=None):
        """Extract the attention weights from GATv2Conv for a specific layer.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            layer_idx: Index of the layer to extract attention from (default: 0)
            mask: Optional mask
            
        Returns:
            Tuple of (edge_index, attention_weights) where attention_weights has
            shape [num_edges+nodes, heads]
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            # Process through previous layers
            for i in range(layer_idx):
                layer = self.layers[i]
                # Pre-normalization
                x_norm = layer['norm'](x)
                
                # Transformer component
                x_trans = x_norm.unsqueeze(0)
                x_trans = layer['attention'](x_trans).squeeze(0)
                x_trans = layer['trans_norm'](x_trans)
                
                # GAT component
                x_gat = layer['gat'](x_norm, edge_index)
                x_gat = layer['gat_norm'](x_gat)
                
                # Combine the outputs with residual connection
                if isinstance(layer['combine'], nn.Identity):  # Sum
                    x_combined = x_trans + x_gat + x  # Residual
                elif isinstance(layer['combine'], nn.Sequential):  # Concat
                    x_combined = layer['combine'](torch.cat([x_trans, x_gat], dim=1)) + x
                
                # Feed-forward network with residual
                x = x_combined + layer['ffn'](x_combined)
            
            # Get the target layer
            layer = self.layers[layer_idx]
            x_norm = layer['norm'](x)
            
            # Return the edge_index and attention weights
            # For GATv2Conv, we use return_attention_weights=True to get attention weights
            _, (att_edge_ind, attention_weights) = layer['gat'](x_norm, edge_index, return_attention_weights=True)
            
            # The attention weights will have shape [num_edges, heads]
            return att_edge_ind, attention_weights

    
    def redraw_projection_matrix(self):
        """Redraw projection matrices in all attention layers."""
        for layer in self.layers:
            layer['attention'].redraw_projection_matrix()







# ---------------------------------------------------------------------------------------------------------------------------

class HybridModel_variant(nn.Module):
    """Hybrid model combining Performer and GATv2Conv layers.
    This model uses PerformerAttention for global context and GATv2Conv for local structure.
    Args:
        in_channels (int): Number of input features per node.
        pe_input_dim (int): Dimension of the positional encoding input.
        pe_dim (int): Dimension of the positional encoding output.
        hidden_channels (int): Number of hidden channels in the Performer and GAT layers.
        heads (int): Number of attention heads in Performer and GAT layers.
        head_channels (int): Number of channels per attention head in Performer.
        num_transformer_layers (int): Number of Performer layers to stack.
        num_gat_layers (int): Number of GAT layers to stack.
        dropout (float): Dropout rate applied in Performer and GAT layers and feed-forward network.
        combine_method (str): Method to combine outputs from Performer and GAT layers ('sum' or 'concat').
    """
    def __init__(
        self,
        in_channels: int,
        pe_input_dim: int,
        pe_dim: int,
        hidden_channels: int,
        heads: int,
        head_channels: int,
        num_transformer_layers: int = 2,
        num_gat_layers: int = 2,
        dropout: float = 0.4,
        combine_method: str = 'concat',  # 'concat', 'sum'
    ):
        super().__init__()
        
        # Shared input embedding
        self.node_emb = nn.Linear(in_channels, hidden_channels - pe_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_input_dim)
        
        # Transformer branch (DeepGraphPerformer)
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_channels),
                'attention': PerformerAttention(
                    channels=hidden_channels,
                    heads=heads,
                    head_channels=head_channels,
                    kernel=torch.nn.ReLU(),
                    dropout=dropout
                ),
                'norm2': nn.LayerNorm(hidden_channels),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels*2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_channels*2, hidden_channels)
                )
            })
            self.transformer_layers.append(layer)
        
        # GAT branch - improved with proper normalization and FFN
        self.gat_layers = nn.ModuleList()
        for _ in range(num_gat_layers):
            layer = nn.ModuleDict({
                # Pre-normalization
                'norm': nn.LayerNorm(hidden_channels),
                
                # GAT component
                'gat': GATv2Conv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    concat=True, 
                    dropout=dropout
                ),
                
                # Post-normalization
                'gat_norm': nn.LayerNorm(hidden_channels*heads),
                
                # Feed-forward network
                'ffn': nn.Sequential(
                    nn.Linear(hidden_channels*heads, hidden_channels*2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_channels*2, hidden_channels)
                )
            })
            self.gat_layers.append(layer)
        
        # Layer normalization for transformer and GAT outputs before combining
        self.transformer_norm = nn.LayerNorm(hidden_channels)
        self.gat_norm = nn.LayerNorm(hidden_channels)
        
        # Combination method
        self.combine_method = combine_method
        if combine_method == 'concat':
            self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Output layer for node prediction
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, pe, edge_index, branch='both'):
        """
        Forward pass with option to use specific branches.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            branch: Which branch to use ('both', 'gat', or 'performer')
        """
        # Initial embedding
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        
        if branch in ['both', 'performer']:
            # Transformer branch
            transformer_x = x
            for layer in self.transformer_layers:
                # Attention block with residual
                residual = transformer_x
                transformer_x_norm = layer['norm1'](transformer_x)
                transformer_x_attn = transformer_x_norm.unsqueeze(0)
                transformer_x_attn = layer['attention'](transformer_x_attn)
                transformer_x = transformer_x_attn.squeeze(0) + residual
                
                # FFN block with residual
                residual = transformer_x
                transformer_x = layer['norm2'](transformer_x)
                transformer_x = layer['ffn'](transformer_x) + residual
            
            transformer_x = self.transformer_norm(transformer_x)
        
        if branch in ['both', 'gat']:
            # GAT branch - improved with proper residuals and FFN
            gat_x = x
            for layer in self.gat_layers:
                # Pre-normalization
                gat_x_norm = layer['norm'](gat_x)
                
                # Apply GAT 
                gat_x_attn = layer['gat'](gat_x_norm, edge_index)
                gat_x_attn = layer['gat_norm'](gat_x_attn)
                
                # Apply FFN with residual
                gat_x_ff = layer['ffn'](gat_x_attn)
                
                # Add residual connection from before GAT (matches GAT_model design)
                gat_x = gat_x + gat_x_ff
            
            gat_x = self.gat_norm(gat_x)
        
        # Combine or use individual branch based on parameter
        if branch == 'both':
            if self.combine_method == 'sum':
                combined_x = transformer_x + gat_x
            elif self.combine_method == 'concat':
                combined_x = torch.cat([transformer_x, gat_x], dim=1)
                combined_x = F.LeakyReLU(self.combine(combined_x))
        elif branch == 'performer':
            combined_x = transformer_x
        elif branch == 'gat':
            combined_x = gat_x
        
        # Final prediction
        return self.output_layer(combined_x)
    
    def redraw_projection_matrix(self):
        """Redraw projection matrices in all attention layers."""
        for layer in self.transformer_layers:
            attention = layer['attention']
            attention.redraw_projection_matrix()
    
    def get_performer_attention(self, x, pe, edge_index, layer_idx=0, mask=None):
        """Extract the attention approximation used by Performer for a specific layer.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            layer_idx: Index of the layer to extract attention from (default: 0)
            mask: Optional mask
            
        Returns:
            Tensor of shape [heads, num_nodes, num_nodes] representing attention weights
        """
        if layer_idx < 0 or layer_idx >= len(self.transformer_layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.transformer_layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            transformer_x = x
            
            # Process through previous transformer layers
            for i in range(layer_idx):
                layer = self.transformer_layers[i]
                # Attention block with residual
                residual = transformer_x
                transformer_x_norm = layer['norm1'](transformer_x)
                transformer_x_attn = transformer_x_norm.unsqueeze(0)
                transformer_x_attn = layer['attention'](transformer_x_attn)
                transformer_x = transformer_x_attn.squeeze(0) + residual
                
                # FFN block with residual
                residual = transformer_x
                transformer_x = layer['norm2'](transformer_x)
                transformer_x = layer['ffn'](transformer_x) + residual
            
            # Get the target layer
            layer = self.transformer_layers[layer_idx]
            transformer_x_norm = layer['norm1'](transformer_x)
            
            # Add batch dimension for attention
            x = transformer_x_norm.unsqueeze(0)
            
            # Get Q, K, V projections
            attention = layer['attention']
            B, N, C = x.shape
            q, k, v = attention.q(x), attention.k(x), attention.v(x)
            
            # Reshape to multi-head format
            q, k, v = map(
                lambda t: t.reshape(B, N, attention.heads, attention.head_channels).permute(
                    0, 2, 1, 3), (q, k, v))
            
            # Apply the kernel approximation to q and k
            fast_attn = attention.fast_attn
            q_prime = generalized_kernel(q, fast_attn.projection_matrix, fast_attn.kernel)
            k_prime = generalized_kernel(k, fast_attn.projection_matrix, fast_attn.kernel)
            
            # Compute approximate attention matrix
            k_prime_sum = k_prime.sum(dim=-2, keepdim=True)
            
            # For explicit attention weights
            attention_weights = []
            for h in range(attention.heads):
                head_q = q_prime[0, h]  # [N, features]
                head_k = k_prime[0, h]  # [N, features]
                
                # Compute normalized attention (shape: [N, N])
                attn_mat = torch.zeros(N, N, device=q.device)
                for i in range(N):
                    q_i = head_q[i:i+1]  # [1, features]
                    # Normalized kernel dot product for each node pair
                    attn_i = (q_i @ head_k.transpose(-1, -2)) / (q_i @ k_prime_sum[0, h].transpose(-1, -2))
                    attn_mat[i] = attn_i
                
                attention_weights.append(attn_mat)
            
            return torch.stack(attention_weights)  # [heads, N, N]
    
    def get_attention_weights(self, x, pe, edge_index, layer_idx=0):
        """Extract the attention weights from GATv2Conv for a specific layer.
        
        Args:
            x: Node features
            pe: Positional encoding
            edge_index: Graph connectivity
            layer_idx: Index of the layer to extract attention from (default: 0)
            
        Returns:
            Tuple of (edge_index, attention_weights) where attention_weights has
            shape [num_edges, heads]
        """
        if layer_idx < 0 or layer_idx >= len(self.gat_layers):
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.gat_layers)-1})")
        
        self.eval()
        with torch.no_grad():
            # Initial embedding
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            gat_x = x
            
            # Process through previous layers using the proper GAT layer structure
            for i in range(layer_idx):
                layer = self.gat_layers[i]
                # Pre-normalization
                gat_x_norm = layer['norm'](gat_x)
                
                # Apply GAT 
                gat_x_attn = layer['gat'](gat_x_norm, edge_index)
                gat_x_attn = layer['gat_norm'](gat_x_attn)
                
                # Apply FFN with residual
                gat_x_ff = layer['ffn'](gat_x_attn)
                
                # Add residual connection from before GAT
                gat_x = gat_x + gat_x_ff
            
            # Get the target layer
            layer = self.gat_layers[layer_idx]
            gat_x_norm = layer['norm'](gat_x)
            
            # Return the edge_index and attention weights for the target layer
            # For GATv2Conv, we need to use return_attention_weights=True
            _, (att_edge_ind, attention_weights) = layer['gat'](
                gat_x_norm, edge_index, return_attention_weights=True)
            
            # The attention weights will have shape [num_edges, heads]
            return att_edge_ind, attention_weights
        

# ---------------------------------------------------------------------------------------------------------------------------

def model_builder(model_name, in_channels, pe_input_dim, pe_dim, hidden_channels, heads, num_layers, dropout):
    if model_name == 'GATv2':
        return GAT_model(
            in_channels=in_channels,
            pe_input_dim=pe_input_dim,
            pe_dim=pe_dim,
            hidden_channels=hidden_channels,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_name == 'Performer':
        return PerformerModel(
            in_channels=in_channels,
            pe_input_dim=pe_input_dim,
            pe_dim=pe_dim,
            hidden_channels=hidden_channels,
            heads=heads,
            head_channels=hidden_channels // heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_name == 'Hybrid':
        return HybridModel_variant(
            in_channels=in_channels,
            pe_input_dim=pe_input_dim,
            pe_dim=pe_dim,
            hidden_channels=hidden_channels,
            heads=heads,
            head_channels=hidden_channels // heads,
            num_transformer_layers=num_layers // 2,
            num_gat_layers=num_layers // 2,
            dropout=dropout,
            combine_method='sum'
        )