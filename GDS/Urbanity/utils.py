from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import geopandas as gpd
from shapely.geometry import LineString
from torch_geometric.data import Data
import numpy as np
import seaborn as sns




def evaluate_and_visualize_predictions(model, CompleteData, path, h3_nodes ,criterion , branch = None):
    if branch is None:
        model.load_state_dict(torch.load(path))
    if branch is not None:
        test_loss = evaluate_hybrid(model, CompleteData, CompleteData.test_mask, criterion ,branch )
    else:
        test_loss = evaluate(model, CompleteData, CompleteData.test_mask, criterion)
    print(f'Final Test Results - Loss: {test_loss:.4f}')
# Generate predictions for all nodes
    model.eval()
    with torch.no_grad():
        if branch is not None:
            predictions = model(CompleteData.x, CompleteData.pe, CompleteData.edge_index, branch ).squeeze().cpu().numpy()
        else:
            predictions = model(CompleteData.x, CompleteData.pe, CompleteData.edge_index).squeeze().cpu().numpy()
#normalize predictions
    scaler = MinMaxScaler()
    predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
# Add predictions back to the geodataframe for visualization
    h3_nodes['predicted_green_view_performer'] = predictions
# Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# Actual Green View Index
    h3_nodes.plot(column='Green View', cmap='Greens', markersize=2,
                     legend=True, ax=ax1, legend_kwds={'label': "Green View Index"})
    ax1.set_title('Actual Green View Index')
    ax1.set_axis_off()
# Predicted Green View Index
    h3_nodes.plot(column='predicted_green_view_performer', cmap='Greens', markersize=2,
                     legend=True, ax=ax2, legend_kwds={'label': "Predicted Green View Index"})
    ax2.set_title('Predicted Green View Index')
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()
    h3_nodes.drop(columns=['predicted_green_view_performer'], inplace=True)


    

def edge_attn_to_dense(edge_index, alpha, num_nodes, num_heads):
    """
    edge_index : [2, E]  (source, target) – same ordering as alpha
    alpha      : [E, heads]
    returns    : heads × N × N tensor
    """
    heads = []
    for h in range(num_heads):
        A = torch.zeros((num_nodes, num_nodes),
                        dtype=alpha.dtype, device=alpha.device)
        src, dst = edge_index
        A[src, dst] = alpha[:, h]  
        heads.append(A)
    return torch.stack(heads, dim=0)       # [H, N, N]

def make_capture_dict(model, data, model_type='GATv2'):
    """
    Uses get_attention_weights to obtain attention matrices for all layers and heads.
    Returns dict {(layer_idx, head_idx): N×N tensor}
    """
    attn_dict = {}
    model.eval()
    
    with torch.no_grad():
        if model_type == 'GATv2':
            if hasattr(model, 'gat_layers'):
                model.layers = model.gat_layers
        elif model_type == 'Performer':
            if hasattr(model, 'transformer_layers'):
                model.layers = model.transformer_layers

        
        for layer_idx in range(len(model.layers)):
            # Get attention weights for this layer
            if model_type == 'GATv2':
                # For GATv2, get_attention_weights returns (edge_index, attention_weights)
                att_edge_index, attention_weights = model.get_attention_weights(
                    data.x, data.pe, data.edge_index, layer_idx=layer_idx
                )
            
                # Convert edge-based attention to dense format
                dense_attention = edge_attn_to_dense(
                    att_edge_index, 
                    attention_weights, 
                    data.num_nodes, 
                    attention_weights.size(1)  # Number of heads
                )
                
                # Store each head's attention matrix separately in the dictionary
                for head_idx in range(dense_attention.size(0)):
                    attn_dict[(layer_idx, head_idx)] = dense_attention[head_idx].cpu()

            elif model_type == 'Performer':
                attention_scores = model.get_performer_attention(data.x, data.pe, data.edge_index, layer_idx=layer_idx)
                attention_scores = attention_scores.squeeze(0)  # Remove batch dimension
                for head_idx in range(attention_scores.size(0)):
                    attn_dict[(layer_idx, head_idx)] = attention_scores[head_idx].cpu()
    
    return attn_dict

def plot_top_k_attention_edges(attention_matrix, h3_nodes, top_k=50):
    """
    Plots the top-k attention edges from the attention matrix.
    
    Args:
        attention_matrix: Attention matrix of shape [heads, num_nodes, num_nodes]
        h3_nodes: GeoDataFrame containing node geometries
        top_k: Number of top attention edges to plot
    """

    node_mapping = {node_id: idx for idx, node_id in enumerate(h3_nodes.index)}
    reverse_mapping = {idx: h3_id for h3_id, idx in node_mapping.items()}
    
    # Find top-k attention pairs
    flattened_attn = attention_matrix.flatten()
    top_indices = np.argsort(flattened_attn)[-top_k:][::-1]
    num_nodes = attention_matrix.shape[1]
    sources = top_indices // num_nodes
    targets = top_indices % num_nodes
    
    # Create geometry and attributes
    edges = []
    for i, j, att in zip(sources, targets, flattened_attn[top_indices]):
        source_h3_id = reverse_mapping[i]
        target_h3_id = reverse_mapping[j]
        source_point = h3_nodes.loc[source_h3_id, 'geometry'].centroid
        target_point = h3_nodes.loc[target_h3_id, 'geometry'].centroid
        edges.append({
            'source': source_h3_id, 'target': target_h3_id, 'attention': float(att),
            'geometry': LineString([source_point, target_point])
        })
    
    # Plot the data
    gdf_edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=h3_nodes.crs)
    
    node_ids = set(gdf_edges['source']).union(set(gdf_edges['target']))
    source_ids = set(gdf_edges['source'])
    target_ids = set(gdf_edges['target'])
    gdf_nodes = h3_nodes[h3_nodes.index.isin(node_ids)]
    gdf_source_nodes = h3_nodes[h3_nodes.index.isin(source_ids)]
    gdf_target_nodes = h3_nodes[h3_nodes.index.isin(target_ids)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # First subplot
    h3_nodes.plot(ax=ax1, color='lightgrey', alpha=0.3, linewidth=0.2)
    gdf_edges.plot(column='attention', cmap='viridis', linewidth=1.5, ax=ax1, 
                   legend=True, legend_kwds={'label': "Attention"})
    gdf_source_nodes.plot(ax=ax1, color='red', markersize=5, alpha=0.5)
    gdf_target_nodes.plot(ax=ax1, color='blue', markersize=5, alpha=0.5)
    # Add legend for source and target nodes
    ax1.scatter([], [], color='red', label='Source Nodes', s=5)
    ax1.scatter([], [], color='blue', label='Target Nodes', s=5)
    ax1.legend()
    ax1.set_title(f'Top {top_k} Attention Edges')
    ax1.set_axis_off()

    # Second subplot with different color scheme
    h3_nodes.plot(ax=ax2, color='lightgrey', alpha=0.3, linewidth=0.2)
    gdf_source_nodes.plot(ax=ax2, color='red', markersize=5, alpha=0.5)
    gdf_target_nodes.plot(ax=ax2, color='blue', markersize=5, alpha=0.5)
    # Add legend for source and target nodes
    ax2.scatter([], [], color='red', label='Source Nodes', s=5)
    ax2.scatter([], [], color='blue', label='Target Nodes', s=5)
    ax2.legend()
    ax2.set_title(f'Top {top_k} Attention Edges (without edges)')
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()

def analyze_head_correlations_within_layer(model, data, model_type, layer_idx=0 ,visualize=True):
    """
    Analyze correlations between different attention heads within the same layer.
    
    Args:
        model: The trained model
        data: The graph data object
        layer_idx: Index of the layer to analyze
        visualize: Whether to create visualizations
        
    Returns:
        correlation_matrix: Matrix of correlations between heads
    """
    print(f"Analyzing correlations between attention heads in Layer {layer_idx}")
    if model_type == 'GATv2':
        # For GAT, get_attention_weights returns (edge_index, attention_weights)
        att_edge_index, attention_weights = model.get_attention_weights(
            data.x, data.pe, data.edge_index, layer_idx=layer_idx
        )
        
        # Convert edge-based attention to dense format
        dense_attention = edge_attn_to_dense(
            att_edge_index, 
            attention_weights, 
            data.num_nodes, 
            attention_weights.size(1)  # Number of heads
        )
        
        # Convert to numpy for correlation calculations
        attention = dense_attention.cpu().numpy()
    elif model_type == 'Performer':
        attention_scores = model.get_performer_attention(data.x, data.pe, data.edge_index, layer_idx=layer_idx)
        attention = attention_scores.squeeze(0)  # Remove batch dimension

    num_heads = attention.shape[0]
    
    # Initialize a correlation matrix
    correlation_matrix = np.zeros((num_heads, num_heads))
    
    # Create a plot for visualizing correlations
    if visualize:
        f, ax = plt.subplots(figsize=(3, 3))
    
    # For each pair of heads, compute correlation of attention patterns
    for i in range(num_heads):
        for j in range(num_heads):
            # Flatten attention matrices for both heads
            head_i_attn = attention[i].flatten()
            head_j_attn = attention[j].flatten()
            
            # Compute correlation
            correlation = np.corrcoef(head_i_attn, head_j_attn)[0, 1]
            correlation_matrix[i, j] = correlation
    
    # Plot the correlation matrix
    if visualize:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   xticklabels=range(num_heads), yticklabels=range(num_heads))
        plt.title(f"Head Correlations in Layer {layer_idx}")
        plt.tight_layout()
        plt.show()
    
    return correlation_matrix


def fuse_heads(layer_mats, method="mean"):
    # layer_mats : list [A^ℓ_h0, A^ℓ_h1, ...]
    stack = torch.stack(layer_mats, 0)    # H × N × N
    if method == "mean":
        return stack.mean(0)
    if method == "rms":
        return (stack.pow(2).mean(0)).sqrt()
    if method == "max":
        return stack.max(0)[0]
    raise ValueError

def make_attention_graph(model, data, model_type ,fuse="mean", corr_thresh=0.5):
    """
    Creates an attention flow graph from model layers.
    
    Args:
        model: The trained model
        data: The graph data object
        model_type: Type of model ("GATv2" or "Performer")
        fuse: Method to fuse attention heads ("mean" or "rms" or "max" or "aware")
        
    Returns:
        Dense N×N tensor representing attention flow through the network
    """
    attn_dict = make_capture_dict(model, data, model_type)  # {(ℓ,h): A}
    
    # Find all unique layer indices in the attention dictionary
    layers = sorted(set(layer for layer, _ in attn_dict.keys()))
    
    fused = []
    for layer in layers:
        # Get all heads for this layer
        heads = sorted(head for l, head in attn_dict.keys() if l == layer)
        
        # Collect attention matrices for all heads in this layer
        mats = [attn_dict[(layer, head)] for head in heads]
        if fuse != "aware":
            # If not using aware fusion, just fuse
            # Fuse the heads for this layer
            fused.append(fuse_heads(mats, fuse))
        else:   # aware means mean if corr > 0 and max if corr < 0
            # Start with the first head
                current_fusion = mats[0]
                
                # Iteratively fuse with subsequent heads based on correlation
                for i in range(1, len(mats)):
                    # Calculate correlation between current fusion and next head
                    fusion_flat = current_fusion.flatten()
                    next_head_flat = mats[i].flatten()
                    corr = torch.corrcoef(torch.stack([fusion_flat, next_head_flat]))[0, 1]
                    
                    # Fuse based on correlation
                    if corr > corr_thresh:
                        # Positive correlation, use mean
                        current_fusion = (current_fusion + mats[i]) / 2
                    else:
                        # If correlation is not above the threshold heads are capturing different information, cannot average
                        current_fusion = torch.maximum(current_fusion, mats[i])
                
                fused.append(current_fusion)
                
    # Multiply attention matrices from last layer to first to get flow of attention through the network
    A_flow = torch.eye(data.num_nodes)
    for A_layer in reversed(fused):
        A_flow = A_layer @ A_flow
        
    return A_flow  

# Training function
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.pe, data.edge_index)
    pred = out[data.train_mask].squeeze()
    loss = criterion(pred , data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(model, data, mask, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.pe, data.edge_index)
        pred = out[mask].squeeze()
        true = data.y[mask]
        loss = criterion(pred, true)
    return loss.item()


def train_performer(model, data, optimizer, criterion, epoch):
    model.train()
    optimizer.zero_grad()
    if epoch < 5:
        redraw_freq = 1
    elif epoch < 150:
        redraw_freq = 30
    else:
        redraw_freq = 1111
    if epoch % redraw_freq == 0:
        model.redraw_projection_matrix()
    out = model(data.x, data.pe, data.edge_index)
    pred = out[data.train_mask].squeeze()
    loss = criterion(pred, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_hybrid(model, data, mask, criterion ,branch='both'):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.pe, data.edge_index, branch=branch)
        pred = out[mask].squeeze()
        true = data.y[mask]
        loss = criterion(pred, true)
    return loss.item()

def analyze_attention_weight_distributions(model, data, num_layers=4):
    """
    Analyzes the distribution of attention weights across different layers.
    
    Args:
        model: The trained Performer model
        data: The graph data object
        num_layers: Number of layers to analyze
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    
    # Storage for attention weights
    all_attention_data = []
    
    # Extract attention weights for each layer
    model.eval()
    with torch.no_grad():
        for layer_idx in range(num_layers):
            print(f"Processing layer {layer_idx}...")
            
            # Get attention scores for this layer
            attention_scores = model.get_performer_attention(
                data.x, data.pe, data.edge_index, layer_idx=layer_idx
            )
            
            # Process each head
            for head_idx in range(attention_scores.size(0)):
                # Extract attention weights for this head
                attn_matrix = attention_scores[head_idx].cpu().numpy()
                
                # Get flattened weights and corresponding indices
                attn_weights = attn_matrix.flatten()
                num_nodes = attn_matrix.shape[0]
                
                # Generate source and target indices
                sources = np.repeat(np.arange(num_nodes), num_nodes)
                targets = np.tile(np.arange(num_nodes), num_nodes)
                
                # Store each attention weight with its source and target
                for source, target, weight in zip(sources, targets, attn_weights):
                    all_attention_data.append({
                        'layer': f'Layer {layer_idx}',
                        'head': f'Head {head_idx}',
                        'source': int(source),
                        'target': int(target),
                        'weight': float(weight)
                    })
                # Basic statistics for this head
                print(f"  Layer {layer_idx}, Head {head_idx}:")
                print(f"    Mean: {np.mean(attn_weights):.6f}")
                print(f"    Median: {np.median(attn_weights):.6f}")
                print(f"    Min: {np.min(attn_weights):.6f}")
                print(f"    Max: {np.max(attn_weights):.6f}")
                print(f"    Std: {np.std(attn_weights):.6f}")
                
                
                
    
    # Create DataFrame for visualization
    attn_df = pd.DataFrame(all_attention_data)
    
    # 1. Plot histograms of attention weights for each layer (combine all heads)
    plt.figure(figsize=(16, 8))
    for i, layer in enumerate(sorted(attn_df['layer'].unique())):
        layer_data = attn_df[attn_df['layer'] == layer]['weight']
        
        # Plot histogram with kernel density estimate
        sns.histplot(layer_data, kde=True, stat='density', alpha=0.6, 
                   label=layer)
    
    plt.title('Distribution of Attention Weights Across Layers', fontsize=14)
    plt.xlabel('Attention Weight', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    # 2. Facet grid to show distribution for each layer/head combination
    g = sns.FacetGrid(attn_df, col='layer', row='head', height=3, aspect=1.5)
    g.map_dataframe(sns.histplot, x='weight', kde=True)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Attention Weight", "Count")
    g.tight_layout()
    plt.show()

    return attn_df