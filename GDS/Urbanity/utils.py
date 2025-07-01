import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import geopandas as gpd
from shapely.geometry import LineString
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import seaborn as sns
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from models import *



def evaluate_and_visualize_predictions(model, CompleteData, path, h3_nodes ,criterion , target = 'Green View' ,branch = None, load = True):
    """ Evaluates the model on the test set and visualizes the predictions.
    Args:
        model: The trained model to evaluate.
        CompleteData: The graph data object containing node features, edge indices, and masks.
        path: Path to the saved model weights.
        h3_nodes: GeoDataFrame containing node geometries and features.
        criterion: Loss function to evaluate the model.
        target: The target variable to visualize.
        branch: Optional branch to evaluate if the model supports hybrid branches.

    Returns:
        fig: Matplotlib figure containing the visualizations of actual and predicted target.
    """
    if branch is None:
        if load:
            print(f'Loading model from {path}')
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
    h3_nodes.plot(column= target, cmap='Greens', markersize=2,
                     legend=True, ax=ax1, legend_kwds={'label': target})
    ax1.set_title(f'Actual {target}')
    ax1.set_axis_off()
# Predicted Green View Index
    h3_nodes.plot(column='predicted_green_view_performer', cmap='Greens', markersize=2,
                     legend=True, ax=ax2, legend_kwds={'label': f"Predicted {target}"})
    ax2.set_title(f'Predicted {target}')
    ax2.set_axis_off()
    plt.suptitle(f'Final Test Loss: {test_loss:.4f}', fontsize=16)
    plt.tight_layout()
    h3_nodes.drop(columns=['predicted_green_view_performer'], inplace=True)
    plt.close(fig)  # Close the figure to avoid display in Jupyter notebooks
    # return the plot
    return fig



def plot_combined_results(model_name, features, city_list, save_path=None,figures_dir='figures'):
    """
    Loads individual city plot images for a given model and feature set,
    stacks them vertically, and displays them as a single figure.

    Args:
        model_name (str): The name of the model (e.g., 'GATv2', 'Hybrid').
        features (list): A list of feature names used for training.
        city_list (list): A list of city names to include in the plot.
        figures_dir (str): The directory where the individual plot images are saved.
    """
    feature_string = '_'.join(features).replace(' ', '')
    
    image_arrays = []
    print(f"Loading images for model '{model_name}' with features '{feature_string}'...")

    # Load the saved figures for each city into a list
    for city in city_list:
        filepath = os.path.join(figures_dir, f'{model_name}_{feature_string}_{city}.png')
        try:
            # Read the image file into a numpy array
            img_data = plt.imread(filepath)
            image_arrays.append(img_data)
        except FileNotFoundError:
            print(f"Warning: Could not find file {filepath}")

    # If we successfully loaded images, stack and display them
    if image_arrays:
        # Vertically stack the numpy arrays of the images
        stacked_image = np.vstack(image_arrays)
        # Display the final combined image
        # Adjust figsize height based on the number of images
        fig_height = 4 * len(image_arrays)
        plt.figure(figsize=(10, fig_height))
    
        plt.imshow(stacked_image)
        plt.axis('off')
        plt.title(f'Results for {model_name} | Features: {feature_string}', pad=20)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

        
    else:
        print("No images were found to display for the specified model and features.")
        return None
        
    return stacked_image




    

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
        model_type: Type of model ("GATv2" or "Performer")
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



def create_spatial_finetune_mask(node_data, edge_index, percentage=0.05, type = 'edge' ,seed=0):
    """
    Create a fine-tuning mask using nearest neighbors to sample nodes uniformly in space.
    
    Args:
        node_data: DataFrame with node data containing 'x' and 'y' coordinates
        edge_index: PyTorch tensor with edge indices
        percentage: Percentage of nodes to sample for fine-tuning
        type: Type of mask to return ('node' or 'edge')
        seed: Random seed for reproducibility
        
    Returns:
        node_mask: Boolean mask for sampled nodes
        edge_mask: Boolean mask for edges connected to sampled nodes
    """
    np.random.seed(seed)
    
    # Extract spatial coordinates
    x_coords = node_data['x'].values
    y_coords = node_data['y'].values
    coordinates = np.column_stack([x_coords, y_coords])
    
    # Determine spatial bounds
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Calculate number of points to sample
    num_nodes = len(node_data)
    num_points_to_sample = int(percentage * num_nodes)
    
    # Generate random points uniformly across the space
    random_points = np.column_stack([
        np.random.uniform(x_min, x_max, num_points_to_sample),
        np.random.uniform(y_min, y_max, num_points_to_sample)
    ])
    
    # Find nearest nodes to each random point
    nn = NearestNeighbors(n_neighbors=1).fit(coordinates)
    distances, indices = nn.kneighbors(random_points)
    
    # Get unique nearest nodes (may be fewer than random points if multiple points map to same node)
    selected_node_indices = np.unique(indices.flatten())
    
    # Create node mask
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[selected_node_indices] = True
    
    # Create edge mask - include edges that connect to selected nodes
    sources, targets = edge_index
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    
    for i, (src, dst) in enumerate(zip(sources, targets)):
        if node_mask[src] or node_mask[dst]:
            edge_mask[i] = True
            
    if type == 'node':
        return node_mask
    elif type == 'edge':
        return edge_mask



def create_region_finetune_mask(node_data, edge_index, type = 'edge' ,region_params=None, seed=0):
    """
    Create a fine-tuning mask for nodes in a specific region of the city.
    
    Args:
        node_data: DataFrame with node data containing 'x' and 'y' coordinates
        edge_index: PyTorch tensor with edge indices
        region_params: Parameters for the region (quadrant): 'NE', 'NW', 'SE', 'SW', 'N', 'S', 'E', 'W', 'C'
        type: Type of mask to return ('node' or 'edge')
        seed: Random seed for reproducibility
        
    Returns:
        node_mask: Boolean mask for nodes in the specified region
        edge_mask: Boolean mask for edges connected to nodes in the region

    """
    
    # Extract spatial coordinates
    x_coords = node_data['x'].values
    y_coords = node_data['y'].values
    
    # Determine spatial bounds of the entire city
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    if region_params is None:
        region_params = 'SW'  # Default to southwest quadrant
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    if region_params == 'NE':
        node_mask = (x_coords >= x_center) & (y_coords >= y_center)
    elif region_params == 'NW':
        node_mask = (x_coords <= x_center) & (y_coords >= y_center)
    elif region_params == 'SE':
        node_mask = (x_coords >= x_center) & (y_coords <= y_center)
    elif region_params == 'SW':
        node_mask = (x_coords <= x_center) & (y_coords <= y_center)
    elif region_params == 'N':
        node_mask = (y_coords >= y_center)
    elif region_params == 'S':
        node_mask = (y_coords <= y_center)
    elif region_params == 'E':
        node_mask = (x_coords >= x_center)
    elif region_params == 'W':
        node_mask = (x_coords <= x_center)
    elif region_params == 'C':
        radius = min(x_max - x_min, y_max - y_min) / 6
        node_mask = ((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2) <= radius ** 2
    else:
        raise ValueError(f"Invalid quadrant: {region_params}")
    
    # Convert NumPy boolean array to PyTorch tensor
    node_mask = torch.tensor(node_mask, dtype=torch.bool)
    
    # Create edge mask - include edges that connect to selected nodes
    sources, targets = edge_index
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    
    for i, (src, dst) in enumerate(zip(sources, targets)):
        if node_mask[src] or node_mask[dst]:
            edge_mask[i] = True
            
    if type == 'node':
        return node_mask
    elif type == 'edge':
        return edge_mask
    
def create_h3_aggregated_graph(amsterdam_nodes, agg_dict, h3_resolution=9):
    # Convert to H3 and get boundaries
    dfh3 = amsterdam_nodes.h3.geo_to_h3(h3_resolution)
    dfh3 = dfh3.h3.h3_to_geo_boundary()
    
    # Aggregate features by H3 cell
    h3_grouped = dfh3.groupby([f'h3_0{h3_resolution}']).agg(agg_dict).reset_index()
    h3_grouped.set_index(f'h3_0{h3_resolution}', inplace=True)
    
    # Get geometry for each H3 cell
    h3_nodes_amst = h3_grouped.h3.h3_to_geo_boundary()
    h3_nodes_amst = gpd.GeoDataFrame(
        h3_grouped, geometry=h3_nodes_amst['geometry'], crs=amsterdam_nodes.crs
    )
    h3_nodes_amst['x'] = h3_nodes_amst.geometry.centroid.x
    h3_nodes_amst['y'] = h3_nodes_amst.geometry.centroid.y

    # Find neighbors for each H3 cell
    neighbors_index = h3_nodes_amst.h3.k_ring(1, explode=True)
    neighbors_index = neighbors_index[neighbors_index['h3_k_ring'] != neighbors_index.index]
    neighbors_index = neighbors_index['h3_k_ring']
    neighbor_df = neighbors_index.reset_index()
    neighbor_pairs = set((row['index'], row['h3_k_ring']) for _, row in neighbor_df.iterrows())

    # Ensure symmetry (optional, can be omitted if not needed)
    symmetric = all((b, a) in neighbor_pairs for a, b in neighbor_pairs)
    if not symmetric:
        missing_pairs = [(b, a) for a, b in neighbor_pairs if (b, a) not in neighbor_pairs]
        # Remove pairs where the first cell is not present in h3_nodes_amst
        missing_pairs = [(cell, neighbor) for cell, neighbor in missing_pairs if h3_nodes_amst[h3_nodes_amst.index == cell].empty]
        neighbors_index = neighbors_index[~neighbors_index.isin([cell for cell, _ in missing_pairs])]
        # Rebuild neighbor_pairs
        neighbor_df = neighbors_index.reset_index()
        neighbor_pairs = set((row['index'], row['h3_k_ring']) for _, row in neighbor_df.iterrows())

    # Build edges GeoDataFrame
    h3_edges_amst = pd.DataFrame(neighbor_pairs, columns=['source', 'target'])
    h3_edges_amst['geometry'] = h3_edges_amst.apply(
        lambda row: LineString([
            h3_nodes_amst.loc[row['source'], 'geometry'].centroid,
            h3_nodes_amst.loc[row['target'], 'geometry'].centroid
        ]), axis=1
    )
    h3_edges_amst = gpd.GeoDataFrame(h3_edges_amst, geometry='geometry', crs=amsterdam_nodes.crs)
    h3_edges_amst = h3_edges_amst[['source', 'target', 'geometry']]
    h3_edges_amst.reset_index(drop=True, inplace=True)
    h3_nodes_amst['Green View'] = h3_nodes_amst['Green View Mean']

    return h3_nodes_amst, h3_edges_amst

def preprocess_data(nodes, edges, X_list = ['x', 'y', 'PopSum'], target = 'Green View Mean', map_type = 'h3'):
    """
    Preprocess the data for training a GNN model.
    Args:
        nodes: DataFrame containing node features
        edges: DataFrame containing edge information
        X_list: List of feature columns to use for node features
        target: Target variable for regression
        map_type: Type of mapping for edges ('h3' or 'geo')
    Returns:
        data: PyTorch Geometric Data object containing processed node features, edge indices, and target variable
    """
    device = 'cpu'
    if 'POI' in X_list:
        poi_cols = ['Civic', 'Commercial','Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
        nodes['POI'] = nodes[poi_cols].sum(axis=1)
        nodes.fillna(0, inplace=True)

    y = nodes[target].values
    X = nodes[X_list].values
    X = np.array(X)
    scaler =  StandardScaler()
    X_scaled = scaler.fit_transform(X)
    node_mapping = {node_id: idx for idx, node_id in enumerate(nodes.index)}
    if map_type == 'h3':
        edge_index = []
        for _, row in edges.iterrows():
            source_idx = node_mapping[row['source']]
            target_idx = node_mapping[row['target']]
            edge_index.append([source_idx, target_idx]) # no need to add the reverse edge since edge already contains it
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    elif map_type == 'geo':
        node_to_id = {}
        for i, node in enumerate(nodes['osmid'].values):
            node_to_id[node] = i

        start_node = [node_to_id[i] for i in edges['u'].values]
        end_node = [node_to_id[i] for i in edges['v'].values]
        start = torch.tensor(start_node, dtype=torch.long)
        end = torch.tensor(end_node, dtype=torch.long)
        edge_index = torch.stack([start, end], dim=0)
    transform = T.AddRandomWalkPE( 20, attr_name= 'pe')
    data = Data(x = torch.FloatTensor(X_scaled), y = torch.FloatTensor(y), edge_index = edge_index )
    data = transform(data)
    np.random.seed(0)
    n_nodes = data.num_nodes
    indices = np.random.permutation(n_nodes)
    train_idx = indices[:int(0.7 * n_nodes)]
    val_idx = indices[int(0.7 * n_nodes):int(0.85 * n_nodes)]
    test_idx = indices[int(0.85 * n_nodes):]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True        
    data.train_mask = train_mask   
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.to(device)


        
    return data


def train_batched_model(model, loader, epochs, lr, device = 'cpu', save_path = None, patience=100, model_type = 'Performer', verbose=True):
    """
    Trains a PyTorch Geometric model using a DataLoader for batched graph data.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The DataLoader for training and validation.
        epochs (int): The maximum number of training epochs.
        lr (float): The learning rate for the optimizer.
        device (str): The device to use for training (e.g., 'cpu' or 'cuda').
        save_path (str): The file path to save the best performing model.
        patience (int): The number of epochs to wait for improvement before early stopping.
        model_type (str): The type of model being trained ('Performer', 'GATv2', or 'Hybrid').
        verbose (bool): Whether to print training progress and metrics.

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The trained model with the best weights loaded.
            - train_losses (list): A list of training losses for each epoch.
            - val_losses (list): A list of validation losses for each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Adjust scheduler patience based on early stopping patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2.5)
    
    best_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []
    if verbose:
        print(f"Starting training on {device}...")

    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0
        epoch_val_loss = 0

        for batch in loader:
            batch = batch.to(device)
            if model_type == 'Performer' or model_type == 'Hybrid':
                loss = train_performer(model, batch, optimizer, criterion, epoch)
            if model_type == 'GATv2':
                loss = train(model, batch, optimizer, criterion)
            epoch_train_loss += loss  
        
        # Validation
        for batch in loader:
            batch = batch.to(device)
            loss = evaluate(model, batch, batch.val_mask, criterion)
            epoch_val_loss += loss

        epoch_train_loss /= len(loader)
        train_losses.append(epoch_train_loss)
        epoch_val_loss /= len(loader)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if verbose:
            if epoch % 10 == 0:
                test_loss = 0
                for batch in loader:
                    batch = batch.to(device)
                    loss = evaluate(model, batch, batch.test_mask, criterion)
                    test_loss += loss
                test_loss /= len(loader)

                print(f'Epoch: {epoch:03d}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, '
                    f'Test Loss: {test_loss:.4f}, Counter: {early_stop_counter}')

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    # Load the best model weights before returning
    if save_path is not None:
        model.load_state_dict(torch.load(save_path))
        if verbose:
            print(f"Finished training. Best model with validation loss {best_loss:.4f} loaded from {save_path}.")
    
    return model, train_losses, val_losses


def finetune(model, data, ft_mask, optimizer, criterion):
    """ Finetunes the model on a subset of edges defined by ft_mask.
    Args:
        model: The trained model to finetune.
        data: The graph data object containing node features, edge indices, and target variable.
        ft_mask: Boolean mask indicating which edges to use for finetuning.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function to compute the loss during finetuning.
    Returns:
        loss: The computed loss for the finetuning step.
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass on full graph
    out = model(data.x, data.pe, data.edge_index)
    predictions = out.squeeze(-1)
    
    # Loss only on masked edges
    loss = criterion(predictions[ft_mask], data.y[ft_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()



