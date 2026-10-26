"""The parts of the GTK tracking code (GTK_tracking_CernPC, commit ) that the notebooks use, copied verbatim
from its src/: the data loader of the events, the Transformer classes, and the clustering of the hits and the track
reconstruction (written by school_build/make_gtk_tracking.py).
"""
import os
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------------------------------------
# Events: the data loader (from src/Dataloader/Transformer/dataset.py)
# ---------------------------------------------------------------------------------------------------------
class TransformerDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.split = split
        self.features = os.listdir(f'{self.data_dir}/features/{split}/') # [:20000]
        self.gt = os.listdir(f'{self.data_dir}/gt_tensors/{split}/') # [:20000]
        self.dataframes = os.listdir(f'{self.data_dir}/dataframe/{split}/') #  [:20000]
        

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Ricava l'indice numerico dal nome file
        name = self.features[idx].split('.')[0].split('_')[-1]
        feature_path = f'{self.data_dir}/features/{self.split}/{self.features[idx]}' 
        gt_path = f'{self.data_dir}/gt_tensors/{self.split}/gt_tensors_{name}.pt'
        df_path = f'{self.data_dir}/dataframe/{self.split}/df_{name}.csv'
        
        
        features_raw = torch.load(feature_path)
        #! time-KTAG time: x,y,z, time - KTAGtime
        if self.config["model"]["time"] == "GTKminusKTAG":
            col4 = (features_raw[:, 3] - features_raw[:, 4]).unsqueeze(1)   # features_raw[:, 3].unsqueeze(1)  #
            features = torch.cat([features_raw[:, :3], col4], dim=-1)
        elif self.config["model"]["time"] == "GTK":
        #! solo tempo GTK
            features = features_raw[:, :4]  

        gt_tensor = torch.load(gt_path)
        dataframe = pd.read_csv(df_path)

        return features, gt_tensor, dataframe, name

# ---------------------------------------------------------------------------------------------------------
# The model: the trained Transformer in PyTorch (from src/models/Transformer.py)
# ---------------------------------------------------------------------------------------------------------
def find_all_combinations(numbers):
    """
    Find all possible combinations of two numbers from the given list,
    including self-combinations (n, n) while maintaining the order.
    """
    result = list(combinations_with_replacement(numbers, 2))
    
    return result


class Encoder(nn.Module):
    def __init__(self, config) :
        self.input_dim =config["model"]["input_dim"]
        self.d_model = config["model"]["d_model"]
        self.nhead = config["model"]["num_head"]
        self.num_layers = config["model"]["num_layers"]
        self.hidden_dim_classifier = config["model"]["hidden_dim_classifier"]
        self.dropout = config["model"]["dropout"]
        self.device = config["model"]["device"]
    
        super(Encoder, self).__init__()   
        self.embedding = nn.Linear(self.input_dim, self.d_model)    

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.hidden_dim_classifier, self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        #MLP
        self.fc1 = nn.Linear(self.d_model*2, self.d_model*2)
        self.fc2 = nn.Linear(self.d_model*2, 2)

        #! MLP Count experiments
        # self.fc11 = nn.Linear(self.d_model*2, self.d_model*2)
        # self.fc2 = nn.Linear(self.d_model*2, 2)


    def forward(self, data, df, max_len): # max_edges):
        data = data.to(self.device) #data.shape = (batch_size, num_nodes, input_dim)

        # padding mask: True dove l'hit e' padding
        dims = [dataframe.shape[0] for dataframe in df]               # hit reali per evento
        positions = torch.arange(data.shape[1], device=data.device)   # 0 ... num_nodes-1
        lengths = torch.tensor(dims, device=data.device)
        padding_mask = positions[None, :] >= lengths[:, None]         # (batch_size, num_nodes)

        #embedding
        data = self.embedding(data)
        #transformer encoder
        transformer_output = self.transformer_encoder(data, src_key_padding_mask=padding_mask)

        #MLP
        #create tensor for MLP
        node_indices = [find_all_combinations(list(range(dim))) for dim in dims]
        mlp_tensor_to_pad = [transformer_output[i,node_indices[i]] for i in range(len(node_indices))]
        pad_amount = [ (max_len**2)- x.shape[0] for x in mlp_tensor_to_pad]
        #[ (max_len**2)- x.shape[0] for x in mlp_tensor_to_pad]
        #[ 4*((max_len+5)**2) -x.shape[0] for x in mlp_tensor_to_pad]
        mlp_tensor = [F.pad(x, (0, 0, 0, 0, 0, padding)) for x,padding in zip(mlp_tensor_to_pad, pad_amount) ]
        mlp_tensor = [x.view(x.shape[0],-1) for x in mlp_tensor]
        mlp_input = torch.cat([x.unsqueeze(0) for x in mlp_tensor], dim =0).to(self.device)

        mlp_output = torch.relu(self.fc1(mlp_input))
        mlp_output = self.fc2(mlp_output)   # logits: la softmax e' in LightningEncoder.forward / _loss

        #! Cambiato qui: MLP0
        # mlp_output = self.fc2(mlp_input)
        # mlp_output = torch.softmax(mlp_output, dim = -1)

        #! Cambiato qui: MLP2
        # mlp_output = torch.relu(self.fc1(mlp_input))
        # mlp_output = torch.relu(self.fc11(mlp_output))
        # mlp_output = self.fc2(mlp_output)
        # mlp_output = torch.softmax(mlp_output, dim = -1)

        return mlp_output

    @torch.no_grad()
    def get_attention_map(self, data, df, max_len):
        """
        Attention weights hit-to-hit di ogni layer, per un singolo evento.
        Stessa convenzione di chiamata di forward(); max_len non e' usato.

        data: tensor (1, num_nodes, input_dim)
        Ritorna: dict {layer_idx: tensor(num_heads, num_hits, num_hits)}

        self_attn e' chiamato direttamente: in eval il TransformerEncoderLayer usa un
        fast path che salta il modulo self_attn, quindi degli hook non verrebbero chiamati.
        Con norm_first=False (default) l'input di self_attn e' l'input del layer.
        """
        assert data.shape[0] == 1, "get_attention_map supporta un solo evento alla volta (batch_size=1)"
        num_hits = df[0].shape[0]
        x = self.embedding(data[:, :num_hits].to(self.device))   # solo hit reali -> niente padding mask

        attn_weights = {}
        for i, layer in enumerate(self.transformer_encoder.layers):
            _, w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
            attn_weights[i] = w[0]      # (num_heads, num_hits, num_hits)
            x = layer(x)                # input del layer successivo
        return attn_weights


class EncoderONNX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['model']['input_dim']
        self.d_model = config['model']['d_model']
        self.num_head = config['model']['num_head']
        self.head_dim = self.d_model // self.num_head
        self.num_layers = config['model']['num_layers']
        self.hidden_dim_classifier = config['model']['hidden_dim_classifier']
        self.dropout = config['model']['dropout']

        # Layers
        self.embedding = nn.Linear(self.input_dim, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_head,
            dim_feedforward=self.hidden_dim_classifier,
            dropout=self.dropout,
            batch_first=True  # Match Lightning model
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model * 2)
        self.fc2 = nn.Linear(self.d_model * 2, 2)

    def encoder_layer(self, layer, x):
        """
        Stesso calcolo di nn.TransformerEncoderLayer (norm_first=False, relu, eval).
        x: (1, N, d_model). Tutti i reshape usano -1 al posto di N -> N resta dinamico.
        """
        # Self-attention: in_proj_weight (3*d_model, d_model) contiene Q, K, V uno dopo l'altro
        q, k, v = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(1, -1, self.num_head, self.head_dim).transpose(1, 2)   # (1, heads, N, head_dim)
        k = k.reshape(1, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = v.reshape(1, -1, self.num_head, self.head_dim).transpose(1, 2)

        weights = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)   # (1, heads, N, N)
        attn = (weights @ v).transpose(1, 2).reshape(1, -1, self.d_model)                 # (1, N, d_model)
        attn = layer.self_attn.out_proj(attn)

        # Residual + LayerNorm, poi feed forward + residual + LayerNorm
        x = layer.norm1(x + attn)
        x = layer.norm2(x + layer.linear2(F.relu(layer.linear1(x))))
        return x

    def forward(self, data, pair_indices):
        """
        data: (N, input_dim)  --> un solo evento, niente padding -> niente mask
        pair_indices: (K, 2) with pairs (i, j)
        output: (1, K, 2) where K = number of pairs
        """
        # 1. Embedding
        x = data.unsqueeze(0)                  # (1, N, input_dim)
        x = self.embedding(x)                  # (1, N, d_model)

        # 2. Transformer encoder
        for layer in self.transformer_encoder.layers:
            x = self.encoder_layer(layer, x)   # (1, N, d_model)

        # 3. Pair indexing
        x_i = x[:, pair_indices[:, 0]]         # (1, K, d_model)
        x_j = x[:, pair_indices[:, 1]]         # (1, K, d_model)

        # 4. MLP finale
        pairs = torch.cat([x_i, x_j], dim=-1)  # (1, K, 2*d_model)
        out = torch.relu(self.fc1(pairs))      # (1, K, 2*d_model)
        out = self.fc2(out)                    # (1, K, 2)
        out = torch.softmax(out, dim=-1)       # (1, K, 2)

        return out

# ---------------------------------------------------------------------------------------------------------
# Clustering of the hits, and track reconstruction from the edges (from src/tools/support_functions_metrics.py)
# ---------------------------------------------------------------------------------------------------------
def edge_dictionary_transformer(model_output, threshold, df):

    #edges
    edges = find_all_combinations(list(range(len(df))))  # Find all combinations of node indices
    src = [edge[0] for edge in edges]
    dst = [edge[1] for edge in edges]
    scores = model_output[:, :, 1].squeeze()  # Assuming the second column is the score

    edge_dict = {}
    for i in range(len(src)):
        if scores[i] > threshold:
            edge_dict[(src[i], dst[i])] = scores[i]

    return edge_dict


def filter_tracks_Framework (tracks, df): #graph_tensor):
    '''
    this function filters the tracks based on the conditions defined on the GTKs
    '''
    # create dictionary of indices of hits belonging to certain gtks stations
    hit_on_gtk3 = False
    hit_on_gtk2 = False
    hit_on_gtk0_or_gtk1 = False  

    #filter the tracks checking the conditions
    filtered_tracks = []
    for track in tracks:
        for hit in track:
            station = df.loc[hit, 'GTKstation']
            if station == 3:
                hit_on_gtk3 = True
            elif station == 2:
                hit_on_gtk2 = True
            elif station in [0, 1]:
                hit_on_gtk0_or_gtk1 = True
        if hit_on_gtk3 and hit_on_gtk2 and hit_on_gtk0_or_gtk1:
            filtered_tracks.append(track)
        # Reset flags for next track
        hit_on_gtk3 = False
        hit_on_gtk2 = False
        hit_on_gtk0_or_gtk1 = False
    
    return filtered_tracks


def create_gt_tracks_Framework (df):
    """
    This function takes a dataframe with columns 'kinepart', 'GTKstation', and 'parent particle' 
    and returns a dictionary where:
    - The keys are 'kinepart' values.
    - The values are lists of admissible tracks (list of row indices).
    
    A track is admissible if:
    1. Its 'kinepart' is always the same.
    2. it has hits on GTK 3 and 2 and optionally on GTK 0 or GTK 1 or both
    """
    
    # Replace 'kinepart' with 'parent particle' values where 'parent particle' is not -1
    # df['kinepart'] = df.apply(
    #     lambda row: row['parent_particle'] if row['parent_particle'] != -1 else row['cl_kinepart'], axis=1)

    tracks = {}

    # Group the dataframe by 'kinepart'
    grouped = df.groupby('cl_kinepart')

    # Loop through each group of the same 'kinepart'
    for kinepart, group in grouped:
        tracks[kinepart] = []
        
        # hits indexeson gtk0
        gtk0 = group[group['GTKstation'] == 0].index
        gtk1 = group[group['GTKstation'] == 1].index
        gtk2 = group[group['GTKstation'] == 2].index
        gtk3 = group[group['GTKstation'] == 3].index

        track = np.concatenate([gtk0, gtk1, gtk2, gtk3])
        # consider only tracks 0,1,2,3 or 1,2,3 or 0,2,3
        if (len(gtk0) > 0 and len(gtk1) > 0 and len(gtk2) > 0 and len(gtk3) > 0) or\
            (len(gtk1) > 0 and len(gtk2) > 0 and len(gtk3) > 0) or\
            (len(gtk0) > 0 and len(gtk2) > 0 and len(gtk3) > 0):
            tracks[kinepart].extend(track)
    
    #remove element with value = []
    tracks = {k: [v] for k, v in tracks.items() if v}

    return tracks


def tp_fp_fn (predicted_tracks, gt_tracks):
    '''
    input:
    - predicted_tracks: list of predicted tracks
    - gt_tracks: dictionary with key = kinepart and value = list of tracks
    output: true positive, false positive and false negative
    '''
    tp = 0
    fp = 0
    fn = 0

    left_particles = list(gt_tracks.keys())
    particles_founder = set()
    for track in predicted_tracks:
        for particle in gt_tracks.keys():
            if sorted(track) in [ sorted(t)for t in gt_tracks[particle]]:
                tp += 1
                #remove from list
                particles_founder.add(particle)
                break
        else:
            fp += 1

    fn = len(left_particles) - len(particles_founder)
    return tp, fp, fn


def find_sequences(pairs):
    '''
    this function create the sequence of a track starting from a dictionary of pairs
    '''
    
    # Create a dictionary to map the first element to the second element
    connection_map = {first: second for first, second in pairs}

    # Initialize an empty list to store the sequences
    sequences = []

    # To find starting points, find elements that are not second elements in any tuple
    all_second_elements = set(connection_map.values())
    starting_points = [first for first in connection_map if first not in all_second_elements]

    # Iterate through starting points to form sequences
    for start in starting_points:
        sequence = [start]
        while start in connection_map:
            start = connection_map[start]
            sequence.append(start)
        sequences.append(sequence)

    return sequences


def allowed_edges(df, edge_dict):
    """
    Returns only connections between nodes belonging to allowed GTK stations.

    Args:
        df (pd.DataFrame): DataFrame with 'GTKstation' column and index corresponding to nodes.
        edge_dict (dict): Dictionary {(i, j): score} with connections between nodes.

    Returns:
        dict: Filtered edge_dict, only with connections between allowed stations.
    """
    # Map node to station
    node_to_station = {
        node: station
        for station in range(4)
        for node in df[df['GTKstation'] == station].index
    }

    # pair of admissible stations
    allowed_station_pairs = {(0, 1), (1, 2), (0, 2), (2, 3) , (1, 3), (0, 3)}

    # filter to obtain only edges between admissible station pairs
    return {
        (i, j): score
        for (i, j), score in edge_dict.items()
        if (node_to_station.get(i), node_to_station.get(j)) in allowed_station_pairs
    }


def best_exiting(edge_dict):
    """
    Selects the best outgoing edge for each source node, based on score.

    Args:
        edge_dict (dict): Dictionary {(i, j): score} with connections between nodes and scores.

    Returns:
        dict: Dictionary {(i, j): score} containing only the outgoing edge with the highest score for each source node i.
    """
    best_exiting_edges = {}
    for (i, j), score in edge_dict.items():
        if not any(k[0] == i and v >= score for k, v in best_exiting_edges.items()):
            # Rimuove eventuale edge precedente con stesso nodo sorgente
            best_exiting_edges = {k: v for k, v in best_exiting_edges.items() if k[0] != i}
            best_exiting_edges[(i, j)] = score
    return best_exiting_edges


def best_entering(best_exiting_edges, df):
    """
    Selects the best entering edges for each destination node

    Args:
        best_exiting_edges (dict): {(i, j): score} with the best outgoing edges.
        df (pd.DataFrame): DataFrame with 'GTKstation' column and index corresponding to nodes.

    Returns:
        dict: best_entering_edges
    """
    # Identifies nodes belonging to each GTK station
    # gtk0 = df[df['GTKstation'] == 0].index
    # gtk1 = df[df['GTKstation'] == 1].index
    # gtk2 = df[df['GTKstation'] == 2].index
    # gtk3 = df[df['GTKstation'] == 3].index

    best_entering_edges = {}

    # Iterate on destination nodes 
    for e_node in set(k[1] for k in best_exiting_edges.keys()):
        # Filter edges that have e_node as destination
        filtered_edges = {k: v for k, v in best_exiting_edges.items() if k[1] == e_node}
        if filtered_edges:
            # Select the edge with the highest score
            best_edge = max(filtered_edges.items(), key=lambda x: x[1])
            best_entering_edges[best_edge[0]] = best_edge[1]

    return best_entering_edges


def make_pred_tracks(edge_dict , df ):
    '''
    This function takes a batch of data, model logits, and a threshold to create predicted tracks
    '''

    # 3.1 Create tracks with the edges
    edge_dict = allowed_edges(df, edge_dict)  
    # 3.2 Best outgoing edge per source node (keeps (i, j): score)
    best_exiting_edges = best_exiting(edge_dict)  # Select the best exiting edge for each source node
    #3.3 Best entering edges with GTK correction
    best_entering_edges = best_entering( best_exiting_edges, df)
    # 3.4  build tracks with remaining edges
    pred_tracks = find_sequences(best_entering_edges)   # Find sequences in the best edges

    return pred_tracks


def cluster_hits_by_distance_and_time(df, features, dist_thr= 2*np.sqrt(2)*0.3 + 0.0001 , time_thr=0.150*5):
                                    #   dist_thr=0.510, time_thr=0.7):
    """
    Incremental clustering: each hit is assigned to an existing cluster if it matches
    at least one hit in that cluster (distance, time, GTKstation), otherwise a new cluster is created.
    If a hit connects multiple clusters, those clusters are merged.
    Returns:
      - new DataFrame with merged hits (features averaged)
      - array of new features
      - dictionary mapping old_indices --> new_cluster_index
    """
    coords = df[['cl_x', 'cl_y']].values
    times = df['cl_time'].values
    gtk = df['GTKstation'].values
    n = len(df)
    clusters = []  # list of lists of indices
    old_to_new = {}

    for i in range(n):
        added_to = []
        for c_idx, cluster in enumerate(clusters):
            for j in cluster:
                if gtk[i] != gtk[j]:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                tdiff = abs(times[i] - times[j])
                if dist <= dist_thr and tdiff <= time_thr:
                    added_to.append(c_idx)
                    break  # Found a match in this cluster, no need to check further
        if not added_to:
            # New cluster
            clusters.append([i])
        else:
            # Merge all clusters this hit connects to, and add the hit
            merged = set([i])
            for idx in sorted(added_to, reverse=True):
                merged.update(clusters[idx])
                del clusters[idx]
            clusters.append(list(merged))

    # Build mapping and new data
    new_rows = []
    for new_idx, indices in enumerate(clusters):
        mean_feats = features[0][indices].mean(dim=0)
        mean_row = df.iloc[indices].mean(numeric_only=True)
        new_rows.append(mean_row)
        for old_idx in indices:
            old_to_new[old_idx] = new_idx
    new_df = pd.DataFrame(new_rows)
    new_features = torch.stack([features[0][indices].mean(dim=0) for indices in clusters]).unsqueeze(0)
    return new_df, new_features, old_to_new


def inverse_cluster_mapping(filtered_indices, old_to_new_mapping):
    """
    Reconstructs the original hit indices from filtered cluster indices and the cluster mapping.
    Returns a list of original indices.
    """
    # Invert mapping: new_cluster_index --> list of old_indices
    new_to_old = {}
    for old, new in old_to_new_mapping.items():
        new_to_old.setdefault(new, []).append(old)
    # Reconstruct original indices
    original_indices = []
    for idx in filtered_indices:
        original_indices.extend(new_to_old.get(idx, []))
    return original_indices
