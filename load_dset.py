import os
from pathlib import Path
import json
from typing import Tuple, List, Dict, Union, Any

import jax
from jax import numpy as jnp, random
import numpy as np
from sklearn.preprocessing import StandardScaler
import jraph
from tqdm import tqdm

"""
Dataset implementation for GraphSAGE PPI and Reddit datasets from https://github.com/williamleif/GraphSAGE
Description of dataset files:
    1. "-G.json": A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
    2. "-id_map.json": A json-stored dictionary mapping the graph node ids to consecutive integers.
    3. "-class_map.json": A json-stored dictionary mapping the graph node ids to classes.
    4. "-feats.npy": A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
"""


def load_data(root: str, prefix: str) -> List[Union[Dict[str, Any], np.ndarray]]:
    """Load necessary files to construct the graph representation and the node features
    Args:
        root (str): Path to dataset folder
        prefix (str): Dataset name prefix
    Returns:
        data (list): List of
    """
    # TODO: List[Dict[str, Any], np.ndarray, Dict[str, List[int]], Dict[str, int]]
    file_suffixes = ["-G.json", "-feats.npy", "-id_map.json", "-class_map.json"]
    filepaths = [Path(root + "/" + prefix + sfx) for sfx in file_suffixes]
    data = [
        json.load(open(fp)) if fp.suffix == ".json" else np.load(fp) for fp in filepaths
    ]
    return data


def normalize_features(graph: Dict, feats: np.ndarray) -> np.ndarray:
    """Normalize the training node features"""
    train_ids = np.array(
        [
            k
            for k, v in graph["nodes"].items()
            if v["test"] is False and v["val"] is False
        ]
    )
    train_feats = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    return scaler.transform(feats)


def load_graph_data(
    dset_dir: str,
    prefix: str,
) -> Tuple[Union[Dict[str, Any], np.ndarray]]:
    """Make the batch iterator for the graph NN
    Args:
    """
    # TODO: Type hint
    dset_path = Path(dset_dir)
    assert dset_path.is_dir()

    graph, feats, id_map, class_map = load_data(dset_dir, prefix)
    return graph, feats, id_map, class_map


def decode_graph(graph, id_map):
    """Decode graph node ids according to id_map
    Args:
    """
    # TODO: Type hint
    for i in tqdm(range(len(graph["nodes"]))):
        graph["nodes"][i]["id"] = id_map[graph["nodes"][i]["id"]]
    return graph


def decode_class_map(class_map, id_map):
    """Decode class_map node ids according to id_map
    Args:
    """
    # TODO: Type hint
    decoded_class_map = {}
    for k, v in class_map.items():
        decoded_class_map[f"{id_map[k]}"] = v
    return decoded_class_map


def construct_better_graph(graph):
    """Construct a graph representation for easy partitioning of splits
    Args:
    """
    # TODO: Type hint
    better_graph = {
        "directed": False,
        "graph": None,
        "nodes": {},
        "links": [],
        "multigraph": False,
    }
    bad_edges = []
    # Pull out id as the node key
    for node in tqdm(graph["nodes"]):
        better_graph["nodes"][node["id"]] = {"test": node["test"], "val": node["val"]}

    # Label the edges for training
    for edge in tqdm(graph["links"]):
        src = edge["source"]
        tgt = edge["target"]
        if src not in better_graph["nodes"] or tgt not in better_graph["nodes"]:
            bad_edges.append(edge)
        else:
            if (
                better_graph["nodes"][src]["val"]
                or better_graph["nodes"][src]["test"]
                or better_graph["nodes"][tgt]["val"]
                or better_graph["nodes"][tgt]["test"]
            ):
                better_graph["links"].append({"src": src, "tgt": tgt, "train": False})
            else:
                better_graph["links"].append({"src": src, "tgt": tgt, "train": True})
    print(f"Discarded {len(bad_edges)} edges with missing nodes")
    return better_graph


def make_senders_receivers(graph, split):
    """Make Jraph compatible jnp arrays for senders and receivers
    Args:
    """
    senders = []
    receivers = []
    for edge in tqdm(graph["links"]):
        senders.append(edge["src"])
        receivers.append(edge["tgt"])
    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    return senders, receivers


def make_graph_tuple(graph: Dict[str, Any], feats: np.ndarray) -> jraph.GraphsTuple:
    """Make a jraph compatible graph tuple
    Args:
        graph (Graph): Graph representation to be converted to GraphsTuple
        feats (np.ndarray): Array of node features
    Returns:
        gt (jraph.GraphsTuple): A new GraphsTuple created using the old graph and features
    """
    # Make feature matrix
    features = jnp.array(feats)

    # Make senders and receivers
    senders = []
    receivers = []
    for edge in tqdm(graph["links"]):
        senders.append(edge["src"])
        receivers.append(edge["tgt"])
    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    # Make edge features
    edges = jnp.ones((senders.size, 1))

    # Declare number of nodes and edges
    n_nodes = len(graph["nodes"])
    n_edges = senders.size

    gt = jraph.GraphsTuple(
        nodes=features,
        senders=senders,
        receivers=receivers,
        edges=edges,
        n_node=n_nodes,
        n_edge=n_edges,
        globals=None,
    )
    return gt


def setup(dset_dir: str, prefix: str):
    """Parse graph and feature data into a jraph compatible GraphsTuple
    Args:
    """
    # TODO: Fix typehints for all functions
    graph, feats, id_map, class_map = load_graph_data(dset_dir, prefix)

    graph = decode_graph(graph, id_map)
    graph = construct_better_graph(graph)

    class_map = decode_class_map(class_map, id_map)

    normed_feats = normalize_features(graph, feats)

    gt = make_graph_tuple(graph, normed_feats)

    return gt, class_map, normed_feats
