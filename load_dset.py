import os
from pathlib import Path
import json
from typing import Tuple, List, Dict

import networkx as nx
from networkx.readwrite import json_graph
from networkx.classes.graph import Graph
import jax
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Dataset implementation for GraphSAGE PPI and Reddit datasets from https://github.com/williamleif/GraphSAGE
Description of dataset files:
    1. "-G.json": A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
    2. "-id_map.json": A json-stored dictionary mapping the graph node ids to consecutive integers.
    3. "-class_map.json": A json-stored dictionary mapping the graph node ids to classes.
    4. "-feats.npy": A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
"""


def load_data(root: str, prefix: str) -> list:
    """Load necessary files to construct the graph representation and the node features
    Args:
        root (str): Path to dataset folder
        prefix (str): Dataset name prefix
    Returns:
        data (list): List of 
    """
    file_suffixes = ["-G.json", "-feats.npy", "-id_map.json", "-class_map.json"]
    filepaths = [Path(root + "/" + prefix + sfx) for sfx in file_suffixes]
    data = [
        json.load(open(fp)) if fp.suffix == ".json" else np.load(fp) for fp in filepaths
    ]
    return data


def clean_annotations(graph: Graph) -> Graph:
    """Remove nodes with no val/test annotations"""
    broken_count = 0
    to_remove = []
    for node in graph.nodes():
        if not "val" in graph.nodes[node] or not "test" in graph.nodes[node]:
            to_remove.append(node)
            broken_count += 1
    for node in to_remove:
        graph.remove_node(node)
    print(f"Removed {broken_count} broken nodes")
    return graph


def annotate_edges(graph: Graph) -> Graph:
    """Annotates edges connected to val/test nodes to be disregarded during training"""
    for edge in graph.edges():
        if (
            graph.nodes[edge[0]]["val"]
            or graph.nodes[edge[0]]["test"]
            or graph.nodes[edge[1]]["val"]
            or graph.nodes[edge[1]]["test"]
        ):
            graph[edge[0]][edge[1]]["train_removed"] = True
        else:
            graph[edge[0]][edge[1]]["train_removed"] = False
    return graph


def normalize_features(graph: Graph, feats: np.ndarray, id_map: Dict) -> np.ndarray:
    """Normalize the node features"""
    train_ids = np.array(
        [
            id_map[str(n)]
            for n in graph.nodes()
            if not graph.nodes[n]["val"] and not graph.nodes[n]["test"]
        ]
    )
    train_feats = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    return scaler.transform(feats)


def load_graph_data(
    dset_dir: str,
    prefix: str,
) -> Tuple[Graph, np.ndarray, Dict, Dict]:
    """Make the batch iterator for the graph NN"""
    dset_path = Path(dset_dir)
    assert dset_path.is_dir()
    print(f"Creating graph dataset from: {dset_dir}")

    graph, feats, id_map, class_map = load_data(dset_dir, prefix)
    graph = json_graph.node_link_graph(graph)
    graph = clean_annotations(graph)
    graph = annotate_edges(graph)
    feats = normalize_features(graph, feats, id_map)

    return graph, feats, id_map, class_map
