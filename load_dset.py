import os
from pathlib import Path
import json
from typing import Tuple, List, Dict, Union, Any, Sequence

import jax
from jax import numpy as jnp, random
import numpy as np
from sklearn.preprocessing import StandardScaler
import jraph

"""
Dataset implementation for GraphSAGE PPI and Reddit datasets from https://github.com/williamleif/GraphSAGE
Description of dataset files:
    1. "-G.json": A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
    2. "-id_map.json": A json-stored dictionary mapping the graph node ids to consecutive integers.
    3. "-class_map.json": A json-stored dictionary mapping the graph node ids to classes.
    4. "-feats.npy": A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
"""


def load_graph_data(
    root: str, prefix: str
) -> List[Union[Dict[str, Sequence[Any]], np.ndarray]]:
    """Load necessary files to construct the graph representation and the node features
    Args:
        root (str): Path to dataset folder
        prefix (str): Dataset name prefix
    Returns:
        data (list): List of
    """
    # TODO: List[Dict[str, Any], np.ndarray, Dict[str, List[int]], Dict[str, int]]
    assert Path(root).is_dir()

    file_suffixes = ["-G.json", "-feats.npy", "-id_map.json", "-class_map.json"]
    filepaths = [Path(root + "/" + prefix + sfx) for sfx in file_suffixes]
    data = [
        json.load(open(fp)) if fp.suffix == ".json" else np.load(fp) for fp in filepaths
    ]
    return data


def normalize_features(
    graph: Dict[str, Sequence[Any]], feats: np.ndarray
) -> np.ndarray:
    """Normalize the training node features"""
    train_ids = np.array(
        [
            i
            for i, node in enumerate(graph["nodes"])
            if node["test"] is False and node["val"] is False
        ]
    )
    train_feats = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    return scaler.transform(feats)


def decode_graph(
    graph: Dict[str, Sequence[Any]], id_map: Dict[int, int]
) -> Dict[str, Sequence[Any]]:
    """Decode graph node id value according to id_map dictionary
    Args:
    """
    for node_idx in range(len(graph["nodes"])):
        decoded_id = id_map[graph["nodes"][node_idx]["id"]]
        graph["nodes"][node_idx]["id"] = decoded_id
    return graph


def decode_class_map(
    class_map: Dict[int, int], id_map: Dict[int, int]
) -> Dict[int, int]:
    """Decode class_map node ids according to id_map
    Args:
    """
    decoded_class_map = {}
    for k, v in class_map.items():
        decoded_class_map[id_map[k]] = v
    return decoded_class_map


def split_graph_data(
    graph: Dict[str, Sequence[Any]]
) -> Tuple[
    Dict[str, Sequence[Any]], Dict[str, Sequence[Any]], Dict[str, Sequence[Any]]
]:
    """Construct new graph representations, where the given graph is split between 
    train/val/test. Train/val/test nodes/features are mutually exclusive between the 
    split graph representations. Edges are allocated as follows:
        - Training edges are any edge connected to one or more training nodes
        - Val/test edges are edges which are exclusively connected to val/test nodes
    Args:
    """
    # TODO: Type hint
    train_graph = {
        "directed": False,
        "graph": None,
        "nodes": {},
        "links": [],
        "multigraph": False,
    }
    val_graph = {
        "directed": False,
        "graph": None,
        "nodes": {},
        "links": [],
        "multigraph": False,
    }
    test_graph = {
        "directed": False,
        "graph": None,
        "nodes": {},
        "links": [],
        "multigraph": False,
    }

    bad_edges = []
    # Split nodes
    # Nodes are mutex between splits
    for node in graph["nodes"]:
        assert not (node["val"] is True and node["test"] is True)
        if node["val"] is True:
            val_graph["nodes"][node["id"]] = {"test": node["test"], "val": node["val"]}
        elif node["test"] is True:
            test_graph["nodes"][node["id"]] = {"test": node["test"], "val": node["val"]}
        else:
            train_graph["nodes"][node["id"]] = {
                "test": node["test"],
                "val": node["val"],
            }

    for edge in graph["links"]:
        src = edge["source"]
        tgt = edge["target"]
        if (
            src not in train_graph["nodes"]
            and src not in val_graph["nodes"]
            and src not in test_graph["nodes"]
        ) or (
            tgt not in train_graph["nodes"]
            and tgt not in val_graph["nodes"]
            and tgt not in test_graph["nodes"]
        ):
            bad_edges.append(edge)
        else:
            if (
                graph["nodes"][src]["val"]
                or graph["nodes"][src]["test"]
                or graph["nodes"][tgt]["val"]
                or graph["nodes"][tgt]["test"]
            ):  # Edges shared between only val/test nodes can be ignored for training
                val_graph["links"].append({"src": src, "tgt": tgt, "train": False})
                test_graph["links"].append({"src": src, "tgt": tgt, "train": False})
            else:
                train_graph["links"].append({"src": src, "tgt": tgt, "train": True})
                val_graph["links"].append({"src": src, "tgt": tgt, "train": False})
                test_graph["links"].append({"src": src, "tgt": tgt, "train": False})

    print(f"Discarded {len(bad_edges)} edges with missing nodes")
    return train_graph, val_graph, test_graph


def make_graphstuple(
    graph: Dict[str, Any], feats: np.ndarray, class_map: Dict[int, int]
) -> Tuple[
    jraph.GraphsTuple,
    jnp.ndarray,
    List[int]
]:
    """Make a jraph compatible graph tuple. Additionally, return an array of the corresponding sorted class mappings
       and a list of the sorted node ids (useful when the graph is split, since nodes may be missing in a given
       split.
    Args:
        graph (Graph): Graph representation to be converted to GraphsTuple
        feats (np.ndarray): Array of node features
    Returns:
        gt (jraph.GraphsTuple): A new GraphsTuple created using the old graph and features
    """
    # Make senders and receivers
    senders = []
    receivers = []
    for edge in graph["links"]:
        senders.append(edge["src"])
        receivers.append(edge["tgt"])

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    # Make edge features
    edges = jnp.ones((senders.size, 1))

    # Declare number of nodes and edges
    n_nodes = len(graph["nodes"])
    n_edges = senders.size

    # split_node_ids and split_class_values need to have identical corresponding
    # values 
    #   - ie. split_node_ids = [1, 3, 5, 9, 19], then:
    #   -     split_vlass_values = [class_map[1], class_map[3], ..., class_map[19]]
    features = jnp.array(feats)
    split_node_ids = sorted(list(graph["nodes"].keys()))
    features = features[jnp.array(split_node_ids)]
    split_class_values = jnp.array([class_map[k] for k in split_node_ids])

    gt = jraph.GraphsTuple(
        nodes=features,
        senders=senders,
        receivers=receivers,
        edges=edges,
        n_node=n_nodes,
        n_edge=n_edges,
        globals=None,
    )

    return gt, split_class_values, split_node_ids


def setup(
    dset_dir: str, prefix: str
) -> Tuple[
    jraph.GraphsTuple,
    jraph.GraphsTuple,
    jraph.GraphsTuple,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray
]:
    """Parse graph and feature data into a jraph compatible GraphsTuple
    Args:
    """
    # TODO: Fix typehints for all functions
    graph, feats, id_map, class_map = load_graph_data(dset_dir, prefix)
    normed_feats = normalize_features(graph, feats)

    class_map = decode_class_map(class_map, id_map)

    graph = decode_graph(graph, id_map)
    train_graph, val_graph, test_graph = split_graph_data(graph)

    train_gt, train_class_map, train_node_ids = make_graphstuple(train_graph, normed_feats, class_map)
    val_gt, val_class_map, val_node_ids = make_graphstuple(val_graph, normed_feats, class_map)
    test_gt, test_class_map, test_node_ids = make_graphstuple(test_graph, normed_feats, class_map)

    return train_gt, val_gt, test_gt, train_class_map, val_class_map, test_class_map, train_node_ids, val_node_ids, test_node_ids


# These are pulled directly from: https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
def get_zacharys_karate_club() -> jraph.GraphsTuple:
  """Returns GraphsTuple representing Zachary's karate club."""
  social_graph = [
      (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
      (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
      (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
      (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
      (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
      (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
      (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
      (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
      (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
      (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
      (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
      (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
      (33, 31), (33, 32)]
  # Add reverse edges.
  social_graph += [(edge[1], edge[0]) for edge in social_graph]
  n_club_members = 34

  return jraph.GraphsTuple(
      n_node=jnp.asarray([n_club_members]),
      n_edge=jnp.asarray([len(social_graph)]),
      # One-hot encoding for nodes, i.e. argmax(nodes) = node index.
      nodes=jnp.eye(n_club_members),
      # No edge features.
      edges=None,
      globals=None,
      senders=jnp.asarray([edge[0] for edge in social_graph]),
      receivers=jnp.asarray([edge[1] for edge in social_graph]))

def get_ground_truth_assignments_for_zacharys_karate_club() -> jnp.ndarray:
  """Returns ground truth assignments for Zachary's karate club."""
  return jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                    0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
