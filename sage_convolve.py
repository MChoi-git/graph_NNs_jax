import os
from pathlib import Path
import json
from functools import partial
from typing import List, Tuple

import networkx as nx
from networkx.classes.graph import Graph
import jax
from jax import numpy as jnp, random, nn
import numpy as np

from load_dset import load_graph_data


def limit_bfs(key: jnp.ndarray, graph: Graph, root: int, max_nodes: int) -> List[int]:
    """Breadth first search constrained by the max_nodes, limiting # of visited nodes
    Args:
        key (jnp.ndarray): RNG key to randomize visit order
        graph (Graph): Graph representation
        root (int): Root node to BFS around
        max_nodes (int): Max number of nodes to visit
    Returns:
        visited (List[int]): List of visited nodes
    """
    visited = [root]
    node_queue = [root]

    while len(node_queue) != 0:
        curr_node = node_queue.pop()
        k_iter, key = random.split(key)  # Each loop iteration should permute neighbours

        # Randomize visit order in the case of max_nodes < deg(root)
        neighbours = list(nx.neighbors(graph, int(curr_node)))
        visit_order = jax.random.permutation(k_iter, jnp.array(neighbours))

        for node in visit_order:

            if node not in visited:
                visited.append(node)

                if len(visited) >= max_nodes:
                    return visited
                else:
                    node_queue.insert(0, node)
    return visited


def sample_node_neighbourhood(
    key: jnp.ndarray, idx: int, max_nodes: int, graph: Graph, feats: np.ndarray
) -> Tuple[List[int], jnp.ndarray]:
    """Get a single graph example in a neighbourhood around node idx. Returns the feature vectors
    Args:
        key (jnp.ndarray): RNG key for the BFS
        idx (int): Node id
        max_nodes (int): Maximum number of nodes to collect in full neighbourhood
        graph (Graph): Graph representation of data
        feats (np.ndarray): Feature vectors for each node
    Returns:
        visited (List[int]): List of visited nodes
        feature_vecs (jnp.ndarray): Corresponding features for the visited nodes
    """
    visited = jnp.array(limit_bfs(key, graph, idx, max_nodes))
    feature_vecs = jnp.take(feats, visited, axis=0)
    return visited, feature_vecs


def aggregate_fn(feature_vecs: jnp.ndarray, agg_type: str) -> jnp.ndarray:
    """Applies aggregation function
    Args:
        feature_vecs (jnp.ndarray): Feature vectors to be aggregated
        agg_type (str): Aggregation type (ie. mean, pool, etc.)
    """
    if agg_type == "mean":
        agg_fn = partial(jnp.mean, axis=-2, keepdims=True)
    new_vec = agg_fn(feature_vecs)
    return new_vec


def sage_convolve(
    key: jnp.ndarray,
    idx: int,
    max_nodes: int,
    graph: Graph,
    feats: jnp.ndarray,
    agg_fn: str = "mean",
) -> Tuple[List[int], jnp.ndarray]:
    """Samples feature vectors in neighbourhood for one node, then applies aggregation function
    Args:
        key (jnp.ndarray): RNG key
        idx (int): Node id to convolve
        max_nodes (int): Maximum amount of nodes to sample in neighbourhood
        graph (Graph): Graph representation of data
        feats (jnp.ndarray): Features
        agg_fn (str): Aggregation function type
    Returns:
        visited (List[int]): 
        cat_aggregated_vec (jnp.ndarray): 
    """
    # Sample neighbourhood
    visited, feature_vecs = jax.lax.stop_gradient(
        sample_node_neighbourhood(key, idx, max_nodes, graph, feats)
    )
    # Apply aggregation fn
    aggregated_vec = aggregate_fn(feature_vecs, agg_fn)

    # Concatenate root node
    cat_aggregated_vec = jnp.concatenate(
        (jnp.array(feats[idx]), aggregated_vec), axis=None
    )
    return visited, cat_aggregated_vec


"""
graph, feats, id_map, class_map = load_graph_data(
    "/h/mchoi/graph_networks/my_graphsage/datasets/toy_ppi_data", "toy-ppi"
)
feats = jnp.array(feats)

key = random.PRNGKey(42069)

key, k1 = random.split(key)
visited, feature_vecs = sample_node(k1, 0, 25, graph, feats)
assert sum(list(nx.neighbors(graph, 0))) == visited[:20].sum()

key, k2 = random.split(key)
param = random.normal(k2, (69, feature_vecs.shape[0] * 2))
test_result = sage_convolve(key, 0, 25, graph, feats)
breakpoint()
"""
