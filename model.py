from typing import Sequence
from functools import partial

import jax
from jax import numpy as jnp, random
import networkx as nx
from networkx.classes.graph import Graph
import flax.linen as nn

from sage_convolve import sage_convolve
from load_dset import load_graph_data


class GraphSAGE(nn.Module):
    """Implementation of the GraphSAGE model from: https://arxiv.org/pdf/1706.02216.pdf. Specifically "Algorithm 1", where the sampling and convolution are done simeltaneously.
    Args:
        features (Sequence[int]): The convolution filter weight matrices features
    Procedure:
        1. 
    """
    features: Sequence[int]

    @nn.compact
    def __call__(
        self, key: jnp.ndarray, graph: Graph, feats: jnp.ndarray, max_nodes: int
    ) -> jnp.ndarray:
        """Takes as input a graph, SAGE's and applies nonlinearity and weights to each node
        Args:
            key (jnp.ndarray): RNG key
            graph (Graph): Graph representation of data
            feats (jnp.ndarray): Node features
            max_nodes (int): Maximum number of neighbour nodes to sample
        Returns:
            new_feats (jnp.ndarray): Convolved node feature data
        """
        node_features = feats
        all_visited_nodes = {}
        from tqdm import tqdm
        for i, kernel in tqdm(enumerate(self.features)):
            # Arrange keys
            key, subkey = random.split(key)
            rng_keys = random.split(subkey, num=(node_features.shape[0]))

            # Arrange nodes to be convolved
            G = graph
            convolve_fn = partial(sage_convolve, max_nodes=max_nodes, graph=G, feats=node_features)
            G_nodes = random.permutation(k2, jnp.array(G.nodes()))

            # Convolve each node
            new_feats = []
            layer_visited_nodes = {}
            for rng_k, node in tqdm(
                zip(rng_keys, G_nodes)
            ):  # This for loop is slow -> vmap
                visited, feature_vec = convolve_fn(rng_k, node)
                layer_visited_nodes[f'node_{node}'] = visited
                new_feats.append(feature_vec)
            new_feats = jnp.array(new_feats)
            all_visited_nodes[f'layer{i}'] = layer_visited_nodes

            # Apply weights and nonlinearity to each nodes updated features
            new_feats = jax.vmap(nn.Dense(kernel, name=f"layer_{i}"))(new_feats)
            node_features = nn.relu(new_feats)
        breakpoint()
        node_features = nn.softmax(node_features)
        return new_feats, all_visited_nodes 


# Init rng keys
key = random.PRNGKey(42069)
k1, k2, k3, k4 = random.split(key, 4)

# Load data
dset_dir = "/h/mchoi/graph_networks/my_graphsage/datasets/toy_ppi_data"
prefix = "toy-ppi"
dset_dir = "/h/mchoi/graph_networks/my_graphsage/datasets/reddit"
prefix = "reddit"
graph, feats, id_map, class_map = load_graph_data(dset_dir, prefix)
graph = nx.relabel_nodes(graph, mapping=id_map) # inplace
breakpoint()

# Init Model
model = GraphSAGE(features=[32, 16, 8, 2])
dummy_graph = graph.subgraph([0] + list(nx.neighbors(graph, 0)))
params = model.init(k2, k3, dummy_graph, feats, 25)

# Test forward pass
new_features, all_visited_nodes= model.apply(params, k4, graph, feats, 25)
breakpoint()
