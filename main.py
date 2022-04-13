import os
from pathlib import Path

import jax
from jax import numpy as jnp, random
import flax
import flax.linen as nn
import networkx
from networkx.classes.graph import Graph

from load_dset import load_graph_data


# Prepare graph data
graph, feats, id_map, class_map = load_graph_data(
    "/h/mchoi/graph_networks/my_graphsage/datasets/toy_ppi_data", "toy-ppi"
)
feats = jnp.array(feats)

rng = random.PRNGKey(42069)

# Prepare model and parameters

