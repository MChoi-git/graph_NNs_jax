from typing import Sequence, Callable

import jax
from jax import numpy as jnp, random
import jraph
import flax.linen as nn
import numpy as np

from load_dset import load_graph_data


class MLP(nn.Module):
    """Stateless one layer MLP"""
    features: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Dense(self.features, name="layer0")(x)
        x = nn.relu(x)
        return x


class GCNLayer(nn.Module):
    features: int
    aggregation_fn: Callable = jax.ops.segment_sum
    add_self_edges: bool = True

    @nn.compact
    def __call__(self, gt: jraph.GraphsTuple):
        new_gt = gt
        nodes, _, receivers, senders, _, n_node, n_edge = new_gt

        # Apply learnable fn phi(n) to each node n in V
        nodes = nn.Dense(self.features)(nodes)
        nodes = nn.relu(nodes)

        # Add self-edges
        if self.add_self_edges is True:
            senders = jnp.concatenate((senders, jnp.arange(nodes.shape[0])))
            receivers = jnp.concatenate((receivers, jnp.arange(nodes.shape[0])))

        # Calculate node degrees and normalize features
        # Need to explicitly tell segment_sum the number of segments to be jittable
        node_deg = jax.ops.segment_sum(jnp.ones_like(receivers), senders, nodes.shape[0])

        # Non-symmetric norm by degrees (undirected)
        nodes = nodes * jax.lax.reciprocal(jnp.maximum(node_deg, 1.0))[:, None]
        nodes = self.aggregation_fn(nodes[senders], receivers, nodes.shape[0])

        return new_gt._replace(nodes=nodes)


class GCNNet(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, gt: jraph.GraphsTuple):
        new_gt = gt
        for i, feat in enumerate(self.features):
            new_gt = GCNLayer(feat)(new_gt)
            if i < len(self.features) - 1:
                nonlin = nn.relu(new_gt.nodes)
                new_gt._replace(nodes=nonlin)
        return new_gt.nodes
