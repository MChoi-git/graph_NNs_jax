import networkx as nx
import jax
from jax import numpy as jnp, random

from load_dset import load_graph_data


def limit_bfs(key, root, graph, max_nodes):
    """Simple BFS starting from root, over graph, terminating when len(visited) > max_nodes
    Args:
    """
    visited = [root]
    queue = [root]
    
    while len(queue) != 0:
        curr_node = queue.pop()

        neighbours = list(nx.neighbors(graph, int(curr_node)))
        visit_order = random.permutation(key, jnp.array(neighbours))

        for node in visit_order:

            if node not in visited:
                visited.append(int(node))

                if len(visited) >= max_nodes:
                    return visited
                else:
                    queue.insert(0, node)
    return visited
                

key = random.PRNGKey(42069)
k1, k2, k3 = random.split(key, 3)
graph, feats, id_map, class_map = load_graph_data(
    "/h/mchoi/graph_networks/my_graphsage/datasets/toy_ppi_data", "toy-ppi"
)
from tqdm import tqdm
samples = {}
for node in tqdm(graph.nodes()):
    samples[f'{node}'] = jnp.array(limit_bfs(k1, node, graph, max_nodes=25))
# Applies mean aggregation for fun :)
func = lambda x: jnp.mean(jnp.take(feats, x, axis=0), axis=0, keepdims=True)
sampled_feats = jax.tree_map(func, samples)
breakpoint()
