import argparse
from typing import Dict, List
from functools import partial

import jax
from jax import numpy as jnp, random
import jraph
import flax.linen as nn
import optax

from model import GCNNet
from load_dset import setup, get_zacharys_karate_club, get_ground_truth_assignments_for_zacharys_karate_club


def boolean_string(s: str) -> bool:
    """Function for proper processing of bools passed in through argparse + commandline"""
    if s not in {"True", "False"}:
        raise ValueError("Not a proper boolean string")
    else:
        return s == "True"


def parse_args():
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument(
        "--dset_dir",
        type=str,
        default="/h/mchoi/graph_networks/graph_NNs_jax/datasets/reddit",
    )
    parser.add_argument("--key", type=int, default=42069)
    parser.add_argument("--prefix", type=str, default="reddit")
    parser.add_argument("--batch_size", type=int, default=69)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sanity_check", default=False, type=boolean_string)
    args = parser.parse_args()
    return args


def cross_entropy_loss(params, model, gt, targets):
    """Computes cross-entropy loss"""

    def cross_entropy(x, t):
        return -(jax.nn.one_hot(t, 2) * nn.log_softmax(x))

    preds = model.apply(params, gt)
    return jnp.mean(jax.vmap(cross_entropy)(preds, targets))


def accuracy(params, model, gt, targets):
    preds = model.apply(params, gt)
    return jnp.mean(jnp.argmax(preds, axis=1) == targets)


def fit(steps, params, model, optimizer, gt, targets: Dict[int, int]):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, gt, targets):
        loss_value, grads = jax.value_and_grad(cross_entropy_loss)(params, model, gt, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    from tqdm import tqdm
    #for i in tqdm(range(steps)):
    for i in range(steps):
        params, opt_state, loss_value = step(params, opt_state, gt, targets) 
        if i % max(steps // 100, 1) == 0:
            acc = accuracy(params, model, gt, targets)
            print(f'step {i} loss: {loss_value} | accuracy: {acc.item()}')


def make_batch_idxs(key, batch_size, gt, class_map):
    """Creates a list of batches, of batch_size, randomly permuted according to key. Zero-pads
       the final batch to be exactly batch_size.
    Args:
    """
    batch_idxs = random.permutation(key, jnp.arange(gt.nodes.shape[0]))
    batches = []
    for idx in range(0, gt.nodes.shape[0], batch_size):
        if idx + batch_size > gt.nodes.shape[0]:
            batches.append(jnp.concatenate((batch_idxs[idx:], jnp.zeros(idx + batch_size - gt.nodes.shape[0]))))
        else:
            batches.append(batch_idxs[idx:idx + batch_size])
    return batches


def batch_graph(batch_idxs: jnp.ndarray, gt: jraph.GraphsTuple, class_map: jnp.ndarray, node_ids: List[int], layers: int) -> jraph.GraphsTuple:
    """Takes a batch consiting of batch_idxs, and collects the necessary nodes/features necessary
       to process the batch.
    Args:
        batch_idxs (jnp.ndarray): An array corresponding to one batch, containing permuted node indices
        gt (jraph.GraphsTuple): The graph data 
        class_map (jnp.ndarray): An ordered array containing the ground-truth class values
        layers (int): Number of layers in the model
    """
    """
    all_nbs = []
    for idx in batch_idxs:
        neighbours = jnp.where(depth_limited_bfs(idx, gt, layers) != 0)
        all_nbs.append(neighbours)
    """
    #TODO: Remove
    batch_idxs = batch_idxs[0]
        
    @jax.jit
    def one_hop_neighbours(idx, gt):
        """Find the 1-hop neighbours to node idx in graph gt.
        Args:
        """
        neighbours = jnp.where(jnp.equal(idx, gt.senders), gt.receivers, 0) # Needs 3 arguments to be jit compatible
        return neighbours

    batch_nodes = []
    for i in range(layers):
        batch_idxs = jax.vmap(partial(one_hop_neighbours, gt=gt))(batch_idxs)
        batch_nodes.append(batch_idxs)
        
    breakpoint()


def sanity_check():
    """Runs a training run on Zachary's Karate Club dataset, taken from:     
       https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
    """
    print(f"Running sanity check...")
    main_key = random.PRNGKey(args.key)
    dummy_gt = get_zacharys_karate_club()
    dummy_targets = get_ground_truth_assignments_for_zacharys_karate_club()

    model = GCNNet(features=[8, 2])

    params = model.init(main_key, dummy_gt)

    optimizer = optax.adam(learning_rate=1e-2)

    params = fit(30, params, model, optimizer, dummy_gt, dummy_targets)
    print(f"Sanity check completed.")

    return params 


def main(args):
    if args.sanity_check is True:
        sanity_check()


    main_key = random.PRNGKey(args.key)
    print("Setting up training data...")
    (
        train_gt,
        val_gt,
        test_gt,
        train_class_map,
        val_class_map,
        test_class_map,
        train_node_ids,
        val_node_ids,
        test_node_ids
    ) = setup(dset_dir=args.dset_dir, prefix=args.prefix)
    print("Training data setup complete.")
    model_feats = [128, 128, 128, 128, 128, 64, 50]
    main_key, subkey = random.split(main_key)

    #TODO: 
    #batch_idxs = make_batch_idxs(subkey, args.batch_size, train_gt, train_class_map)
    #batches = batch_graph(batch_idxs, train_gt, train_class_map, train_node_ids, len(model_feats))

    # Setup model and params
    main_key, subkey = random.split(main_key)
    model = GCNNet(features=model_feats)
    params = model.init(subkey, train_gt)

    # Setup optimizer
    optimizer = optax.adam(learning_rate=args.lr)

    # Train model
    params = fit(args.steps, params, model, optimizer, train_gt, train_class_map)
    breakpoint()

if __name__ == "__main__":
    args = parse_args()
    main(args)
