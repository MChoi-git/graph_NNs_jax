import argparse

import jax
from jax import numpy as jnp, random
import flax.linen as nn
import optax

from model import MyGCN
from load_dset import setup

#TODO: Batch graphs according to train/val/test


def parse_args():
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument(
        "--dset_dir",
        type=str,
        default="/h/mchoi/graph_networks/graph_NNs_jax/datasets/reddit",
    )
    parser.add_argument("--prefix", type=str, default="reddit")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    return args


def cross_entropy_loss(params, model, gt, targets):
    """Computes cross-entropy loss"""

    def cross_entropy(x, t):
        return -(jax.nn.one_hot(t, 50) * nn.log_softmax(x))

    preds = model.apply(params, gt)
    return jnp.mean(jax.vmap(cross_entropy)(preds, targets))


def fit(params, model, optimizer, gt, targets):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, gt, targets):
        loss_value, grads = jax.value_and_grad(cross_entropy_loss)(params, model, gt, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    from tqdm import tqdm
    for i in tqdm(range(10)):
        params, opt_state, loss_value = step(params, opt_state, gt, targets) 
        if i % 1 == 0:
            #TODO: accuracy function
            print(f'step {i} loss: {loss_value}')


def main(args):
    # Prep data struct
    gt, class_map = setup(dset_dir=args.dset_dir, prefix=args.prefix)
    class_map = jnp.array(jax.tree_util.tree_leaves(class_map))
    assert gt.nodes.shape[0] == class_map.shape[0]

    # Setup model and params
    model = MyGCN(features=[1024, 512, 256, 64, 50])
    params = model.init(random.PRNGKey(42069), gt)

    # Setup optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    params = fit(params, model, optimizer, gt, class_map)
    breakpoint()

if __name__ == "__main__":
    args = parse_args()
    main(args)
