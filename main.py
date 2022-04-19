import argparse

import jax
from jax import numpy as jnp, random
import flax.linen as nn

from model import MyGCN
from load_dset import setup


def parse_args():
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument(
        "--dset_dir",
        type=str,
        default="/h/mchoi/graph_networks/graph_NNs_jax/datasets/reddit",
    )
    parser.add_argument("--prefix", type=str, default="reddit")
    args = parser.parse_args()
    return args


def main(args):
    gt, class_map, feats = setup(dset_dir=args.dset_dir, prefix=args.prefix)

    gcn_net = MyGCN(features=[256, 256])
    params = gcn_net.init(random.PRNGKey(42069), gt)
    y = gcn_net.apply(params, gt)
    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
