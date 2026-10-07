#!/usr/bin/env python3
"""Train Conv8+BN with sample_weight = (disc_loss + 1) ** 2."""
from train_loss_plus_one import LossWeighting, main as train_weighted

WEIGHTING = LossWeighting(multiplier=1, offset=1, power=2,
                         formula="(disc_loss + 1) ** 2", label="loss_plus_one_squared")


def main(argv=None):
    train_weighted(argv, weighting=WEIGHTING)


if __name__ == "__main__":
    main()
