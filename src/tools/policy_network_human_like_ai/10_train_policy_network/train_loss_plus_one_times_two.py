#!/usr/bin/env python3
"""Train Conv8+BN with sample_weight = (disc_loss + 1) * 2."""
from train_loss_plus_one import LinearWeighting, main as train_linear

WEIGHTING = LinearWeighting(multiplier=2, offset=2, formula="(disc_loss + 1) * 2",
                           label="loss_plus_one_times_two")


def main(argv=None):
    train_linear(argv, weighting=WEIGHTING)


if __name__ == "__main__":
    main()
