#!/usr/bin/env python3
"""Train Conv8+BN with sample_weight = 2 * disc_loss, including zero weights."""
from train_loss_plus_one import LinearWeighting, main as train_linear

WEIGHTING = LinearWeighting(multiplier=2, offset=0, formula="2 * disc_loss", label="loss_times_two")


def main(argv=None):
    train_linear(argv, weighting=WEIGHTING)


if __name__ == "__main__":
    main()
