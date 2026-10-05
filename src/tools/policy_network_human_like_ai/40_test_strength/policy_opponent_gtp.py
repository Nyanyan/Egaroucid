#!/usr/bin/env python3
"""Policy-model geometric mixtures and a uniformly random legal-move baseline."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "30_blend_with_egaroucid"))
from blend_gtp_engine import BlendGtpEngine
from blend_policy_with_egaroucid import BoardState, PolicyBatchClient, parse_side, policy_to_coord


def geometric_model_distribution(normal, weighted, legal, beta, *, logits=True):
    """Normalized P_normal**(1-beta) * P_weighted**beta on legal moves."""
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    result = np.zeros(64, dtype=np.float64)
    legal = np.asarray(legal, dtype=np.int64)
    if not len(legal):
        return result

    def values(source):
        selected = np.asarray(source, dtype=np.float64)[legal]
        if not np.all(np.isfinite(selected)):
            raise ValueError("non-finite policy output on legal moves")
        if logits:
            return selected
        if np.any(selected < 0):
            raise ValueError("negative policy probability")
        with np.errstate(divide="ignore"):
            return np.log(selected)

    # Do not multiply unused log(0) by zero at the endpoints.
    scores = (values(normal) if beta == 0 else values(weighted) if beta == 1
              else (1 - beta) * values(normal) + beta * values(weighted))
    peak = np.max(scores)
    if not np.isfinite(peak):
        raise ValueError("models have no shared positive-probability legal move")
    weights = np.exp(scores - peak)
    result[legal] = weights / weights.sum()
    return result


class PolicyOpponent(BlendGtpEngine):
    def __init__(self, args):
        self.args = args
        self.state = BoardState.initial()
        self.last_move_stone_loss = None
        self.game_seed = args.seed
        self.rng = random.Random(self.game_seed)
        self.blender = ExitStack()  # inherited GTP loop closes all clients
        self.normal = self.weighted = None
        if args.mode == "geometric":
            if not 0 <= args.beta <= 1:
                raise ValueError("beta must be in [0, 1]")
            if args.beta < 1:
                self.normal = PolicyBatchClient(args.host, args.normal_port, args.timeout_sec)
                self.blender.callback(self.normal.close)
            if args.beta > 0:
                self.weighted = PolicyBatchClient(args.host, args.weighted_port, args.timeout_sec)
                self.blender.callback(self.weighted.close)

    def clear_board(self):
        self.rng.seed(self.game_seed)
        return super().clear_board()

    def genmove(self, color):
        side = parse_side(color)
        legal = self.state.legal_policies(side)
        if not legal:
            self.state.side = side ^ 1
            return "pass"
        if self.args.mode == "random":
            move = self.rng.choice(legal)
        else:
            normal = self.normal.predict(self.state, side)[0] if self.normal is not None else None
            weighted = self.weighted.predict(self.state, side)[0] if self.weighted is not None else None
            probabilities = geometric_model_distribution(normal, weighted, legal, self.args.beta,
                                                          logits=self.args.output == "logits")
            move = max(legal, key=lambda policy: float(probabilities[policy]))
        self.state.apply_move(side, move)
        return policy_to_coord(move)

    def dispatch(self, command, args):
        if command.lower() == "list_commands" and self.args.mode == "random":
            keep_running, commands = super().dispatch(command, args)
            return keep_running, commands + "\nset_game_seed"
        if command.lower() == "name":
            return True, "UniformRandom" if self.args.mode == "random" else "GeometricPolicyModels"
        if command.lower() == "set_game_seed":
            if self.args.mode != "random" or len(args) != 1:
                raise ValueError("set_game_seed requires one integer for the random opponent")
            self.game_seed = int(args[0])
            self.rng.seed(self.game_seed)
            return True, ""
        return super().dispatch(command, args)


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("geometric", "random"), required=True)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--normal-port", type=int)
    parser.add_argument("--weighted-port", type=int)
    parser.add_argument("--timeout-sec", type=float, default=30)
    parser.add_argument("--output", choices=("logits", "probabilities"), default="logits")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    PolicyOpponent(make_argparser().parse_args()).loop()
