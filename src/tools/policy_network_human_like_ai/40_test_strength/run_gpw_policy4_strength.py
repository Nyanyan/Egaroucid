#!/usr/bin/env python3
"""Run the four unblended policies from GPW-26 Section 3.3 on fresh XOT samples."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import battle_blend_strength as battle
import run_gpw_strength as gpw
from strength_tournament import PlayerSpec


HERE = Path(__file__).resolve().parent


class FourPolicyPlayers(gpw.PaperPlayers):
    """Reuse the paper's model checks and servers, with only A and B added."""

    resource_alphas = (1.0, 1.0)

    def add_specs(self, args, original_specs, port, original_port=0):
        specs = [replace(spec, name="normal") if spec.name == "alpha_1.0" else spec
                 for spec in original_specs]
        loss_b_port = self.extra_policy_servers["loss_b"].port if port and self.extra_policy_servers else 0
        for name, model_port in (("loss_a", port), ("loss_b", loss_b_port)):
            # beta=1 contacts only the weighted-model server; no mixture or search.
            command = (sys.executable, str(HERE / "policy_opponent_gtp.py"),
                       "--mode", "geometric", "--beta", "1.0",
                       "--host", args.policy_server_host, "--weighted-port", str(model_port),
                       "--output", "logits", "--timeout-sec", str(args.policy_server_timeout_sec))
            specs.append(PlayerSpec(name, command, args.blend_processes_per_player, True, 1.0))
        if [spec.name for spec in specs] != ["egaroucid_l21", "normal", "loss_a", "loss_b"]:
            raise ValueError("Four-policy roster requires Egaroucid L21, normal, loss A and loss B")
        return specs

    def update_configuration(self, configuration):
        super().update_configuration(configuration)
        configuration["participants"]["paper_roster"] = {
            "source": "GPW-26 camera-ready Section 3.3 and Tables 1 and 2",
            "total_players": 4,
            "policies": ["egaroucid_l21", "normal", "loss_a", "loss_b"],
            "selection": "argmax over legal moves",
            "training": self.evidence,
        }
        configuration["inputs"]["implementation_sources"]["four_policy_roster"] = battle.file_identity(Path(__file__))


def make_argparser():
    parser = gpw.make_argparser()
    parser.description = __doc__
    parser.set_defaults(baseline_levels="21", alphas="1.0", edax_levels="",
                        seed=58, match_sets_per_pair=50,
                        output_dir=HERE / "output/gpw4_independent_xot_50sets_seed58")
    return parser


def prepare(args):
    if (battle.parse_int_list(args.baseline_levels) != [21]
            or battle.parse_float_list(args.alphas) != [1.0]
            or battle.parse_int_list(args.edax_levels)):
        raise ValueError("Four-policy roster requires --baseline-levels 21 --alphas 1.0 --edax-levels=")
    if args.policy_backend != "tensorflow":
        raise ValueError("The three trained CNN models require TensorFlow")
    evidence = {name: gpw.model_evidence(path, gpw.MODEL_FORMULAS[name]) for name, path in (
        ("normal", args.policy_model), ("loss_a", args.loss_a_model), ("loss_b", args.loss_b_model))}
    players = FourPolicyPlayers(args, evidence)
    battle.validate_args(args, players.resource_alphas)
    return players


def main(argv=None):
    args = make_argparser().parse_args(argv)
    players = prepare(args)
    with battle.OutputRunLock(args.output_dir):
        battle.run_tournament(args, extension=players)


if __name__ == "__main__":
    main()
