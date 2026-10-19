#!/usr/bin/env python3
"""Run the 30-player GPW round robin, including Egaroucid and Edax level 3."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import battle_blend_strength as battle
from strength_tournament import PlayerSpec


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
TRAINED = HERE.parent / "10_train_policy_network/trained"
LEVELS = (1, 3, 5, 9, 13, 17, 21)
ALPHAS = (.2, .4, .6, .8, 1.0)
NORMAL_MODEL = WORKSPACE / "experiment_bat/20261002/results/train/selected_model.h5"
LOSS_A_MODEL = TRAINED / "conv8_bn_loss_plus_one_squared_e10_20261006/train/selected_model.h5"
LOSS_B_MODEL = TRAINED / "prepared_conv8_bn_e10_20261003_221904_955/selected_model.h5"
MODEL_FORMULAS = {"normal": None, "loss_a": "(disc_loss + 1) ** 2", "loss_b": "2 ** (disc_loss / 2)"}


def model_evidence(path, formula):
    """Bind each model to the paper's architecture, training run and weighting."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    metadata = {}
    for name in ("selected_summary.json", "training_identity.json", "prepared_dataset.json"):
        source = path.parent / name
        metadata[name] = json.loads(source.read_text(encoding="utf-8"))
    summary = metadata["selected_summary.json"]
    settings = metadata["training_identity.json"]["settings"]
    dataset = metadata["prepared_dataset.json"]
    actual_formula = (dataset.get("sample_weights") or {}).get("formula")
    if actual_formula != formula:
        raise ValueError(f"Wrong training weight formula for {path}: {actual_formula!r}; expected {formula!r}")
    if (summary.get("name") != "conv8_bn" or summary.get("spec", {}).get("architecture") != "conv8_bn"
            or summary.get("epochs_ran") != 10 or summary.get("adopted_epoch") != 10
            or settings.get("epochs") != 10 or settings.get("batch_size") != 512):
        raise ValueError(f"Model must be conv8_bn trained for 10 epochs with batch size 512: {path}")
    if dataset.get("augmentation_factor") != 8:
        raise ValueError(f"Model must use eight-fold board augmentation: {path}")
    for split, games in (("train", 120768), ("val", 6707), ("test", 6706)):
        if dataset["splits"][split]["games"] != games:
            raise ValueError(f"Model {split} game count differs from the paper: {path}")
    return {"sample_weight_formula": formula or "uniform", "architecture": "conv8_bn",
            "epochs": 10, "batch_size": 512,
            "files": {name: battle.file_identity(path.parent / name) for name in metadata}}


class PaperPlayers:
    """Fresh roster adapter; no previous tournament results are imported."""
    resource_alphas = (1.0,) * 11  # Two five-policy NN families and random.
    policy_server_count = 2  # A and B, in addition to the normal-model server.

    def __init__(self, args, evidence):
        self.model = Path(args.loss_a_model).resolve()
        self.loss_b_model = Path(args.loss_b_model).resolve()
        self.normal_model = Path(args.policy_model).resolve()
        self.evidence = evidence
        self.extra_policy_servers = {}

    def missing_imports(self, tasks, loaded_results):
        return []

    def add_specs(self, args, original_specs, port, original_port=0):
        specs = [replace(spec, name=f"normal_egaroucid_{spec.name}")
                 if spec.name.startswith("alpha_") else spec for spec in original_specs]
        script = str(HERE / "policy_opponent_gtp.py")
        loss_b_port = self.extra_policy_servers["loss_b"].port if port and self.extra_policy_servers else 0
        for name, pi2_port, pi1_port in (("loss_a_normal", original_port, port),
                                        ("loss_b_loss_a", port, loss_b_port)):
            for alpha in ALPHAS:
                command = (sys.executable, script, "--mode", "geometric", "--beta", str(alpha),
                           "--host", args.policy_server_host,
                           "--normal-port", str(pi2_port), "--weighted-port", str(pi1_port),
                           "--output", "logits", "--timeout-sec", str(args.policy_server_timeout_sec))
                specs.append(PlayerSpec(f"{name}_alpha_{alpha:.1f}", command,
                                        args.blend_processes_per_player, True, 1.0))
        specs.append(PlayerSpec("random", (sys.executable, script, "--mode", "random", "--seed", str(args.seed)),
                                args.blend_processes_per_player, False, 1.0, random_seed=args.seed))
        if len(specs) != 30 or len({spec.name for spec in specs}) != 30:
            raise ValueError("GPW roster with Egaroucid and Edax level 3 requires exactly 30 distinct players")
        return specs

    def update_configuration(self, configuration):
        configuration["participants"]["paper_roster"] = {
            "source": "GPW-26 camera-ready Tables 2 and 3, plus Egaroucid and Edax level 3",
            "total_players": 30,
            "policy_families": [
                {"pi1": "normal", "pi2": "egaroucid_l21", "alphas": list(ALPHAS)},
                {"pi1": "loss_a", "pi2": "normal", "alphas": list(ALPHAS)},
                {"pi1": "loss_b", "pi2": "loss_a", "alphas": list(ALPHAS)},
            ],
            "formula": "normalized pi1**alpha * pi2**(1-alpha)",
            "selection": "argmax over legal moves",
            "random": {"kind": "uniform_random", "seed": configuration["schedule"]["xot_seed"]},
            "training": self.evidence,
        }
        configuration["inputs"]["loss_a_model"] = battle.file_identity(self.model)
        configuration["inputs"]["loss_b_model"] = battle.file_identity(self.loss_b_model)
        configuration["inputs"]["implementation_sources"].update({
            "paper_roster": battle.file_identity(Path(__file__)),
            "policy_opponent_gtp": battle.file_identity(HERE / "policy_opponent_gtp.py"),
        })
        configuration["policy_inference"]["server_count"] = 3
        configuration["policy_inference"]["models"] = ["normal", "loss_a", "loss_b"]

    def start_server(self, args, manager, run_id):
        settings = copy.copy(args)
        settings.policy_model = self.model
        loss_a = battle.start_policy_server(settings, manager, run_id + "_loss_a")
        settings.policy_model = self.loss_b_model
        self.extra_policy_servers["loss_b"] = battle.start_policy_server(settings, manager, run_id + "_loss_b")
        return loss_a

    def validate_runtime(self, normal, loss_a):
        for name, server in (("loss_a", loss_a), *self.extra_policy_servers.items()):
            if (server.backend, server.backend_version, server.device) != (
                    normal.backend, normal.backend_version, normal.device):
                raise ValueError(f"Policy runtime differs between normal and {name}")


def make_argparser():
    parser = battle.make_argparser()
    parser.description = __doc__
    parser.set_defaults(baseline_levels=",".join(map(str, LEVELS)), edax_levels=",".join(map(str, LEVELS)),
                        alphas="0.2,0.4,0.6,0.8,1.0", policy_model=NORMAL_MODEL, logit_blend=True,
                        match_sets_per_pair=50, blend_egaroucid_level=21, score_temperature=1.0,
                        parallel_games=70, parallel_match_sets=35, blend_processes_per_player=4,
                        adaptive_engine_threads=True, engine_thread_budget=70,
                        minimum_available_memory_mib=8192.0, sequential_tail_match_sets=35,
                        output_dir=HERE / "output/gpw30_independent_xot_50sets")
    parser.add_argument("--loss-a-model", type=Path, default=LOSS_A_MODEL,
                        help="(disc_loss + 1)**2 model from Table 1")
    parser.add_argument("--loss-b-model", type=Path, default=LOSS_B_MODEL,
                        help="2**(disc_loss/2) model from Table 1")
    return parser


def prepare(args):
    if (tuple(battle.parse_int_list(args.baseline_levels)) != LEVELS
            or tuple(battle.parse_int_list(args.edax_levels)) != LEVELS
            or tuple(battle.parse_float_list(args.alphas)) != ALPHAS):
        raise ValueError("GPW roster requires levels 1,3,5,9,13,17,21 and alphas .2,.4,.6,.8,1; use the generic runner for other rosters")
    if args.blend_egaroucid_level != 21 or args.score_temperature != 1 or not args.logit_blend:
        raise ValueError("GPW policy blending requires Egaroucid L21, tau=1, and logit blending")
    if args.policy_backend != "tensorflow":
        raise ValueError("GPW roster requires TensorFlow for all three trained CNN models")
    evidence = {name: model_evidence(path, MODEL_FORMULAS[name]) for name, path in (
        ("normal", args.policy_model), ("loss_a", args.loss_a_model), ("loss_b", args.loss_b_model))}
    players = PaperPlayers(args, evidence)
    battle.validate_args(args, players.resource_alphas)
    return players


def main(argv=None):
    args = make_argparser().parse_args(argv)
    players = prepare(args)
    with battle.OutputRunLock(args.output_dir):
        battle.run_tournament(args, extension=players)


if __name__ == "__main__":
    main()
