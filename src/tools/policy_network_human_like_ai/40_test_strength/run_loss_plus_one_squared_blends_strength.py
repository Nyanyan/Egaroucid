#!/usr/bin/env python3
"""Append normal/squared-loss policy blends to the completed 19-player tournament."""
from collections import Counter
from dataclasses import replace
from pathlib import Path

import run_loss_plus_one_strength as base
import run_loss_plus_one_squared_strength as squared
from extend_strength_tournament import ModelExtension, compatible_playing_source, input_path

HERE = Path(__file__).resolve().parent
LABEL = "loss_plus_one_squared"


def blend_name(alpha):
    return f"{LABEL}_blend_alpha_{alpha:g}"


class SquaredBlendsExtension(squared.LossPlusOneSquaredExtension):
    @property
    def retained_player_count(self):
        return self.old_count + 2  # Original players, squared NN, random.

    def make_args(self, options):
        if not options.include_random or not options.model_blend_betas:
            raise ValueError("Keep the random opponent and specify at least one blend alpha")
        args = ModelExtension.make_args(self, options)
        if args.output_dir == self.opponent_reference or self.opponent_reference in args.output_dir.parents:
            raise ValueError("Use a separate output outside the squared-model reference")
        return args

    def validate_reference(self, args):
        source = self.opponent_manifest["configuration"]
        if source["participants"]["additional_policy"] != {
                "label": LABEL, "alphas": [1.0], "appended_after_original_players": True}:
            raise ValueError("Reference must contain the standalone squared-loss model")
        if source["inputs"]["additional_policy_model"]["sha256"] != base.sha256_file(self.model):
            raise ValueError("Squared model differs from the completed reference")
        opponents = source["participants"].get("policy_opponents", [])
        if len(opponents) != 1 or opponents[0]["name"] != "random":
            raise ValueError("Reference must contain only the original players, squared NN, and random")
        for key in ("policy_inference", "runtime_versions"):
            if source[key] != self.configuration[key]:
                raise ValueError(f"Squared reference differs from the normal reference: {key}")
        for key in ("blend_egaroucid_level", "score_temperature", "logit_blend", "hint_cache"):
            if source["engine"][key] != self.configuration["engine"][key]:
                raise ValueError(f"Squared reference playing settings differ: {key}")
        for key in ("engine_runner", "blend_gtp_engine", "blend_policy", "policy_batch_server", "policy_opponent_gtp"):
            identity = source["inputs"]["implementation_sources"][key]
            if not compatible_playing_source(key, identity["sha256"], base.sha256_file(input_path(identity))):
                raise ValueError(f"Squared reference playing implementation changed: {key}")
        # Checks the complete source, original matches, random games, and XOT order.
        super().validate_reference(args)
        runtime = base.read_json(self.opponent_reference / "strength_runtime_manifest.json")["configuration"]
        expected = self.runtime["configuration"]
        for key in ("policy_server", "gpu_model_and_driver"):
            if runtime[key] != expected[key]:
                raise ValueError(f"Squared reference runtime differs: {key}")
        if runtime.get("additional_policy_server") != expected["policy_server"]:
            raise ValueError("Squared reference additional policy server differs")
        tasks = base.make_match_set_tasks(self.retained_player_count, self.openings,
                                         source["schedule"]["match_sets_per_pair"], args.seed)
        _, results = base.ResultStore(self.opponent_reference).load({t.task_id: t for t in tasks}, read_only=True)
        # Appending after random preserves every old participant index and transcript.
        self.results = [r for r in results if r.set_index < args.match_sets_per_pair]
        expected_count = self.retained_player_count * (self.retained_player_count - 1) // 2 * args.match_sets_per_pair
        if len(self.results) != expected_count:
            raise ValueError("Squared reference is missing retained match sets")
        self.verify_source_unchanged()

    def add_specs(self, args, original_specs, port, original_port=0):
        specs = ModelExtension.add_specs(self, args, original_specs, port, original_port)
        random = specs.pop()
        specs.insert(self.old_count + 1, random)
        for index, alpha in enumerate(self.model_blend_betas, self.retained_player_count):
            specs[index] = replace(specs[index], name=blend_name(alpha))
        return specs

    def update_configuration(self, configuration):
        ModelExtension.update_configuration(self, configuration)
        opponents = configuration["participants"]["policy_opponents"]
        random = opponents.pop()
        for opponent, alpha in zip(opponents, self.model_blend_betas):
            opponent["name"] = blend_name(alpha)
            opponent["alpha"] = alpha
        opponents.insert(0, random)
        configuration["imported_reference"]["imported_match_sets"] = (
            self.old_count * (self.old_count - 1) // 2 * configuration["schedule"]["match_sets_per_pair"])
        sources = configuration["inputs"]["implementation_sources"]
        sources["loss_plus_one_runner"] = base.battle.file_identity(Path(base.__file__))
        sources["loss_plus_one_squared_runner"] = base.battle.file_identity(Path(squared.__file__))
        sources["loss_plus_one_squared_blends_runner"] = base.battle.file_identity(Path(__file__))
        configuration["squared_blend_experiment"] = {
            "new_players": [blend_name(a) for a in self.model_blend_betas],
            "alphas": list(self.model_blend_betas),
            "formula": "normalized P_normal^(1-alpha) * P_squared^alpha",
            "alpha_is_internal_beta": True,
            "scope": "all pairs involving an appended blend, including blend versus blend",
            "retained_player_count": self.retained_player_count,
            "reference": {"directory": str(self.opponent_reference),
                          "experiment_id": self.opponent_manifest["experiment_id"],
                          "files": self.opponent_files, "imported_match_sets": len(self.results),
                          "participant_indices_unchanged": True},
        }


def make_argparser():
    parser = squared.make_argparser()
    parser.description = __doc__
    parser.set_defaults(reference_dir=HERE / "output/conv8_bn_loss_plus_one_squared_single_20261006",
                        output_dir=HERE / "output/conv8_bn_loss_plus_one_squared_blends_20261007")
    parser.add_argument("--alphas", default="0.2,0.4,0.6,0.8",
                        help="Normal/squared model blend weights: 0=normal, 1=squared (internal beta)")
    return parser


def prepare(args):
    squared.verify_completed_training(args.model)
    extension = SquaredBlendsExtension(args.reference_dir, args.normal_reference_dir, args.model)
    options = base.extension_parser().parse_args([
        "--output-dir", str(args.output_dir), "--include-random", "--model-blend-betas", args.alphas,
        "--match-sets-per-pair", str(args.match_sets_per_pair)])
    engine = extension.opponent_manifest["configuration"]["engine"]
    defaults = {"parallel_games": engine["parallel_games"],
                "blend_processes_per_player": engine["blend_processes_per_player"],
                "engine_thread_budget": engine.get("adaptive_search_threads", {}).get("total_threads", 70),
                "sequential_tail_match_sets": engine.get("sequential_tail", {}).get("remaining_match_sets", 35)}
    for key, default in defaults.items():
        setattr(options, key, getattr(args, key) if getattr(args, key) is not None else default)
    options.egaroucid_hint_timeout_sec = engine["egaroucid_hint_timeout_sec"]
    options.gtp_command_timeout_sec = engine["gtp_command_timeout_sec"]
    runtime_args = extension.make_args(options)
    print("Validating the completed squared-model tournament and original XOT sequence...", flush=True)
    extension.validate_reference(runtime_args)
    levels, alphas = base.battle.validate_args(runtime_args, extension.resource_alphas)
    specs = extension.add_specs(runtime_args, base.battle.build_player_specs(runtime_args, levels, alphas, 0), 0)
    tasks = base.make_match_set_tasks(extension.player_count, extension.openings,
                                     runtime_args.match_sets_per_pair, runtime_args.seed)
    imported = extension.missing_imports(tasks, [])
    imported_ids = {r.task_id for r in imported}
    pending = [t for t in tasks if t.task_id not in imported_ids]
    counts = Counter((t.p0_idx, t.p1_idx) for t in pending)
    expected = Counter({(i, j): runtime_args.match_sets_per_pair
                        for i in range(len(specs)) for j in range(i + 1, len(specs))
                        if j >= extension.retained_player_count})
    if counts != expected:
        raise ValueError("New schedule must contain exactly N sets for each pair involving an appended blend")
    print("Added players: " + ", ".join(s.name for s in specs[extension.retained_player_count:]), flush=True)
    print(f"Participants: {extension.retained_player_count} retained + {len(extension.model_blend_betas)} blends = {len(specs)}", flush=True)
    print(f"Imported: {len(imported):,} sets; new: {len(pending):,} sets / {len(pending) * 2:,} games", flush=True)
    print(f"Each pair: {runtime_args.match_sets_per_pair} color-swapped sets; first XOT openings with seed {runtime_args.seed}", flush=True)
    print(f"Output: {runtime_args.output_dir}", flush=True)
    base.prepare_extension_output(runtime_args, extension, dry_run=True)
    return runtime_args, extension


def main(argv=None):
    args = make_argparser().parse_args(argv)
    runtime_args, extension = prepare(args)
    if args.dry_run:
        print("Dry-run passed. No engines started; no results written.", flush=True)
        return
    with base.battle.OutputRunLock(runtime_args.output_dir):
        base.prepare_extension_output(runtime_args, extension)
        base.battle.run_tournament(runtime_args, extension=extension)


if __name__ == "__main__":
    main()
