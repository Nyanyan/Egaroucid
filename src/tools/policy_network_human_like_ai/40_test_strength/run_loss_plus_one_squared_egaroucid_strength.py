#!/usr/bin/env python3
"""Append squared-loss NN / Egaroucid L21 blends, reusing every completed pair."""
from collections import Counter
import copy
from dataclasses import replace
from pathlib import Path

import run_loss_plus_one_strength as base
import run_loss_plus_one_squared_strength as squared
from run_loss_plus_one_squared_blends_strength import blend_name as model_blend_name
from extend_strength_tournament import ModelExtension, compatible_playing_source, input_path

HERE = Path(__file__).resolve().parent
LABEL = "loss_plus_one_squared"


def blend_name(alpha):
    return f"{LABEL}_egaroucid21_alpha_{base.battle.format_alpha(alpha)}"


class SquaredEgaroucidExtension(squared.LossPlusOneSquaredExtension):
    def __init__(self, reference_dir, normal_reference_dir, model, alphas):
        self.egaroucid_alphas = tuple(base.battle.parse_float_list(alphas))
        if (not self.egaroucid_alphas
                or any(not 0 < a < 1 for a in self.egaroucid_alphas)
                or len(set(self.egaroucid_alphas)) != len(self.egaroucid_alphas)
                or len({blend_name(a) for a in self.egaroucid_alphas}) != len(self.egaroucid_alphas)):
            raise ValueError("--alphas requires distinct values strictly between 0 and 1; alpha=1 is already retained")
        super().__init__(reference_dir, normal_reference_dir, model)
        opponents = self.opponent_manifest["configuration"]["participants"].get("policy_opponents", [])
        if (not opponents or opponents[0].get("name") != "random"
                or any(p.get("kind") != "geometric_policy" for p in opponents[1:])):
            raise ValueError("Reference must contain random followed by normal/squared NN blends")
        self.retained_betas = tuple(p["beta"] for p in opponents[1:])

    @property
    def retained_player_count(self):
        return self.old_count + 2 + len(self.retained_betas)

    @property
    def resource_alphas(self):
        return (*super().resource_alphas, *self.egaroucid_alphas)

    def make_args(self, options):
        options.include_random = True
        options.model_blend_betas = ",".join(map(str, self.retained_betas))
        args = ModelExtension.make_args(self, options)
        if args.output_dir == self.opponent_reference or self.opponent_reference in args.output_dir.parents:
            raise ValueError("Use a separate output outside the completed reference")
        if args.blend_egaroucid_level != 21 or not args.logit_blend:
            raise ValueError("Reference must use Egaroucid level 21 and logit blending")
        return args

    def retained_configuration(self, configuration):
        ModelExtension.update_configuration(self, configuration)
        opponents = configuration["participants"]["policy_opponents"]
        random = opponents.pop()
        for opponent, beta in zip(opponents, self.model_blend_betas):
            opponent.update(name=model_blend_name(beta), alpha=beta)
        opponents.insert(0, random)

    def validate_reference(self, args):
        source = self.opponent_manifest["configuration"]
        expected_config = copy.deepcopy(self.configuration)
        self.retained_configuration(expected_config)
        if source["participants"] != expected_config["participants"]:
            raise ValueError("Reference roster or blend formula differs from the retained players")
        if source["inputs"]["additional_policy_model"]["sha256"] != base.sha256_file(self.model):
            raise ValueError("Squared model differs from the completed reference")
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
        # Validates both references, their complete logs, original results, and XOT order.
        super().validate_reference(args)
        runtime = base.read_json(self.opponent_reference / "strength_runtime_manifest.json")["configuration"]
        expected_runtime = self.runtime["configuration"]
        for key in ("policy_server", "gpu_model_and_driver"):
            if runtime[key] != expected_runtime[key]:
                raise ValueError(f"Squared reference runtime differs: {key}")
        if runtime.get("additional_policy_server") != expected_runtime["policy_server"]:
            raise ValueError("Squared reference additional policy server differs")
        tasks = base.make_match_set_tasks(self.retained_player_count, self.openings,
                                         source["schedule"]["match_sets_per_pair"], args.seed)
        _, results = base.ResultStore(self.opponent_reference).load({t.task_id: t for t in tasks}, read_only=True)
        self.results = [r for r in results if r.set_index < args.match_sets_per_pair]
        expected_count = self.retained_player_count * (self.retained_player_count - 1) // 2 * args.match_sets_per_pair
        if len(self.results) != expected_count:
            raise ValueError("Reference is missing retained match sets")
        self.verify_source_unchanged()

    def add_specs(self, args, original_specs, port, original_port=0):
        specs = ModelExtension.add_specs(self, args, original_specs, port, original_port)
        random = specs.pop()
        specs.insert(self.old_count + 1, random)
        for index, beta in enumerate(self.model_blend_betas, self.old_count + 2):
            specs[index] = replace(specs[index], name=model_blend_name(beta))
        new_args = copy.copy(args)
        new_args.edax_levels = ""
        extra = base.battle.build_player_specs(new_args, [], self.egaroucid_alphas, port)
        specs.extend(replace(spec, name=blend_name(spec.alpha)) for spec in extra)
        return specs

    def update_configuration(self, configuration):
        self.retained_configuration(configuration)
        configuration["participants"]["policy_opponents"].extend(
            {"name": blend_name(alpha), "kind": "egaroucid_policy_blend", "alpha": alpha,
             "model": "additional_policy_model", "egaroucid_level": 21,
             "score_temperature": configuration["engine"]["score_temperature"], "selection": "argmax",
             "formula": "alpha * squared_logits + (1-alpha) * egaroucid_score / temperature"}
            for alpha in self.egaroucid_alphas)
        configuration["imported_reference"]["imported_match_sets"] = (
            self.old_count * (self.old_count - 1) // 2 * configuration["schedule"]["match_sets_per_pair"])
        sources = configuration["inputs"]["implementation_sources"]
        sources["loss_plus_one_runner"] = base.battle.file_identity(Path(base.__file__))
        sources["loss_plus_one_squared_runner"] = base.battle.file_identity(Path(squared.__file__))
        sources["loss_plus_one_squared_egaroucid_runner"] = base.battle.file_identity(Path(__file__))
        configuration["squared_egaroucid_experiment"] = {
            "new_players": [blend_name(a) for a in self.egaroucid_alphas],
            "alphas": list(self.egaroucid_alphas), "alpha_weight": "squared NN",
            "retained_player_count": self.retained_player_count,
            "scope": "all pairs involving an appended L21 blend, including blend versus blend",
            "reference": {"directory": str(self.opponent_reference),
                          "experiment_id": self.opponent_manifest["experiment_id"],
                          "files": self.opponent_files, "imported_match_sets": len(self.results),
                          "participant_indices_unchanged": True},
        }


def make_argparser():
    parser = squared.make_argparser()
    parser.description = __doc__
    parser.set_defaults(reference_dir=HERE / "output/conv8_bn_loss_plus_one_squared_blends_20261007",
                        output_dir=HERE / "output/conv8_bn_loss_plus_one_squared_egaroucid21_20261007")
    parser.add_argument("--alphas", default="0.2,0.4,0.6,0.8",
                        help="Squared NN weights in L21 blends, strictly between 0 and 1")
    return parser


def prepare(args):
    squared.verify_completed_training(args.model)
    extension = SquaredEgaroucidExtension(args.reference_dir, args.normal_reference_dir, args.model, args.alphas)
    options = base.extension_parser().parse_args([
        "--output-dir", str(args.output_dir), "--match-sets-per-pair", str(args.match_sets_per_pair)])
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
    print("Validating completed tournaments, models, and the original XOT sequence...", flush=True)
    extension.validate_reference(runtime_args)
    levels, alphas = base.battle.validate_args(runtime_args, extension.resource_alphas)
    specs = extension.add_specs(runtime_args, base.battle.build_player_specs(runtime_args, levels, alphas, 0), 0)
    tasks = base.make_match_set_tasks(extension.player_count, extension.openings,
                                     runtime_args.match_sets_per_pair, runtime_args.seed)
    imported = extension.missing_imports(tasks, [])
    imported_ids = {r.task_id for r in imported}
    pending = [t for t in tasks if t.task_id not in imported_ids]
    expected = Counter({(i, j): runtime_args.match_sets_per_pair
                        for i in range(len(specs)) for j in range(i + 1, len(specs))
                        if j >= extension.retained_player_count})
    if Counter((t.p0_idx, t.p1_idx) for t in pending) != expected:
        raise ValueError("New schedule must contain only pairs involving an appended L21 blend")
    print("Added players: " + ", ".join(s.name for s in specs[extension.retained_player_count:]), flush=True)
    print(f"Participants: {extension.retained_player_count} retained + {len(extension.egaroucid_alphas)} blends = {len(specs)}", flush=True)
    print(f"Imported: {len(imported):,} sets; new: {len(pending):,} sets / {len(pending) * 2:,} games", flush=True)
    print(f"Each pair: {runtime_args.match_sets_per_pair} color-swapped sets; independent XOT samples per pair with seed {runtime_args.seed}", flush=True)
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
