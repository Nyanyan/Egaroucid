#!/usr/bin/env python3
"""Add a model's positive-alpha players to a completed strength tournament."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

import battle_blend_strength as battle
from strength_tournament import PlayerSpec, ResultStore, make_manifest, make_match_set_tasks, sha256_file


# Reviewed runner updates preserve move rules and fixed search levels. Optional
# concurrency adjustments are recorded separately in the tournament manifest.
COMPATIBLE_ENGINE_RUNNER_HASHES = (
    "b0a88aeaec37fe7fc3f4251ed956abb011bee7d3f038b351bb4031dc4aa6ff84",
    "4a7535f8b6757664f20c1f234e6726ab68764c50dc6b56a4d0f2a76d7438df8b",
    "c47f660542d2033d5d9e870f97931c1ad2d922fc034b8b4a4043901b8aae2676",
    "048bec284e7ac005e39bdc6744b6c073c98f10233559707d035f161d555dcb1a",
    "2869bb15f9f3c693ca8e22e297ff3a3c5fd8cc8f6ebbdf5bba3ad5ec069cb9de",
    "ba6b4e0ac2964f70152f4c976718669171d98aa124a32675982f63d1cc97c1bc",
    "9941e628d91b0c1c25fc36993030ff1999a0f7bef06395043209af640f218dc8",
    "946c2a446d760c3413727623ae290fc663148eae8116bc1ef3fa5a24447e64dc",
    "692cbce92d972678e0726deda2cfecc2fabb87028c76520c05ae08430d3b43b5",
)
COMPATIBLE_BLEND_GTP_HASHES = (
    "c24bbe6de070a7e86646a7d62180133c5d7a5ffafdef8761c6fda1983001afe6",
    "ce9865e8df8f75014c868b96a19960b79b3b3a8b90895b6917ab91355def10aa",
)


def compatible_playing_source(key, previous, current):
    return previous == current or (
        key == "engine_runner" and previous in COMPATIBLE_ENGINE_RUNNER_HASHES
        and current in COMPATIBLE_ENGINE_RUNNER_HASHES
    ) or (
        key == "blend_gtp_engine" and previous in COMPATIBLE_BLEND_GTP_HASHES
        and current in COMPATIBLE_BLEND_GTP_HASHES
    )


def input_path(identity):
    path = Path(identity["path"])
    return path if path.is_absolute() else battle.REPO_ROOT / path


class ModelExtension:
    def __init__(self, reference_dir, model, label):
        self.reference_dir = Path(reference_dir).resolve()
        self.model = Path(model).resolve()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label):
            raise ValueError("--label must start with a letter and use letters, digits, _ or -")
        self.label = label
        manifest_path = self.reference_dir / "strength_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.configuration = self.manifest["configuration"]
        if make_manifest(self.configuration) != self.manifest:
            raise ValueError("Reference manifest fingerprint is inconsistent")
        config = self.configuration
        if (config["schedule"].get("opening_sampling") != "independent_per_pair_without_replacement_v1"
                or config["schedule"].get("same_opening_sequence_for_every_pair") is not False
                or config["engine"].get("tt_reset") != "after_every_game_clear_or_discard_process"
                or config["engine"].get("hint_cache") is not False
                or config["engine"].get("memory_hint_cache") is not False):
            raise ValueError(
                "Reference uses the old tournament protocol. Rerun the reference in a new output directory "
                "with independent XOT sampling, per-game TT resets, and no hint cache; old results cannot be imported."
            )
        if config["participants"].get("additional_policy"):
            raise ValueError("Choose the original single-model tournament as reference")
        if config["schedule"]["max_match_sets"] is not None:
            raise ValueError("Reference must be a complete round robin without a task limit")
        if config["policy_inference"]["backend"] != "tensorflow":
            raise ValueError("This launcher requires a TensorFlow reference model")
        self.alphas = tuple(a for a in config["participants"]["alphas"] if a > 0)
        if not self.alphas:
            raise ValueError("Reference has no positive-alpha policies")
        self.old_count = sum(len(config["participants"][key])
                             for key in ("baseline_levels", "alphas", "edax_levels"))
        self.runtime = json.loads((self.reference_dir / "strength_runtime_manifest.json").read_text(encoding="utf-8"))
        if self.runtime["configuration"]["experiment_id"] != self.manifest["experiment_id"]:
            raise ValueError("Reference runtime belongs to another tournament")
        self.source_identities = {
            name: battle.file_identity(self.reference_dir / name)
            for name in ("strength_manifest.json", "strength_runtime_manifest.json", "strength_games.jsonl", "strength_results.json")
        }
        self.results = []
        self.model_blend_betas = ()
        self.include_random = False

    @property
    def resource_alphas(self):
        # Additional pure-policy and random wrappers never start a search engine.
        return (*self.alphas, *([1.0] * (len(self.model_blend_betas) + int(self.include_random))))

    @property
    def player_count(self):
        return self.old_count + len(self.resource_alphas)

    def make_args(self, options):
        self.model_blend_betas = tuple(battle.parse_float_list(options.model_blend_betas or ""))
        if (any(not 0 < beta < 1 for beta in self.model_blend_betas)
                or len(set(self.model_blend_betas)) != len(self.model_blend_betas)):
            raise ValueError("--model-blend-betas requires unique values strictly between 0 and 1")
        self.include_random = bool(options.include_random)
        args = battle.make_argparser().parse_args([])
        config = self.configuration
        participants, schedule, engine, inference = (config[k] for k in ("participants", "schedule", "engine", "policy_inference"))
        for key in ("baseline_levels", "alphas", "edax_levels"):
            setattr(args, key, ",".join(map(str, participants[key])))
        reference_sets = schedule["match_sets_per_pair"]
        args.match_sets_per_pair = (min(50, reference_sets) if options.match_sets_per_pair is None
                                    else options.match_sets_per_pair)
        if not 1 <= args.match_sets_per_pair <= reference_sets:
            raise ValueError("--match-sets-per-pair must be within the reference opening sequence")
        args.seed = schedule["xot_seed"]
        for key in ("parallel_match_sets", "parallel_games", "extra_engine_memory_mib", "finish_low_levels_first",
                    "minimum_available_memory_mib", "baseline_processes_per_player", "blend_processes_per_player",
                    "engine_threads", "blend_egaroucid_level", "score_temperature", "logit_blend",
                    "gtp_command_timeout_sec", "egaroucid_hint_timeout_sec"):
            setattr(args, key, engine[key])
        for key in ("gtp_command_timeout_sec", "egaroucid_hint_timeout_sec"):
            if getattr(options, key) is not None:
                setattr(args, key, getattr(options, key))
            if not 0 < getattr(args, key) < float("inf"):
                raise ValueError(f"{key} must be finite and positive")
        if options.gtp_command_timeout_sec is None:
            args.gtp_command_timeout_sec = max(args.gtp_command_timeout_sec,
                                               args.egaroucid_hint_timeout_sec + 100.0)
        if args.gtp_command_timeout_sec <= args.egaroucid_hint_timeout_sec:
            raise ValueError("GTP timeout must exceed the Egaroucid hint timeout")
        thread_policy = engine.get("adaptive_search_threads", {})
        args.adaptive_engine_threads = (thread_policy.get("enabled", True)
                                        if options.adaptive_engine_threads is None else options.adaptive_engine_threads)
        args.sequential_tail_match_sets = (35 if options.sequential_tail_match_sets is None
                                           else options.sequential_tail_match_sets)
        if options.engine_threads is not None:
            args.engine_threads = options.engine_threads
        # The old CPU-feedback cap of four is not a fixed-total allocation.
        if thread_policy.get("allocation") != "fixed_total_between_games":
            thread_policy = {}
        for option, source, default in (("max_engine_threads", "maximum_per_process", 64),
                                        ("engine_thread_budget", "total_threads", 70),
                                        ("estimated_search_thread_memory_mib", "estimated_extra_thread_memory_mib", 64.0)):
            value = getattr(options, option)
            setattr(args, option, thread_policy.get(source, default) if value is None else value)
        args.edax_processes_per_player = engine["edax"]["processes_per_player"] if engine["edax"] else 2
        args.no_hint_cache = True
        args.policy_backend = inference["backend"]
        args.allow_policy_cpu = inference["required_device"] != "GPU"
        for target, source in (("policy_max_batch_size", "max_batch_size"), ("policy_batch_wait_ms", "batch_wait_ms"),
                               ("policy_inference_threads", "inference_threads"), ("policy_server_timeout_sec", "request_timeout_sec"),
                               ("policy_server_startup_timeout_sec", "startup_timeout_sec")):
            setattr(args, target, inference[source])
        for target, source in (("openings", "xot_openings"), ("egaroucid_exe", "egaroucid_executable"),
                               ("edax_exe", "edax_executable"), ("edax_eval_file", "edax_evaluation_data"),
                               ("policy_model", "policy_model")):
            if source in config["inputs"]:
                setattr(args, target, input_path(config["inputs"][source]))
        # With two model families, bound initial engine memory to roughly the
        # reference budget. Search levels and the model's move policy stay fixed;
        # search concurrency can be adjusted at game boundaries.
        old_alpha_count = len(participants["alphas"])
        args.blend_processes_per_player = max(2, 2 * ((engine["blend_processes_per_player"] * old_alpha_count)
                                                     // (old_alpha_count + len(self.alphas)) // 2))
        if options.blend_processes_per_player is not None:
            args.blend_processes_per_player = options.blend_processes_per_player
        capacities = (len(participants["baseline_levels"]) * args.baseline_processes_per_player // 2
                      + (old_alpha_count + len(self.resource_alphas)) * args.blend_processes_per_player // 2
                      + len(participants["edax_levels"]) * args.edax_processes_per_player // 2)
        args.parallel_games = min(engine["parallel_games"], 2 * (capacities // 2))
        if options.parallel_games is not None:
            args.parallel_games = options.parallel_games
        args.parallel_match_sets = (args.parallel_games + 1) // 2
        args.output_dir = options.output_dir.resolve()
        if args.output_dir == self.reference_dir or self.reference_dir in args.output_dir.parents:
            raise ValueError("Use a separate --output-dir outside the reference directory")
        args.resume = True
        args.dry_run = False
        args.prepare_only = options.prepare_only
        args.status_every_match_sets = 20
        return args

    def validate_reference(self, args):
        for key, identity in self.configuration["inputs"].items():
            if key == "implementation_sources":
                continue
            if sha256_file(input_path(identity)) != identity["sha256"]:
                raise ValueError(f"Reference input changed: {key}")
        # The new orchestration imports old results; move selection and GTP
        # behavior must still use the exact implementations of the reference.
        for key in ("engine_runner", "blend_gtp_engine", "blend_policy", "policy_batch_server"):
            identity = self.configuration["inputs"]["implementation_sources"][key]
            if not compatible_playing_source(key, identity["sha256"], sha256_file(input_path(identity))):
                raise ValueError(f"Reference playing implementation changed: {key}")
        if battle.package_runtime_versions(args.policy_backend) != self.configuration["runtime_versions"]:
            raise ValueError("Python/package versions differ from the reference")
        levels, alphas = battle.validate_args(args, self.resource_alphas)
        names = [s.name for s in battle.build_player_specs(args, levels, alphas, 0)]
        aggregate = json.loads((self.reference_dir / "strength_results.json").read_text(encoding="utf-8"))
        if aggregate["experiment_id"] != self.manifest["experiment_id"] or names != [p["name"] for p in aggregate["players"]]:
            raise ValueError("Reference participant order or experiment ID differs")
        self.openings = battle.read_openings(args.openings, args.seed)
        battle.validate_scheduled_openings(self.openings, args.match_sets_per_pair)
        digest = hashlib.sha256(("\n".join(self.openings) + "\n").encode("ascii")).hexdigest()
        if digest != self.configuration["schedule"]["shuffled_opening_sequence_sha256"]:
            raise ValueError("XOT opening sequence differs from reference")
        source_tasks = make_match_set_tasks(self.old_count, self.openings,
                                           self.configuration["schedule"]["match_sets_per_pair"], args.seed)
        if battle.task_plan_hash(source_tasks) != self.configuration["schedule"]["task_plan_sha256"]:
            raise ValueError("Reference task plan differs")
        _, self.results = ResultStore(self.reference_dir).load({t.task_id: t for t in source_tasks}, read_only=True)
        if len(self.results) != len(source_tasks) or aggregate["completed_match_sets"] != len(source_tasks):
            raise ValueError("Complete the reference tournament before adding a new model")
        self.results = [r for r in self.results if r.set_index < args.match_sets_per_pair]
        self.verify_source_unchanged()

    def verify_source_unchanged(self):
        for name, identity in self.source_identities.items():
            if sha256_file(self.reference_dir / name) != identity["sha256"]:
                raise ValueError(f"Reference changed during validation: {name}")

    def update_configuration(self, configuration):
        configuration["participants"]["additional_policy"] = {
            "label": self.label, "alphas": list(self.alphas), "appended_after_original_players": True,
        }
        configuration["inputs"]["additional_policy_model"] = battle.file_identity(self.model)
        configuration["inputs"]["implementation_sources"]["extension_runner"] = battle.file_identity(Path(__file__))
        opponents = [{"name": f"model_blend_beta_{beta:g}", "kind": "geometric_policy", "beta": beta,
                      "normal_model": "policy_model", "weighted_model": "additional_policy_model",
                      "selection": "argmax", "formula": "normalized P_normal^(1-beta) * P_weighted^beta"}
                     for beta in self.model_blend_betas]
        if self.include_random:
            opponents.append({"name": "random", "kind": "uniform_random", "seed": configuration["schedule"]["xot_seed"],
                              "seeding": "sha256(seed,p0_idx,p1_idx,set_index,p0_is_black)"})
        if opponents:
            configuration["participants"]["policy_opponents"] = opponents
            configuration["inputs"]["implementation_sources"]["policy_opponent_gtp"] = battle.file_identity(
                Path(__file__).with_name("policy_opponent_gtp.py"))
        configuration["imported_reference"] = {
            "directory": str(self.reference_dir), "experiment_id": self.manifest["experiment_id"],
            "files": self.source_identities, "imported_match_sets": len(self.results),
            "original_configuration": self.configuration,
        }

    def add_specs(self, args, original_specs, port, original_port=0):
        new_args = copy.copy(args)
        new_args.edax_levels = ""
        extra = battle.build_player_specs(new_args, [], self.alphas, port)
        specs = list(original_specs) + [replace(s, name=f"{self.label}_{s.name}") for s in extra]
        script = str(Path(__file__).with_name("policy_opponent_gtp.py"))
        for beta in self.model_blend_betas:
            specs.append(PlayerSpec(f"model_blend_beta_{beta:g}",
                (sys.executable, script, "--mode", "geometric", "--beta", str(beta),
                 "--normal-port", str(original_port), "--weighted-port", str(port),
                 "--output", "logits" if args.logit_blend else "probabilities",
                 "--timeout-sec", str(args.policy_server_timeout_sec)),
                args.blend_processes_per_player, True, 1.0))
        if self.include_random:
            specs.append(PlayerSpec("random", (sys.executable, script, "--mode", "random", "--seed", str(args.seed)),
                                    args.blend_processes_per_player, False, 1.0, random_seed=args.seed))
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("Additional participant names collide with original participants")
        return specs

    def missing_imports(self, tasks, loaded_results):
        by_pair_set = {(t.p0_idx, t.p1_idx, t.set_index): t for t in tasks}
        existing = {r.task_id: r for r in loaded_results}
        imports = []
        for result in self.results:
            task = by_pair_set[(result.p0_idx, result.p1_idx, result.set_index)]
            if task.opening != result.opening:
                raise ValueError("Imported match uses a different XOT opening")
            remapped = replace(result, task_id=task.task_id)
            if task.task_id in existing:
                if existing[task.task_id] != remapped:
                    raise ValueError("Imported match has changed in the extended result log")
            else:
                imports.append(remapped)
        return imports

    def start_server(self, args, manager, run_id):
        new_args = copy.copy(args)
        new_args.policy_model = self.model
        return battle.start_policy_server(new_args, manager, run_id + "_" + self.label)

    def validate_runtime(self, original, additional):
        expected = self.runtime["configuration"]["policy_server"]
        for info in (original, additional):
            actual = {"backend": info.backend, "backend_version": info.backend_version, "device": info.device}
            if actual != expected:
                raise ValueError("Policy server runtime differs from the reference")
        if original.device == "GPU" and battle.gpu_runtime_identity() != self.runtime["configuration"]["gpu_model_and_driver"]:
            raise ValueError("GPU/driver differs from the reference")


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, help="Inferred from an existing --output-dir when omitted")
    parser.add_argument("--model", type=Path, help="Inferred from an existing --output-dir when omitted")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel-games", type=int)
    parser.add_argument("--blend-processes-per-player", type=int)
    parser.add_argument("--model-blend-betas", help="Comma-separated mixtures: beta=0 normal, beta=1 loss_weighted")
    parser.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-engine-threads", action=argparse.BooleanOptionalAction, default=None,
                        help="Adapt search concurrency between games (enabled for legacy references)")
    parser.add_argument("--engine-threads", type=int)
    parser.add_argument("--max-engine-threads", type=int)
    parser.add_argument("--engine-thread-budget", type=int)
    parser.add_argument("--sequential-tail-match-sets", type=int,
                        help="Global remaining-set threshold for sequential 32-thread games (default: 35; 0 disables)")
    parser.add_argument("--estimated-search-thread-memory-mib", type=float)
    parser.add_argument("--egaroucid-hint-timeout-sec", type=float,
                        help="Maximum wait per Egaroucid hint command; inherits the saved run when omitted")
    parser.add_argument("--gtp-command-timeout-sec", type=float,
                        help="Outer command timeout; must exceed the hint timeout")
    parser.add_argument("--match-sets-per-pair", type=int,
                        help="Use the first N independently sampled openings of each reference pair (default: 50, or all if fewer)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Back up and resize an existing extension; update reports without starting games")
    parser.add_argument("--dry-run", action="store_true", help="Validate reference and plan; start no engines and write no results")
    return parser


def main(argv=None):
    options = make_argparser().parse_args(argv)
    pending_path = options.output_dir / "strength_pending_opponents.json"
    pending_options = None
    if pending_path.exists():
        pending_options = json.loads(pending_path.read_text(encoding="utf-8"))
        validators = {
            "model_blend_betas": lambda value: isinstance(value, str),
            "include_random": lambda value: type(value) is bool,
            "adaptive_engine_threads": lambda value: type(value) is bool,
            "engine_thread_budget": lambda value: type(value) is int and value >= 4,
            "engine_threads": lambda value: type(value) is int and value >= 1,
            "max_engine_threads": lambda value: type(value) is int and value >= 1,
            "parallel_games": lambda value: type(value) is int and value >= 2,
            "sequential_tail_match_sets": lambda value: type(value) is int and value >= 0,
            "egaroucid_hint_timeout_sec": lambda value: type(value) in (int, float) and 0 < value < float("inf"),
            "gtp_command_timeout_sec": lambda value: type(value) in (int, float) and 0 < value < float("inf"),
        }
        if (not isinstance(pending_options, dict) or not pending_options
                or any(key not in validators or not validators[key](value)
                       for key, value in pending_options.items())):
            raise ValueError("Invalid pending opponent configuration")
        for key, value in pending_options.items():
            if getattr(options, key) is None:
                setattr(options, key, value)
    existing_path = options.output_dir / "strength_manifest.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))["configuration"]
        if "imported_reference" not in existing:
            raise ValueError("--output-dir must be an extension tournament")
        if options.reference_dir is None:
            options.reference_dir = Path(existing["imported_reference"]["directory"])
        if options.model is None:
            options.model = input_path(existing["inputs"]["additional_policy_model"])
        if options.label is None:
            options.label = existing["participants"]["additional_policy"]["label"]
        opponents = existing["participants"].get("policy_opponents", [])
        if options.model_blend_betas is None and opponents:
            options.model_blend_betas = ",".join(str(p["beta"]) for p in opponents if p["kind"] == "geometric_policy")
        if options.include_random is None and opponents:
            options.include_random = any(p["kind"] == "uniform_random" for p in opponents)
        for key in ("parallel_games", "blend_processes_per_player",
                    "gtp_command_timeout_sec", "egaroucid_hint_timeout_sec"):
            if getattr(options, key) is None:
                value = existing["engine"][key]
                if key == "gtp_command_timeout_sec" and options.egaroucid_hint_timeout_sec is not None:
                    value = max(value, options.egaroucid_hint_timeout_sec + 100.0)
                setattr(options, key, value)
        policy = existing["engine"].get("adaptive_search_threads", {})
        if options.sequential_tail_match_sets is None:
            options.sequential_tail_match_sets = existing["engine"].get("sequential_tail", {}).get("remaining_match_sets", 35)
        if policy.get("allocation") != "fixed_total_between_games":
            policy = {key: value for key, value in policy.items()
                      if key in ("enabled", "estimated_extra_thread_memory_mib")}
        for option, key in (("adaptive_engine_threads", "enabled"),
                            ("max_engine_threads", "maximum_per_process"),
                            ("engine_thread_budget", "total_threads"),
                            ("estimated_search_thread_memory_mib", "estimated_extra_thread_memory_mib")):
            if getattr(options, option) is None and key in policy:
                setattr(options, option, policy[key])
        if options.engine_threads is None:
            options.engine_threads = existing["engine"]["engine_threads"]
    if options.reference_dir is None or options.model is None:
        raise ValueError("A new extension requires --reference-dir and --model")
    if options.label is None:
        options.label = "loss_weighted"
    extension = ModelExtension(options.reference_dir, options.model, options.label)
    args = extension.make_args(options)
    print("Validating completed reference tournament...", flush=True)
    extension.validate_reference(args)
    tasks = make_match_set_tasks(extension.player_count, extension.openings,
                                 args.match_sets_per_pair, args.seed)
    imports = extension.missing_imports(tasks, [])
    imported_ids = {r.task_id for r in imports}
    if any(t.p1_idx < extension.old_count for t in tasks if t.task_id not in imported_ids):
        raise ValueError("Unexpected unfinished match between original participants")
    print(f"participants: {extension.old_count} + {len(extension.resource_alphas)} = {extension.player_count}", flush=True)
    if extension.model_blend_betas or extension.include_random:
        print(f"model mixture betas: {extension.model_blend_betas}; uniform random: {extension.include_random}", flush=True)
    print(f"inherited sets: {len(imports):,}; new sets: {len(tasks) - len(imports):,}; new games: {2 * (len(tasks) - len(imports)):,}", flush=True)
    print(f"sets per pair: {args.match_sets_per_pair}; XOT seed: {args.seed}; engine threads: {args.engine_threads}", flush=True)
    print(f"parallel games: {args.parallel_games}; blend processes per player: {args.blend_processes_per_player}", flush=True)
    print(f"adaptive search threads: {args.adaptive_engine_threads}; maximum: {args.max_engine_threads}; "
          f"total thread budget: {args.engine_thread_budget}; apply: next game", flush=True)
    print(f"timeouts: Egaroucid hint {args.egaroucid_hint_timeout_sec:g}s; "
          f"outer GTP command {args.gtp_command_timeout_sec:g}s", flush=True)
    print(f"sequential tail: remaining <= {args.sequential_tail_match_sets} match sets; "
          "32 search threads; one color game at a time; clear TT and reuse supported engines", flush=True)
    print(f"new model: {extension.model} ({'present' if extension.model.is_file() else 'waiting for training'})", flush=True)
    if options.dry_run:
        from strength_recovery import prepare_extension_output
        prepare_extension_output(args, extension, dry_run=True)
        print("dry-run complete: reference unchanged; no engines started; no results written", flush=True)
        return
    with battle.OutputRunLock(args.output_dir):
        from strength_recovery import prepare_extension_output
        prepare_extension_output(args, extension)
        if (pending_options is not None
                and all(getattr(options, key) == value for key, value in pending_options.items())
                and json.loads(pending_path.read_text(encoding="utf-8")) == pending_options):
            pending_path.unlink()
        if options.prepare_only:
            return
        battle.run_tournament(args, extension=extension)


if __name__ == "__main__":
    main()
