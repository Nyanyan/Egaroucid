#!/usr/bin/env python3
"""Play only the loss+1 network against the previous non-exponential opponents."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import battle_blend_strength as battle
from extend_strength_tournament import ModelExtension, make_argparser as extension_parser
from strength_recovery import prepare_extension_output
from strength_tournament import ResultStore, make_manifest, make_match_set_tasks, sha256_file

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
REFERENCE_FILES = ("strength_manifest.json", "strength_runtime_manifest.json",
                   "strength_games.jsonl", "strength_results.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def remap_result_players(result, mapping):
    """Keep both recorded games intact while changing only participant indices."""
    p0, p1 = mapping[result.p0_idx], mapping[result.p1_idx]
    if p0 >= p1:
        raise ValueError("import mapping must preserve pair order")
    games = tuple(replace(game, p0_idx=p0, p1_idx=p1,
                          black_idx=mapping[game.black_idx], white_idx=mapping[game.white_idx])
                  for game in result.color_games)
    return replace(result, p0_idx=p0, p1_idx=p1, color_games=games)


class LossPlusOneExtension(ModelExtension):
    def __init__(self, reference_dir, normal_reference_dir, model, label="loss_plus_one"):
        self.opponent_reference = Path(reference_dir).resolve()
        self.opponent_manifest = read_json(self.opponent_reference / "strength_manifest.json")
        source = self.opponent_manifest["configuration"]
        if make_manifest(source) != self.opponent_manifest:
            raise ValueError("opponent reference fingerprint differs")
        if normal_reference_dir is None:
            normal_reference_dir = source["imported_reference"]["directory"]
        super().__init__(normal_reference_dir, model, label)
        if source["imported_reference"]["experiment_id"] != self.manifest["experiment_id"]:
            raise ValueError("opponent reference belongs to a different normal tournament")
        self.opponent_files = {name: battle.file_identity(self.opponent_reference / name) for name in REFERENCE_FILES}
        self.alphas = (1.0,)
        self.excluded_names = []
        self.random_import_count = 0
        self.random_source_index = None

    def make_args(self, options):
        if options.model_blend_betas or not options.include_random:
            raise ValueError("this run requires only the new standalone model and the existing random opponent")
        args = super().make_args(options)
        if args.output_dir == self.opponent_reference or self.opponent_reference in args.output_dir.parents:
            raise ValueError("Use a separate output outside the opponent reference")
        return args

    def validate_reference(self, args):
        # Validates all original inputs, playing code, completed pairs, and XOT order.
        super().validate_reference(args)
        source = self.opponent_manifest["configuration"]
        schedule, participants = source["schedule"], source["participants"]
        if (schedule["max_match_sets"] is not None
                or schedule["match_sets_per_pair"] < args.match_sets_per_pair
                or schedule["xot_seed"] != args.seed
                or schedule["shuffled_opening_sequence_sha256"] != self.configuration["schedule"]["shuffled_opening_sequence_sha256"]):
            raise ValueError("opponent reference must contain the same complete opening prefix")
        for key in ("baseline_levels", "alphas", "edax_levels"):
            if participants[key] != self.configuration["participants"][key]:
                raise ValueError(f"normal opponent roster differs: {key}")
        if source["inputs"]["policy_model"] != self.configuration["inputs"]["policy_model"]:
            raise ValueError("normal model differs between references")
        levels, alphas = battle.validate_args(args, self.resource_alphas)
        normal_names = [spec.name for spec in battle.build_player_specs(args, levels, alphas, 0)]
        additional = participants["additional_policy"]
        expected_names = (normal_names + [f"{additional['label']}_alpha_{a:.1f}" for a in additional["alphas"]]
                          + [p["name"] for p in participants["policy_opponents"]])
        aggregate = read_json(self.opponent_reference / "strength_results.json")
        runtime = read_json(self.opponent_reference / "strength_runtime_manifest.json")
        if (aggregate["experiment_id"] != self.opponent_manifest["experiment_id"]
                or [p["name"] for p in aggregate["players"]] != expected_names
                or make_manifest(runtime["configuration"]) != runtime
                or runtime["configuration"]["experiment_id"] != self.opponent_manifest["experiment_id"]):
            raise ValueError("opponent reference participant order or runtime differs")
        random_spec = [p for p in participants["policy_opponents"] if p["name"] == "random"]
        if len(random_spec) != 1 or random_spec[0]["kind"] != "uniform_random" or random_spec[0]["seed"] != args.seed:
            raise ValueError("expected the previous uniform random player with the same seed")
        self.random_source_index = expected_names.index("random")
        if self.random_source_index < self.old_count:
            raise ValueError("random player must follow the normal reference players")
        tasks = make_match_set_tasks(len(expected_names), self.openings, schedule["match_sets_per_pair"], args.seed)
        if battle.task_plan_hash(tasks) != schedule["task_plan_sha256"]:
            raise ValueError("opponent reference task plan differs")
        _, results = ResultStore(self.opponent_reference).load({t.task_id: t for t in tasks}, read_only=True)
        if len(results) != len(tasks) or aggregate["completed_match_sets"] != len(tasks):
            raise ValueError("opponent reference tournament is incomplete")
        baseline = {(r.p0_idx, r.p1_idx, r.set_index): r for r in self.results}
        random_results = []
        mapping = {i: i for i in range(self.old_count)}
        mapping[self.random_source_index] = self.old_count + 1  # New model is at old_count.
        for result in results:
            if result.set_index >= args.match_sets_per_pair:
                continue
            if result.p1_idx < self.old_count:
                original = baseline[result.p0_idx, result.p1_idx, result.set_index]
                if replace(result, task_id=original.task_id) != original:
                    raise ValueError("normal-player results differ between references")
            elif result.p1_idx == self.random_source_index and result.p0_idx < self.old_count:
                random_results.append(remap_result_players(result, mapping))
        if len(random_results) != self.old_count * args.match_sets_per_pair:
            raise ValueError("missing prior random-versus-normal match sets")
        # Pair-index seeds change if random is moved after removing players.
        # Replay those pairs under their new schedules instead of importing
        # games with different openings (and old random seeds).
        if self.random_source_index == self.old_count + 1:
            self.results.extend(random_results)
            self.random_import_count = len(random_results)
        else:
            self.random_import_count = 0
        self.excluded_names = [name for name in expected_names[self.old_count:] if name != "random"]
        self.verify_source_unchanged()

    def verify_source_unchanged(self):
        super().verify_source_unchanged()
        for name, identity in self.opponent_files.items():
            if sha256_file(self.opponent_reference / name) != identity["sha256"]:
                raise ValueError(f"opponent reference changed: {name}")

    def update_configuration(self, configuration):
        super().update_configuration(configuration)
        # The base importer covers normal players; random games have a second source.
        configuration["imported_reference"]["imported_match_sets"] = len(self.results) - self.random_import_count
        configuration["inputs"]["implementation_sources"]["loss_plus_one_runner"] = battle.file_identity(Path(__file__))
        configuration[f"{self.label}_experiment"] = {
            "new_player": f"{self.label}_alpha_1.0", "new_player_index": self.old_count,
            "excluded_players": self.excluded_names,
            "scope": "new standalone network and any random-player pairs whose indices changed",
            "random_reference": {"directory": str(self.opponent_reference), "files": self.opponent_files,
                                 "experiment_id": self.opponent_manifest["experiment_id"],
                                 "imported_match_sets": self.random_import_count,
                                 "original_player_index": self.random_source_index,
                                 "current_player_index": self.old_count + 1,
                                 "seeding": "Imported games retain their original random draws and transcripts; new games use current pair indices"},
        }


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=HERE / "output/conv8_bn_loss_weighted_20261003_221904")
    parser.add_argument("--normal-reference-dir", type=Path, help="Defaults to the original tournament recorded in the reference")
    parser.add_argument("--model", type=Path, default=HERE.parent / "10_train_policy_network/trained/conv8_bn_loss_plus_one_e10_20261006/train/selected_model.h5")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/conv8_bn_loss_plus_one_single_20261006")
    parser.add_argument("--match-sets-per-pair", type=int, default=50, help="Use this many openings from the start of the previous XOT sequence")
    parser.add_argument("--parallel-games", type=int)
    parser.add_argument("--blend-processes-per-player", type=int)
    parser.add_argument("--engine-thread-budget", type=int)
    parser.add_argument("--sequential-tail-match-sets", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Verify prior results and the schedule without starting games or writing output")
    return parser


def prepare(args, extension_class=LossPlusOneExtension):
    if not args.model.is_file():
        raise FileNotFoundError(f"Completed model not found: {args.model}")
    extension = extension_class(args.reference_dir, args.normal_reference_dir, args.model)
    options = extension_parser().parse_args(["--output-dir", str(args.output_dir), "--include-random",
                                            "--match-sets-per-pair", str(args.match_sets_per_pair)])
    engine = extension.opponent_manifest["configuration"]["engine"]
    for key in ("parallel_games", "blend_processes_per_player", "engine_thread_budget", "sequential_tail_match_sets"):
        defaults = {"parallel_games": engine["parallel_games"],
                    "blend_processes_per_player": engine["blend_processes_per_player"],
                    "engine_thread_budget": engine.get("adaptive_search_threads", {}).get("total_threads", 70),
                    "sequential_tail_match_sets": engine.get("sequential_tail", {}).get("remaining_match_sets", 35)}
        setattr(options, key, getattr(args, key) if getattr(args, key) is not None else defaults[key])
    options.egaroucid_hint_timeout_sec = engine["egaroucid_hint_timeout_sec"]
    options.gtp_command_timeout_sec = engine["gtp_command_timeout_sec"]
    runtime_args = extension.make_args(options)
    print("Validating completed tournaments and the original XOT sequence...", flush=True)
    extension.validate_reference(runtime_args)
    levels, alphas = battle.validate_args(runtime_args, extension.resource_alphas)
    specs = extension.add_specs(runtime_args, battle.build_player_specs(runtime_args, levels, alphas, 0), 0)
    tasks = make_match_set_tasks(extension.player_count, extension.openings, runtime_args.match_sets_per_pair, runtime_args.seed)
    imported = extension.missing_imports(tasks, [])
    imported_ids = {r.task_id for r in imported}
    pending = [t for t in tasks if t.task_id not in imported_ids]
    new_index = extension.old_count
    random_index = extension.old_count + 1
    replay_random = extension.random_import_count == 0
    if any(new_index not in (t.p0_idx, t.p1_idx)
           and not (replay_random and t.p1_idx == random_index and t.p0_idx < new_index) for t in pending):
        raise ValueError("unexpected new match between existing opponents")
    by_opponent = Counter(t.p1_idx if t.p0_idx == new_index else t.p0_idx for t in pending
                          if new_index in (t.p0_idx, t.p1_idx))
    if by_opponent != Counter({i: runtime_args.match_sets_per_pair for i in range(len(specs)) if i != new_index}):
        raise ValueError("new model must have exactly N match sets against each opponent")
    print(f"New player: {specs[new_index].name}", flush=True)
    print("Opponents: " + ", ".join(s.name for i, s in enumerate(specs) if i != new_index), flush=True)
    print("Excluded: " + ", ".join(extension.excluded_names), flush=True)
    print(f"Imported: {len(imported):,} sets; new: {len(pending):,} sets / {len(pending) * 2:,} games", flush=True)
    print(f"Each opponent: {runtime_args.match_sets_per_pair} color-swapped sets; independent XOT samples per pair with seed {runtime_args.seed}", flush=True)
    # Validate resumability now; dry-run must not create even an empty output directory.
    prepare_extension_output(runtime_args, extension, dry_run=True)
    return runtime_args, extension


def main(argv=None):
    args = make_argparser().parse_args(argv)
    runtime_args, extension = prepare(args)
    if args.dry_run:
        print("Dry-run passed. No engines started; no results written.", flush=True)
        return
    with battle.OutputRunLock(runtime_args.output_dir):
        prepare_extension_output(runtime_args, extension)
        battle.run_tournament(runtime_args, extension=extension)


if __name__ == "__main__":
    main()
