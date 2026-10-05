"""Audited opening-prefix reduction and participant append, preserving old logs."""

import copy
from datetime import datetime, timezone
from dataclasses import replace
import json
from pathlib import Path
import shutil

import battle_blend_strength as battle
from extend_strength_tournament import compatible_playing_source
from strength_tournament import ResultStore, make_manifest, make_match_set_tasks, sha256_file


LEGACY_ORCHESTRATION = {
    "battle_orchestrator": ("cb8a3c84fdb922af41771f56140833a64a06df9bd5fc8f47aa7ed450849e5a95",
                            "05c45bf911a3ef082fb39b40ba23cfb78fb0c3241212f3e912b72faaa3293205",
                            "d91f3ec7de3f83d20f2044f44a0d63efa100f8c18a8d3deb56e0aefb2400f4fc",
                            "1afe96f9f3d988aa902f463ff69250a21d0259dcbb6ccf10dbb74ac3526f2d1b",
                            "f16a590ded5024daea59f07566deb5550c839faac673961fd0fe60a6f0942e74",
                            "5e8380a2b822efccd99a56130794c26a043af0e03baf8686a3ad81ce7fee54b9",
                            "b0017ffe7746b4d31de5b6d29090f3ce547383512f869db5b7c248e918cefe2e",
                            "b2657d6db2777a32b05f855b8d65c1ca4a6e1d39d83d89082c681db9ca5719c2"),
    "extension_runner": ("019a39668b28e5eb2d6c3d98d821eba4f84d417d09afa214c73ccc521f1e8671",
                         "5bd95e8c26b38720cef5062cac2e37930b39b7511706b102f3500aadf7827665",
                         "2cbc33a7d3ea9f211e249b801b80f77f2d6a2446343dfeab66529161729ee34f",
                         "3265d6aeb4b7f1b9ed58dfa77ec84b2f5fdfe06f22f73c00d045eab149b85eab",
                         "88e2f9e0364029d367c59d71389585c62027b954498a4e8ed3dcee5bdce6f31d",
                         "9e46919c56bd98f574c1588dc9bd1bb89b851b193e6f96b25af2b2a0060a9b21",
                         "d95b16aad908a50d3286bd97130765474f8bd0e4aa456aea9902997b3cccc257",
                         "e1d8c0deebadcd4d08e32129ff4a68a3b263bc41f80bd6e46b1051b53878458f",
                         "6ce25b6bd0fd58444f14152cb0867b267ceeb562dafd7971137446144252c1cc"),
    "engine_resources": ("1601f80f77d2bedfb4b75e91de90d7ca49b568d9f0037c29f11f610672c27dcc",
                         "99e485212b4b6f5a43dc70418865887334ed79d4661e72122c48c395d1930636",
                         "2194e4bf7e5fb92ddcc6f36351c492238cdba61b47605905ea6c1e89d28f88f2",
                         "6ed6286b99763b56a01eaf360fdbc29efd99bbde99e3f91f5fb582f1cb343bdf",
                         "faf0674edc916f73d4d221ca3f2216bb439e0aa3698af808ebc675edc126ec95",
                         "61b9add9a27ed482c0bbafda948743f0efcec3d4f3253a7fcc2b6dd69ba04642",
                         "11d2d893fb204da861b0827e94d41f55bba82b3b6860964964fb8837a7fb82e1"),
    "tournament_model": ("5ce944dcbe21fab7e7ab512e2031185a92362b23693887b7a6118d14bc677c6d",
                         "10f1898e18c3167baa8265747a28dfe9ebe6cbe3b7a2e637cfcfe0560ae21215",
                         "1ba940dd6bd70af8bcc0c79148911a0e3485e45c9cb7b1181d780c74f019371d"),
}


def validate_prefix_change(previous, current):
    """Accept a shorter opening prefix, appended opponents, and reviewed code."""
    allowed = copy.deepcopy(previous)
    old_opponents = allowed["participants"].get("policy_opponents", [])
    new_opponents = current["participants"].get("policy_opponents", [])
    if len(new_opponents) < len(old_opponents) or new_opponents[:len(old_opponents)] != old_opponents:
        raise ValueError("Existing opponents cannot be removed, reordered, or changed")
    if new_opponents:
        allowed["participants"]["policy_opponents"] = copy.deepcopy(new_opponents)
    old_schedule, new_schedule = allowed["schedule"], current["schedule"]
    if not 1 <= new_schedule["match_sets_per_pair"] <= old_schedule["match_sets_per_pair"]:
        raise ValueError("Recovery can only retain a prefix of the existing opening sequence")
    for key in ("match_sets_per_pair", "task_plan_sha256", "total_match_sets"):
        old_schedule[key] = new_schedule[key]
    allowed["imported_reference"]["imported_match_sets"] = current["imported_reference"]["imported_match_sets"]
    # Search concurrency is a requested runtime adjustment. Keep the original
    # manifest in the recovery snapshot so imported results retain provenance.
    for key in ("engine_threads", "adaptive_search_threads", "parallel_games", "parallel_match_sets",
                "gtp_command_timeout_sec", "egaroucid_hint_timeout_sec", "sequential_tail"):
        if key in current["engine"]:
            allowed["engine"][key] = current["engine"][key]
    sources = allowed["inputs"]["implementation_sources"]
    for key, identity in current["inputs"]["implementation_sources"].items():
        old = sources.get(key)
        if old == identity:
            continue
        if old is None and key == "policy_opponent_gtp" and new_opponents and not old_opponents:
            sources[key] = identity
            continue
        if old is None or old["path"] != identity["path"]:
            raise ValueError(f"Unexpected implementation source change: {key}")
        if (key in LEGACY_ORCHESTRATION and old["sha256"] in LEGACY_ORCHESTRATION[key]
                or compatible_playing_source(key, old["sha256"], identity["sha256"])):
            sources[key] = identity
        else:
            raise ValueError(f"Unreviewed implementation change: {key}")
    if allowed != current:
        raise ValueError("Recovery settings differ beyond appended opponents, the opening prefix, and reviewed implementation")


def prepare_extension_output(args, extension, *, dry_run=False):
    levels, alphas = battle.validate_args(args, extension.resource_alphas)
    count = extension.player_count
    tasks = make_match_set_tasks(count, extension.openings, args.match_sets_per_pair, args.seed)
    configuration = battle.build_manifest_configuration(args, levels, alphas, extension.openings, tasks)
    extension.update_configuration(configuration)
    manifest = make_manifest(configuration)
    output = Path(args.output_dir)
    manifest_path = output / "strength_manifest.json"
    if not manifest_path.exists():
        if (output / "strength_games.jsonl").exists():
            raise ValueError("Cannot recover results without their original manifest")
        if not getattr(args, "prepare_only", False):
            return
        raise ValueError("--prepare-only requires an existing extension tournament")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if make_manifest(previous["configuration"]) != previous:
        raise ValueError("Existing extension manifest fingerprint is inconsistent")
    if previous == manifest:
        print("Existing output already matches the requested prefix and implementation.", flush=True)
        return
    validate_prefix_change(previous["configuration"], configuration)
    old_participants = previous["configuration"]["participants"]
    old_count = (sum(len(old_participants[key]) for key in ("baseline_levels", "alphas", "edax_levels"))
                 + len(old_participants["additional_policy"]["alphas"])
                 + len(old_participants.get("policy_opponents", [])))
    old_tasks = make_match_set_tasks(old_count, extension.openings,
                                    previous["configuration"]["schedule"]["match_sets_per_pair"], args.seed)
    if battle.task_plan_hash(old_tasks) != previous["configuration"]["schedule"]["task_plan_sha256"]:
        raise ValueError("Existing task plan differs from its manifest")
    by_pair_set = {(t.p0_idx, t.p1_idx, t.set_index): t for t in tasks}
    for task in old_tasks:
        if task.set_index < args.match_sets_per_pair:
            if by_pair_set[(task.p0_idx, task.p1_idx, task.set_index)].opening != task.opening:
                raise ValueError("An existing pair's opening sequence has changed")
    _, old_results = ResultStore(output).load({t.task_id: t for t in old_tasks}, read_only=True)
    retained = [replace(r, task_id=by_pair_set[(r.p0_idx, r.p1_idx, r.set_index)].task_id)
                for r in old_results if r.set_index < args.match_sets_per_pair]
    retained.extend(extension.missing_imports(tasks, retained))
    runtime_path = output / "strength_runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if make_manifest(runtime["configuration"]) != runtime:
        raise ValueError("Existing runtime fingerprint is inconsistent")
    if runtime["configuration"]["experiment_id"] != previous["experiment_id"]:
        raise ValueError("Existing runtime belongs to another tournament")
    runtime_config = copy.deepcopy(runtime["configuration"])
    runtime_config["experiment_id"] = manifest["experiment_id"]
    remaining = len(tasks) - len(retained)
    print(f"Schedule recovery: players {old_count} -> {count}; sets 0..{args.match_sets_per_pair - 1}; "
          f"retain {len(retained):,}; remaining {remaining:,} sets ({2 * remaining:,} games)", flush=True)
    if dry_run:
        return
    # Keep complete, byte-for-byte copies before replacing any live artifact.
    backup = output / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    before, staged = backup / "before", backup / "after"
    before.mkdir(parents=True)
    staged.mkdir()
    for path in output.iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".tsv", ".txt"):
            shutil.copy2(path, before / path.name)
    store = ResultStore(staged)
    store.open()
    try:
        by_id = {t.task_id: t for t in tasks}
        for result in retained:
            store.append_result(by_id[result.task_id], result)
    finally:
        store.close()
    # Preserve failures for diagnostics, including any outside the new prefix.
    if (before / "strength_failed_tasks.jsonl").exists():
        if old_count == count:
            shutil.copy2(before / "strength_failed_tasks.jsonl", staged / "strength_failed_tasks.jsonl")
        else:
            failures = []
            for line in (before / "strength_failed_tasks.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                old_task = row["task"]
                task = by_pair_set.get((old_task["p0_idx"], old_task["p1_idx"], old_task["set_index"]))
                row["original_task"] = old_task
                row["original_experiment_id"] = previous["experiment_id"]
                if task is not None:
                    row["task"] = task.identity()
                else:
                    row["excluded_from_current_schedule"] = True
                failures.append(json.dumps(row, ensure_ascii=False))
            (staged / "strength_failed_tasks.jsonl").write_text("\n".join(failures) + "\n", encoding="utf-8")
    _, verified = ResultStore(staged).load(by_id, read_only=True)
    if verified != retained:
        raise ValueError("Staged results differ from retained results")
    specs = extension.add_specs(args, battle.build_player_specs(args, levels, alphas, 0), 0)
    stats = battle.TournamentStats(count)
    for result in retained:
        stats.record(result)
    battle.write_outputs(specs, stats, staged, len(retained), len(tasks),
                         battle.target_match_sets_by_pair(tasks, count), manifest["experiment_id"])
    battle.atomic_write_json(staged / "strength_manifest.json", manifest)
    battle.atomic_write_json(staged / "strength_runtime_manifest.json", make_manifest(runtime_config))
    battle.atomic_write_json(staged / "strength_progress.json", {
        "schema_version": 3, "experiment_id": manifest["experiment_id"],
        "completed_match_sets": len(retained), "total_match_sets": len(tasks),
        "completed_actual_games": 2 * len(retained), "total_actual_games": 2 * len(tasks),
        "remaining_match_sets": remaining, "remaining_actual_games": 2 * remaining,
        "stop_reason": "ready_to_resume" if remaining else "already_complete",
        "elapsed_sec_this_run": 0.0,
    })
    receipt = {
        "previous_experiment_id": previous["experiment_id"], "experiment_id": manifest["experiment_id"],
        "selection": f"Original set_index 0..{args.match_sets_per_pair - 1} for every pair; no resampling",
        "completed_before": len(old_results), "retained_match_sets": len(retained),
        "previous_participant_count": old_count, "participant_count": count,
        "remaining_match_sets": remaining,
        "raw_sha256_before": sha256_file(before / "strength_games.jsonl"),
        "raw_sha256_after": sha256_file(staged / "strength_games.jsonl"),
        "previous_thread_policy": previous["configuration"]["engine"].get("adaptive_search_threads"),
        "thread_policy_for_remaining_games": configuration["engine"].get("adaptive_search_threads"),
        "previous_sequential_tail": previous["configuration"]["engine"].get("sequential_tail"),
        "sequential_tail_for_remaining_games": configuration["engine"].get("sequential_tail"),
        "previous_timeouts": {key: previous["configuration"]["engine"][key]
                              for key in ("gtp_command_timeout_sec", "egaroucid_hint_timeout_sec")},
        "timeouts_for_remaining_games": {key: configuration["engine"][key]
                                        for key in ("gtp_command_timeout_sec", "egaroucid_hint_timeout_sec")},
    }
    extension.verify_source_unchanged()
    for path in before.iterdir():
        if sha256_file(output / path.name) != sha256_file(path):
            raise ValueError(f"Output changed while preparing recovery: {path.name}")
    battle.atomic_write_json(backup / "recovery.json", receipt)
    for path in staged.iterdir():
        temporary = output / (path.name + ".recovery.tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(output / path.name)
    # Old bootstrap tables describe the old schedule; their backups remain.
    for pattern in ("strength_elo_ci95.*", "strength_elo_bootstrap_*.csv"):
        for path in output.glob(pattern):
            if (before / path.name).is_file():
                path.unlink()
    print(f"Recovery prepared; complete original results saved in {before}", flush=True)
