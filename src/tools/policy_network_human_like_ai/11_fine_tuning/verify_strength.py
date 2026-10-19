"""Verify the completed 3,000 new games and export the new model's results."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

from experiment_common import DEFAULT_RESULTS, MODEL_ID

OUTPUT = DEFAULT_RESULTS / "strength"
MODEL = MODEL_ID


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    result = read(OUTPUT / "strength_results.json")
    progress = read(OUTPUT / "strength_progress.json")
    manifest = read(OUTPUT / "strength_manifest.json")
    assert result["experiment_id"] == progress["experiment_id"] == manifest["experiment_id"]
    assert progress["remaining_match_sets"] == 0 and progress["stop_reason"] in ("finished", "already_complete")
    assert result["completed_match_sets"] == result["total_match_sets"] == 23250
    assert result["completed_actual_games"] == result["total_actual_games"] == 46500
    names = [player["name"] for player in result["players"]]
    assert len(names) == len(set(names)) == 31 and names[-1] == MODEL
    reference = manifest["configuration"]["imported_reference"]
    reference_dir = Path(reference["directory"])
    for name, evidence in reference["files"].items():
        assert digest(reference_dir / name) == evidence["sha256"]
    original = {}
    with (reference_dir / "strength_games.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)["result"]
            entry.pop("task_id")
            key = entry["p0_idx"], entry["p1_idx"], entry["set_index"]
            assert key not in original
            original[key] = entry
    counts = defaultdict(lambda: dict(sets=0, wins=0, draws=0, losses=0,
                                    actual_games=0, actual_wins=0, actual_draws=0, actual_losses=0,
                                    disc_sum=0))
    seen, imported = set(), 0
    with (OUTPUT / "strength_games.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            game_set = entry["result"]
            for key, value in entry["task"].items():
                assert game_set[key] == value
            key = game_set["p0_idx"], game_set["p1_idx"], game_set["set_index"]
            assert key not in seen
            seen.add(key)
            if game_set["p1_idx"] < 30:
                game_set.pop("task_id")
                assert game_set == original[key]
                imported += 1
                continue
            assert game_set["p1_idx"] == 30 and 0 <= game_set["set_index"] < 50
            opponent = names[game_set["p0_idx"]]
            row = counts[opponent]
            difference = -game_set["p0_disc_diff"]
            row["sets"] += 1
            row["wins"] += difference > 0
            row["draws"] += difference == 0
            row["losses"] += difference < 0
            row["disc_sum"] += difference
            colors = game_set["color_games"]
            assert len(colors) == 2 and {game["p0_is_black"] for game in colors} == {True, False}
            assert sum(game["p0_disc_diff"] for game in colors) / 2 == game_set["p0_disc_diff"]
            for game in colors:
                assert game["transcript"].startswith(game_set["opening"])
                black, white = game["black_stones"], game["white_stones"]
                empty = 64 - black - white
                assert empty >= 0
                black_difference = black - white
                black_difference += empty if black_difference > 0 else -empty if black_difference < 0 else 0
                assert game["p0_disc_diff"] == black_difference * (1 if game["p0_is_black"] else -1)
                difference = -game["p0_disc_diff"]
                row["actual_games"] += 1
                row["actual_wins"] += difference > 0
                row["actual_draws"] += difference == 0
                row["actual_losses"] += difference < 0
    assert imported == 21750 and len(seen) == 23250
    assert len(counts) == 30 and all(row["sets"] == 50 and row["actual_games"] == 100 for row in counts.values())
    paired = [row for row in result["pair_results"] if row["player"] == MODEL]
    assert len(paired) == 30
    for row in paired:
        recounted = counts[row["opponent"]]
        for field, key in (("paired_match_sets", "sets"), ("paired_set_win", "wins"),
                           ("paired_set_draw", "draws"), ("paired_set_loss", "losses"),
                           ("actual_games", "actual_games"), ("actual_game_win", "actual_wins"),
                           ("actual_game_draw", "actual_draws"), ("actual_game_loss", "actual_losses")):
            assert row[field] == recounted[key]
        assert abs(row["avg_paired_disc_diff_descriptive"] - recounted["disc_sum"] / 50) < 1e-10
        assert row["paired_set_score"] == (recounted["wins"] + recounted["draws"] / 2) / 50
    model = next(row for row in result["summary"] if row["name"] == MODEL)
    assert model["paired_match_sets"] == 1500 and model["actual_games"] == 3000
    elo_order = sorted(result["summary"], key=lambda row: row["paired_set_elo_descriptive"], reverse=True)
    elo_rank = next(i + 1 for i, row in enumerate(elo_order) if row["name"] == MODEL)
    with (OUTPUT / "quest_vs_30_opponents.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    report = dict(status="passed", model=MODEL, summary=model, relative_elo_rank=elo_rank,
                  relative_elo_mean=1500, participants=31, new_match_sets=1500, new_actual_games=3000,
                  imported_match_sets=21750, reference_games_unchanged=True, new_results_recounted=True,
                  opponents=paired, experiment_id=result["experiment_id"],
                  sources={name:digest(OUTPUT/name) for name in ("strength_results.json", "strength_games.jsonl", "strength_manifest.json")})
    (OUTPUT / "quest_result_verified.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key not in ("opponents", "sources")}), flush=True)


if __name__ == "__main__":
    main()
