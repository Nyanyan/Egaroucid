"""Join saved Quest disc losses and WTHOR predictions without engine searches."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_wthor_predictions(paths, names, random_path):
    reference = None
    selected = {}
    sources = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            current = {key: saved[key].copy() for key in
                       ("records", "game_ids", "symmetry_target_masks")}
            model_names = saved["model_names"].tolist()
            moves = saved["predicted_policies"]
            require(moves.shape == (len(current["records"]), len(model_names)),
                    "WTHOR prediction dimensions differ")
            require(len(model_names) == len(set(model_names)), "Duplicate WTHOR model IDs")
            if reference is None:
                reference = current
            else:
                require(all(np.array_equal(reference[key], current[key]) for key in current),
                        "WTHOR inputs use different rows, games or symmetry masks")
            # Earlier input files have explicit precedence for overlapping models.
            for name in names:
                if name in model_names and name not in selected:
                    selected[name] = moves[:, model_names.index(name)].copy()
            sources.append(dict(file=Path(path).name, sha256=sha256(path)))
    require(reference is not None, "At least one WTHOR prediction file is required")
    if "random" in names:
        require(random_path is not None, "Random predictions are required")
        with np.load(random_path, allow_pickle=False) as saved:
            require(str(saved["reference_sha256"]) in {v["sha256"] for v in sources},
                    "Random predictions reference a different WTHOR sample")
            selected["random"] = saved["moves"].copy()
        sources.append(dict(file=Path(random_path).name, sha256=sha256(random_path)))
    require(set(names).issubset(selected), "Some models lack WTHOR predictions")
    matrix = np.column_stack([selected[name] for name in names])
    require(matrix.shape == (len(reference["records"]), len(names))
            and np.all((matrix >= 0) & (matrix < 64)), "Invalid WTHOR policies")
    bits = np.left_shift(np.uint64(1), matrix.astype(np.uint64))
    return (bits & reference["symmetry_target_masks"][:, None]) != 0, sources


def prepare(quest_results, predictions, random_predictions, wthor_human_losses, output):
    quest_results, output = Path(quest_results), Path(output)
    report = read_json(quest_results / "results.json")
    names = report["identity"]["series"]
    require(names[0] == "human" and len(names) == len(set(names)), "Invalid Quest model IDs")
    names = names[1:]
    losses = np.load(quest_results / "row_losses.npy", allow_pickle=False)
    with np.load(quest_results / "sample.npz", allow_pickle=False) as saved:
        bands = saved["band_ids"]
    require(losses.shape == (len(bands), len(names) + 1)
            and np.isfinite(losses).all() and (losses >= 0).all(), "Invalid Quest losses")
    hits, sources = merge_wthor_predictions(predictions, names, random_predictions)
    human = np.load(wthor_human_losses, allow_pickle=False)
    require(human.shape == (len(hits),) and np.isfinite(human).all()
            and (human >= 0).all(), "WTHOR human losses have a different denominator")
    strata = np.unique(bands)
    require(len(strata) == len(report["identity"]["bands"]), "Quest strata differ")
    pieces = [losses[bands == band, 1:] for band in strata]
    require(all(len(part) >= 2 for part in pieces), "Each Quest stratum needs at least two rows")
    require(len({len(part) for part in pieces}) == 1, "Expected equally sized Quest strata")
    x = np.mean([part.mean(axis=0) for part in pieces], axis=0)
    covariance_x = sum(np.cov(part, rowvar=False, ddof=1) / len(part) for part in pieces) / len(pieces)**2
    y = hits.mean(axis=0)
    covariance_y = np.cov(hits.astype(float), rowvar=False, ddof=1) / len(hits)
    expected = {row["model"]: row["mean_disc_loss"] for row in report["overall"]}
    require(all(np.isclose(x[i], expected[name], atol=1e-12, rtol=0)
                for i, name in enumerate(names)), "Verified Quest means differ")
    labels_path = quest_results / "model_labels.json"
    labels = read_json(labels_path) if labels_path.exists() else {}
    rows = [dict(model=name, label=labels.get(name, name), mean_disc_loss=float(x[i]),
                 wthor_accuracy=float(y[i]), wthor_hits=int(hits[:, i].sum()),
                 quest_positions=len(losses), wthor_positions=len(hits))
            for i, name in enumerate(names)]
    human_by_band = {row["band"]: row for row in report["by_band"] if row["model"] == "human"}
    references = [dict(key="wthor", label="WTHOR全体", value=float(human.mean()),
                       positions=len(human), color="#34495E", linestyle="--")]
    for band, color, style in (("2000-2200", "#277DA8", ":"),
                               ("1400-1600", "#AD6A20", "-."),
                               ("800-1000", "#C05066", "--")):
        row = human_by_band[band]
        references.append(dict(key="quest_" + band, label="オセロクエスト " + band,
                               value=row["mean_disc_loss"], positions=row["positions"],
                               color=color, linestyle=style))
    for path in (quest_results / "row_losses.npy", quest_results / "sample.npz",
                 quest_results / "results.json", Path(wthor_human_losses)):
        sources.append(dict(file=path.name, sha256=sha256(path)))
    result = dict(rows=rows, references=references, covariance_x=covariance_x.tolist(),
                  covariance_y=covariance_y.tolist(), covariance_model_order=names,
                  reference_level=21, quest_band_count=len(pieces),
                  quest_rows_per_band=len(pieces[0]), sources=sources,
                  assumptions=["Fixed models; independent Quest and WTHOR samples.",
                               "Quest strata have equal weights. Within each stratum and WTHOR, rows are independent.",
                               "Covariance is estimated jointly across models; no game-cluster or duplicate-board correction."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest-results", type=Path, required=True)
    parser.add_argument("--wthor-predictions", type=Path, action="append", required=True)
    parser.add_argument("--random-predictions", type=Path)
    parser.add_argument("--wthor-human-losses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.quest_results, args.wthor_predictions, args.random_predictions,
                   args.wthor_human_losses, args.output)
    print(f"Prepared {len(data['rows'])} models and {len(data['references'])} human reference lines.")


if __name__ == "__main__":
    main()
