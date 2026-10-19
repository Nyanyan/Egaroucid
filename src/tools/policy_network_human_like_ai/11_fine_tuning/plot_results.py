"""Draw two verified 31-player Elo versus agreement figures and export their data."""
from pathlib import Path
import csv
import sys

from experiment_common import (BASE, DEFAULT_RUN, DEFAULT_RESULTS, MODEL_ID, read, write,
                               file_sha256, validate_model)
import numpy as np
import evaluate_gpw_human_match as gpw
sys.path.insert(0, str(BASE / "50_rating_vs_human_match"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from label_editor import arrange_unplaced_labels
from plot_rating_vs_human_match import configure_plotting

COLORS = dict(E="#249972", D="#2468B4", P="#D62728", R="#747C85", Q="#7844B5")
MARKERS = dict(E="o", D="s", P="^", R="X", Q="*")


def label(name):
    if name == MODEL_ID:
        return "今回のファインチューニング", "Q", None
    if name == "random":
        return "R", "R", None
    for prefix, family in (("egaroucid_l", "E"), ("edax_l", "D")):
        if name.startswith(prefix):
            return family + name[len(prefix):], family, int(name[len(prefix):])
    for prefix, offset in (("normal_egaroucid_alpha_", 0), ("loss_a_normal_alpha_", 1), ("loss_b_loss_a_alpha_", 2)):
        if name.startswith(prefix):
            alpha = float(name[len(prefix):]) + offset
            endpoint = {1: "通常NN", 2: "石損A", 3: "石損B"}.get(alpha)
            return f"α={alpha:.1f}" + (f"（{endpoint}）" if endpoint else ""), "P", alpha
    raise ValueError(name)


def load_comparisons(output):
    wthor = read(output / "wthor/summary.json")
    new = read(output / "quest_new/summary.json")
    baseline = read(output / "quest_baselines/summary.json")
    for result in (wthor, new, baseline):
        if result["status"] != "passed":
            raise ValueError("Incomplete agreement results")
    for folder, result in (("wthor", wthor), ("quest_new", new), ("quest_baselines/test", baseline)):
        if file_sha256(output / folder / "predictions.npz") != result["predictions_sha256"]:
            raise ValueError(f"Prediction file changed: {folder}")
    with np.load(output / "quest_baselines/test/predictions.npz", allow_pickle=False) as old:
        with np.load(output / "quest_new/predictions.npz", allow_pickle=False) as trained:
            if (not np.array_equal(old["records"], trained["records"])
                    or not np.array_equal(old["symmetry_target_masks"], trained["symmetry_target_masks"])):
                raise ValueError("Quest baselines and new model did not use the same test sample")
            recounted = gpw.summarize_and_verify(dict(name="test", records=old["records"], losses=None),
                old["predicted_policies"], old["symmetry_target_masks"])
            if recounted != baseline["results"]:
                raise ValueError("Quest reference prediction recount differs")
    return dict(wthor=wthor["comparison"], quest=baseline["results"] + [new["result"]])


def render(rows, dataset, output):
    configure_plotting()
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.subplots_adjust(left=.08, right=.97, top=.80, bottom=.20)
    by_name = {row["model"]: row for row in rows}
    for family in ("E", "D", "P"):
        series = sorted((row for row in rows if row["family"] == family), key=lambda row: row["order"])
        if family == "P":
            series.insert(0, by_name["egaroucid_l21"])
        ax.plot([r["elo"] for r in series], [r["top1_accuracy_percent"] for r in series],
                color=COLORS[family], lw=1.3, alpha=.75, zorder=2)
    labels = {}
    for row in rows:
        x, y, family = row["elo"], row["top1_accuracy_percent"], row["family"]
        ax.errorbar(x, y, xerr=[[x-row["elo_low"]], [row["elo_high"]-x]],
                    yerr=[[y-100*row["top1_ci95_lower"]], [100*row["top1_ci95_upper"]-y]],
                    fmt=MARKERS[family], color=COLORS[family], ms=15 if family == "Q" else 6,
                    ecolor=COLORS[family], elinewidth=.9, capsize=2.5, zorder=5)
        text = row["label"]
        if family == "Q":
            text += f"\n一致率 {y:.2f}% ／ Elo {x:.0f}"
        labels[row["model"]] = ax.annotate(text, (x, y), xytext=(12, 18), textcoords="offset points",
            fontsize=11 if family == "Q" else 9.5, color=COLORS[family],
            weight="bold" if family == "Q" else "normal", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=.65, color=COLORS[family]), zorder=6)
    y_low = min(100*r["top1_ci95_lower"] for r in rows)
    y_high = max(100*r["top1_ci95_upper"] for r in rows)
    ax.set(xlim=(min(0, min(r["elo_low"] for r in rows)-130), max(r["elo_high"] for r in rows)+160),
           ylim=(5*np.floor(y_low/5)-3, 5*np.ceil(y_high/5)+5),
           xlabel="強さ：相対Elo（31モデルの平均＝1500）", ylabel="着手一致率（%）")
    ax.grid(alpha=.16)
    title = "WTHOR" if dataset == "wthor" else "オセロクエスト：レート800未満"
    count = 100000 if dataset == "wthor" else 6692
    fig.suptitle(f"{title} — 強さと着手一致率", x=.08, ha="left", y=.965, fontsize=22, weight="bold")
    fig.text(.08, .913, f"共通のテスト{count:,}局面で比較 ／ 通常WTHORモデルから800未満へ追加学習", fontsize=12, color="#526173")
    legends = [("E", "Egaroucid（E：レベル）"), ("D", "Edax（D：レベル）"),
               ("P", "既存NN・融合方策"), ("R", "ランダム"), ("Q", "今回のモデル")]
    fig.legend(handles=[Line2D([], [], color=COLORS[key], marker=MARKERS[key], lw=1.3 if key in "EDP" else 0,
                               label=text, ms=8) for key, text in legends],
               loc="upper left", bbox_to_anchor=(.073, .882), ncol=5, frameon=False, columnspacing=1.5)
    dataset_note = ("WTHOR：既存実験と同じ未増幅testの100,000局面（seed 613）。" if dataset == "wthor" else
                    "Quest：学習・検証から棋譜単位で分離した未増幅testの6,692局面全件。手番側のレートが800未満。")
    fig.text(.08, .035, dataset_note + "\n"
        "一致率は合法手から選んだTop-1。盤面を保つ対称変換で等価な着手も一致として数える。\n"
        "Eloは既存43,500局＋追加3,000局から推定した相対値。オセロクエストの実レートとは別の尺度。\n"
        "誤差棒：Eloはペア内セットbootstrap 20,000反復、一致率は95% Wilson区間（棋譜内の相関は未補正）。\n"
        "α=0：Egaroucid L21、α=1：通常NN、α=2：石損A、α=3：石損B。線はレベル・α順に接続。",
        fontsize=9.1, linespacing=1.65, color="#526173")
    arrange_unplaced_labels(fig, labels, set())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {key: Text.get_window_extent(value, renderer) for key, value in labels.items()}
    overlaps = [(a, b) for i, a in enumerate(boxes) for b in list(boxes)[i+1:] if boxes[a].overlaps(boxes[b])]
    if overlaps:
        raise ValueError(f"Overlapping figure labels: {overlaps}")
    files = []
    for extension in ("png", "svg"):
        path = output / f"elo_vs_agreement_{dataset}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.2)
        files.append(dict(path=str(path), sha256=file_sha256(path)))
    plt.close(fig)
    return files


def main():
    output = DEFAULT_RESULTS
    evidence = validate_model(DEFAULT_RUN / "selected_model.h5")
    verified = read(output / "strength/quest_result_verified.json")
    ratings = read(output / "strength/strength_elo_ci95.json")
    for filename, expected in verified["sources"].items():
        if file_sha256(output / "strength" / filename) != expected:
            raise ValueError(f"Strength results changed: {filename}")
    if (verified["status"] != "passed" or ratings["experiment_id"] != verified["experiment_id"]
            or ratings["bootstrap"]["replicates"] != 20000):
        raise ValueError("Incomplete or mismatched Elo analysis")
    manifest = read(output / "strength/strength_manifest.json")
    if manifest["configuration"]["inputs"]["quest_model"]["sha256"] != evidence["selected_model.h5"]["sha256"]:
        raise ValueError("Tournament model differs from trained model")
    rates = {r["name"]: r for r in ratings["results"]}
    comparisons = load_comparisons(output)
    final_rows, artifacts = [], []
    for dataset, records in comparisons.items():
        if set(rates) != {r["model"] for r in records} or len(records) != 31:
            raise ValueError("Expected identical 31-player rosters")
        rows = []
        for record in records:
            name = record["model"]
            display, family, order = label(name)
            rate = rates[name]
            rows.append(dict(**{**record, "dataset": dataset}, label=display, family=family, order=order,
                        elo=rate["paired_set_elo"], elo_low=rate["paired_set_elo_ci95_low"],
                        elo_high=rate["paired_set_elo_ci95_high"]))
        if abs(sum(r["elo"] for r in rows)/31 - 1500) > 1e-8:
            raise ValueError("Elo centering differs")
        final_rows.extend(rows)
        artifacts.extend(render(rows, dataset, output))
    with (output / "comparison_31_models.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    write(output / "report_data.json", dict(status="passed", model=MODEL_ID, model_evidence=evidence,
        results=final_rows, artifacts=artifacts, experiment_id=ratings["experiment_id"],
        verification=dict(quest_identical_test_records=True, quest_saved_predictions_recounted=True,
                          imported_games_unchanged=True, no_label_overlap=True)))
    print("Saved both figures and verified 62 model/dataset results.", flush=True)


if __name__ == "__main__":
    main()
