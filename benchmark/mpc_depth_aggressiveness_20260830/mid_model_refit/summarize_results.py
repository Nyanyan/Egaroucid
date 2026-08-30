#!/usr/bin/env python3
"""ridge sweepを、holdoutを選択に使わず日本語Markdownへまとめる。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEVELS = ("74", "88", "93")
OFFSETS = ("-4", "-2", "0", "2", "4")


def load_reports() -> dict[float, dict[str, object]]:
    reports: dict[float, dict[str, object]] = {}
    for path in sorted(HERE.glob("fit_results_ridge_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        ridge = float(data["method"]["ridge"])
        reports[ridge] = data
    if not reports:
        path = HERE / "fit_results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        reports[float(data["method"]["ridge"])] = data
    return reports


def node_average(metrics: dict[str, object]) -> float:
    return sum(float(metrics["levels"][level]["estimated_node_ratio"]) for level in LEVELS) / len(LEVELS)


def feasible(candidate: dict[str, object]) -> bool:
    current = candidate["cv_aggregate"]["current"]
    refit = candidate["cv_aggregate"]["safety_refit"]
    return all(
        int(refit["levels"][level]["margin_excess_ge_2"]) <= int(current["levels"][level]["margin_excess_ge_2"])
        and int(refit["levels"][level]["margin_excess_ge_4"]) <= int(current["levels"][level]["margin_excess_ge_4"])
        for level in LEVELS
    )


def triple(metrics: dict[str, object], key: str) -> str:
    return "/".join(str(metrics["levels"][level][key]) for level in LEVELS)


def percent_ratio(new: float, old: float) -> str:
    return f"{100.0 * new / old:.3f}%"


def main() -> int:
    reports = load_reports()
    ridge_rows: list[dict[str, object]] = []
    filtered: dict[str, tuple[float, dict[str, object]]] = {}
    for offset in OFFSETS:
        choices = []
        for ridge, report in sorted(reports.items()):
            candidate = report["candidates"][offset]
            current_cv = candidate["cv_aggregate"]["current"]
            refit_cv = candidate["cv_aggregate"]["safety_refit"]
            current_holdout = candidate["holdout"]["current"]
            refit_holdout = candidate["holdout"]["safety_refit"]
            is_feasible = feasible(candidate)
            cv_relative = node_average(refit_cv) / node_average(current_cv)
            holdout_relative = node_average(refit_holdout) / node_average(current_holdout)
            row = {
                "offset": int(offset),
                "ridge": ridge,
                "cv_constraint": is_feasible,
                "cv_node_relative": cv_relative,
                "cv_ge2_current_74_88_93": triple(current_cv, "margin_excess_ge_2"),
                "cv_ge2_refit_74_88_93": triple(refit_cv, "margin_excess_ge_2"),
                "cv_ge4_current_74_88_93": triple(current_cv, "margin_excess_ge_4"),
                "cv_ge4_refit_74_88_93": triple(refit_cv, "margin_excess_ge_4"),
                "cv_wrong_cuts_current": triple(current_cv, "simulated_wrong_cut_count"),
                "cv_wrong_cuts_refit": triple(refit_cv, "simulated_wrong_cut_count"),
                "holdout_node_relative": holdout_relative,
                "holdout_ge2_current_74_88_93": triple(current_holdout, "margin_excess_ge_2"),
                "holdout_ge2_refit_74_88_93": triple(refit_holdout, "margin_excess_ge_2"),
                "holdout_ge4_current_74_88_93": triple(current_holdout, "margin_excess_ge_4"),
                "holdout_ge4_refit_74_88_93": triple(refit_holdout, "margin_excess_ge_4"),
                "holdout_wrong_cuts_current": triple(current_holdout, "simulated_wrong_cut_count"),
                "holdout_wrong_cuts_refit": triple(refit_holdout, "simulated_wrong_cut_count"),
            }
            ridge_rows.append(row)
            if is_feasible:
                choices.append((cv_relative, ridge, candidate))
        if not choices:
            raise RuntimeError(f"offset {offset}: CV制約を満たすridgeがありません")
        _, ridge, candidate = min(choices, key=lambda value: (value[0], value[1]))
        filtered[offset] = (ridge, candidate)

    with (HERE / "ridge_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ridge_rows[0]))
        writer.writeheader()
        writer.writerows(ridge_rows)

    current = next(iter(reports.values()))["current_coefficients"]
    with (HERE / "cv_filtered_coefficients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["offset", "ridge", "kind", "a", "b", "c", "d", "e", "f", "g_effective"])
        for offset, (ridge, candidate) in filtered.items():
            writer.writerow([offset, ridge, "current", *current])
            writer.writerow([offset, ridge, "refit", *candidate["safety_coefficients"]])

    header = [
        "#pragma once",
        "// CVで2石/4石超過が現行以下となるridgeのうち、推定node比最小の比較候補。",
        "// 採用を示すものではない。gは実効値であり、MPC_PROBCUT_G_OFFSETを足してはならない。",
        "namespace mid_mpc_cv_filtered_20260830 {",
        "struct Coefficients { double a, b, c, d, e, f, g; };",
    ]
    for offset, (ridge, candidate) in filtered.items():
        name = "m" + str(-int(offset)) if int(offset) < 0 else "p" + offset
        values = ", ".join(f"{float(value):.17g}" for value in candidate["safety_coefficients"])
        header.append(f"inline constexpr Coefficients offset_{name}{{{values}}}; // ridge={ridge:g}")
    header += ["} // namespace mid_mpc_cv_filtered_20260830", ""]
    (HERE / "candidate_coefficients_cv_filtered.hpp").write_text("\n".join(header), encoding="utf-8")

    first_report = next(iter(reports.values()))
    inputs = first_report["inputs"]
    overlap = first_report["development_holdout_overlap"]
    metadata_overlap = first_report["metadata_overlap"]
    lines = [
        "# 中盤MPC誤差モデル a～g 再学習レポート",
        "",
        "## 何を変更したか",
        "",
        "現行と同じ次式を使い、`a`～`g`の7係数を直接再学習した。標準偏差や丸め済みmarginへ固定倍率を掛ける候補は含めていない。",
        "",
        "```text",
        "x = a * (石数 / 64) + b * (浅い探索深度 / 60) + c * (深い探索深度 / 60)",
        "sigma = d*x^3 + e*x^2 + f*x + g",
        "margin = ceil(z * sigma)",
        "```",
        "",
        "現行ソースの `g` は基礎値 `1.5719077939546169` に既定offset `0.3`を加えるため、比較に使った実効値は `1.8719077939546169`。以下の新しい `g` はすべて最終的な実効値であり、実装時にoffsetをもう一度加えてはならない。",
        "",
        "係数は浅い探索深度の差ごとに別々にfitした。実戦相当として比較した深い探索深度12～16での対応は次の通り。0は探索を行わず静的評価値を使う。",
        "",
        "|浅い深度差|深さ12|深さ13|深さ14|深さ15|深さ16|",
        "|---:|---:|---:|---:|---:|---:|",
        "|-4|0|1|0|3|2|",
        "|-2|2|3|2|5|4|",
        "|0（現行）|4|5|4|7|6|",
        "|+2|6|7|6|9|8|",
        "|+4|8|9|8|11|10|",
        "",
        "## データ分離",
        "",
        "|項目|件数|",
        "|---|---:|",
        f"|係数学習用の元行数|{inputs['development_rows_before_quarantine']:,}|",
        f"|holdoutと同じ対局連結成分だったため除外した行|{inputs['development_rows_removed_by_root_quarantine']:,}|",
        f"|除外後の開発盤面|{inputs['development_boards']:,}|",
        f"|除外後の対局連結成分|{inputs['development_roots']:,}|",
        f"|独立holdout盤面|{inputs['holdout_boards']:,}|",
        f"|独立holdout対局連結成分|{inputs['holdout_roots']:,}|",
        f"|除外前の開発/holdout重複盤面|{overlap['before_quarantine_boards']:,}|",
        f"|除外前の開発/holdout重複対局連結成分|{overlap['before_quarantine_roots']:,}|",
        f"|除外後の開発/holdout重複盤面|{overlap['after_quarantine_boards']:,}|",
        f"|除外後の開発/holdout重複対局連結成分|{overlap['after_quarantine_roots']:,}|",
        f"|再収集値と既存holdoutの共通行|{inputs['holdout_original_rows_rechecked']:,}|",
        f"|共通行のscore不一致|{inputs['holdout_regeneration_score_mismatches']:,}|",
        "",
        "同一盤面が別対局にも現れた場合は、それらの対局を同じ連結成分にまとめた。5-fold CVはこの連結成分単位で分割したため、同一対局由来局面や重複盤面が学習側とvalidation側を跨がない。独立holdoutの誤差値は係数作成・ridge選別には使っていない。",
        "",
        "### 元メタデータ集合どうしの重複監査",
        "",
        "`model_dev`は`mid_mpc_model_dev`、`ggs_dev/ggs_holdout`は`mid_probcut_dev/holdout`の盤面由来を示す。tuning三集合は係数学習には使っていない。",
        "",
        "|集合A|集合B|同一盤面|同じ対局連結成分|",
        "|---|---|---:|---:|",
    ]
    for pair, values in sorted(metadata_overlap.items()):
        left, right = pair.split("__", 1)
        lines.append(f"|{left}|{right}|{values['board_overlap']}|{values['root_overlap']}|")
    lines += [
        "",
        "## 指標の定義",
        "",
        "`deep_value - shallow_value`を浅い探索の誤差とした。`ceil(z*sigma)`を超えた残差について、超過が2石以上・4石以上となる行数を数えた。NWSの境界値betaを-16～16へ一様に置き、浅い探索が深い探索値と矛盾するcutを出す回数も数えた。",
        "",
        "推定node比は、各betaでMPC probeを行い、cut時は深い探索を省略し、cutしない場合は浅い探索と深い探索の両方を行う、という近似で計算した。これは実際の探索木で測ったnode数ではなく、係数比較用の同一条件proxyである。浅い探索そのもののnode数は100%探索・1スレッド・各局面で置換表を空にして実測した。",
        "",
        "## ridge全候補（開発5-fold CV）",
        "",
        "`CV制約`は74/88/93の全水準で、2石以上と4石以上の超過行数が現行以下だったことを示す。`node相対`は新係数の推定node比を現行係数で割った値で、100%未満なら減少。誤差数は74/88/93の順。",
        "",
        "|浅い深度差|ridge|CV制約|CV node相対|CV 2石以上 現行→新|CV 4石以上 現行→新|holdout node相対|holdout 2石以上 現行→新|holdout 4石以上 現行→新|",
        "|---:|---:|:---:|---:|---|---|---:|---|---|",
    ]
    for row in ridge_rows:
        lines.append(
            f"|{row['offset']:+d}|{row['ridge']:g}|{'○' if row['cv_constraint'] else '×'}|"
            f"{100.0 * row['cv_node_relative']:.3f}%|"
            f"{row['cv_ge2_current_74_88_93']}→{row['cv_ge2_refit_74_88_93']}|"
            f"{row['cv_ge4_current_74_88_93']}→{row['cv_ge4_refit_74_88_93']}|"
            f"{100.0 * row['holdout_node_relative']:.3f}%|"
            f"{row['holdout_ge2_current_74_88_93']}→{row['holdout_ge2_refit_74_88_93']}|"
            f"{row['holdout_ge4_current_74_88_93']}→{row['holdout_ge4_refit_74_88_93']}|"
        )

    lines += [
        "",
        "## CV制約内で推定node比が最小だった係数セット",
        "",
        "これは機械的な表の絞り込みであり、採用判断ではない。各浅い深度差について、上表でCV制約が○のridgeだけを残し、その中でCVの推定node比が最小のものを表示する。",
        "",
        "|浅い深度差|ridge|a|b|c|d|e|f|g（実効値）|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for offset, (ridge, candidate) in filtered.items():
        values = candidate["safety_coefficients"]
        lines.append("|" + "|".join([
            f"{int(offset):+d}", f"{ridge:g}", *[f"{float(value):.10f}" for value in values]
        ]) + "|")
    lines += [
        "",
        "現行係数は全候補共通で、`a=0.8335834704, b=-4.7177890997, c=1.1467905782, d=-0.5274699259, e=6.5091001394, f=3.9546352082, g=1.8719077940（実効値）`。",
        "",
        "### gの表記",
        "",
        "`rms_refit`はGaussian NLLでa～gを直接fitした係数、`safety_refit`は同じ係数を出発点に、開発側だけで2石/4石超過制約を満たすようg係数を直接再調整した係数である。`safety_refit`でもa～fは`rms_refit`で再学習した値のままで、現行値とは異なる。どちらも固定倍率を使わない。",
        "",
        "|浅い深度差|ridge|現行g基礎値|現行offset|現行g実効値|rms_refit g実効値|safety_refit g実効値|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for offset, (ridge, candidate) in filtered.items():
        lines.append(
            f"|{int(offset):+d}|{ridge:g}|1.57190779395|+0.3|1.87190779395|"
            f"{float(candidate['rms_coefficients'][6]):.11f}|{float(candidate['safety_coefficients'][6]):.11f}|"
        )
    lines += [
        "",
        "新しい2列のgは、いずれも追加offsetなしでそのまま式へ入れる実効値である。",
        "",
        "### rms_refit / safety_refit のCV・独立holdout表",
        "",
        "`node相対`は同じ浅い探索深度の現行係数を100%とした値。2石/4石以上と誤cut数は74/88/93の順。",
        "",
        "|浅い深度差|データ|モデル|NLL|標準化RMS|node相対|2石以上|4石以上|誤cut数|",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for offset, (_, candidate) in filtered.items():
        for dataset_name, collection in (("CV", candidate["cv_aggregate"]), ("独立holdout", candidate["holdout"])):
            baseline = collection["current"]
            baseline_nodes = node_average(baseline)
            for model_name in ("current", "rms_refit", "safety_refit"):
                metrics = collection[model_name]
                lines.append(
                    f"|{int(offset):+d}|{dataset_name}|{model_name}|"
                    f"{metrics['weighted_gaussian_nll']:.4f}|{metrics['weighted_normalized_rms']:.4f}|"
                    f"{100.0 * node_average(metrics) / baseline_nodes:.3f}%|"
                    f"{triple(metrics, 'margin_excess_ge_2')}|{triple(metrics, 'margin_excess_ge_4')}|"
                    f"{triple(metrics, 'simulated_wrong_cut_count')}|"
                )
    lines += [
        "",
        "## 絞り込み後係数のCV・独立holdout比較",
        "",
        "|浅い深度差|CV node相対|holdout node相対|CV NLL 現行→新|holdout NLL 現行→新|CV浅い探索node平均|holdout浅い探索node平均|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for offset, (ridge, candidate) in filtered.items():
        current_cv = candidate["cv_aggregate"]["current"]
        refit_cv = candidate["cv_aggregate"]["safety_refit"]
        current_holdout = candidate["holdout"]["current"]
        refit_holdout = candidate["holdout"]["safety_refit"]
        lines.append(
            f"|{int(offset):+d}|{percent_ratio(node_average(refit_cv), node_average(current_cv))}|"
            f"{percent_ratio(node_average(refit_holdout), node_average(current_holdout))}|"
            f"{current_cv['weighted_gaussian_nll']:.4f}→{refit_cv['weighted_gaussian_nll']:.4f}|"
            f"{current_holdout['weighted_gaussian_nll']:.4f}→{refit_holdout['weighted_gaussian_nll']:.4f}|"
            f"{refit_cv['measured_shallow_nodes_mean']:,.0f}|{refit_holdout['measured_shallow_nodes_mean']:,.0f}|"
        )
    lines += [
        "",
        "この係数再学習だけでは、CV制約を保ったまま推定node数が明確に減る組合せは得られていない。浅い深度差+4・ridge 3ではCVが99.998%だった一方、独立holdoutは100.167%だった。他の絞り込み後係数も独立holdoutで100.072～100.478%だった。これは採否ではなく、今回の係数式・データ・制約で観測した比較結果である。",
        "",
        "## 再実行",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File .\\run_all.ps1 -Python python -Compiler clang++",
        "# holdoutの浅い探索値を再収集しない場合",
        "powershell -ExecutionPolicy Bypass -File .\\run_all.ps1 -Python python -Compiler clang++ -SkipHoldoutCollection",
        "```",
        "",
        "生データは `fit_results_ridge_*.json`、全ridge表は `ridge_sweep.csv`、絞り込み後係数は `cv_filtered_coefficients.csv` と `candidate_coefficients_cv_filtered.hpp` に保存した。",
        "",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
