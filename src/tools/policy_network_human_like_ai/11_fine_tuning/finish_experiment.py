"""Wait for the visible workers, verify results, estimate Elo, and draw both plots."""
from datetime import datetime
from pathlib import Path
import os
import re
import subprocess
import sys
import time

from experiment_common import HERE, BASE, DEFAULT_RESULTS, MODEL_ID, read, write
from battle_blend_strength import OutputRunLock


def worker_finished(name, launch_name):
    resource = DEFAULT_RESULTS / f"{name}_resource.json"
    launch = read(DEFAULT_RESULTS / launch_name)
    # PowerShell's round-trip format has seven fractional digits; Python 3.9 accepts six.
    timestamp = re.sub(r"(\.\d{6})\d+(?=[+-]|$)", r"\1", launch["started_at"]).replace("Z", "+00:00")
    launched_at = datetime.fromisoformat(timestamp).timestamp()
    if not resource.exists() or resource.stat().st_mtime < launched_at:
        return False
    result = read(resource)
    if result["returncode"] != 0:
        raise RuntimeError(f"{name} failed (exit {result['returncode']}); inspect {name}.log")
    return True


def main():
    output = DEFAULT_RESULTS
    with OutputRunLock(output / "finalization"):
        def status(state, **extra):
            write(output / "analysis_status.json", dict(state=state, pid=os.getpid(), updated_at_unix=time.time(), **extra))
        try:
            status("waiting_for_workers")
            def wait_for_worker(name, launch_name):
                while not worker_finished(name, launch_name):
                    details = ""
                    log = output / f"{name}.log"
                    if log.exists():
                        with log.open("rb") as stream:
                            stream.seek(max(0, log.stat().st_size - 32768))
                            lines = stream.read().decode("utf-8", errors="replace").splitlines()
                        progress = [line for line in lines if "[エンジン探索進捗]" in line]
                        if progress:
                            details = progress[-1]
                    print(f"Waiting for {name}. {details}", flush=True)
                    status("waiting_for_workers", waiting_for=name, progress=details,
                           strength_complete=worker_finished("strength", "strength_visible_launch.json"))
                    time.sleep(60)

            wait_for_worker("strength", "strength_visible_launch.json")
            stages = [
                ("verifying_games", [sys.executable, "-u", str(HERE / "verify_strength.py")]),
                ("estimating_elo", [sys.executable, "-u", str(BASE / "40_test_strength/reanalyze_strength_elo.py"),
                                    "--output-dir", str(output / "strength"), "--bootstrap-replicates", "20000", "--workers", "8"]),
                ("plotting", [sys.executable, "-u", str(HERE / "plot_results.py")]),
            ]
            for stage, command in stages:
                if stage == "plotting":
                    wait_for_worker("quest_baselines", "quest_visible_launch.json")
                status(stage)
                print(stage, flush=True)
                if stage == "estimating_elo" and (output / "strength/strength_elo_ci95.json").exists():
                    saved = read(output / "strength/strength_elo_ci95.json")
                    current = read(output / "strength/strength_manifest.json")
                    if (saved["experiment_id"] == current["experiment_id"]
                            and saved["bootstrap"]["replicates"] == 20000):
                        print("Reuse the completed Elo bootstrap.", flush=True)
                        continue
                subprocess.run(command, cwd=HERE, check=True)
            report = read(output / "report_data.json")
            rows = {row["dataset"]: row for row in report["results"] if row["model"] == MODEL_ID}
            normal = {row["dataset"]: row for row in report["results"] if row["model"] == "normal_egaroucid_alpha_1.0"}
            q = rows["quest"]
            lines = ["# レート800未満へのファインチューニング結果", "",
                "通常WTHORモデルから全層を10エポック追加学習し、検証損失が最小だったモデルを評価しました。", "",
                "| テスト集合 | 局面数 | 新モデルの一致数 | 新モデルの一致率 | 元の通常WTHORモデル |", "| --- | ---: | ---: | ---: | ---: |"]
            for key, name in (("quest", "オセロクエスト800未満"), ("wthor", "WTHOR")):
                row = rows[key]
                lines.append(f"| {name} | {row['positions']:,} | {row['top1_hits']:,} | {row['top1_accuracy_percent']:.3f}% | {normal[key]['top1_accuracy_percent']:.3f}% |")
            lines += ["", f"新モデルの相対Eloは **{q['elo']:.1f}**、95%区間は **{q['elo_low']:.1f}〜{q['elo_high']:.1f}** でした。",
                "既存30モデルと各100局、計3,000局を追加し、既存43,500局と合わせて31モデルのEloを推定しました。31モデルの平均を1500とする尺度で、オセロクエストの実レートとは異なります。", "",
                "一致率は合法手のTop-1を使い、盤面を保つ対称変換で等価な手も一致として数えます。Questは学習・検証と棋譜単位で分離した未増幅test全件、WTHORは既存実験と同じ未増幅testの10万局面です。", "",
                "![WTHOR](elo_vs_agreement_wthor.png)", "", "![オセロクエスト800未満](elo_vs_agreement_quest.png)", "",
                "CSVは `comparison_31_models.csv`、数値と検証記録は `report_data.json` に保存しています。グラフのElo誤差棒はペア内セットのbootstrap 20,000反復、一致率の誤差棒はWilsonの95%区間です。一致率の区間では、同じ棋譜内の相関を補正していません。", ""]
            (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
            status("complete", results=str(output / "report_data.json"))
            print("COMPLETE: verified results and both graphs saved.", flush=True)
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
