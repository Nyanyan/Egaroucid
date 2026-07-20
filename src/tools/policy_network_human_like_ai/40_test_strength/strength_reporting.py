#!/usr/bin/env python3
"""Reports and resource monitoring for the strength tournament."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BIN_DIR = REPO_ROOT / "bin"

import sys

sys.path.insert(0, str(BIN_DIR))

from elo_rating_backcal import fit_elo_from_winrates  # noqa: E402
from strength_engine import ProcessManager  # noqa: E402
from strength_tournament import (  # noqa: E402
    PlayerSpec,
    TournamentStats,
    atomic_write_json,
    atomic_write_text,
    score_confidence_interval,
)


def format_elapsed(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def estimate_elos(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
) -> Dict[str, float]:
    """Return descriptive Elo point estimates with finite pair smoothing.

    Every unordered player pair shares its XOT opening sequence with every
    other pair.  Consequently, the fitted ratings are useful summaries of this
    tournament, but the usual independent-game uncertainty calculation is not
    applicable.  A half-win and half-loss are added to every pair so an
    observed 0% or 100% score still produces a finite estimate.
    """
    player_count = len(specs)
    win_rates = np.full((player_count, player_count), np.nan, dtype=float)
    match_sets = np.zeros((player_count, player_count), dtype=float)
    for player_idx in range(player_count):
        for opponent_idx in range(player_count):
            if player_idx == opponent_idx:
                continue
            wins, draws, losses = stats.results[player_idx][opponent_idx]
            n = wins + draws + losses
            if n == 0:
                # A neutral pseudo-result for an unplayed pair would make the
                # partial-tournament Elo look more complete than the data.
                return {}
            match_sets[player_idx, opponent_idx] = float(n + 1)
            win_rates[player_idx, opponent_idx] = (
                wins + 0.5 * draws + 0.5
            ) / (n + 1)
    try:
        ratings = fit_elo_from_winrates(
            win_rates,
            games=match_sets,
            names=[spec.name for spec in specs],
        )
    except ValueError:
        return {}
    return {spec.name: float(ratings[spec.name]) for spec in specs}


def _csv_text(fieldnames: Sequence[str], rows: Sequence[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _tsv_text(rows: Sequence[Sequence[object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def _score(wins: int, draws: int, losses: int) -> Optional[float]:
    n = wins + draws + losses
    return (wins + 0.5 * draws) / n if n else None


def _logical_player_command(spec: PlayerSpec) -> List[str]:
    """Remove the run-specific ephemeral policy-server port from metadata."""

    command = list(spec.command)
    try:
        port_index = command.index("--policy-server-port") + 1
    except ValueError:
        return command
    if port_index < len(command):
        command[port_index] = "<managed-policy-server-port>"
    return command


def _validated_target_matrix(
    specs: Sequence[PlayerSpec],
    target_match_sets_by_pair: Sequence[Sequence[int]],
) -> List[List[int]]:
    player_count = len(specs)
    if len(target_match_sets_by_pair) != player_count:
        raise ValueError(
            "target_match_sets_by_pair must have one row per player"
        )
    matrix: List[List[int]] = []
    for player_idx, source_row in enumerate(target_match_sets_by_pair):
        if len(source_row) != player_count:
            raise ValueError(
                "target_match_sets_by_pair must be a square player matrix"
            )
        row = [int(value) for value in source_row]
        if any(value < 0 for value in row):
            raise ValueError(
                "target_match_sets_by_pair cannot contain negative targets"
            )
        if row[player_idx] != 0:
            raise ValueError(
                "target_match_sets_by_pair diagonal entries must be zero"
            )
        matrix.append(row)
    for player_idx in range(player_count):
        for opponent_idx in range(player_idx + 1, player_count):
            if matrix[player_idx][opponent_idx] != matrix[opponent_idx][player_idx]:
                raise ValueError(
                    "target_match_sets_by_pair must be symmetric"
                )
    return matrix


def build_summary_rows(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
    ratings: Dict[str, float],
    target_match_sets_by_pair: Sequence[Sequence[int]],
) -> List[dict]:
    target_matrix = _validated_target_matrix(
        specs,
        target_match_sets_by_pair,
    )
    rows = []
    for player_idx, spec in enumerate(specs):
        wins, draws, losses = stats.player_record(player_idx)
        paired_n = wins + draws + losses
        actual_wins = sum(
            record[0] for record in stats.actual_results[player_idx]
        )
        actual_draws = sum(
            record[1] for record in stats.actual_results[player_idx]
        )
        actual_losses = sum(
            record[2] for record in stats.actual_results[player_idx]
        )
        actual_n = actual_wins + actual_draws + actual_losses
        average_disc_diff = (
            sum(stats.disc_diff[player_idx]) / paired_n
            if paired_n
            else None
        )
        elo = ratings.get(spec.name)
        rows.append(
            {
                "name": spec.name,
                "planned_match_sets": sum(target_matrix[player_idx]),
                "paired_match_sets": paired_n,
                "paired_set_win": wins,
                "paired_set_draw": draws,
                "paired_set_loss": losses,
                "paired_set_score_descriptive": _score(
                    wins,
                    draws,
                    losses,
                ),
                "avg_paired_disc_diff_descriptive": average_disc_diff,
                "actual_games": actual_n,
                "actual_game_win": actual_wins,
                "actual_game_draw": actual_draws,
                "actual_game_loss": actual_losses,
                "actual_game_score_descriptive": _score(
                    actual_wins,
                    actual_draws,
                    actual_losses,
                ),
                "paired_set_elo_descriptive": elo,
            }
        )
    return rows


def build_pair_rows(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
    target_match_sets_by_pair: Sequence[Sequence[int]],
) -> List[dict]:
    target_matrix = _validated_target_matrix(
        specs,
        target_match_sets_by_pair,
    )
    rows = []
    for player_idx, spec in enumerate(specs):
        for opponent_idx, opponent in enumerate(specs):
            if player_idx == opponent_idx:
                continue
            wins, draws, losses = stats.results[player_idx][opponent_idx]
            paired_n = wins + draws + losses
            paired_ci = (
                score_confidence_interval(wins, draws, losses)
                if paired_n
                else None
            )
            actual_wins, actual_draws, actual_losses = (
                stats.actual_results[player_idx][opponent_idx]
            )
            actual_n = actual_wins + actual_draws + actual_losses
            rows.append(
                {
                    "player": spec.name,
                    "opponent": opponent.name,
                    "planned_match_sets": target_matrix[player_idx][
                        opponent_idx
                    ],
                    "paired_match_sets": paired_n,
                    "paired_set_win": wins,
                    "paired_set_draw": draws,
                    "paired_set_loss": losses,
                    "paired_set_score": _score(wins, draws, losses),
                    "paired_set_score_ci95_low": (
                        paired_ci.low if paired_ci else None
                    ),
                    "paired_set_score_ci95_high": (
                        paired_ci.high if paired_ci else None
                    ),
                    "paired_set_score_ci95_half_width": (
                        paired_ci.half_width if paired_ci else None
                    ),
                    "avg_paired_disc_diff_descriptive": (
                        stats.disc_diff[player_idx][opponent_idx] / paired_n
                        if paired_n
                        else None
                    ),
                    "actual_games": actual_n,
                    "actual_game_win": actual_wins,
                    "actual_game_draw": actual_draws,
                    "actual_game_loss": actual_losses,
                    "actual_game_score_descriptive": _score(
                        actual_wins,
                        actual_draws,
                        actual_losses,
                    ),
                }
            )
    return rows


def build_text_report(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
    target_match_sets_by_pair: Sequence[Sequence[int]],
    ratings: Dict[str, float],
) -> str:
    target_matrix = _validated_target_matrix(
        specs,
        target_match_sets_by_pair,
    )
    lines = [
        "Participant-wide descriptive summary",
        (
            "A set contains two color-swapped games, and its result is the "
            "sign of their mean disc difference."
        ),
        (
            "All player pairs use the same set-index/XOT schedule, so the "
            "all-opponent score and Elo below are descriptive point estimates "
            "without confidence intervals. Pair-specific score confidence "
            "intervals are available in strength_pair_results.csv."
        ),
        "",
        "\t".join(
            [
                "player",
                "sets",
                "W",
                "D",
                "L",
                "score(desc)",
                "avg_disc(desc)",
                "Elo(desc)",
            ]
        ),
    ]
    for player_idx, spec in enumerate(specs):
        wins, draws, losses = stats.player_record(player_idx)
        n = wins + draws + losses
        score = _score(wins, draws, losses)
        average_disc = (
            sum(stats.disc_diff[player_idx]) / n if n else None
        )
        elo = ratings.get(spec.name)
        lines.append(
            "\t".join(
                [
                    spec.name,
                    str(n),
                    str(wins),
                    str(draws),
                    str(losses),
                    "-" if score is None else f"{score:.4f}",
                    (
                        "-"
                        if average_disc is None
                        else f"{average_disc:+.3f}"
                    ),
                    "-" if elo is None else f"{elo:.1f}",
                ]
            )
        )

    names = [spec.name for spec in specs]
    lines.extend(
        ["", "Paired-set score matrix (descriptive point estimates)"]
    )
    lines.append("\t".join(["vs >"] + names))
    for player_idx, spec in enumerate(specs):
        cells = [spec.name]
        for opponent_idx in range(len(specs)):
            if player_idx == opponent_idx:
                cells.append("-")
                continue
            score = _score(*stats.results[player_idx][opponent_idx])
            cells.append("-" if score is None else f"{score:.4f}")
        lines.append("\t".join(cells))

    lines.extend(["", "Progress (paired sets played/target)"])
    lines.append("\t".join(["vs >"] + names + ["all"]))
    for player_idx, spec in enumerate(specs):
        cells = [spec.name]
        for opponent_idx in range(len(specs)):
            cells.append(
                "-"
                if player_idx == opponent_idx
                else (
                    f"{stats.n_played[player_idx][opponent_idx]}"
                    f"/{target_matrix[player_idx][opponent_idx]}"
                )
            )
        target_per_player = sum(target_matrix[player_idx])
        cells.append(
            f"{sum(stats.n_played[player_idx])}/{target_per_player}"
        )
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def write_outputs(
    specs: Sequence[PlayerSpec],
    stats: TournamentStats,
    output_dir: Path,
    completed_match_sets: int,
    total_match_sets: int,
    target_match_sets_by_pair: Sequence[Sequence[int]],
    experiment_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_matrix = _validated_target_matrix(
        specs,
        target_match_sets_by_pair,
    )
    planned_total = sum(
        target_matrix[player_idx][opponent_idx]
        for player_idx in range(len(specs))
        for opponent_idx in range(player_idx + 1, len(specs))
    )
    if planned_total != total_match_sets:
        raise ValueError(
            "total_match_sets does not match target_match_sets_by_pair: "
            f"{total_match_sets} != {planned_total}"
        )
    ratings = estimate_elos(specs, stats)
    summary_rows = build_summary_rows(
        specs,
        stats,
        ratings,
        target_matrix,
    )
    pair_rows = build_pair_rows(specs, stats, target_matrix)
    atomic_write_text(
        output_dir / "strength_summary.csv",
        _csv_text(list(summary_rows[0]) if summary_rows else ["name"], summary_rows),
    )
    atomic_write_text(
        output_dir / "strength_pair_results.csv",
        _csv_text(
            list(pair_rows[0]) if pair_rows else ["player"],
            pair_rows,
        ),
    )

    names = [spec.name for spec in specs]
    score_matrix: List[List[object]] = [
        ["player"]
        + names
        + [
            "all_opponents_score_descriptive",
            "paired_set_elo_descriptive",
        ]
    ]
    disc_matrix: List[List[object]] = [
        ["player"]
        + names
        + [
            "all_opponents_avg_disc_diff_descriptive",
            "paired_set_elo_descriptive",
        ]
    ]
    progress_matrix: List[List[object]] = [
        ["player"] + names + ["all_opponents"]
    ]
    for player_idx, spec in enumerate(specs):
        score_row: List[object] = [spec.name]
        disc_row: List[object] = [spec.name]
        progress_row: List[object] = [spec.name]
        for opponent_idx in range(len(specs)):
            if player_idx == opponent_idx:
                score_row.append("-")
                disc_row.append("-")
                progress_row.append("-")
                continue
            record = stats.results[player_idx][opponent_idx]
            n = stats.n_played[player_idx][opponent_idx]
            score = _score(*record)
            score_row.append("-" if score is None else score)
            disc_row.append(
                stats.disc_diff[player_idx][opponent_idx] / n
                if n
                else "-"
            )
            progress_row.append(
                f"{n}/{target_matrix[player_idx][opponent_idx]}"
            )
        player_record = stats.player_record(player_idx)
        total_n = sum(player_record)
        overall_score = _score(*player_record)
        score_row.append(
            "-" if overall_score is None else overall_score
        )
        disc_row.append(
            (
                sum(stats.disc_diff[player_idx]) / total_n
                if total_n
                else "-"
            )
        )
        elo = ratings.get(spec.name)
        score_row.append(elo)
        disc_row.append(elo)
        progress_row.append(
            f"{sum(stats.n_played[player_idx])}"
            f"/{sum(target_matrix[player_idx])}"
        )
        score_matrix.append(score_row)
        disc_matrix.append(disc_row)
        progress_matrix.append(progress_row)
    atomic_write_text(
        output_dir / "strength_paired_set_score_matrix.tsv",
        _tsv_text(score_matrix),
    )
    atomic_write_text(
        output_dir / "strength_paired_disc_diff_matrix.tsv",
        _tsv_text(disc_matrix),
    )
    atomic_write_text(
        output_dir / "strength_progress_matrix.tsv",
        _tsv_text(progress_matrix),
    )
    for legacy_name in (
        "strength_win_rate_matrix.tsv",
        "strength_disc_diff_matrix.tsv",
    ):
        (output_dir / legacy_name).unlink(missing_ok=True)
    report = build_text_report(specs, stats, target_matrix, ratings)
    atomic_write_text(output_dir / "strength_report.txt", report)

    atomic_write_json(
        output_dir / "strength_results.json",
        {
            "schema_version": 3,
            "experiment_id": experiment_id,
            "completed_match_sets": completed_match_sets,
            "total_match_sets": total_match_sets,
            "completed_actual_games": completed_match_sets * 2,
            "total_actual_games": total_match_sets * 2,
            "planned_match_sets_by_pair": target_matrix,
            "scoring": {
                "actual_games_per_match_set": 2,
                "paired_disc_diff": (
                    "(p0 black disc difference + p0 white disc difference) / 2"
                ),
                "paired_set_outcome": "sign of paired_disc_diff",
                "paired_set_score": (
                    "(wins + 0.5 * draws) / paired match sets"
                ),
                "pair_score_ci95_independent_observation": (
                    "one color-swapped XOT match set within a fixed player pair"
                ),
                "pair_score_ci95": (
                    "Envelope of a Student-t interval and a Wilson boundary "
                    "safeguard for scores in {0, 0.5, 1}, clipped to [0, 1]"
                ),
                "shared_opening_schedule": (
                    "Every player pair uses the same XOT opening at the same "
                    "set index, so observations are correlated across pairs"
                ),
                "participant_summary": (
                    "descriptive point estimates only; no confidence interval "
                    "is reported after aggregating across opponents"
                ),
                "paired_set_elo_descriptive": (
                    "finite descriptive point estimate with 0.5 pseudo-win "
                    "and 0.5 pseudo-loss added independently to every player "
                    "pair; no confidence interval"
                ),
                "actual_game_score": (
                    "descriptive only; no independent-game confidence interval "
                    "because both colors share one XOT opening"
                ),
            },
            "players": [
                {
                    "name": spec.name,
                    "command": _logical_player_command(spec),
                }
                for spec in specs
            ],
            "summary": summary_rows,
            "pair_results": pair_rows,
        },
    )


class PerformanceMonitor:
    def __init__(
        self,
        output_dir: Path,
        interval_sec: float,
        manager: ProcessManager,
        run_id: str,
    ):
        self.output_dir = output_dir
        self.interval_sec = float(interval_sec)
        self.manager = manager
        self.run_id = str(run_id)
        if not self.run_id or any(
            character in self.run_id for character in "\\/:*?\"<>|"
        ):
            raise ValueError("run_id must be a non-empty filename-safe value")
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []
        self.started_at = 0.0

    def start(self) -> None:
        try:
            import psutil  # noqa: F401
        except ImportError:
            return
        self.started_at = time.time()
        self.thread = threading.Thread(
            target=self._loop,
            name="strength-performance-monitor",
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _gpu_sample() -> Tuple[Optional[float], Optional[float]]:
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5.0,
            )
            if process.returncode != 0:
                return None, None
            line = process.stdout.strip().splitlines()[0]
            utilization, memory_mib = [
                float(part.strip()) for part in line.split(",")[:2]
            ]
            return utilization, memory_mib
        except (
            OSError,
            ValueError,
            IndexError,
            subprocess.TimeoutExpired,
        ):
            return None, None

    def _loop(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)
        sample_index = 0
        gpu_sample: Tuple[Optional[float], Optional[float]] = (None, None)
        while not self.stop_event.wait(self.interval_sec):
            if sample_index % max(1, round(5.0 / self.interval_sec)) == 0:
                gpu_sample = self._gpu_sample()
            memory = psutil.virtual_memory()
            try:
                children = psutil.Process(os.getpid()).children(recursive=True)
            except psutil.Error:
                children = []
            sample = {
                "run_id": self.run_id,
                "sampled_at_unix_sec": time.time(),
                "elapsed_sec": time.time() - self.started_at,
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "system_memory_used_mib": float(
                    memory.used / (1024.0 * 1024.0)
                ),
                "available_memory_mib": float(
                    memory.available / (1024.0 * 1024.0)
                ),
                "system_memory_percent": float(memory.percent),
                "gpu_percent": gpu_sample[0],
                "gpu_memory_used_mib": gpu_sample[1],
                "child_processes": len(children),
                "configured_minimum_available_memory_mib": (
                    self.manager.minimum_available_memory_mib
                ),
            }
            self.samples.append(sample)
            if (
                sample["available_memory_mib"]
                < self.manager.minimum_available_memory_mib
            ):
                self.manager.low_memory_event.set()
            atomic_write_json(
                self.output_dir / "performance_live.json",
                {
                    **sample,
                    "low_memory_limit_reached": (
                        self.manager.low_memory_event.is_set()
                    ),
                },
            )
            sample_index += 1

    @staticmethod
    def _aggregate(
        samples: Sequence[dict],
        key: str,
        operation,
    ) -> Optional[float]:
        values = [
            float(sample[key])
            for sample in samples
            if sample.get(key) is not None
        ]
        return operation(values) if values else None

    @staticmethod
    def _sample_fields() -> List[str]:
        return [
            "run_id",
            "sampled_at_unix_sec",
            "elapsed_sec",
            "cpu_percent",
            "system_memory_used_mib",
            "available_memory_mib",
            "system_memory_percent",
            "gpu_percent",
            "gpu_memory_used_mib",
            "child_processes",
            "configured_minimum_available_memory_mib",
        ]

    @classmethod
    def _read_samples(cls, path: Path) -> List[dict]:
        numeric_fields = set(cls._sample_fields()) - {"run_id"}
        rows: List[dict] = []
        with path.open("r", encoding="utf-8", newline="") as source:
            for raw in csv.DictReader(source):
                row: dict = {"run_id": raw.get("run_id", "")}
                for field in numeric_fields:
                    text = raw.get(field, "")
                    row[field] = (
                        None
                        if text in (None, "")
                        else float(text)
                    )
                rows.append(row)
        return rows

    @classmethod
    def _summary(
        cls,
        samples: Sequence[dict],
        *,
        latest_run_id: str,
        low_memory_limit_reached: bool,
    ) -> dict:
        average = lambda values: sum(values) / len(values)
        run_ids = sorted(
            {
                str(sample["run_id"])
                for sample in samples
                if sample.get("run_id")
            }
        )
        configured_limits = sorted(
            {
                float(sample["configured_minimum_available_memory_mib"])
                for sample in samples
                if sample.get(
                    "configured_minimum_available_memory_mib"
                )
                is not None
            }
        )
        return {
            "latest_run_id": latest_run_id,
            "run_ids": run_ids,
            "run_count": len(run_ids),
            "samples": len(samples),
            "average_cpu_percent": cls._aggregate(
                samples, "cpu_percent", average
            ),
            "maximum_cpu_percent": cls._aggregate(
                samples, "cpu_percent", max
            ),
            "average_system_memory_used_mib": cls._aggregate(
                samples, "system_memory_used_mib", average
            ),
            "maximum_system_memory_used_mib": cls._aggregate(
                samples, "system_memory_used_mib", max
            ),
            "minimum_available_memory_mib": cls._aggregate(
                samples, "available_memory_mib", min
            ),
            "configured_minimum_available_memory_mib_values": (
                configured_limits
            ),
            "low_memory_limit_reached_in_latest_run": (
                low_memory_limit_reached
            ),
            "average_gpu_percent": cls._aggregate(
                samples, "gpu_percent", average
            ),
            "maximum_gpu_percent": cls._aggregate(
                samples, "gpu_percent", max
            ),
            "maximum_gpu_memory_used_mib": cls._aggregate(
                samples, "gpu_memory_used_mib", max
            ),
            "maximum_child_processes": cls._aggregate(
                samples, "child_processes", max
            ),
        }

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=max(10.0, self.interval_sec * 2.0))
        fields = self._sample_fields()
        segment_path = (
            self.output_dir
            / f"performance_samples_{self.run_id}.csv"
        )
        atomic_write_text(
            segment_path,
            _csv_text(fields, self.samples),
        )
        atomic_write_json(
            self.output_dir
            / f"performance_summary_{self.run_id}.json",
            {
                "sample_interval_sec": self.interval_sec,
                **self._summary(
                    self.samples,
                    latest_run_id=self.run_id,
                    low_memory_limit_reached=(
                        self.manager.low_memory_event.is_set()
                    ),
                ),
            },
        )

        all_samples: List[dict] = []
        for path in sorted(
            self.output_dir.glob("performance_samples_*.csv")
        ):
            all_samples.extend(self._read_samples(path))
        atomic_write_text(
            self.output_dir / "performance_samples.csv",
            _csv_text(fields, all_samples),
        )
        atomic_write_json(
            self.output_dir / "performance_summary.json",
            {
                "sample_interval_sec_latest_run": self.interval_sec,
                **self._summary(
                    all_samples,
                    latest_run_id=self.run_id,
                    low_memory_limit_reached=(
                        self.manager.low_memory_event.is_set()
                    ),
                ),
            },
        )
