#!/usr/bin/env python3
"""Data model, scheduling, persistence, and statistics for strength matches."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import median, NormalDist
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RESULT_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PlayerSpec:
    """Immutable player definition.

    ``setboard_before_genmove`` is true for the blended GTP wrapper. Such a
    player does not keep game state and therefore borrows an engine only while
    selecting a move.
    """

    name: str
    command: Tuple[str, ...]
    processes_per_player: int
    setboard_before_genmove: bool
    alpha: Optional[float] = None

    @property
    def concurrent_match_set_capacity(self) -> int:
        return self.processes_per_player // 2


@dataclass(frozen=True)
class MatchSetTask:
    """Two color-swapped games from one XOT opening."""

    task_id: int
    p0_idx: int
    p1_idx: int
    set_index: int
    opening: str

    def identity(self) -> dict:
        return {
            "task_id": self.task_id,
            "p0_idx": self.p0_idx,
            "p1_idx": self.p1_idx,
            "set_index": self.set_index,
            "opening": self.opening,
        }


@dataclass(frozen=True)
class GameResult:
    p0_idx: int
    p1_idx: int
    p0_is_black: bool
    black_idx: int
    white_idx: int
    p0_disc_diff: int
    black_stones: int
    white_stones: int
    transcript: str


@dataclass(frozen=True)
class MatchSetResult:
    task_id: int
    p0_idx: int
    p1_idx: int
    set_index: int
    opening: str
    p0_disc_diff: float
    p0_black_disc_diff: int
    p0_white_disc_diff: int
    color_games: Tuple[GameResult, GameResult]

    @property
    def p0_score(self) -> float:
        if self.p0_disc_diff > 0.0:
            return 1.0
        if self.p0_disc_diff < 0.0:
            return 0.0
        return 0.5

    def to_dict(self) -> dict:
        data = asdict(self)
        data["actual_games"] = 2
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatchSetResult":
        color_games_raw = data.get("color_games")
        if not isinstance(color_games_raw, list) or len(color_games_raw) != 2:
            raise ValueError("a match-set result must contain exactly two color games")
        color_games = tuple(GameResult(**game) for game in color_games_raw)
        return cls(
            task_id=int(data["task_id"]),
            p0_idx=int(data["p0_idx"]),
            p1_idx=int(data["p1_idx"]),
            set_index=int(data["set_index"]),
            opening=str(data["opening"]),
            p0_disc_diff=float(data["p0_disc_diff"]),
            p0_black_disc_diff=int(data["p0_black_disc_diff"]),
            p0_white_disc_diff=int(data["p0_white_disc_diff"]),
            color_games=color_games,  # type: ignore[arg-type]
        )


def combine_color_games(
    task: MatchSetTask,
    games: Sequence[GameResult],
) -> MatchSetResult:
    if len(games) != 2:
        raise ValueError("each XOT match set must contain exactly two games")
    by_color = {game.p0_is_black: game for game in games}
    if set(by_color) != {False, True}:
        raise ValueError("a match set must contain one game with p0 as each color")
    black_game = by_color[True]
    white_game = by_color[False]
    for game in (black_game, white_game):
        if (game.p0_idx, game.p1_idx) != (task.p0_idx, task.p1_idx):
            raise ValueError("color-game participants do not match the task")
    average_disc_diff = (
        float(black_game.p0_disc_diff) + float(white_game.p0_disc_diff)
    ) / 2.0
    return MatchSetResult(
        task_id=task.task_id,
        p0_idx=task.p0_idx,
        p1_idx=task.p1_idx,
        set_index=task.set_index,
        opening=task.opening,
        p0_disc_diff=average_disc_diff,
        p0_black_disc_diff=black_game.p0_disc_diff,
        p0_white_disc_diff=white_game.p0_disc_diff,
        color_games=(black_game, white_game),
    )


def make_match_set_tasks(
    player_count: int,
    openings: Sequence[str],
    match_sets_per_pair: int,
    seed: int,
) -> List[MatchSetTask]:
    """Build a deterministic, balanced schedule.

    Every pair receives the same shuffled XOT opening at a given set index.
    Pair order is shuffled once to avoid a fixed level-based scheduling bias.
    """

    if player_count < 2:
        raise ValueError("at least two players are required")
    if match_sets_per_pair < 1:
        raise ValueError("match_sets_per_pair must be positive")
    if not openings:
        raise ValueError("at least one XOT opening is required")

    pairs = [
        (p0_idx, p1_idx)
        for p0_idx in range(player_count)
        for p1_idx in range(p0_idx + 1, player_count)
    ]
    random.Random(seed).shuffle(pairs)
    tasks: List[MatchSetTask] = []
    for set_index in range(match_sets_per_pair):
        opening = openings[set_index % len(openings)]
        for p0_idx, p1_idx in pairs:
            tasks.append(
                MatchSetTask(
                    task_id=len(tasks),
                    p0_idx=p0_idx,
                    p1_idx=p1_idx,
                    set_index=set_index,
                    opening=opening,
                )
            )
    return tasks


def limit_tasks(
    tasks: Sequence[MatchSetTask],
    max_match_sets: Optional[int],
) -> List[MatchSetTask]:
    if max_match_sets is None:
        return list(tasks)
    if max_match_sets < 1:
        raise ValueError("--max-match-sets must be positive")
    return list(tasks[:max_match_sets])


def target_match_sets_by_pair(
    tasks: Sequence[MatchSetTask],
    player_count: int,
) -> List[List[int]]:
    """Count the actual scheduled observations for every player pair."""

    matrix = [
        [0 for _ in range(player_count)]
        for _ in range(player_count)
    ]
    for task in tasks:
        if not (
            0 <= task.p0_idx < player_count
            and 0 <= task.p1_idx < player_count
            and task.p0_idx != task.p1_idx
        ):
            raise ValueError("task participant index is out of range")
        matrix[task.p0_idx][task.p1_idx] += 1
        matrix[task.p1_idx][task.p0_idx] += 1
    return matrix


class PendingTaskQueue:
    """Small O(number-of-pairs) scheduler for a large round-robin.

    The old implementation repeatedly scanned all remaining tasks. At 60,000
    match sets that makes the scheduler progressively more expensive. This
    queue stores one deque per player pair, so each scheduling decision scans
    at most the 120 pairs in the default 16-player tournament.
    """

    MAX_SET_INDEX_LOOKAHEAD = 1

    def __init__(
        self,
        tasks: Iterable[MatchSetTask],
        player_count: Optional[int] = None,
    ):
        by_pair: Dict[Tuple[int, int], Deque[MatchSetTask]] = defaultdict(deque)
        task_list = list(tasks)
        if player_count is None:
            player_count = (
                1
                + max(
                    (
                        max(task.p0_idx, task.p1_idx)
                        for task in task_list
                    ),
                    default=-1,
                )
            )
        self._remaining_by_player = [0 for _ in range(player_count)]
        for task in task_list:
            by_pair[(task.p0_idx, task.p1_idx)].append(task)
            self._remaining_by_player[task.p0_idx] += 1
            self._remaining_by_player[task.p1_idx] += 1
        self._by_pair = dict(by_pair)
        self._pairs = list(self._by_pair)
        self._cursor = 0
        self._remaining = sum(len(items) for items in self._by_pair.values())

    def __bool__(self) -> bool:
        return self._remaining > 0

    def __len__(self) -> int:
        return self._remaining

    def push_front(self, task: MatchSetTask) -> None:
        pair = (task.p0_idx, task.p1_idx)
        if pair not in self._by_pair:
            self._by_pair[pair] = deque()
            self._pairs.append(pair)
        self._by_pair[pair].appendleft(task)
        self._remaining += 1
        self._remaining_by_player[task.p0_idx] += 1
        self._remaining_by_player[task.p1_idx] += 1

    def remaining_for_player(self, player_idx: int) -> int:
        return self._remaining_by_player[player_idx]

    def pop_schedulable(
        self,
        active_counts: Sequence[int],
        capacities: Sequence[int],
        duration_weights: Optional[Sequence[float]] = None,
    ) -> Optional[MatchSetTask]:
        if not self._remaining or not self._pairs:
            return None
        if duration_weights is None:
            duration_weights = [1.0 for _ in capacities]
        oldest_pending_set_index = min(
            pair_tasks[0].set_index
            for pair_tasks in self._by_pair.values()
            if pair_tasks
        )
        best: Optional[Tuple[int, float, int, int]] = None
        for offset in range(len(self._pairs)):
            pair_index = (self._cursor + offset) % len(self._pairs)
            p0_idx, p1_idx = self._pairs[pair_index]
            pair_tasks = self._by_pair[(p0_idx, p1_idx)]
            if not pair_tasks:
                continue
            if (
                pair_tasks[0].set_index
                > oldest_pending_set_index + self.MAX_SET_INDEX_LOOKAHEAD
            ):
                continue
            if (
                active_counts[p0_idx] >= capacities[p0_idx]
                or active_counts[p1_idx] >= capacities[p1_idx]
            ):
                continue
            p0_work = (
                self._remaining_by_player[p0_idx]
                * max(0.001, float(duration_weights[p0_idx]))
                / capacities[p0_idx]
            )
            p1_work = (
                self._remaining_by_player[p1_idx]
                * max(0.001, float(duration_weights[p1_idx]))
                / capacities[p1_idx]
            )
            # Longest remaining participant workload first. The smaller sum
            # term breaks ties in favor of a pair that advances two loaded
            # participants at once. Set index is the primary key so every
            # schedulable pair receives the current shared XOT opening before
            # a pair advances far into later openings. At most one opening of
            # lookahead is allowed when all older tasks are capacity-blocked;
            # this keeps workers useful without letting one fast pair consume
            # its whole schedule. ``-offset`` preserves rotating fairness
            # within the same opening wave.
            score = max(p0_work, p1_work) + 0.01 * (p0_work + p1_work)
            candidate = (
                -pair_tasks[0].set_index,
                score,
                -offset,
                pair_index,
            )
            if best is None or candidate > best:
                best = candidate
        if best is None:
            return None
        pair_index = best[3]
        p0_idx, p1_idx = self._pairs[pair_index]
        self._cursor = (pair_index + 1) % len(self._pairs)
        self._remaining -= 1
        self._remaining_by_player[p0_idx] -= 1
        self._remaining_by_player[p1_idx] -= 1
        return self._by_pair[(p0_idx, p1_idx)].popleft()


def normalized_duration_weights(
    duration_weights: Sequence[float],
    duration_observations: Sequence[int],
) -> List[float]:
    """Return stable relative duration estimates for scheduling.

    A player's first completed set includes process startup and model/cache
    warm-up. Comparing that value with the old synthetic default of one second
    can make one pair monopolize the queue. Unobserved players therefore use
    the median observed duration, and observed outliers are bounded to one
    half through twice that median. Until any observation exists all players
    receive the same unit weight.
    """

    if len(duration_weights) != len(duration_observations):
        raise ValueError(
            "duration weights and observation counts must have equal length"
        )
    observed = [
        float(weight)
        for weight, count in zip(
            duration_weights,
            duration_observations,
        )
        if count > 0 and math.isfinite(float(weight)) and float(weight) > 0.0
    ]
    if not observed:
        return [1.0 for _ in duration_weights]
    center = median(observed)
    lower = center * 0.5
    upper = center * 2.0
    normalized = []
    for weight, count in zip(duration_weights, duration_observations):
        value = float(weight)
        if count <= 0 or not math.isfinite(value) or value <= 0.0:
            value = center
        normalized.append(min(upper, max(lower, value)))
    return normalized


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    half_width: float


def student_t_critical_975(degrees_of_freedom: int) -> float:
    """Return an accurate dependency-free approximation to t(df, 0.975)."""

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    z = NormalDist().inv_cdf(0.975)
    v = float(degrees_of_freedom)
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    z9 = z7 * z2
    return (
        z
        + (z3 + z) / (4.0 * v)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * v * v)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
        / (384.0 * v**3)
        + (
            79.0 * z9
            + 776.0 * z7
            + 1482.0 * z5
            - 1920.0 * z3
            - 945.0 * z
        )
        / (92160.0 * v**4)
    )


def score_confidence_interval(
    wins: int,
    draws: int,
    losses: int,
) -> ConfidenceInterval:
    """95% t interval for paired-match-set scores in {0, 0.5, 1}."""

    n = wins + draws + losses
    if n < 2:
        return ConfidenceInterval(0.0, 1.0, 0.5)
    mean = (wins + 0.5 * draws) / n
    sum_squares = wins + 0.25 * draws
    variance = max(0.0, (sum_squares - n * mean * mean) / (n - 1))
    margin = student_t_critical_975(n - 1) * math.sqrt(variance / n)
    t_low = max(0.0, mean - margin)
    t_high = min(1.0, mean + margin)

    # A plain t interval collapses to a point after all wins, all losses, or
    # all draws. Use the envelope with a Wilson score interval as a boundary
    # safeguard. For the usual mixed outcomes the t interval is wider.
    z = NormalDist().inv_cdf(0.975)
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    wilson_center = (mean + z_squared / (2.0 * n)) / denominator
    wilson_margin = (
        z
        * math.sqrt(
            mean * (1.0 - mean) / n
            + z_squared / (4.0 * n * n)
        )
        / denominator
    )
    low = max(0.0, min(t_low, wilson_center - wilson_margin))
    high = min(1.0, max(t_high, wilson_center + wilson_margin))
    return ConfidenceInterval(low, high, max(mean - low, high - mean))


def conservative_score_half_width(
    match_sets: int,
    expected_score: float = 0.5,
) -> float:
    """Planning half-width assuming the worst case of no draws."""

    if match_sets < 2:
        return 0.5
    if not 0.0 <= expected_score <= 1.0:
        raise ValueError("expected_score must be between zero and one")
    variance = expected_score * (1.0 - expected_score)
    return student_t_critical_975(match_sets - 1) * math.sqrt(
        variance / match_sets
    )


class TournamentStats:
    """Main-thread-only aggregate of durably stored match-set results."""

    def __init__(self, player_count: int):
        self.player_count = player_count
        self.results = [
            [[0, 0, 0] for _ in range(player_count)]
            for _ in range(player_count)
        ]
        self.disc_diff = [
            [0.0 for _ in range(player_count)]
            for _ in range(player_count)
        ]
        self.n_played = [
            [0 for _ in range(player_count)]
            for _ in range(player_count)
        ]
        self.actual_results = [
            [[0, 0, 0] for _ in range(player_count)]
            for _ in range(player_count)
        ]
        self.actual_disc_diff = [
            [0.0 for _ in range(player_count)]
            for _ in range(player_count)
        ]
        self.actual_n_played = [
            [0 for _ in range(player_count)]
            for _ in range(player_count)
        ]

    def record(self, result: MatchSetResult) -> None:
        p0_idx = result.p0_idx
        p1_idx = result.p1_idx
        diff = result.p0_disc_diff
        if diff > 0.0:
            self.results[p0_idx][p1_idx][0] += 1
            self.results[p1_idx][p0_idx][2] += 1
        elif diff < 0.0:
            self.results[p0_idx][p1_idx][2] += 1
            self.results[p1_idx][p0_idx][0] += 1
        else:
            self.results[p0_idx][p1_idx][1] += 1
            self.results[p1_idx][p0_idx][1] += 1
        self.disc_diff[p0_idx][p1_idx] += diff
        self.disc_diff[p1_idx][p0_idx] -= diff
        self.n_played[p0_idx][p1_idx] += 1
        self.n_played[p1_idx][p0_idx] += 1
        for game in result.color_games:
            game_diff = game.p0_disc_diff
            if game_diff > 0:
                self.actual_results[p0_idx][p1_idx][0] += 1
                self.actual_results[p1_idx][p0_idx][2] += 1
            elif game_diff < 0:
                self.actual_results[p0_idx][p1_idx][2] += 1
                self.actual_results[p1_idx][p0_idx][0] += 1
            else:
                self.actual_results[p0_idx][p1_idx][1] += 1
                self.actual_results[p1_idx][p0_idx][1] += 1
            self.actual_disc_diff[p0_idx][p1_idx] += game_diff
            self.actual_disc_diff[p1_idx][p0_idx] -= game_diff
            self.actual_n_played[p0_idx][p1_idx] += 1
            self.actual_n_played[p1_idx][p0_idx] += 1

    def pair_record(self, player_idx: int, opponent_idx: int) -> Tuple[int, int, int]:
        wins, draws, losses = self.results[player_idx][opponent_idx]
        return wins, draws, losses

    def player_record(self, player_idx: int) -> Tuple[int, int, int]:
        records = self.results[player_idx]
        return (
            sum(record[0] for record in records),
            sum(record[1] for record in records),
            sum(record[2] for record in records),
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(configuration: Mapping[str, object]) -> dict:
    canonical = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "configuration": configuration,
    }


def ensure_manifest(
    path: Path,
    manifest: Mapping[str, object],
    resume: bool,
    results_exist: bool,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment_id") != manifest.get("experiment_id"):
            raise ValueError(
                f"{path} describes a different experiment; "
                "use the original settings or a new --output-dir"
            )
        return
    if resume and results_exist:
        raise ValueError(
            "cannot safely resume results without strength_manifest.json; "
            "use a new --output-dir"
        )
    atomic_write_json(path, manifest)


class ResultStore:
    """Append-only match results with strict resume validation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results_path = output_dir / "strength_games.jsonl"
        self.failures_path = output_dir / "strength_failed_tasks.jsonl"
        self._results_file = None
        self._failures_file = None

    def load(
        self,
        tasks_by_id: Mapping[int, MatchSetTask],
    ) -> Tuple[set[int], List[MatchSetResult]]:
        if not self.results_path.exists():
            return set(), []
        completed: set[int] = set()
        results: List[MatchSetResult] = []
        with self.results_path.open("rb+") as source:
            line_start = 0
            line_number = 0
            while True:
                raw = source.readline()
                if not raw:
                    break
                line_number += 1
                next_offset = source.tell()
                if not raw.strip():
                    line_start = next_offset
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if (
                        next_offset == self.results_path.stat().st_size
                        and not raw.endswith(b"\n")
                    ):
                        source.truncate(line_start)
                        break
                    raise ValueError(
                        f"invalid JSON at {self.results_path}:{line_number}"
                    )
                if int(row.get("schema_version", -1)) != RESULT_SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported result schema at line {line_number}; "
                        "start the rewritten tournament in a new output directory"
                    )
                task_data = row.get("task")
                if not isinstance(task_data, dict):
                    raise ValueError(f"missing task identity at line {line_number}")
                task_id = int(task_data.get("task_id", -1))
                expected = tasks_by_id.get(task_id)
                if expected is None or task_data != expected.identity():
                    raise ValueError(
                        f"result task identity mismatch at line {line_number}"
                    )
                if task_id in completed:
                    raise ValueError(f"duplicate completed task_id {task_id}")
                result_data = row.get("result")
                if not isinstance(result_data, dict):
                    raise ValueError(f"missing result at line {line_number}")
                result = MatchSetResult.from_dict(result_data)
                if (
                    result.task_id != task_id
                    or result.p0_idx != expected.p0_idx
                    or result.p1_idx != expected.p1_idx
                    or result.set_index != expected.set_index
                    or result.opening != expected.opening
                ):
                    raise ValueError(
                        f"result identity mismatch at line {line_number}"
                    )
                colors = {game.p0_is_black for game in result.color_games}
                if colors != {False, True} or any(
                    (game.p0_idx, game.p1_idx)
                    != (expected.p0_idx, expected.p1_idx)
                    for game in result.color_games
                ):
                    raise ValueError(
                        f"invalid color games at line {line_number}"
                    )
                black_game = next(
                    game for game in result.color_games if game.p0_is_black
                )
                white_game = next(
                    game for game in result.color_games if not game.p0_is_black
                )
                expected_average = (
                    black_game.p0_disc_diff + white_game.p0_disc_diff
                ) / 2.0
                if (
                    result.p0_black_disc_diff != black_game.p0_disc_diff
                    or result.p0_white_disc_diff != white_game.p0_disc_diff
                    or abs(result.p0_disc_diff - expected_average) > 1.0e-12
                ):
                    raise ValueError(
                        f"inconsistent match-set score at line {line_number}"
                    )
                completed.add(task_id)
                results.append(result)
                if (
                    next_offset == self.results_path.stat().st_size
                    and not raw.endswith(b"\n")
                ):
                    # A crash can occur after the JSON object reaches disk but
                    # before its delimiter. Without repairing it, the next
                    # append would create ``{...}{...}`` and poison resume.
                    source.seek(next_offset)
                    source.write(b"\n")
                    source.flush()
                    os.fsync(source.fileno())
                    next_offset = source.tell()
                line_start = next_offset
        return completed, results

    def open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = self.results_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def append_result(
        self,
        task: MatchSetTask,
        result: MatchSetResult,
    ) -> None:
        if self._results_file is None:
            raise RuntimeError("result store is not open")
        row = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task": task.identity(),
            "result": result.to_dict(),
        }
        self._results_file.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._results_file.flush()

    def append_failure(
        self,
        task: MatchSetTask,
        attempt: int,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        if self._failures_file is None:
            self._failures_file = self.failures_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
        row = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task": task.identity(),
            "attempt": attempt,
            "error": repr(error),
            "traceback": traceback_text,
        }
        self._failures_file.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._failures_file.flush()

    def checkpoint(self) -> None:
        if self._results_file is not None:
            self._results_file.flush()
            os.fsync(self._results_file.fileno())
        if self._failures_file is not None:
            self._failures_file.flush()
            os.fsync(self._failures_file.fileno())

    def close(self) -> None:
        for stream_name in ("_results_file", "_failures_file"):
            stream = getattr(self, stream_name)
            if stream is not None:
                stream.close()
                setattr(self, stream_name, None)
