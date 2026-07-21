"""Generate resumable, contest-time root teachers for uncovered GGS starts.

Input is the immutable JSON report produced by ``collect_ggs_roots.py``.
Every teacher search disables both ordinary and contest books, uses one engine
process per root, and checkpoints after each completed root.  The output data
rows are accepted directly by ``build_root_table.py --root-results``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from collect_ggs_roots import REPORT_SCHEMA, sha256_file
from othello import Board, coord_to_index


TEACHER_SCHEMA = "ggs_root_teacher_state_v3"
TEACHER_FORMAT = "# ggs_root_teacher_v1"
VERIFICATION_CANDIDATE_COUNT = 8
RESULT_RE = re.compile(
    r"^\|\s*(?P<level>[^|]+)\|\s*(?P<depth>[^|]+)\|\s*"
    r"(?P<move>[a-h][1-8])\|\s*(?P<score>[+-]?\d+)\|\s*"
    r"(?P<time>[^|]+)\|\s*(?P<nodes>\d+)\|\s*(?P<nps>\d+)\|\s*$",
    re.MULTILINE,
)
DEPTH_RE = re.compile(r"(?P<depth>\d+)@(?P<selectivity>\d+)%")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".state.json")


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def load_uncovered_roots(coverage_path: Path) -> list[str]:
    try:
        report = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read coverage report {coverage_path}: {error}") from error
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{coverage_path}: unexpected coverage report schema")
    if report.get("root_discs") != 14:
        raise ValueError(f"{coverage_path}: expected 14-disc roots")
    roots = report.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{coverage_path}: roots is missing or empty")
    result: set[str] = set()
    for index, entry in enumerate(roots):
        if not isinstance(entry, dict) or not isinstance(entry.get("canonical_board"), str):
            raise ValueError(f"{coverage_path}: invalid root entry {index}")
        if entry.get("deep_book") or entry.get("root_table"):
            continue
        board = entry["canonical_board"]
        if Board.from_text(board).n_discs() != 14:
            raise ValueError(f"{coverage_path}: root entry {index} is not a 14-disc board")
        result.add(board)
    if not result:
        raise ValueError(f"{coverage_path}: no uncovered 14-disc roots")
    return sorted(result)


def parse_search_results(output: str, board: str) -> list[dict[str, int | str]]:
    rows = [match.groupdict() for match in RESULT_RE.finditer(output)]
    if not rows:
        raise ValueError("expected at least one search-result row, found none")
    legal_moves = Board.from_text(board).legal_moves()
    result: list[dict[str, int | str]] = []
    seen_moves: set[str] = set()
    for row in rows:
        move = str(row["move"])
        if move in seen_moves:
            raise ValueError(f"engine returned duplicate teacher move {move}")
        seen_moves.add(move)
        if coord_to_index(move) not in legal_moves:
            raise ValueError(f"engine returned illegal teacher move {move}")
        result.append({
            "move": move,
            "score": int(str(row["score"])),
            "level": str(row["level"]).strip(),
            "depth": str(row["depth"]).strip(),
            "time": str(row["time"]).strip(),
            "nodes": int(str(row["nodes"])),
            "nps": int(str(row["nps"])),
        })
    return result


def parse_search_result(output: str, board: str) -> dict[str, int | str]:
    results = parse_search_results(output, board)
    if len(results) != 1:
        raise ValueError(f"expected exactly one search-result row, found {len(results)}")
    return results[0]


def search_root(
    exe: Path,
    board: str,
    time_seconds: float,
    threads: int,
    hash_level: int,
) -> dict[str, int | str]:
    if time_seconds <= 0 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid time, thread, or hash setting")
    command = [
        str(exe),
        "-time", f"{time_seconds:g}",
        "-t", str(threads),
        "-hash", str(hash_level),
        "-nobook",
    ]
    commands = f"setboard {board}\ngo\nquit\n"
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=commands,
            text=True,
            capture_output=True,
            timeout=max(60.0, time_seconds * 2.0 + 30.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"teacher search failed for {board}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher engine exited {completed.returncode} for {board}: {completed.stderr[-400:]}"
        )
    combined_output = completed.stdout + "\n" + completed.stderr
    try:
        return parse_search_result(combined_output, board)
    except ValueError as error:
        raise RuntimeError(
            f"teacher result parsing failed for {board}: {error}; output={combined_output[-400:]}"
        ) from error


def search_root_at_level(
    exe: Path,
    board: str,
    level: int,
    threads: int,
    hash_level: int,
) -> dict[str, int | str]:
    if level < 1 or threads <= 0 or not 0 <= hash_level <= 29:
        raise ValueError("invalid level, thread, or hash setting")
    command = [
        str(exe),
        "-l", str(level),
        "-t", str(threads),
        "-hash", str(hash_level),
        "-nobook",
    ]
    commands = f"setboard {board}\nhint 1\nquit\n"
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=commands,
            text=True,
            capture_output=True,
            timeout=180.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"teacher fallback failed for {board}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher fallback exited {completed.returncode} for {board}: {completed.stderr[-400:]}"
        )
    combined_output = completed.stdout + "\n" + completed.stderr
    try:
        return parse_search_result(combined_output, board)
    except ValueError as error:
        raise RuntimeError(
            f"teacher fallback parsing failed for {board}: {error}; output={combined_output[-400:]}"
        ) from error


def search_root_candidates_at_level(
    exe: Path,
    board: str,
    level: int,
    threads: int,
    hash_level: int,
    n_candidates: int = 8,
) -> list[dict[str, int | str]]:
    if level < 1 or threads <= 0 or not 0 <= hash_level <= 29 or n_candidates < 1:
        raise ValueError("invalid level, thread, hash setting, or candidate count")
    command = [
        str(exe),
        "-l", str(level),
        "-t", str(threads),
        "-hash", str(hash_level),
        "-nobook",
    ]
    commands = f"setboard {board}\\nhint {n_candidates}\\nquit\\n"
    try:
        completed = subprocess.run(
            command,
            cwd=exe.parent,
            input=commands,
            text=True,
            capture_output=True,
            timeout=180.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"teacher verification failed for {board}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher verification exited {completed.returncode} for {board}: "
            f"{completed.stderr[-400:]}"
        )
    combined_output = completed.stdout + "\\n" + completed.stderr
    try:
        return parse_search_results(combined_output, board)
    except ValueError as error:
        raise RuntimeError(
            f"teacher verification parsing failed for {board}: {error}; "
            f"output={combined_output[-400:]}"
        ) from error


def validate_quality(result: dict[str, int | str], min_depth: int, min_selectivity: int) -> None:
    if min_depth < 1 or not 1 <= min_selectivity <= 100:
        raise ValueError("invalid minimum teacher quality")
    depth = DEPTH_RE.fullmatch(str(result["depth"]))
    if depth is None:
        raise ValueError(f"invalid teacher depth {result['depth']!r}")
    actual_depth = int(depth.group("depth"))
    actual_selectivity = int(depth.group("selectivity"))
    if actual_depth < min_depth or actual_selectivity < min_selectivity:
        raise ValueError(
            f"teacher search quality {actual_depth}@{actual_selectivity}% is below "
            f"{min_depth}@{min_selectivity}%"
        )


def _new_state(
    coverage_path: Path,
    exe: Path,
    roots: list[str],
    time_seconds: float,
    threads: int,
    hash_level: int,
    min_depth: int,
    min_selectivity: int,
    fallback_level: int,
    method: str,
    teacher_level: int,
    verify_level: int,
) -> dict[str, Any]:
    return {
        "schema": TEACHER_SCHEMA,
        "coverage": {
            "path": coverage_path.resolve().as_posix(),
            "sha256": sha256_file(coverage_path),
        },
        "engine": {
            "path": exe.resolve().as_posix(),
            "sha256": sha256_file(exe),
        },
        "time_seconds": time_seconds,
        "threads": threads,
        "hash_level": hash_level,
        "min_depth": min_depth,
        "min_selectivity": min_selectivity,
        "fallback_level": fallback_level,
        "method": method,
        "teacher_level": teacher_level,
        "verify_level": verify_level,
        "verification_candidate_count": VERIFICATION_CANDIDATE_COUNT,
        "roots": roots,
        "results": {},
    }


def _load_state(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume teacher state {path}: {error}") from error
    for key in (
        "schema", "coverage", "engine", "time_seconds", "threads", "hash_level",
        "min_depth", "min_selectivity", "fallback_level", "method", "teacher_level",
        "verify_level", "verification_candidate_count", "roots",
    ):
        if state.get(key) != expected.get(key):
            raise ValueError(f"{path}: resume mismatch for {key}")
    if not isinstance(state.get("results"), dict):
        raise ValueError(f"{path}: results is invalid")
    if not set(state["results"]).issubset(set(expected["roots"])):
        raise ValueError(f"{path}: results include an unexpected root")
    return state


def _write_outputs(output: Path, state: dict[str, Any]) -> None:
    roots = state["roots"]
    results = state["results"]
    rows = []
    for board in roots:
        result = results.get(board)
        if result is not None:
            rows.append(f"{board} {result['score']} {result['move']}:{result['score']}")
    teacher_text = "\n".join(
        [
            TEACHER_FORMAT,
            f"# coverage_sha256 {state['coverage']['sha256']}",
            f"# engine_sha256 {state['engine']['sha256']}",
            f"# time_seconds {state['time_seconds']:g}",
            f"# threads {state['threads']}",
            f"# hash_level {state['hash_level']}",
            f"# min_depth {state['min_depth']}",
            f"# min_selectivity {state['min_selectivity']}",
            f"# fallback_level {state['fallback_level']}",
            f"# method {state['method']}",
            f"# teacher_level {state['teacher_level']}",
            f"# verify_level {state['verify_level']}",
            f"# verification_candidate_count {state['verification_candidate_count']}",
            f"# completed {len(rows)}/{len(roots)}",
            *rows,
            "",
        ]
    )
    _atomic_write_text(output, teacher_text)
    _atomic_write_text(
        _state_path(output),
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "schema": "ggs_root_teacher_manifest_v3",
        "output": {
            "path": output.resolve().as_posix(),
            "sha256": sha256_file(output),
            "completed": len(rows),
            "requested": len(roots),
        },
        "coverage": state["coverage"],
        "engine": state["engine"],
        "time_seconds": state["time_seconds"],
        "threads": state["threads"],
        "hash_level": state["hash_level"],
        "min_depth": state["min_depth"],
        "min_selectivity": state["min_selectivity"],
        "fallback_level": state["fallback_level"],
        "method": state["method"],
        "teacher_level": state["teacher_level"],
        "verify_level": state["verify_level"],
        "verification_candidate_count": state["verification_candidate_count"],
        "results": {board: results[board] for board in sorted(results)},
    }
    _atomic_write_text(
        _manifest_path(output),
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def generate_teachers(
    coverage_path: Path,
    exe: Path,
    output: Path,
    time_seconds: float,
    threads: int,
    hash_level: int,
    min_depth: int = 33,
    min_selectivity: int = 74,
    fallback_level: int = 33,
    method: str = "hint",
    teacher_level: int = 33,
    verify_level: int = 0,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    if not exe.is_file():
        raise FileNotFoundError(f"engine executable not found: {exe}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if fallback_level < 0:
        raise ValueError("fallback_level must not be negative")
    if method not in {"hint", "time_then_hint", "time_then_verify"}:
        raise ValueError("method must be hint, time_then_hint, or time_then_verify")
    if teacher_level < 1:
        raise ValueError("teacher_level must be positive")
    if verify_level < 0:
        raise ValueError("verify_level must not be negative")
    if method == "time_then_verify" and verify_level < 1:
        raise ValueError("time_then_verify requires a positive verify_level")
    if method == "time_then_verify" and fallback_level < min_depth:
        raise ValueError(
            "time_then_verify requires fallback_level at least min_depth "
            "so a shallow time search cannot lower teacher quality"
        )
    roots = load_uncovered_roots(coverage_path)
    if limit is not None:
        roots = roots[:limit]
    expected = _new_state(
        coverage_path, exe, roots, time_seconds, threads, hash_level, min_depth, min_selectivity,
        fallback_level, method, teacher_level, verify_level,
    )
    state_path = _state_path(output)
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume state not found: {state_path}")
        state = _load_state(state_path, expected)
    else:
        if state_path.exists() or output.exists():
            raise FileExistsError(f"output exists; use --resume or choose a new output: {output}")
        state = expected
        _write_outputs(output, state)

    for board in roots:
        if board in state["results"]:
            continue
        if method == "hint":
            result = search_root_at_level(exe, board, teacher_level, threads, hash_level)
            validate_quality(result, min_depth, min_selectivity)
            result["method"] = f"hint_level_{teacher_level}"
        elif method == "time_then_hint":
            result = search_root(exe, board, time_seconds, threads, hash_level)
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = "time"
            except ValueError:
                if fallback_level == 0:
                    raise
                result = search_root_at_level(exe, board, fallback_level, threads, hash_level)
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = f"hint_level_{fallback_level}"
        else:
            result = search_root(exe, board, time_seconds, threads, hash_level)
            primary = result
            try:
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = f"time_verified_hint_level_{verify_level}"
            except ValueError:
                result = search_root_at_level(exe, board, fallback_level, threads, hash_level)
                validate_quality(result, min_depth, min_selectivity)
                result["method"] = (
                    f"time_fallback_hint_level_{fallback_level}_verified_hint_level_{verify_level}"
                )
                result["primary"] = primary
            verification = search_root_at_level(exe, board, verify_level, threads, hash_level)
            verification_candidates = [verification]
            verification_top_moves = [str(verification["move"])]
            verification_mode = "top1_exact"
            if str(result["move"]) != str(verification["move"]):
                verification_candidates = search_root_candidates_at_level(
                    exe, board, verify_level, threads, hash_level,
                    VERIFICATION_CANDIDATE_COUNT,
                )
                verification = verification_candidates[0]
                best_verification_score = max(
                    int(row["score"]) for row in verification_candidates
                )
                verification_top_moves = sorted(
                    str(row["move"])
                    for row in verification_candidates
                    if int(row["score"]) == best_verification_score
                )
                verification_mode = "top8_tie_check"
                if str(result["move"]) not in verification_top_moves:
                    raise ValueError(
                        f"teacher move {result['move']} is not tied for best at level-{verify_level}: "
                        f"top moves {','.join(verification_top_moves)} score "
                        f"{best_verification_score}"
                    )
            result["verification"] = verification
            result["verification_candidates"] = verification_candidates
            result["verification_top_moves"] = verification_top_moves
            result["verification_mode"] = verification_mode
        state["results"][board] = result
        _write_outputs(output, state)
        print(f"completed {len(state['results'])}/{len(roots)} {board}", flush=True)
    return {"completed": len(state["results"]), "requested": len(roots)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--hash", dest="hash_level", type=int, default=29)
    parser.add_argument("--min-depth", type=int, default=33)
    parser.add_argument("--min-selectivity", type=int, default=74)
    parser.add_argument("--fallback-level", type=int, default=33)
    parser.add_argument("--method", choices=("hint", "time_then_hint", "time_then_verify"), default="hint")
    parser.add_argument("--teacher-level", type=int, default=33)
    parser.add_argument("--verify-level", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = generate_teachers(
        args.coverage,
        args.exe.resolve(),
        args.output,
        args.time_seconds,
        args.threads,
        args.hash_level,
        args.min_depth,
        args.min_selectivity,
        args.fallback_level,
        args.method,
        args.teacher_level,
        args.verify_level,
        args.resume,
        args.limit,
    )
    print(f"teacher roots complete {result['completed']}/{result['requested']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
