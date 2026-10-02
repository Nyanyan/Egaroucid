"""Edax 4.6 native-protocol hints for the shared WTHOR evaluation pipeline."""

from pathlib import Path
import queue
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "30_blend_with_egaroucid"))
from blend_policy_with_egaroucid import (
    EgaroucidHintRunner, coord_to_policy, policy_to_coord,
)


EDAX_LEVELS = (1, 5, 9, 13, 17, 21)
DEFAULT_EDAX_EXE = (
    Path(__file__).resolve().parents[4]
    / "bin" / "versions" / "edax_4_6" / "wEdax-x86-64.exe"
)
EDAX_HASH_BITS = 22
# Quiet native hint rows contain depth, score, elapsed time, node count,
# optional speed, and a principal variation. Only its first move is a hint.
HINT_ROW = re.compile(
    r"^\s*\d+(?:@\d+%)?\s+([<>?]?)([+-]\d+)\s+"
    r"\d+:\d+(?:\.\d+)?\s+\d+(?:\s+\d+)?\s+([a-h][1-8])\b",
    re.IGNORECASE,
)


def parse_edax_hint_output(raw: str):
    scores = {}
    for line in raw.splitlines():
        match = HINT_ROW.match(line)
        if match is None:
            continue
        policy = coord_to_policy(match[3])
        if policy in scores:
            raise ValueError(f"duplicate Edax hint move: {match[3]}")
        # Edax result_print emits a bound marker separately from result->score.
        # Keep that numeric search score (e.g. <-63 is -63, not an inferred -64).
        scores[policy] = float(match[2])
    if not scores:
        raise ValueError(f"Edax returned no hint moves: {raw[-500:]}")
    return scores


class EdaxHintRunner(EgaroucidHintRunner):
    """Reuse process lifecycle/deadlines, adapting Edax's prompt and hints."""

    def __init__(self, exe, *, eval_file=None, **kwargs):
        super().__init__(exe, **kwargs)
        self.eval_file = (
            Path(eval_file) if eval_file is not None
            else self.exe.resolve().parent / "data" / "eval.dat"
        )

    def command(self):
        return [
            str(self.exe.resolve()), "-q", "-l", str(self.level),
            "-n", str(self.threads), "-h", str(EDAX_HASH_BITS),
            "-book-usage", "off", "-ponder", "off",
            "-eval-file", str(self.eval_file.resolve()),
        ]

    def _ensure_process(self):
        fresh = self.proc is None or self.proc.poll() is not None
        proc = super()._ensure_process()
        if fresh:
            # Native Edax treats CR and LF separately. Also synchronize past
            # startup prompts before sending the first position.
            proc.stdin.reconfigure(newline="\n")
            proc.stdin.write("version\n")
            proc.stdin.flush()
            for _ in range(3):
                if "Edax version" in self._read_until_prompt():
                    break
            else:
                raise RuntimeError("Edax version handshake failed")
        return proc

    def _read_until_prompt(self):
        chars = []
        line_has_content = False
        deadline = time.monotonic() + self.timeout_sec
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.proc.kill()
                raise InterruptedError("Edax command cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.proc.kill()
                raise TimeoutError(f"Edax prompt timeout after {self.timeout_sec} sec")
            try:
                ch = self.stdout_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
            if isinstance(ch, BaseException):
                raise RuntimeError("failed to read Edax stdout") from ch
            if ch == "":
                raise RuntimeError(f"Edax process ended: {''.join(chars[-500:])}")
            chars.append(ch)
            # A lower-bound score such as >+63 is part of a hint row.
            # Native Edax's actual command prompt is at the start of a line.
            if ch == ">" and not line_has_content:
                return "".join(chars)
            if ch in "\r\n":
                line_has_content = False
            elif not ch.isspace():
                line_has_content = True

    def close(self):
        proc = self.proc
        try:
            super().close()
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=2.0)
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        pipe.close()

    def hint_scores(self, state, side, hint_count=3):
        raw = self.raw_hint(state, side, hint_count)
        scores = parse_edax_hint_output(raw)
        bounds = {coord_to_policy(match[3]): match[1]
                  for line in raw.splitlines() if (match := HINT_ROW.match(line))}
        # Normalize to the cache's shared move-order format. Insertion order
        # preserves Edax's top-N order, including score ties.
        normalized = "\n".join(
            f"0 | 0 | 0 | {policy_to_coord(policy)} | {score}"
            + (f" | Edax bound {bounds[policy]}" if bounds[policy] else "")
            for policy, score in scores.items()
        )
        return scores, normalized
