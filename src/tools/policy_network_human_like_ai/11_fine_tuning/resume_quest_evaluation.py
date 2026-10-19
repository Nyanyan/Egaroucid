"""Resume cached Quest evaluation with a longer watchdog, preserving search settings."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

import evaluate_quest_baselines as evaluation
from experiment_common import DEFAULT_RESULTS, file_sha256, write


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--recovery-timeout-sec", type=float, default=14400)
    recovery, original_args = parser.parse_known_args()
    if recovery.recovery_timeout_sec < 1800:
        parser.error("Recovery timeout must be at least 1800 seconds")
    original_collect = evaluation.gpw.collect_comparison_hints

    def collect(groups, workers, options, *args, **kwargs):
        options = {engine: dict(settings) for engine, settings in options.items()}
        before = dict(options["egaroucid"])
        options["egaroucid"].update(timeout_sec=recovery.recovery_timeout_sec, max_retries=0)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(options["egaroucid"]["output_dir"]).parents[1]
        write(output / "runtime_recoveries" / f"{stamp}.json", dict(
            started_at=datetime.now().astimezone().isoformat(),
            reason="One level-21 position exceeded the 1800-second watchdog three times",
            wrapper_sha256=file_sha256(Path(__file__)),
            original_evaluator_sha256=file_sha256(Path(evaluation.__file__)),
            old_timeout_sec=before["timeout_sec"],
            new_timeout_sec=recovery.recovery_timeout_sec,
            old_max_retries=before["max_retries"], new_max_retries=0,
            search_settings_unchanged=True,
            note="Only wall-clock watchdog and retry count change; completed cache entries are reused"))
        print(f"Resume cached evaluation: Egaroucid watchdog {recovery.recovery_timeout_sec:g}s; "
              "same levels, one thread per engine, hint counts and test positions.", flush=True)
        return original_collect(groups, workers, options, *args, **kwargs)

    evaluation.gpw.collect_comparison_hints = collect
    sys.argv = [sys.argv[0]] + original_args
    evaluation.main()


if __name__ == "__main__":
    main()
