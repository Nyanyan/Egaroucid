"""Publish a contest root table only after its fixed local match has passed audit.

This is deliberately separate from :mod:`build_root_table`.  The generic
builder can create a temporary table for experiments.  This program is the
only command-line entry point that writes ``trained/contest_root_table.egcb``:
it reruns the audit recorded in a JSON file, rebuilds the table from the
frozen teacher rows, and records every input fingerprint beside the published
file.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import audit_root_table_matches
import build_root_table
from book_artifact import file_lock
from config import TRAINED_DIR


PUBLICATION_SCHEMA = "verified_contest_root_table_publication_v1"
PUBLICATION_SUFFIX = ".publication.json"
REQUIRED_BOOTSTRAP_SEED = audit_root_table_matches.REQUIRED_MATCH_SEED
REQUIRED_BOOTSTRAP_REPETITIONS = 100_000


def publication_path_for_root_table(path: Path) -> Path:
    return path.with_suffix(path.suffix + PUBLICATION_SUFFIX)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return payload


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _path_and_sha256(record: object, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"audit has no valid {label} fingerprint")
    path_text = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected, str):
        raise ValueError(f"audit has no valid {label} path or SHA-256")
    path = Path(path_text)
    if not path.is_file():
        raise ValueError(f"audit {label} file is missing: {path}")
    actual = build_root_table.sha256_file(path)
    if actual != expected:
        raise ValueError(f"audit {label} SHA-256 does not match {path}")
    return path.resolve()


def _canonical_json(value: object) -> object:
    """Normalize tuples produced by the audit into the JSON file representation."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _require_passed_audit(audit: dict[str, Any]) -> None:
    if audit.get("schema") != audit_root_table_matches.AUDIT_SCHEMA:
        raise ValueError("audit has an unsupported schema")
    if audit.get("bootstrap_seed") != REQUIRED_BOOTSTRAP_SEED:
        raise ValueError(
            f"audit bootstrap_seed must be {REQUIRED_BOOTSTRAP_SEED}; "
            "a different value is not a pre-specified publication condition"
        )
    if audit.get("bootstrap_repetitions") != REQUIRED_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            f"audit bootstrap_repetitions must be {REQUIRED_BOOTSTRAP_REPETITIONS}"
        )
    if audit.get("valid") is not True or audit.get("eligible_for_adoption") is not True:
        raise ValueError("audit does not permit publication")
    if audit.get("failures") != []:
        raise ValueError("audit contains failed checks")
    if audit.get("level_31_verification_required") is not True:
        raise ValueError("audit does not require level-31 verification")
    score_interval = audit.get("score_interval")
    margin_interval = audit.get("margin_interval")
    if (
        not isinstance(score_interval, (list, tuple))
        or len(score_interval) != 2
        or not isinstance(margin_interval, (list, tuple))
        or len(margin_interval) != 2
        or not isinstance(score_interval[0], (int, float))
        or not isinstance(margin_interval[0], (int, float))
        or score_interval[0] <= 0.5
        or margin_interval[0] <= 0.0
    ):
        raise ValueError("audit confidence-interval lower bounds do not permit publication")


def _rerun_and_compare_audit(audit: dict[str, Any], staging_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    results = _path_and_sha256(audit.get("results"), "results")
    prepared = _path_and_sha256(audit.get("prepared_input"), "prepared input")
    metadata = _path_and_sha256(audit.get("metadata"), "metadata")
    rerun = audit_root_table_matches.audit_match_results(
        results,
        prepared,
        metadata,
        staging_dir / "audit.md",
        REQUIRED_BOOTSTRAP_SEED,
        REQUIRED_BOOTSTRAP_REPETITIONS,
        int(audit.get("minimum_processed", 0)),
        int(audit.get("minimum_accepted", 0)),
    )
    if _canonical_json(rerun) != audit:
        raise ValueError("saved audit JSON differs from a fresh audit of its recorded inputs")
    _require_passed_audit(rerun)
    # Check again after the rerun.  The files are outside the publication lock,
    # so the first check alone would allow a changed prepared input to be used
    # by the later table reconstruction.
    _path_and_sha256(audit.get("results"), "results")
    _path_and_sha256(audit.get("prepared_input"), "prepared input")
    _path_and_sha256(audit.get("metadata"), "metadata")
    return results, prepared, metadata, _read_json(prepared, "prepared input")


def _build_staging_table(prepared: dict[str, Any], staging_dir: Path) -> tuple[Path, Path]:
    table_record = prepared.get("table")
    snapshot_record = prepared.get("snapshot")
    table = _path_and_sha256(table_record, "prepared table")
    snapshot = _path_and_sha256(snapshot_record, "teacher-row snapshot")
    expected_manifest = table.with_suffix(table.suffix + ".manifest.json")
    if not expected_manifest.is_file():
        raise ValueError(f"prepared table manifest is missing: {expected_manifest}")
    expected_manifest_sha = table_record.get("manifest_sha256") if isinstance(table_record, dict) else None
    if build_root_table.sha256_file(expected_manifest) != expected_manifest_sha:
        raise ValueError("prepared table manifest SHA-256 does not match")
    rebuilt = staging_dir / build_root_table.ROOT_TABLE_FILENAME
    build_root_table.build_root_table([], rebuilt, root_result_files=[snapshot])
    if build_root_table.sha256_file(rebuilt) != build_root_table.sha256_file(table):
        raise ValueError("table rebuilt from the frozen teacher rows differs from the audited table")
    expected_entries = table_record.get("entries") if isinstance(table_record, dict) else None
    validation = build_root_table.validate_root_table(rebuilt, expected_root_discs=14)
    if validation["entries"] != expected_entries:
        raise ValueError("rebuilt table entry count differs from the audited table")
    return rebuilt, snapshot


def _published_table_manifest(staging_manifest: Path, target: Path) -> dict[str, Any]:
    manifest = _read_json(staging_manifest, "staging table manifest")
    if manifest.get("schema") != "contest_root_table_manifest_v1":
        raise ValueError("staging table manifest has an unsupported schema")
    manifest["output"] = {
        "path": target.resolve().as_posix(),
        "bytes": target.stat().st_size,
        "sha256": build_root_table.sha256_file(target),
    }
    return manifest


def publish_verified_root_table(
    audit_json: Path,
    *,
    replace_existing: bool = False,
    output: Path = TRAINED_DIR / build_root_table.ROOT_TABLE_FILENAME,
) -> dict[str, Any]:
    """Rerun an audit and atomically publish its exact verified table.

    The output path is intentionally constrained to the tournament root-table
    location.  A later cohort must be prepared and audited as one complete
    table; this function never silently mixes unaudited existing rows.
    """
    target = output.resolve()
    required_target = (TRAINED_DIR / build_root_table.ROOT_TABLE_FILENAME).resolve()
    if target != required_target:
        raise ValueError(f"publication output is fixed at {required_target}")
    audit_path = audit_json.resolve()
    audit = _read_json(audit_path, "audit JSON")
    audit_sha256 = build_root_table.sha256_file(audit_path)
    _require_passed_audit(audit)
    targets = (
        target,
        build_root_table.manifest_path_for_root_table(target),
        publication_path_for_root_table(target),
    )
    if not replace_existing and any(path.exists() for path in targets):
        raise FileExistsError(
            "a tournament root table or its publication records already exist; "
            "use --replace-existing only after a full new table has passed audit"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".publication.lock")
    with file_lock(lock):
        if not replace_existing and any(path.exists() for path in targets):
            raise FileExistsError("a tournament root table was published while waiting for the lock")
        with tempfile.TemporaryDirectory(prefix=".verified-root-table-", dir=target.parent) as raw_staging:
            staging_dir = Path(raw_staging)
            results, prepared_path, metadata, prepared = _rerun_and_compare_audit(audit, staging_dir)
            rebuilt, snapshot = _build_staging_table(prepared, staging_dir)
            if build_root_table.sha256_file(audit_path) != audit_sha256:
                raise ValueError("audit JSON changed while publication was being checked")
            _path_and_sha256(audit.get("results"), "results")
            _path_and_sha256(audit.get("prepared_input"), "prepared input")
            _path_and_sha256(audit.get("metadata"), "metadata")
            rebuilt_manifest = build_root_table.manifest_path_for_root_table(rebuilt)
            # First replace the table.  The manifest and publication record are
            # each written atomically immediately after it and bind its SHA-256.
            _atomic_write_bytes(target, rebuilt.read_bytes())
            table_manifest = _published_table_manifest(rebuilt_manifest, target)
            manifest_path = build_root_table.manifest_path_for_root_table(target)
            _atomic_write_json(manifest_path, table_manifest)
            publication = {
                "schema": PUBLICATION_SCHEMA,
                "published_table": {
                    "path": target.as_posix(),
                    "sha256": build_root_table.sha256_file(target),
                    "entries": build_root_table.validate_root_table(target, expected_root_discs=14)["entries"],
                },
                "published_table_manifest": {
                    "path": manifest_path.as_posix(),
                    "sha256": build_root_table.sha256_file(manifest_path),
                },
                "audit": {"path": audit_path.as_posix(), "sha256": audit_sha256},
                "results": {"path": results.as_posix(), "sha256": build_root_table.sha256_file(results)},
                "prepared_input": {"path": prepared_path.as_posix(), "sha256": build_root_table.sha256_file(prepared_path)},
                "metadata": {"path": metadata.as_posix(), "sha256": build_root_table.sha256_file(metadata)},
                "teacher_row_snapshot": {"path": snapshot.as_posix(), "sha256": build_root_table.sha256_file(snapshot)},
                "match_statistics": {
                    key: audit[key]
                    for key in ("matches", "wins", "draws", "losses", "score_rate", "mean_margin", "score_interval", "margin_interval")
                },
                "publication_conditions": {
                    "bootstrap_seed": REQUIRED_BOOTSTRAP_SEED,
                    "bootstrap_repetitions": REQUIRED_BOOTSTRAP_REPETITIONS,
                    "level_31_verification_required": True,
                    "audit_rerun_matches_saved_audit": True,
                },
            }
            _atomic_write_json(publication_path_for_root_table(target), publication)
    return publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    result = publish_verified_root_table(args.audit_json, replace_existing=args.replace_existing)
    print(
        f"published entries={result['published_table']['entries']} "
        f"sha256={result['published_table']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
