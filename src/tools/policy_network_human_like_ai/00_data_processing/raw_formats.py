"""Raw-format adapters. Every adapter yields a transcript and source provenance."""

import csv
from itertools import chain
import re


def normalize_transcript(value: str) -> str:
    transcript = "".join(value.split()).lower()
    if not re.fullmatch(r"(?:[a-h][1-8]){1,60}", transcript):
        raise ValueError("expected 1..60 coordinate moves such as f5d6c3 (passes are implicit)")
    return transcript


def read_wthor_csv(path):
    # Older exports are a single headerless column; newer exports have metadata.
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        first = next(rows, None)
        if first is None:
            return
        headers = [cell.strip().lower() for cell in first]
        if "transcript" in headers:
            column = headers.index("transcript")
            numbered_rows = enumerate(rows, 2)
        elif len(first) == 1:
            column = 0
            numbered_rows = enumerate(chain([first], rows), 1)
        else:
            raise ValueError(f"unsupported CSV header in {path}; expected a transcript column")
        for row_number, row in numbered_rows:
            if not row or all(not cell.strip() for cell in row):
                continue
            if column >= len(row):
                raise ValueError(f"missing transcript in {path}:{row_number}")
            try:
                transcript = normalize_transcript(row[column])
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: {error}") from error
            yield transcript, row_number


def read_transcript_text(path):
    with path.open(encoding="utf-8-sig") as source:
        for row_number, line in enumerate(source, 1):
            if line.strip():
                yield normalize_transcript(line), row_number


# To add a source format, implement its reader and register it here. Stages 02-04
# depend only on the normalized text and numeric game-ID contract, not adapters.
RAW_FORMATS = {
    "wthor-csv": ("*.csv", read_wthor_csv),
    "transcript-text": ("*.txt", read_transcript_text),
}
