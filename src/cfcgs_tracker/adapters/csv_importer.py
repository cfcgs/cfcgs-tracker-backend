import csv
from io import StringIO
from typing import Any, Iterator

from src.cfcgs_tracker.adapters.xlsx_importer import normalize_header


def read_csv_rows(file_bytes: bytes) -> list[dict[str, Any]]:
    batches = list(iter_csv_row_batches(file_bytes, batch_size=10_000))
    return [row for batch in batches for row in batch]


def iter_csv_row_batches(
    file_bytes: bytes,
    *,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    content = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(content))
    batch: list[dict[str, Any]] = []

    for row in reader:
        if not row:
            continue

        normalized_row = {
            normalize_header(header): value
            for header, value in row.items()
            if header
        }

        if any(value not in (None, "") for value in normalized_row.values()):
            batch.append(normalized_row)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
