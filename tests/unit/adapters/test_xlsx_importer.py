from io import BytesIO

from openpyxl import Workbook

from src.cfcgs_tracker.adapters.xlsx_importer import (
    iter_sheet_row_batches,
    normalize_header,
)


def _build_workbook_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "records"
    sheet.append(["Year", "Source URL", "Mitigation (%)", "Fund/Name"])
    sheet.append([2024, "https://example.org", 10, "Fund A"])
    sheet.append([None, None, None, None])
    sheet.append([2025, "https://example.com", 20, "Fund B"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_normalize_header_removes_accents_symbols_and_spaces():
    assert normalize_header("Mitigação (%) / Valor") == "mitigacao_valor"
    assert normalize_header("Fund/Name") == "fund_name"


def test_iter_sheet_row_batches_normalizes_headers_and_skips_empty_rows():
    file_bytes = _build_workbook_bytes()

    batches = list(
        iter_sheet_row_batches(
            file_bytes,
            sheet_index=0,
            batch_size=1,
        )
    )

    assert len(batches) == 2
    assert batches[0][0] == {
        "year": 2024,
        "source_url": "https://example.org",
        "mitigation": 10,
        "fund_name": "Fund A",
    }
    assert batches[1][0]["year"] == 2025
