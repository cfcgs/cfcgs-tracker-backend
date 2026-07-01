from src.cfcgs_tracker.adapters.csv_importer import (
    iter_csv_row_batches,
    read_csv_rows,
)


def test_iter_csv_row_batches_normalizes_headers_and_skips_blank_rows():
    file_bytes = (
        "Year,Project Title,Source URL\n"
        "2024,Project A,https://example.org\n"
        ",,\n"
        "2025,Project B,https://example.com\n"
    ).encode("utf-8")

    batches = list(iter_csv_row_batches(file_bytes, batch_size=1))

    assert len(batches) == 2
    assert batches[0][0] == {
        "year": "2024",
        "project_title": "Project A",
        "source_url": "https://example.org",
    }
    assert batches[1][0]["year"] == "2025"


def test_read_csv_rows_decodes_utf8_sig_and_flattens_batches():
    file_bytes = (
        "\ufeffFund,Fund Type,Fund focus\n"
        "Fund A,Multilateral,Adaptation\n"
        "Fund B,Bilateral,Mitigation\n"
    ).encode("utf-8")

    rows = read_csv_rows(file_bytes)

    assert rows == [
        {
            "fund": "Fund A",
            "fund_type": "Multilateral",
            "fund_focus": "Adaptation",
        },
        {
            "fund": "Fund B",
            "fund_type": "Bilateral",
            "fund_focus": "Mitigation",
        },
    ]
