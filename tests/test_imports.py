from http import HTTPStatus

from tests.conftest import ClimateFinanceImportRowFactory, FundImportRowFactory


def test_create_import_xlsx(client, token, user, file_xlsx):
    user_id = user.id
    response = client.post(
        "/imports/",
        files={
            "file": file_xlsx(
                file_name="test_import.xlsx",
                fund_rows=FundImportRowFactory.build_batch(2),
                climate_rows=ClimateFinanceImportRowFactory.build_batch(3),
            )
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    payload = response.json()

    assert response.status_code == HTTPStatus.CREATED
    assert payload["file_type"] == "xlsx"
    assert payload["file_name"] == "test_import.xlsx"
    assert payload["status"] == "success"
    assert payload["rows_received"] == 5
    assert payload["rows_inserted"] == 5
    assert payload["rows_updated"] == 0
    assert payload["rows_duplicated"] == 0
    assert payload["rows_failed"] == 0
    assert payload["user_id"] == user_id
    assert payload["id"] == 1
    assert payload["started_at"] is not None
    assert payload["finished_at"] is not None


def test_create_import_climate_finance_csv(client, user, token, file_csv):
    user_id = user.id
    response = client.post(
        "/imports/",
        files={
            "file": file_csv(
                file_kind="climate_finance",
                climate_rows=ClimateFinanceImportRowFactory.build_batch(10),
            )
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    payload = response.json()

    assert payload["file_type"] == "csv"
    assert payload["file_name"] == "test_climate_finance.csv"
    assert payload["status"] == "success"
    assert payload["rows_received"] == 10
    assert payload["rows_inserted"] == 10
    assert payload["rows_updated"] == 0
    assert payload["rows_duplicated"] == 0
    assert payload["rows_failed"] == 0
    assert payload["user_id"] == user_id
    assert payload["id"] == 1
    assert payload["started_at"] is not None
    assert payload["finished_at"] is not None


def test_create_import_fund_csv(client, token, file_csv):
    response = client.post(
        "/imports/",
        files={
            "file": file_csv(
                file_kind="fund",
            )
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    payload = response.json()

    assert payload["file_type"] == "csv"
    assert payload["file_name"] == "test_fund_status.csv"
    assert payload["status"] == "success"
    assert payload["rows_received"] == 1
    assert payload["rows_inserted"] == 1
    assert payload["rows_updated"] == 0
    assert payload["rows_duplicated"] == 0
    assert payload["rows_failed"] == 0
    assert payload["user_id"] == 1
    assert payload["id"] == 1
    assert payload["started_at"] is not None
    assert payload["finished_at"] is not None


def test_read_import_job_as_admin(client, user, token, admin_token, file_csv):
    create_response = client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {token}"},
    )

    import_job_id = create_response.json()["id"]

    response = client.get(
        f"/imports/{import_job_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    payload = response.json()

    assert response.status_code == HTTPStatus.OK
    assert payload["id"] == import_job_id
    assert payload["file_type"] == "csv"
    assert payload["user_email"] == user.email


def test_read_import_job_as_importer_reads_own_job(
    client,
    token,
    file_csv,
):
    create_response = client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {token}"},
    )

    import_job_id = create_response.json()["id"]

    response = client.get(
        f"/imports/{import_job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == import_job_id


def test_read_import_job_as_importer_cannot_read_other_user_job(
    client,
    token,
    other_token,
    file_csv,
):
    create_response = client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {token}"},
    )

    import_job_id = create_response.json()["id"]

    response = client.get(
        f"/imports/{import_job_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "Not enough permissions"}


def test_read_import_jobs_as_admin(client, user, token, admin_token, file_csv):
    client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = client.get(
        "/imports/",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    payload = response.json()

    assert response.status_code == HTTPStatus.OK
    assert len(payload["import_jobs"]) == 1
    assert payload["import_jobs"][0]["id"] == 1
    assert payload["import_jobs"][0]["user_email"] == user.email


def test_read_import_jobs_as_importer_returns_only_own_jobs(
    client,
    token,
    other_token,
    file_csv,
):
    own_response = client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {token}"},
    )
    client.post(
        "/imports/",
        files={"file": file_csv(file_kind="fund")},
        headers={"Authorization": f"Bearer {other_token}"},
    )

    response = client.get(
        "/imports/",
        headers={"Authorization": f"Bearer {token}"},
    )

    payload = response.json()

    assert response.status_code == HTTPStatus.OK
    assert len(payload["import_jobs"]) == 1
    assert payload["import_jobs"][0]["id"] == own_response.json()["id"]


def test_read_import_jobs_requires_authenticated_import_operator(client):
    response = client.get(
        "/imports/",
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {"detail": "Not authenticated"}


def test_create_import_without_filename(client, token):
    boundary = "test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename=""\r\n'
        "Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n\r\n"
        "fake-content\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    response = client.post(
        "/imports/",
        content=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )

    payload = response.json()
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert payload["detail"] == "File name is required"


def test_create_import_with_invalid_csv_headers(client, token, file_csv):
    content = ("Foo,Bar,Baz\n1,2,3\n").encode("utf-8")

    response = client.post(
        "/imports/",
        files={
            "file": ("invalid.csv", content, "text/csv"),
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert (
        response.json()["detail"]
        == "Unsupported CSV headers. Could not identify climate finance or fund status layout."
    )


def test_create_import_invalid_extension_should_return_value_error(
    client, token, file_csv
):
    response = client.post(
        "/imports/",
        files={
            "file": (
                file_csv(file_name="invalid", file_kind="climate_finance")
            )
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert (
        response.json()["detail"]
        == "Unsupported file type. Use .csv or .xlsx."
    )
