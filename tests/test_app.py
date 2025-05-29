from http import HTTPStatus

from starlette.testclient import TestClient

from src.cfcgs_tracker.app import app


def test_read_root_should_return_200():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "Hello World"}
