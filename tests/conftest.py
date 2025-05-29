import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from src.cfcgs_tracker.app import app

@pytest.fixture()
def client():
    return TestClient(app)

@pytest.fixture()
def session():
    engine = create_engine('sqlite:///:memory:')
    return engine