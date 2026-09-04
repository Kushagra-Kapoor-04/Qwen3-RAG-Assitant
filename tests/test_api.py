import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client(monkeypatch):
    mock_service = MagicMock()
    mock_service.is_ready.return_value = True
    mock_service.query.return_value = MagicMock(
        question="What is RAG?",
        answer="RAG combines retrieval with generation.",
        sources=["doc1.pdf"],
        is_grounded=True,
        was_regenerated=False,
        regeneration_attempts=0,
        processing_time_ms=42.0,
    )
    mock_service.get_query_history.return_value = []

    monkeypatch.setattr(api_main, "_query_service", mock_service)
    monkeypatch.setattr(api_main, "get_query_service", lambda: mock_service)

    return TestClient(api_main.app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_success(client):
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAG combines retrieval with generation."
    assert data["is_grounded"] is True
    assert data["sources"] == ["doc1.pdf"]


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_service_not_ready(client, monkeypatch):
    mock_service = MagicMock()
    mock_service.is_ready.return_value = False
    monkeypatch.setattr(api_main, "get_query_service", lambda: mock_service)

    response = client.post("/query", json={"question": "Hello?"})
    assert response.status_code == 503


def test_history_endpoint(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json()["count"] == 0
