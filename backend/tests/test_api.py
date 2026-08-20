import io

from fastapi.testclient import TestClient

from app.main import app


def test_research_streams_a_response():
    with TestClient(app) as client:
        r = client.post("/api/research", json={"request": "define entropy"})
        assert r.status_code == 200
        assert "Research summary" in r.text


def test_blank_request_is_rejected():
    with TestClient(app) as client:
        r = client.post("/api/research", json={"request": "  "})
        assert r.status_code == 400


def test_upload_returns_file_metadata():
    with TestClient(app) as client:
        files = [("files", ("notes.txt", io.BytesIO(b"hello notes"), "text/plain"))]
        r = client.post("/api/sources", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["uploaded"][0]["name"] == "notes.txt"
        assert body["uploaded"][0]["size"] == 11
