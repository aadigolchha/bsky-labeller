from fastapi.testclient import TestClient
from analyzer import app  # Import your app

client = TestClient(app)

def test_election_label():
    response = client.post(
        "/analyze",
        json={"text": "I saw a huge election today", "did": "did:plc:123"}
    )
    assert response.status_code == 200
    assert response.json() == {"label": "election-label", "score": 1.0, "text": "i saw a huge election today"}

def test_vaccine_label():
    response = client.post(
        "/analyze",
        json={"text": "I saw a huge vaccine today", "did": "did:plc:123"}
    )
    assert response.status_code == 200
    assert response.json() == {"label": "vaccine-label", "score": 1.0, "text": "i saw a huge vaccine today"}

def test_no_label():
    response = client.post(
        "/analyze",
        json={"text": "Just a normal day", "did": "did:plc:123"}
    )
    assert response.status_code == 200
    assert response.json() == {"label": None}