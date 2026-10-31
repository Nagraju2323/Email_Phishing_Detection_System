from email.message import EmailMessage

import pytest
from fastapi.testclient import TestClient

import app as phishing_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(phishing_app, "DB_PATH", tmp_path / "test-history.sqlite3")
    phishing_app.init_db()
    with TestClient(phishing_app.app) as test_client:
        yield test_client


def _eml_bytes(subject, body, sender="sender@example.org", recipient="user@example.org"):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return message.as_bytes()


def test_health_dashboard_and_static_frontend(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["algorithm"] == "Logistic Regression"

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["total_scanned"] == 0
    assert dashboard["phishing_detected"] == 0
    assert dashboard["legitimate_detected"] == 0

    frontend = client.get("/")
    assert frontend.status_code == 200
    assert "EMAIL PHISHING" in frontend.text
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize(
    "subject,content",
    [
        (
            "Project meeting",
            "Hi team, our project meeting is confirmed for Thursday at 2 PM. Please review the agenda before we meet. Thanks, Alex.",
        ),
        (
            "Shared document",
            "A document is waiting. Open https://files-share.example.invalid/session and sign in to view it.",
        ),
        (
            "Urgent password reset",
            "URGENT: Your account will be suspended today. Verify your password now at https://secure-login.example.invalid/reset to keep access.",
        ),
        (
            "OTP verification required",
            "Reply with the one-time passcode sent to your phone immediately to verify your account and prevent suspension.",
        ),
        (
            "Banking alert",
            "A transfer is on hold. Confirm your bank account number, card details, and online banking password at https://bank-check.example.invalid/confirm.",
        ),
    ],
)
def test_text_patterns_return_real_model_predictions(client, subject, content):
    response = client.post("/api/scan-text", json={"subject": subject, "content": content})
    assert response.status_code == 200
    result = response.json()
    assert result["classification"] in {"phishing", "legitimate"}
    assert 0 <= result["prediction_probability"] <= 1
    assert 0 <= result["phishing_probability"] <= 1
    assert "does not guarantee" in result["warning"]


def test_empty_text_is_rejected(client):
    response = client.post("/api/scan-text", json={"subject": "Empty", "content": "   "})
    assert response.status_code == 422


def test_invalid_file_extension_is_rejected(client):
    response = client.post(
        "/api/scan-eml",
        files={"file": ("message.txt", b"Subject: hello\n\nbody", "text/plain")},
    )
    assert response.status_code == 400


def test_valid_plain_text_eml(client):
    raw = _eml_bytes("Invoice notice", "Please review the invoice attached to this message.")
    response = client.post(
        "/api/scan-eml",
        files={"file": ("notice.eml", raw, "message/rfc822")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["subject"] == "Invoice notice"
    assert result["sender"] == "sender@example.org"


def test_html_only_eml_ignores_script_and_extracts_anchor_url(client):
    raw = (
        b"From: sender@example.org\r\nTo: user@example.org\r\nSubject: Account alert\r\n"
        b"MIME-Version: 1.0\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><h1>Verify your account</h1><a href='https://verify.example.invalid'>Continue</a>"
        b"<script>SHOULD_NOT_BE_EXTRACTED</script></body></html>"
    )
    response = client.post(
        "/api/scan-eml",
        files={"file": ("html-only.eml", raw, "message/rfc822")},
    )
    assert response.status_code == 200
    result = response.json()
    assert "https://verify.example.invalid" in result["urls"]
    assert "SHOULD_NOT_BE_EXTRACTED" not in str(result)


def test_multipart_eml_prefers_plain_text_and_ignores_attachment(client):
    message = EmailMessage()
    message["From"] = "sender@example.org"
    message["To"] = "user@example.org"
    message["Subject"] = "Quarterly update"
    message.set_content("The quarterly report is ready for review.")
    message.add_alternative("<html><body>HTML alternative <a href='https://example.org'>link</a></body></html>", subtype="html")
    message.add_attachment(b"not opened", maintype="application", subtype="octet-stream", filename="report.bin")
    response = client.post(
        "/api/scan-eml",
        files={"file": ("multipart.eml", message.as_bytes(), "message/rfc822")},
    )
    assert response.status_code == 200
    result = response.json()
    assert any("attachment" in indicator.lower() for indicator in result["indicators"])
    assert "https://example.org" in result["urls"]


def test_oversized_eml_is_rejected(client):
    raw = b"From: sender@example.org\r\nSubject: Large\r\n\r\n" + b"x" * (1_048_577)
    response = client.post(
        "/api/scan-eml",
        files={"file": ("large.eml", raw, "message/rfc822")},
    )
    assert response.status_code == 413


def test_history_stores_metadata_only_and_can_be_cleared(client):
    secret_body = "private body text SHOULD_NOT_BE_STORED"
    response = client.post(
        "/api/scan-text",
        json={"subject": "Metadata test", "content": secret_body},
    )
    assert response.status_code == 200
    history = client.get("/api/history").json()["items"]
    assert len(history) == 1
    assert history[0]["subject"] == "Metadata test"
    assert secret_body not in str(history)
    assert client.get("/api/dashboard").json()["total_scanned"] == 1
    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert client.get("/api/history").json()["items"] == []
