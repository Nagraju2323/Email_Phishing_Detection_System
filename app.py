"""Local FastAPI web application for the saved email phishing model."""
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import ipaddress
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Annotated
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from preprocessing import normalize_text


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "comparison_logistic_regression_model.pkl"
VECTORIZER_PATH = ROOT / "comparison_logistic_regression_tfidf.pkl"
DB_PATH = Path(os.environ.get("PHISHING_APP_DB_PATH", ROOT / "app_data" / "scan_history.sqlite3"))
MAX_EML_BYTES = 1_048_576
MAX_TEXT_CHARACTERS = 100_000
MAX_HISTORY_ROWS = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishing_app")

URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\"']+")
URL_TRAILING_PUNCTUATION = ".,;:!?)]}"
SUSPICIOUS_TERMS = {
    "Urgency or threat": ("urgent", "immediately", "suspended", "suspension", "expire", "final warning", "limited time"),
    "Credential request": ("password", "passcode", "otp", "one-time code", "verification code", "login credentials"),
    "Account verification": ("verify your account", "confirm your account", "validate your account", "unlock your account"),
    "Financial request": ("bank account", "banking details", "card number", "payment details", "wire transfer"),
}
TERM_MATCHERS = {
    category: [(term, re.compile(r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)")) for term in terms]
    for category, terms in SUSPICIOUS_TERMS.items()
}


class ScanTextRequest(BaseModel):
    subject: str = Field(default="", max_length=500)
    content: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)


class VisibleHTMLParser(HTMLParser):
    """Extract visible text and anchor destinations without rendering HTML."""

    HIDDEN_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.urls = []
        self._hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.HIDDEN_TAGS:
            self._hidden_depth += 1
        if tag == "a" and self._hidden_depth == 0:
            href = dict(attrs).get("href")
            if href:
                self.urls.append(href.strip())

    def handle_endtag(self, tag):
        if tag.lower() in self.HIDDEN_TAGS and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data):
        if self._hidden_depth == 0 and data.strip():
            self.text_parts.append(data.strip())


def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if not isinstance(model, LogisticRegression):
        raise TypeError("The selected artifact is not a Logistic Regression model.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("The selected model must use classes 0 and 1.")
    if not hasattr(vectorizer, "transform"):
        raise TypeError("The selected vectorizer does not support transform().")
    if int(model.n_features_in_) != len(vectorizer.vocabulary_):
        raise ValueError("The selected model and vectorizer feature dimensions do not match.")
    return model, vectorizer


try:
    MODEL, VECTORIZER = _load_artifacts()
    MODEL_LOAD_ERROR = None
except Exception:
    logger.exception("Could not load the selected Logistic Regression artifacts.")
    MODEL, VECTORIZER = None, None
    MODEL_LOAD_ERROR = "Model artifacts are unavailable or incompatible."


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(
    title="Email Phishing Detection System",
    description="Local email scanning with the saved Logistic Regression and TF-IDF artifacts.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "testserver"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    if request.url.path == "/api/scan-eml":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_EML_BYTES + 65_536:
                    return JSONResponse({"detail": "Upload exceeds the 1 MiB limit."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length."}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with _connect_db() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                subject TEXT NOT NULL,
                sender TEXT NOT NULL,
                classification TEXT NOT NULL CHECK(classification IN ('phishing', 'legitimate')),
                prediction_probability REAL NOT NULL,
                phishing_probability REAL NOT NULL
            )"""
        )


def _clean_header(value, limit=300):
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", "").split())[:limit]


def _extract_urls(text, extra_urls=()):
    found = []
    seen = set()
    for candidate in list(URL_PATTERN.findall(text)) + list(extra_urls):
        candidate = candidate.strip().rstrip(URL_TRAILING_PUNCTUATION)[:500]
        if candidate.lower().startswith("www."):
            candidate = "https://" + candidate
        if candidate.lower().startswith(("http://", "https://")) and candidate not in seen:
            seen.add(candidate)
            found.append(candidate)
        if len(found) >= 100:
            break
    return found


def _url_indicators(urls):
    indicators = []
    if urls:
        indicators.append(f"Contains {len(urls)} web link(s); destinations were not visited.")
    for link in urls:
        hostname = urlparse(link).hostname or ""
        try:
            ipaddress.ip_address(hostname)
            indicators.append("A link uses a raw IP address.")
            break
        except ValueError:
            pass
    if any("xn--" in (urlparse(link).hostname or "").lower() for link in urls):
        indicators.append("A link contains an internationalized/punycode hostname.")
    return indicators


def _analyze_email(subject, content, sender="", recipient="", attachment_count=0, extra_urls=()):
    subject = _clean_header(subject, 500)
    sender = _clean_header(sender, 300)
    recipient = _clean_header(recipient, 300)
    content = "\n".join(str(content).splitlines()).strip()
    if not content:
        raise HTTPException(status_code=422, detail="No readable email text was found.")
    model_input = f"Subject: {subject}\n{content}" if subject else content
    # Include HTML anchor destinations in the model text as well as the result
    # indicators; otherwise an HTML-only email hides its most useful URL tokens.
    if extra_urls:
        model_input += "\nLinks: " + " ".join(str(url)[:500] for url in extra_urls[:100])
    normalized = normalize_text(model_input)
    if not normalized:
        raise HTTPException(status_code=422, detail="Email text is empty after preprocessing.")
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)

    features = VECTORIZER.transform([normalized])
    probabilities = MODEL.predict_proba(features)[0]
    phishing_probability = float(probabilities[list(MODEL.classes_).index(1)])
    predicted_class = 1 if phishing_probability >= 0.5 else 0
    prediction_probability = phishing_probability if predicted_class == 1 else 1.0 - phishing_probability
    urls = _extract_urls(model_input, extra_urls)
    lower_text = (subject + "\n" + content).casefold()
    matched_terms = []
    categories = []
    for category, matchers in TERM_MATCHERS.items():
        matches = [term for term, matcher in matchers if matcher.search(lower_text)]
        if matches:
            categories.append(category)
            matched_terms.extend(matches)

    indicators = list(_url_indicators(urls))
    if categories:
        indicators.append("Potentially risky wording: " + ", ".join(categories) + ".")
    else:
        indicators.append("No configured urgency, credential, account-verification, or financial-request terms matched.")
    if sender:
        indicators.append("Sender information was present in the message.")
    if attachment_count:
        indicators.append(
            f"{attachment_count} attachment part(s) were not opened or analyzed; attachment safety is unknown."
        )
    if len(content) < 80:
        indicators.append("The message text is short, so there is limited evidence for classification.")

    classification = "phishing" if predicted_class == 1 else "legitimate"
    label = "PHISHING EMAIL DETECTED" if predicted_class == 1 else "CLASSIFIED AS LEGITIMATE"
    warning = (
        "This model result is an indicator, not proof. A legitimate classification does not guarantee that an email is safe. "
        "Verify unexpected requests through a known, independent contact method."
    )
    scan_id, scanned_at = _save_scan(
        subject=subject,
        sender=sender,
        classification=classification,
        prediction_probability=prediction_probability,
        phishing_probability=phishing_probability,
    )
    return {
        "id": scan_id,
        "scanned_at": scanned_at,
        "subject": subject,
        "sender": sender,
        "recipient": recipient,
        "classification": classification,
        "label": label,
        "prediction_probability": prediction_probability,
        "phishing_probability": phishing_probability,
        "suspicious_keywords": sorted(set(matched_terms)),
        "keyword_categories": categories,
        "urls": urls,
        "indicators": indicators,
        "warning": warning,
        "model": "Logistic Regression + TF-IDF",
    }


def _save_scan(subject, sender, classification, prediction_probability, phishing_probability):
    scanned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect_db() as connection:
        cursor = connection.execute(
            """INSERT INTO scans
            (scanned_at, subject, sender, classification, prediction_probability, phishing_probability)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (scanned_at, subject[:300], sender[:300], classification,
             float(prediction_probability), float(phishing_probability)),
        )
        return int(cursor.lastrowid), scanned_at


def _row_to_history(row):
    return {
        "id": row["id"],
        "scanned_at": row["scanned_at"],
        "subject": row["subject"],
        "sender": row["sender"],
        "classification": row["classification"],
        "prediction_probability": row["prediction_probability"],
        "phishing_probability": row["phishing_probability"],
    }


def _parse_eml(raw_bytes):
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable email message.") from exc
    recognized_headers = {"from", "to", "subject", "date", "mime-version", "content-type"}
    if not any(name.lower() in recognized_headers for name in message.keys()):
        raise HTTPException(status_code=400, detail="The .eml file does not contain recognizable email headers.")

    plain_parts = []
    html_parts = []
    extracted_links = []
    attachment_count = 0
    for part in message.walk():
        if part.is_multipart():
            continue
        disposition = part.get_content_disposition()
        if disposition == "attachment" or part.get_filename():
            attachment_count += 1
            continue
        content_type = part.get_content_type().lower()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            payload = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = payload.decode(charset, errors="replace")
            except LookupError:
                payload = payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes):
            payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if content_type == "text/plain":
            plain_parts.append(str(payload))
        else:
            parser = VisibleHTMLParser()
            try:
                parser.feed(str(payload))
                parser.close()
            except Exception:
                logger.info("Ignored malformed HTML in an uploaded email.")
            html_parts.append(" ".join(parser.text_parts))
            extracted_links.extend(parser.urls)

    # Prefer plain text when present; HTML is used for HTML-only messages.
    body = "\n".join(plain_parts if plain_parts else html_parts)
    return {
        "subject": _clean_header(message.get("subject", ""), 500),
        "sender": _clean_header(message.get("from", ""), 300),
        "recipient": _clean_header(message.get("to", ""), 300),
        "body": body,
        "urls": extracted_links,
        "attachment_count": attachment_count,
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok" if MODEL is not None and VECTORIZER is not None else "degraded",
        "model_loaded": MODEL is not None and VECTORIZER is not None,
        "algorithm": "Logistic Regression",
        "vectorizer_loaded": VECTORIZER is not None,
    }


@app.get("/api/dashboard")
def dashboard():
    if not DB_PATH.exists():
        init_db()
    with _connect_db() as connection:
        counts = connection.execute(
            """SELECT COUNT(*) AS total,
                SUM(CASE WHEN classification='phishing' THEN 1 ELSE 0 END) AS phishing,
                SUM(CASE WHEN classification='legitimate' THEN 1 ELSE 0 END) AS legitimate
                FROM scans"""
        ).fetchone()
        recent_rows = connection.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 8"
        ).fetchall()
    total = int(counts["total"] or 0)
    phishing = int(counts["phishing"] or 0)
    legitimate = int(counts["legitimate"] or 0)
    return {
        "total_scanned": total,
        "phishing_detected": phishing,
        "legitimate_detected": legitimate,
        "phishing_rate": (phishing / total) if total else 0.0,
        "recent_scans": [_row_to_history(row) for row in recent_rows],
        "model": {
            "algorithm": "Logistic Regression",
            "features": "TF-IDF word unigrams and bigrams",
            "status": "ready" if MODEL is not None and VECTORIZER is not None else "unavailable",
        },
    }


@app.post("/api/scan-text")
def scan_text(request: ScanTextRequest):
    if not request.content.strip():
        raise HTTPException(status_code=422, detail="Email content cannot be blank.")
    return _analyze_email(request.subject, request.content)


@app.post("/api/scan-eml")
async def scan_eml(file: Annotated[UploadFile, File(description="Upload an .eml email file")]):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".eml"):
        raise HTTPException(status_code=400, detail="Only .eml files are accepted.")
    raw_bytes = await file.read(MAX_EML_BYTES + 1)
    await file.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded .eml file is empty.")
    if len(raw_bytes) > MAX_EML_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 1 MiB limit.")
    if b"\x00" in raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid text email.")
    parsed = _parse_eml(raw_bytes)
    if not parsed["body"].strip():
        raise HTTPException(status_code=422, detail="The .eml file contains no readable text/plain or text/html body.")
    return _analyze_email(
        parsed["subject"], parsed["body"], parsed["sender"], parsed["recipient"],
        parsed["attachment_count"], parsed["urls"],
    )


@app.get("/api/history")
def history(limit: int = 100):
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    if not DB_PATH.exists():
        init_db()
    with _connect_db() as connection:
        rows = connection.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"items": [_row_to_history(row) for row in rows]}


@app.delete("/api/history")
def clear_history():
    if not DB_PATH.exists():
        init_db()
    with _connect_db() as connection:
        connection.execute("DELETE FROM scans")
        connection.execute("DELETE FROM sqlite_sequence WHERE name='scans'")
    return {"cleared": True}
