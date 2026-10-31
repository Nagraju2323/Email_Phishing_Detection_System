# Email Phishing Detection System

A local FastAPI application for scanning pasted email text or `.eml` files with the existing selected Logistic Regression model and its matching TF-IDF vectorizer.

## Model artifacts

The application loads these existing files from the project root:

- `comparison_logistic_regression_model.pkl`
- `comparison_logistic_regression_tfidf.pkl`

At startup, it checks that the classifier is Logistic Regression, its classes are `[0, 1]`, and its feature count matches the vectorizer vocabulary. It never trains or fits model artifacts. Prediction calls `vectorizer.transform()` only.

## Requirements

- Python 3.10 or later
- The model and vectorizer files listed above
- Dependencies in `requirements.txt`

The scikit-learn and Joblib versions are pinned to match the saved artifacts. The upload endpoint uses FastAPI's multipart support through `python-multipart`.

## Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The server binds to loopback by default; keep it local unless you add authentication and deployment-grade protections.

The first scan creates `app_data/scan_history.sqlite3`. History stores only scan time, subject, sender, classification, and probabilities. It does not store email bodies or extracted URLs. Use **Clear history** to remove stored scan metadata.

## Dashboard and scanning

- Dashboard counters and recent scans come only from this application's SQLite scan records. A new installation shows zeros and an empty history.
- Paste a subject and message body, or upload an `.eml` file up to 1 MiB.
- `.eml` parsing supports multipart, `text/plain`, and `text/html`. Script/style content is excluded, attachments are not opened or analyzed, and URLs are displayed as plain text and are never visited.
- The frontend inserts email-derived values with DOM text nodes, not HTML. The API adds restrictive security headers and accepts local hostnames only.
- A **Legitimate** result is not a guarantee that an email is safe. Verify unexpected requests through a known channel.

## API

- `GET /api/health` — model load/compatibility status.
- `GET /api/dashboard` — live totals and recent scan records.
- `POST /api/scan-text` — JSON body: `{"subject":"...", "content":"..."}`.
- `POST /api/scan-eml` — multipart form field named `file` with an `.eml` file.
- `GET /api/history?limit=100` — recent metadata only.
- `DELETE /api/history` — clear history.

## Tests

Run the API and email parsing checks with:

```powershell
python -m pytest -q
```

The tests use a temporary SQLite database and do not alter the model files or datasets.
