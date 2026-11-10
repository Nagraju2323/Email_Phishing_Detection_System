"use strict";

const $ = (selector) => document.querySelector(selector);
const scanError = $("#scan-error");
const resultPanel = $("#result-panel");
let toastTimer;

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3200);
}

function showError(message) {
  scanError.textContent = message;
  scanError.classList.remove("hidden");
}

function clearError() {
  scanError.textContent = "";
  scanError.classList.add("hidden");
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The service returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(payload.detail || "The request could not be completed.");
  }
  return payload;
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function appendTextList(container, values, fallback) {
  container.replaceChildren();
  if (!values || values.length === 0) {
    const item = document.createElement("li");
    item.textContent = fallback;
    container.append(item);
    return;
  }
  for (const value of values) {
    const item = document.createElement("li");
    item.textContent = value;
    container.append(item);
  }
}

function renderResult(result) {
  const isPhishing = result.classification === "phishing";
  const summary = $("#result-summary");
  summary.className = `result-summary ${isPhishing ? "phishing" : "legitimate"}`;
  summary.replaceChildren();

  const symbol = document.createElement("span");
  symbol.className = "result-symbol";
  symbol.textContent = isPhishing ? "!" : "✓";
  const label = document.createElement("strong");
  label.textContent = result.label;
  summary.append(symbol, label);
  if (result.subject) {
    const subject = document.createElement("span");
    subject.className = "result-subject";
    subject.textContent = result.subject;
    summary.append(subject);
  }

  $("#result-probability").textContent = formatPercent(result.prediction_probability);
  $("#result-model-class").textContent = `Model classification: ${result.model_classification}`;
  $("#result-phishing-probability").textContent = formatPercent(result.phishing_probability);
  $("#result-probability-bar").style.width = `${Math.max(0, Math.min(100, result.prediction_probability * 100))}%`;
  appendTextList($("#result-indicators"), result.indicators, "No additional indicators were extracted.");
  appendTextList($("#result-urls"), result.urls, "No web URLs extracted.");

  const keywords = $("#result-keywords");
  keywords.replaceChildren();
  if (!result.suspicious_keywords.length) {
    const empty = document.createElement("span");
    empty.className = "keyword-empty";
    empty.textContent = "No configured terms matched.";
    keywords.append(empty);
  } else {
    for (const word of result.suspicious_keywords) {
      const chip = document.createElement("span");
      chip.className = "keyword-chip";
      chip.textContent = word;
      keywords.append(chip);
    }
  }
  $("#result-warning-text").textContent = result.warning;
  $("#result-explanation").textContent = result.decision_explanation;
  resultPanel.classList.remove("hidden");
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHistory(items) {
  const body = $("#history-body");
  body.replaceChildren();
  $("#history-empty").classList.toggle("hidden", items.length > 0);
  for (const item of items) {
    const row = document.createElement("tr");
    const subject = document.createElement("td");
    subject.className = "subject-cell";
    subject.textContent = item.subject || "(no subject)";
    const sender = document.createElement("td");
    sender.className = "muted-cell";
    sender.textContent = item.sender || "—";
    const date = document.createElement("td");
    date.className = "muted-cell";
    date.textContent = new Date(item.scanned_at).toLocaleString();
    const classification = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `classification-pill ${item.classification}`;
    badge.textContent = item.classification === "phishing" ? "● PHISHING" : "● LEGITIMATE";
    classification.append(badge);
    const probability = document.createElement("td");
    probability.className = "muted-cell";
    probability.textContent = formatPercent(item.prediction_probability);
    row.append(subject, sender, date, classification, probability);
    body.append(row);
  }
}

async function refreshDashboard() {
  try {
    const [dashboard, history] = await Promise.all([api("/api/dashboard"), api("/api/history?limit=100")]);
    $("#stat-total").textContent = Number(dashboard.total_scanned).toLocaleString();
    $("#stat-phishing").textContent = Number(dashboard.phishing_detected).toLocaleString();
    $("#stat-legitimate").textContent = Number(dashboard.legitimate_detected).toLocaleString();
    $("#stat-rate").textContent = formatPercent(dashboard.phishing_rate);
    $("#rate-bar").style.width = `${Math.max(0, Math.min(100, dashboard.phishing_rate * 100))}%`;
    renderHistory(history.items);

    const online = dashboard.model.status === "ready";
    $("#api-status").textContent = online ? "SYSTEM OPERATIONAL" : "MODEL UNAVAILABLE";
    $(".api-indicator").classList.toggle("offline", !online);
    const modelReady = $("#model-ready");
    modelReady.replaceChildren();
    const readyDot = document.createElement("i");
    const readyText = document.createTextNode(online ? " READY" : " UNAVAILABLE");
    modelReady.append(readyDot, readyText);
    modelReady.classList.toggle("offline", !online);
    $("#model-status-text").textContent = online ? "Artifacts compatible" : "Check local model files";
  } catch (error) {
    $("#api-status").textContent = "SERVICE OFFLINE";
    $(".api-indicator").classList.add("offline");
    $("#model-ready").textContent = "UNAVAILABLE";
    $("#model-ready").classList.add("offline");
    $("#model-status-text").textContent = "Service unavailable";
    showToast(error.message);
  }
}

function setBusy(button, busy, busyText) {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

async function submitTextScan() {
  clearError();
  resultPanel.classList.add("hidden");
  const content = $("#email-content").value;
  if (!content.trim()) {
    showError("Enter email content before scanning.");
    $("#email-content").focus();
    return;
  }
  const button = $("#analyze-button");
  setBusy(button, true, "Analyzing…");
  try {
    const result = await api("/api/scan-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject: $("#email-subject").value, content }),
    });
    renderResult(result);
    await refreshDashboard();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function submitFileScan() {
  clearError();
  resultPanel.classList.add("hidden");
  const file = $("#eml-file").files[0];
  if (!file) {
    showError("Choose an .eml file before scanning.");
    return;
  }
  if (!file.name.toLowerCase().endsWith(".eml")) {
    showError("Only .eml files are accepted.");
    return;
  }
  if (file.size > 1_048_576) {
    showError("The .eml file exceeds the 1 MiB upload limit.");
    return;
  }
  const button = $("#upload-button");
  setBusy(button, true, "Analyzing…");
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const result = await api("/api/scan-eml", { method: "POST", body: form });
    renderResult(result);
    await refreshDashboard();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(button, false);
  }
}

function clearPasteInputs() {
  $("#email-subject").value = "";
  $("#email-content").value = "";
  $("#character-count").textContent = "0 / 100,000";
  clearError();
  resultPanel.classList.add("hidden");
}

function clearFileInput() {
  $("#eml-file").value = "";
  $("#file-name").textContent = "No file selected";
  clearError();
  resultPanel.classList.add("hidden");
}

$("#tab-paste").addEventListener("click", () => {
  $("#tab-paste").classList.add("active");
  $("#tab-upload").classList.remove("active");
  $("#tab-paste").setAttribute("aria-selected", "true");
  $("#tab-upload").setAttribute("aria-selected", "false");
  $("#paste-pane").classList.remove("hidden");
  $("#upload-pane").classList.add("hidden");
  clearError();
});
$("#tab-upload").addEventListener("click", () => {
  $("#tab-upload").classList.add("active");
  $("#tab-paste").classList.remove("active");
  $("#tab-upload").setAttribute("aria-selected", "true");
  $("#tab-paste").setAttribute("aria-selected", "false");
  $("#upload-pane").classList.remove("hidden");
  $("#paste-pane").classList.add("hidden");
  clearError();
});
$("#analyze-button").addEventListener("click", submitTextScan);
$("#upload-button").addEventListener("click", submitFileScan);
$("#clear-button").addEventListener("click", clearPasteInputs);
$("#upload-clear-button").addEventListener("click", clearFileInput);
$("#dismiss-result").addEventListener("click", () => resultPanel.classList.add("hidden"));
$("#email-content").addEventListener("input", (event) => {
  $("#character-count").textContent = `${event.target.value.length.toLocaleString()} / 100,000`;
});
$("#eml-file").addEventListener("change", (event) => {
  $("#file-name").textContent = event.target.files[0]?.name || "No file selected";
  clearError();
});
$("#clear-history").addEventListener("click", async () => {
  if (!window.confirm("Clear all stored scan history? This does not affect the model.")) return;
  try {
    await api("/api/history", { method: "DELETE" });
    await refreshDashboard();
    showToast("Scan history cleared.");
  } catch (error) {
    showToast(error.message);
  }
});

const dropzone = $("#dropzone");
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $("#eml-file").files = transfer.files;
  $("#file-name").textContent = file.name;
});

refreshDashboard();
