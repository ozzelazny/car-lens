/*
 * Phase 6.1 recognize UI - vanilla JS, no framework.
 *
 * Behaviour:
 *  1. On load, ping /health and surface the model name in the footer.
 *  2. Drag-drop OR file picker -> handleFile(file).
 *  3. handleFile validates type/size, renders preview, POSTs to
 *     /api/recognize, and renders predictions.
 *  4. "Try another" resets to the drop zone.
 *
 * Endpoints are relative to the current host. The nginx config in
 * ../nginx.conf proxies /api/* and /health to the backend service so
 * the browser sees a same-origin API.
 */

const MAX_BYTES = 20 * 1024 * 1024; // mirrored from app.py MAX_UPLOAD_BYTES

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const resultPane = document.getElementById("result");
const previewImg = document.getElementById("preview-img");
const predList = document.getElementById("prediction-list");
const elapsedNode = document.getElementById("elapsed");
const errorPane = document.getElementById("error");
const resetBtn = document.getElementById("reset-btn");
const healthStatus = document.getElementById("health-status");

function showError(msg) {
  errorPane.textContent = msg;
  errorPane.classList.remove("hidden");
}

function clearError() {
  errorPane.textContent = "";
  errorPane.classList.add("hidden");
}

function resetUi() {
  clearError();
  resultPane.classList.add("hidden");
  predList.innerHTML = "";
  elapsedNode.textContent = "";
  previewImg.src = "";
  fileInput.value = "";
}

resetBtn.addEventListener("click", resetUi);

// ---- drag-drop wiring ------------------------------------------------

["dragenter", "dragover"].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});
dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length > 0) {
    handleFile(files[0]);
  }
});

// Click-anywhere-on-zone triggers the file picker, but only outside
// the inner <label> so we don't get a double-fire when the user clicks
// the "click to choose" link.
dropZone.addEventListener("click", (e) => {
  if (e.target && e.target.tagName === "LABEL") return;
  fileInput.click();
});
fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f);
});

// ---- core upload + render -------------------------------------------

async function handleFile(file) {
  clearError();
  if (!file.type || !file.type.startsWith("image/")) {
    showError(`Not an image file (type: ${file.type || "unknown"}).`);
    return;
  }
  if (file.size > MAX_BYTES) {
    showError(
      `File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB > 20 MB limit.`,
    );
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  const fd = new FormData();
  fd.append("image", file, file.name || "upload.jpg");

  try {
    const resp = await fetch("/api/recognize", {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body && body.detail) detail = `${detail}: ${body.detail}`;
      } catch (_) {
        // body wasn't JSON; ignore
      }
      showError(detail);
      return;
    }
    const data = await resp.json();
    renderPredictions(data);
  } catch (err) {
    showError(`Network error: ${err.message || err}`);
  }
}

function renderPredictions(data) {
  predList.innerHTML = "";
  for (const pred of data.predictions || []) {
    const pct = (pred.confidence * 100).toFixed(1);
    const li = document.createElement("li");
    const label = document.createElement("div");
    label.className = "pred-label";
    const name = document.createElement("span");
    name.textContent = pred.display || pred.class_id;
    const pctNode = document.createElement("span");
    pctNode.className = "pct";
    pctNode.textContent = `${pct}%`;
    label.appendChild(name);
    label.appendChild(pctNode);

    const bar = document.createElement("div");
    bar.className = "pred-bar";
    const fill = document.createElement("div");
    fill.className = "pred-bar-fill";
    bar.appendChild(fill);
    li.appendChild(label);
    li.appendChild(bar);
    predList.appendChild(li);
    // Defer setting width so the transition animates from 0.
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    });
  }
  elapsedNode.textContent =
    typeof data.elapsed_ms === "number"
      ? `Inference: ${data.elapsed_ms.toFixed(1)} ms`
      : "";
  resultPane.classList.remove("hidden");
}

// ---- backend health check on load -----------------------------------

async function pingHealth() {
  try {
    const resp = await fetch("/health");
    if (!resp.ok) {
      healthStatus.textContent = `unhealthy (HTTP ${resp.status})`;
      return;
    }
    const body = await resp.json();
    healthStatus.textContent = `ok - ${body.model} - ${body.n_classes} classes - ${body.device}`;
  } catch (err) {
    healthStatus.textContent = `unreachable (${err.message || err})`;
  }
}

pingHealth();
