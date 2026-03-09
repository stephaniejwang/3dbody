/**
 * Upload module — handles video upload, reference selection, and API polling.
 *
 * Dependencies: None (vanilla JS). Three.js loaded separately by viewer.js.
 */

// Auto-detect API base: same origin in production, localhost in dev
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : window.location.origin;
const POLL_INTERVAL_MS = 1000;

// Progress stages for the progress bar
const PROGRESS_STAGES = {
    "Uploading video...": 10,
    "Queued for processing": 15,
    "Extracting frames...": 20,
    "Loading object detection model...": 30,
    "Detecting scale...": 40,
    "Detecting pose...": 55,
    "Recovering body mesh...": 70,
    "Applying scale calibration...": 80,
    "Computing measurements...": 90,
};

// DOM elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const clearFile = document.getElementById("clear-file");
const uploadBtn = document.getElementById("upload-btn");
const progressOverlay = document.getElementById("progress-overlay");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const errorMessage = document.getElementById("error-message");
const errorText = document.getElementById("error-text");
const dismissError = document.getElementById("dismiss-error");
const heightInputGroup = document.getElementById("height-input-group");
const heightValue = document.getElementById("height-value");
const shoeSize = document.getElementById("shoe-size");
const reuploadBtn = document.getElementById("reupload-btn");

const cameraInput = document.getElementById("camera-input");
const cameraBtn = document.getElementById("camera-btn");

let selectedFile = null;
let heightUnit = "cm"; // "cm" or "in"
let shoeUnit = "us"; // "us", "eu", or "uk"
let selectedGender = "male"; // "male" or "female"
let currentJobId = null;
let recalUnit = "cm";

// ===== File selection =====

dropZone.addEventListener("click", () => fileInput.click());

// Camera record button (opens native camera on mobile)
cameraBtn?.addEventListener("click", (e) => {
    e.stopPropagation(); // Don't trigger drop zone click
    cameraInput.click();
});
cameraInput?.addEventListener("change", () => {
    if (cameraInput.files.length > 0) handleFile(cameraInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

clearFile.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    dropZone.style.display = "";
    uploadBtn.disabled = true;
});

function handleFile(file) {
    const ext = file.name.toLowerCase().split(".").pop();
    if (!["mp4", "mov"].includes(ext)) {
        showError("Only mp4 and mov files are supported.");
        return;
    }
    if (file.size > 200 * 1024 * 1024) {
        showError("File exceeds 200MB limit.");
        return;
    }
    selectedFile = file;
    fileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    fileInfo.classList.remove("hidden");
    dropZone.style.display = "none";
    uploadBtn.disabled = false;
    hideError();
}

// ===== Reference mode =====

const referenceRadios = document.querySelectorAll('input[name="reference"]');
const heightOptionalTag = document.getElementById("height-optional-tag");

function updateHeightRequired() {
    const mode = document.querySelector('input[name="reference"]:checked')?.value;
    const isRequired = mode === "none";
    if (heightOptionalTag) {
        heightOptionalTag.textContent = isRequired ? "(required)" : "(optional)";
        heightOptionalTag.className = isRequired ? "required-tag" : "optional-tag";
    }
}

referenceRadios.forEach((radio) => {
    radio.addEventListener("change", updateHeightRequired);
});

// Height unit toggle
document.querySelectorAll("#height-input-group .unit-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("#height-input-group .unit-btn").forEach((b) =>
            b.classList.remove("active")
        );
        btn.classList.add("active");
        heightUnit = btn.dataset.unit;
    });
});

// Shoe size unit toggle
document.querySelectorAll(".shoe-unit-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".shoe-unit-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        shoeUnit = btn.dataset.unit;
    });
});

// Gender toggle
document.querySelectorAll(".gender-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".gender-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        selectedGender = btn.dataset.gender;
    });
});

// Init
updateHeightRequired();

// ===== Upload =====

uploadBtn.addEventListener("click", startUpload);

async function startUpload() {
    if (!selectedFile) return;
    hideError();

    let referenceMode = document.querySelector('input[name="reference"]:checked').value;

    // Parse height if provided
    let heightCm = null;
    const rawVal = parseFloat(heightValue.value);
    if (!isNaN(rawVal) && rawVal > 0) {
        let val = rawVal;
        if (heightUnit === "in") {
            val = val * 2.54; // convert to cm
        }
        if (val < 50 || val > 250) {
            showError("Height must be between 50 cm and 250 cm.");
            return;
        }
        heightCm = val;
    }

    // Parse shoe size if provided
    let shoeSizeVal = null;
    const rawShoe = parseFloat(shoeSize?.value);
    if (!isNaN(rawShoe) && rawShoe > 0) {
        shoeSizeVal = rawShoe;
    }

    // "none" reference mode requires height or shoe size
    if (referenceMode === "none") {
        if (heightCm === null && shoeSizeVal === null) {
            showError("Please enter your height or shoe size when not using a reference object.");
            return;
        }
        // Backend expects "height_cm" as the reference mode
        referenceMode = "height_cm";
    }

    // Build form data
    const formData = new FormData();
    formData.append("video", selectedFile);
    formData.append("reference_mode", referenceMode);
    formData.append("gender", selectedGender);
    if (heightCm !== null) {
        formData.append("height_cm", heightCm.toString());
    }
    if (shoeSizeVal !== null) {
        formData.append("shoe_size", shoeSizeVal.toString());
        formData.append("shoe_unit", shoeUnit);
    }

    // Show progress
    showProgress("Uploading video...");

    try {
        const resp = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || `Upload failed (${resp.status})`);
        }

        const { job_id } = await resp.json();
        currentJobId = job_id;
        pollJob(job_id);
    } catch (err) {
        hideProgress();
        showError(err.message || "Upload failed. Is the backend running?");
    }
}

// ===== Job polling =====

async function pollJob(jobId) {
    let retries404 = 0;
    const poll = async () => {
        try {
            const resp = await fetch(`${API_BASE}/status/${jobId}`);
            if (resp.status === 404) {
                retries404++;
                if (retries404 >= 3) {
                    hideProgress();
                    showError("Job lost — the server likely restarted. Please re-upload.");
                    return;
                }
                // Could be a brief race condition, retry
                setTimeout(poll, POLL_INTERVAL_MS);
                return;
            }
            if (!resp.ok) throw new Error(`Status check failed (${resp.status})`);
            retries404 = 0;

            const data = await resp.json();
            updateProgress(data.progress || data.status);

            if (data.status === "done") {
                hideProgress();
                showViewer(data.result);
                return;
            }
            if (data.status === "error") {
                hideProgress();
                showError(data.error || "Processing failed.");
                return;
            }

            // Continue polling
            setTimeout(poll, POLL_INTERVAL_MS);
        } catch (err) {
            hideProgress();
            showError(err.message || "Lost connection to server.");
        }
    };
    poll();
}

// ===== Progress UI =====

function showProgress(msg) {
    progressOverlay.classList.remove("hidden");
    progressText.textContent = msg;
    progressBar.style.width = "5%";
}

function updateProgress(msg) {
    progressText.textContent = msg;
    const pct = PROGRESS_STAGES[msg] || null;
    if (pct) {
        progressBar.style.width = pct + "%";
    }
}

function hideProgress() {
    progressOverlay.classList.add("hidden");
    progressBar.style.width = "0%";
}

// ===== Error UI =====

function showError(msg) {
    errorText.textContent = msg;
    errorMessage.classList.remove("hidden");
}

function hideError() {
    errorMessage.classList.add("hidden");
}

dismissError.addEventListener("click", hideError);

// ===== Screen switching =====

function showViewer(result) {
    document.getElementById("upload-screen").classList.remove("active");
    document.getElementById("viewer-screen").classList.add("active");

    // Dispatch custom event for viewer.js
    window.dispatchEvent(new CustomEvent("body-result", { detail: result }));
}

// Re-upload
reuploadBtn?.addEventListener("click", () => {
    document.getElementById("viewer-screen").classList.remove("active");
    document.getElementById("upload-screen").classList.add("active");
    selectedFile = null;
    currentJobId = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    dropZone.style.display = "";
    uploadBtn.disabled = true;
});

// ===== Recalibration =====

const recalBtn = document.getElementById("recal-btn");
const recalHeight = document.getElementById("recal-height");
const recalStatus = document.getElementById("recal-status");

// Unit toggle for recalibration input
document.querySelectorAll(".recal-unit").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".recal-unit").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        recalUnit = btn.dataset.unit;
    });
});

recalBtn?.addEventListener("click", async () => {
    if (!currentJobId) return;

    let val = parseFloat(recalHeight.value);
    if (isNaN(val) || val <= 0) {
        recalStatus.textContent = "Enter a valid height.";
        recalStatus.className = "recal-status error";
        recalStatus.classList.remove("hidden");
        return;
    }

    // Convert to cm if inches
    let heightCm = recalUnit === "in" ? val * 2.54 : val;
    if (heightCm < 50 || heightCm > 250) {
        recalStatus.textContent = "Height must be between 50 cm and 250 cm.";
        recalStatus.className = "recal-status error";
        recalStatus.classList.remove("hidden");
        return;
    }

    recalBtn.disabled = true;
    recalStatus.textContent = "Recalibrating...";
    recalStatus.className = "recal-status";
    recalStatus.classList.remove("hidden");

    try {
        const resp = await fetch(`${API_BASE}/recalibrate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: currentJobId, height_cm: heightCm }),
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || `Recalibration failed (${resp.status})`);
        }

        const { result } = await resp.json();

        // Update the viewer with recalibrated result
        window.dispatchEvent(new CustomEvent("body-result", { detail: result }));

        recalStatus.textContent = `Recalibrated to ${heightCm.toFixed(1)} cm — measurements updated.`;
        recalStatus.className = "recal-status success";
    } catch (err) {
        recalStatus.textContent = err.message;
        recalStatus.className = "recal-status error";
    } finally {
        recalBtn.disabled = false;
    }
});
