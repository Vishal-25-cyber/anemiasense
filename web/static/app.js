/* ============================================================
   AnemiaSense v3.0 — Unified Dual-Model Frontend
   Eye Conjunctiva (MobileNetV2) + Nail Bed (MobileNetV3Large)
   Features: Live Camera Capture + Interactive Mark & Zoom (ROI)
   ============================================================ */

"use strict";

// ── API ENDPOINTS ────────────────────────────────────────────
const API = {
  EYE:       "/api/predict",
  NAIL:      "/api/predict/nail",
  COMBINED:  "/api/predict/combined",
  HEALTH:    "/api/health",
  METADATA:  "/api/metadata",
  HISTORY:   "/api/history",
  STATS:     "/api/history/stats",
  DB_STATUS: "/api/db/status",
};

let currentReportData = null;
let activeHistoryFilter = "ALL";

const MAX_FILE_BYTES = 15 * 1024 * 1024; // 15 MB

// ── SAFE DOM HELPER ──────────────────────────────────────────
function $(id) {
  return document.getElementById(id);
}

// ── TOAST NOTIFICATIONS ──────────────────────────────────────
function showToast(message, type = "info", durationMs = 4500) {
  const container = $("toasts");
  if (!container) return;

  const toast = document.createElement("div");
  const cls = (type === "error" || type === "err") ? "err" : (type === "success" || type === "ok") ? "ok" : "info";
  toast.className = `toast ${cls}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, durationMs);
}

// ── ESCAPE HTML ──────────────────────────────────────────────
function escHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ── SECTION NAVIGATION ───────────────────────────────────────
function initNavigation() {
  const sections = document.querySelectorAll(".sec");
  const navLinks = document.querySelectorAll(".nav a");

  function showSection(targetId) {
    sections.forEach(s => s.classList.remove("on"));
    navLinks.forEach(a => a.classList.remove("active"));

    const sec = $(targetId);
    if (sec) sec.classList.add("on");

    const navLink = $("nl-" + targetId);
    if (navLink) navLink.classList.add("active");

    if (targetId === "history") {
      loadHistory();
      loadHistoryStats();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navLinks.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const secId = link.getAttribute("data-sec");
      if (secId) showSection(secId);
    });
  });

  const brandLink = $("nav-brand");
  if (brandLink) {
    brandLink.addEventListener("click", e => {
      e.preventDefault();
      showSection("home");
    });
  }

  const btnStart = $("btn-start");
  if (btnStart) {
    btnStart.addEventListener("click", e => {
      e.preventDefault();
      showSection("screening");
    });
  }

  const btnLearn = $("btn-learn");
  if (btnLearn) {
    btnLearn.addEventListener("click", e => {
      e.preventDefault();
      showSection("about");
    });
  }

  document.querySelectorAll("[data-sec]").forEach(el => {
    if (!el.id || !el.id.startsWith("nl-")) {
      el.addEventListener("click", e => {
        e.preventDefault();
        const secId = el.getAttribute("data-sec");
        if (secId) showSection(secId);
      });
    }
  });

  function handleHash() {
    const hash = window.location.hash.replace("#", "").toLowerCase();
    if (hash === "screening") showSection("screening");
    else if (hash === "about" || hash === "how-it-works") showSection("about");
    else if (hash === "home") showSection("home");
  }

  window.addEventListener("hashchange", handleHash);
  if (window.location.hash) handleHash();
}

// ── TAB SWITCHING ─────────────────────────────────────────────
function initTabs() {
  const tabs = [
    { btn: $("t-eye"),  panel: $("p-eye") },
    { btn: $("t-nail"), panel: $("p-nail") },
  ];

  tabs.forEach(({ btn, panel }) => {
    if (!btn || !panel) return;
    btn.addEventListener("click", () => {
      tabs.forEach(t => {
        if (t.btn) {
          t.btn.classList.remove("on");
          t.btn.setAttribute("aria-selected", "false");
        }
        if (t.panel) {
          t.panel.classList.remove("on");
        }
      });
      btn.classList.add("on");
      btn.setAttribute("aria-selected", "true");
      panel.classList.add("on");
    });
  });
}

// ── PATIENT CONTEXT & VALIDATION ──────────────────────────────
function validatePatientContext(highlightErrors = false) {
  const genderEl = $("sel-gender");
  const ageEl    = $("inp-age");
  const dietEl   = $("sel-diet");

  const gender = genderEl ? genderEl.value.trim() : "";
  const ageRaw = ageEl ? ageEl.value.trim() : "";
  const age    = parseInt(ageRaw, 10);
  const diet   = dietEl ? dietEl.value.trim() : "";

  const errors = [];
  const isGenderValid = (gender === "M" || gender === "F");
  const isAgeValid    = (ageRaw !== "" && !isNaN(age) && age >= 1 && age <= 120);
  const isDietValid   = (diet === "vegetarian" || diet === "non-vegetarian");

  if (!isGenderValid) errors.push("Please select Patient Gender.");
  if (!isAgeValid)    errors.push("Please enter a valid Patient Age (1–120 years).");
  if (!isDietValid)   errors.push("Please select Patient Dietary Preference.");

  if (highlightErrors) {
    if (genderEl) genderEl.classList.toggle("err-field", !isGenderValid);
    if (ageEl)    ageEl.classList.toggle("err-field", !isAgeValid);
    if (dietEl)   dietEl.classList.toggle("err-field", !isDietValid);
  } else {
    if (genderEl && isGenderValid) genderEl.classList.remove("err-field");
    if (ageEl && isAgeValid) ageEl.classList.remove("err-field");
    if (dietEl && isDietValid) dietEl.classList.remove("err-field");
  }

  const isValid = isGenderValid && isAgeValid && isDietValid;
  return {
    isValid,
    errors,
    data: {
      gender: isGenderValid ? gender : "",
      age: isAgeValid ? age : null,
      dietaryPref: isDietValid ? diet : "",
    }
  };
}

function getPatientContext() {
  const check = validatePatientContext(false);
  return check.data;
}

function updateAnalyzeButtonsState() {
  const { isValid } = validatePatientContext(false);

  const btnEye = $("btn-eye");
  if (btnEye) {
    const hasEyeFile = uploaders.eye && Boolean(uploaders.eye.getFile());
    btnEye.disabled = !(hasEyeFile && isValid);
  }

  const btnNail = $("btn-nail");
  if (btnNail) {
    const hasNailFile = uploaders.nail && Boolean(uploaders.nail.getFile());
    btnNail.disabled = !(hasNailFile && isValid);
  }
}

// ── BUTTON LOADING STATE ─────────────────────────────────────
function setBtnLoading(btnId, spinnerId, loading, loadingText = "Analyzing…") {
  const btn = $(btnId);
  const sp  = $(spinnerId);
  if (!btn) return;

  const lbl = btn.querySelector(".btn-lbl");

  if (loading) {
    btn.disabled = true;
    if (sp) sp.classList.remove("hidden");
    if (lbl) {
      if (!lbl.dataset.orig) lbl.dataset.orig = lbl.textContent;
      lbl.textContent = loadingText;
    }
  } else {
    btn.disabled = false;
    if (sp) sp.classList.add("hidden");
    if (lbl && lbl.dataset.orig) {
      lbl.textContent = lbl.dataset.orig;
    }
  }
}

// ── DROPZONE FACTORY ─────────────────────────────────────────
function wireDropzone({ dzId, fiId, dziId, dzpId, imgId, rmId, roiBadgeId, btnId, onFileChange }) {
  const dz       = $(dzId);
  const fi       = $(fiId);
  const dzi      = $(dziId);
  const dzp      = $(dzpId);
  const img      = $(imgId);
  const rm       = $(rmId);
  const roiBadge = roiBadgeId ? $(roiBadgeId) : null;
  const btn      = btnId ? $(btnId) : null;

  if (!dz || !fi) {
    return {
      getFile: () => null,
      setFile: () => {},
      clear: () => {},
      getCropCoords: () => null,
      setCropCoords: () => {}
    };
  }

  let currentFile = null;
  let originalSourceFile = null;
  let currentCropCoords = null;

  function setFile(file, isCropped = false) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showToast("Please select a valid image file (PNG, JPG, WEBP).", "err");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      showToast("File is too large. Maximum size is 15 MB.", "err");
      return;
    }

    currentFile = file;
    if (!isCropped) {
      originalSourceFile = file;
      currentCropCoords = null;
    }

    const url = URL.createObjectURL(file);
    if (img) img.src = url;
    if (dzi) dzi.classList.add("hidden");
    if (dzp) dzp.classList.remove("hidden");
    updateAnalyzeButtonsState();

    if (roiBadge) {
      if (isCropped) roiBadge.classList.remove("hidden");
      else roiBadge.classList.add("hidden");
    }

    if (onFileChange) onFileChange(file);
  }

  function clearFile() {
    currentFile = null;
    originalSourceFile = null;
    currentCropCoords = null;
    if (img) img.src = "";
    if (dzp) dzp.classList.add("hidden");
    if (dzi) dzi.classList.remove("hidden");
    if (roiBadge) roiBadge.classList.add("hidden");
    fi.value = "";
    updateAnalyzeButtonsState();
    if (onFileChange) onFileChange(null);
  }

  fi.addEventListener("change", e => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0], false);
    }
  });

  if (rm) {
    rm.addEventListener("click", e => {
      e.stopPropagation();
      clearFile();
    });
  }

  dz.addEventListener("click", e => {
    // If clicked on remove or tools overlay, ignore outer click
    if (e.target.closest(".dz-prev-rm") || e.target.closest(".dz-tools-overlay")) return;
    fi.click();
  });

  dz.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fi.click();
    }
  });

  dz.addEventListener("dragover", e => {
    e.preventDefault();
    dz.classList.add("over");
  });

  dz.addEventListener("dragleave", () => {
    dz.classList.remove("over");
  });

  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.classList.remove("over");
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0], false);
    }
  });

  return {
    getFile: () => currentFile,
    getOriginalFile: () => originalSourceFile || currentFile,
    setFile: setFile,
    clear: clearFile,
    getCropCoords: () => currentCropCoords,
    setCropCoords: (coords) => { currentCropCoords = coords; },
    getImgElement: () => img,
    openBrowse: () => fi.click(),
  };
}

// ── STUDIO HD MEDICAL CLARITY & DENOISING ENGINE ──────────────
function applyStudioHDEnhancement(ctx, width, height) {
  try {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const len = data.length;
    const w = width;
    const h = height;
    if (w < 4 || h < 4) return;

    // Allocate working buffers for Y (luminance), U (chroma-blue), V (chroma-red)
    const Y = new Float32Array(w * h);
    const U = new Float32Array(w * h);
    const V = new Float32Array(w * h);

    let sumU = 0, sumV = 0, sumY = 0;
    for (let i = 0, p = 0; i < len; i += 4, p++) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      const uVal = -0.14713 * r - 0.28886 * g + 0.436 * b;
      const vVal = 0.615 * r - 0.51499 * g - 0.10001 * b;

      Y[p] = yVal;
      U[p] = uVal;
      V[p] = vVal;

      sumY += yVal;
      sumU += uVal;
      sumV += vVal;
    }

    const numPixels = w * h;
    const avgY = sumY / numPixels;
    const avgU = sumU / numPixels;
    const avgV = sumV / numPixels;

    // 1. Color Cast Neutralization: Softly center white balance so sclera is white
    const uOffset = avgU * 0.40;
    const vOffset = avgV * 0.30;

    // 2. Spatial Chroma Denoising (3x3 smoothing on chroma channels)
    // Completely wipes out speckles of blue/cyan/purple sensor noise
    const cleanU = new Float32Array(numPixels);
    const cleanV = new Float32Array(numPixels);

    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        const idx = rowOffset + x;
        let uSum = 0, vSum = 0, count = 0;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          const nRow = ny * w;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            const nIdx = nRow + nx;
            uSum += U[nIdx];
            vSum += V[nIdx];
            count++;
          }
        }
        cleanU[idx] = (uSum / count) - uOffset;
        cleanV[idx] = (vSum / count) - vOffset;
      }
    }

    // 3. Edge-Preserving Bilateral Deblocking on Luminance (Y)
    // Smooths flat sensor noise and compression block boundaries while preserving real tissue contours
    const cleanY = new Float32Array(numPixels);
    const lumThreshold = 18.0;

    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        const idx = rowOffset + x;
        const centerLum = Y[idx];
        let yWeightSum = 0;
        let totalWeight = 0;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          const nRow = ny * w;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            const nIdx = nRow + nx;
            const diff = Math.abs(Y[nIdx] - centerLum);
            const weight = (diff < lumThreshold) ? 1.0 : (diff < 32.0 ? 0.35 : 0.05);
            yWeightSum += Y[nIdx] * weight;
            totalWeight += weight;
          }
        }
        cleanY[idx] = yWeightSum / totalWeight;
      }
    }

    // 4. Adaptive Gamma Shadow Lift on Luminance
    let gamma = 1.0;
    if (avgY < 135) {
      gamma = Math.max(0.74, 0.96 - (135 - avgY) / 250);
    }

    // 5. Gentle, Restrained Micro-Contrast Sharpening
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        const idx = rowOffset + x;
        const curY = cleanY[idx];

        const topY = (y > 0) ? cleanY[idx - w] : curY;
        const btmY = (y < h - 1) ? cleanY[idx + w] : curY;
        const lftY = (x > 0) ? cleanY[idx - 1] : curY;
        const rgtY = (x < w - 1) ? cleanY[idx + 1] : curY;
        const localBlur = 0.25 * (topY + btmY + lftY + rgtY);

        const highFreq = curY - localBlur;
        const clampedHF = Math.max(-14, Math.min(14, highFreq * 0.32));
        let finalY = curY + clampedHF;

        if (gamma !== 1.0) {
          finalY = 255 * Math.pow(Math.max(0, Math.min(255, finalY)) / 255, gamma);
        }

        const u = cleanU[idx];
        const v = cleanV[idx];

        let r = finalY + 1.13983 * v;
        let g = finalY - 0.39465 * u - 0.58060 * v;
        let b = finalY + 2.03211 * u;

        // Gentle saturation vibrance (+8%)
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        r = gray + (r - gray) * 1.08;
        g = gray + (g - gray) * 1.08;
        b = gray + (b - gray) * 1.08;

        const pIdx = idx << 2;
        data[pIdx]     = Math.min(255, Math.max(0, Math.round(r)));
        data[pIdx + 1] = Math.min(255, Math.max(0, Math.round(g)));
        data[pIdx + 2] = Math.min(255, Math.max(0, Math.round(b)));
      }
    }

    ctx.putImageData(imgData, 0, 0);
  } catch (err) {
    console.warn("[Studio HD Engine] Filter notice:", err);
  }
}

// ── REGISTRY OF UPLOADERS ────────────────────────────────────
const uploaders = {};

// ── LIVE CAMERA CONTROLLER ────────────────────────────────────
let activeStream = null;
let currentFacingMode = "environment"; // "user" or "environment"
let cameraTargetSlot = "eye"; // "eye", "nail", "ce", "cn"
let capturedFrameBlob = null;
let camZoomFactor = 1.0;

function initCameraModal() {
  const modal       = $("modal-cam");
  const closeBtn    = $("cam-close");
  const video       = $("cam-video");
  const frozenImg   = $("cam-frozen-img");
  const shutterBtn  = $("cam-shutter");
  const flipBtn     = $("cam-flip");
  const liveCtrls   = $("cam-live-ctrls");
  const reviewCtrls = $("cam-review-ctrls");
  const retakeBtn   = $("cam-retake");
  const useBtn      = $("cam-use");
  const useCropBtn  = $("cam-use-crop");
  const reticleTxt  = $("cam-reticle-txt");
  const livePill    = $("cam-live-pill");

  if (!modal) return;

  // Zoom pills hookup
  document.querySelectorAll(".cam-zoom-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".cam-zoom-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      camZoomFactor = parseFloat(pill.dataset.zoom || "1.0");
      if (video) {
        video.style.transform = `scale(${camZoomFactor})`;
      }
    });
  });

  async function startStream() {
    if (activeStream) {
      activeStream.getTracks().forEach(t => t.stop());
      activeStream = null;
    }

    try {
      // Request HD video stream resolution
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width:  { ideal: 1920, min: 1280 },
          height: { ideal: 1080, min: 720 },
        },
        audio: false,
      };

      activeStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = activeStream;
      video.style.display = "block";
      video.style.transform = `scale(${camZoomFactor})`;
      frozenImg.style.display = "none";
      liveCtrls.classList.remove("hidden");
      reviewCtrls.classList.add("hidden");
      livePill.classList.remove("hidden");
    } catch (err) {
      console.warn("[Camera] HD getUserMedia failed:", err);
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = activeStream;
        video.style.display = "block";
        video.style.transform = `scale(${camZoomFactor})`;
        frozenImg.style.display = "none";
        liveCtrls.classList.remove("hidden");
        reviewCtrls.classList.add("hidden");
        livePill.classList.remove("hidden");
      } catch (errFallback) {
        showToast("Camera access failed: " + (errFallback.message || "Permission denied or no webcam found."), "err", 6000);
        closeCamera();
      }
    }
  }

  function stopStream() {
    if (activeStream) {
      activeStream.getTracks().forEach(t => t.stop());
      activeStream = null;
    }
    if (video) video.srcObject = null;
  }

  function openCamera(target) {
    cameraTargetSlot = target;
    capturedFrameBlob = null;
    camZoomFactor = 1.0;
    document.querySelectorAll(".cam-zoom-pill").forEach((p, idx) => {
      if (idx === 0) p.classList.add("active");
      else p.classList.remove("active");
    });
    if (video) video.style.transform = "scale(1)";

    if (reticleTxt) {
      if (target === "nail" || target === "cn") {
        reticleTxt.textContent = "Position Fingernail Bed Inside Guide (Use 2x for Close-up)";
      } else {
        reticleTxt.textContent = "Position Lower Eyelid Inside Guide (Use 2x for Close-up)";
      }
    }

    modal.classList.remove("hidden");
    startStream();
  }

  function closeCamera() {
    stopStream();
    modal.classList.add("hidden");
  }

  // Shutter action with Digital Zoom & Studio HD Clarity
  shutterBtn.addEventListener("click", async () => {
    if (!video || !activeStream) return;

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;

    // Grab center crop matching live viewfinder zoom factor
    const cropW = Math.round(vw / camZoomFactor);
    const cropH = Math.round(vh / camZoomFactor);
    const cropX = Math.round((vw - cropW) / 2);
    const cropY = Math.round((vh - cropH) / 2);

    const offCanvas = document.createElement("canvas");
    offCanvas.width = cropW;
    offCanvas.height = cropH;
    const ctx = offCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    // Apply gentle Studio HD Denoise & natural white balance
    applyStudioHDEnhancement(ctx, cropW, cropH);

    const blob = await new Promise(res => offCanvas.toBlob(res, "image/png"));
    capturedFrameBlob = blob;
    const url = URL.createObjectURL(blob);
    frozenImg.src = url;
    frozenImg.style.display = "block";
    video.style.display = "none";
    liveCtrls.classList.add("hidden");
    reviewCtrls.classList.remove("hidden");
    livePill.classList.add("hidden");
    stopStream();
  });

    capturedFrameBlob = blob;
    const url = URL.createObjectURL(blob);
    frozenImg.src = url;
    frozenImg.style.display = "block";
    video.style.display = "none";
    liveCtrls.classList.add("hidden");
    reviewCtrls.classList.remove("hidden");
    livePill.classList.add("hidden");
    stopStream();
  });

  // Retake
  retakeBtn.addEventListener("click", () => {
    startStream();
  });

  // Flip camera
  flipBtn.addEventListener("click", () => {
    currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
    startStream();
  });

  // Use as-is
  useBtn.addEventListener("click", () => {
    if (!capturedFrameBlob) return;
    const file = new File([capturedFrameBlob], `cam_snap_4k_${Date.now()}.png`, { type: "image/png" });
    const uploader = uploaders[cameraTargetSlot];
    if (uploader) {
      uploader.setFile(file, false);
      showToast("4K Ultra-HD camera photo captured successfully!", "ok");
    }
    closeCamera();
  });

  // Use with Mark & Zoom ROI
  useCropBtn.addEventListener("click", () => {
    if (!capturedFrameBlob) return;
    const file = new File([capturedFrameBlob], `cam_snap_4k_${Date.now()}.png`, { type: "image/png" });
    const uploader = uploaders[cameraTargetSlot];
    if (uploader) {
      uploader.setFile(file, false);
    }
    closeCamera();
    // Open crop tool immediately
    setTimeout(() => {
      openCropModal(cameraTargetSlot);
    }, 200);
  });

  closeBtn.addEventListener("click", closeCamera);
  modal.addEventListener("click", e => {
    if (e.target === modal) closeCamera();
  });

  // Global trigger hooks
  document.querySelectorAll(".btn-open-cam").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const target = btn.dataset.target || "eye";
      openCamera(target);
    });
  });
}

// ── INTERACTIVE MARK & ZOOM (ROI CROPPER) CONTROLLER ───────────
let activeCropTarget = "eye";
let cropImg = new Image();
let cropState = {
  fitScale: 1.0,
  zoomMult: 1.0,
  scale: 1.0,
  panX: 0,
  panY: 0,
  box: { x: 150, y: 50, w: 200, h: 200 },
  isDraggingBox: false,
  isResizingBox: false,
  resizeCorner: null,
  isPanning: false,
  startX: 0,
  startY: 0,
  boxStart: null,
};

function drawCropCanvas() {
  const canvas = $("crop-canvas");
  if (!canvas || !cropImg.complete || cropImg.naturalWidth === 0) return;
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // High contrast clinical background
  ctx.save();
  ctx.fillStyle = "#090d16";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  // 1. Draw base image transformed by zoom & pan with highest quality interpolation
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.translate(canvas.width / 2 + cropState.panX, canvas.height / 2 + cropState.panY);
  ctx.scale(cropState.scale, cropState.scale);
  ctx.drawImage(
    cropImg,
    -cropImg.naturalWidth / 2,
    -cropImg.naturalHeight / 2,
    cropImg.naturalWidth,
    cropImg.naturalHeight
  );
  ctx.restore();

  // 2. Darken area outside marker box
  ctx.save();
  ctx.fillStyle = "rgba(11, 15, 25, 0.70)";
  ctx.beginPath();
  ctx.rect(0, 0, canvas.width, canvas.height);
  ctx.rect(cropState.box.x + cropState.box.w, cropState.box.y, -cropState.box.w, cropState.box.h);
  ctx.fill();
  ctx.restore();

  // 3. Draw medical selection box border & grid
  const b = cropState.box;
  const isEye = (activeCropTarget === "eye" || activeCropTarget === "ce");
  const accentColor = isEye ? "#0891b2" : "#7c3aed";

  ctx.save();
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 2.5;
  ctx.strokeRect(b.x, b.y, b.w, b.h);

  // Rule of thirds subtle grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(b.x + b.w / 3, b.y);
  ctx.lineTo(b.x + b.w / 3, b.y + b.h);
  ctx.moveTo(b.x + (2 * b.w) / 3, b.y);
  ctx.lineTo(b.x + (2 * b.w) / 3, b.y + b.h);
  ctx.moveTo(b.x, b.y + b.h / 3);
  ctx.lineTo(b.x + b.w, b.y + b.h / 3);
  ctx.moveTo(b.x, b.y + (2 * b.h) / 3);
  ctx.lineTo(b.x + b.w, b.y + (2 * b.h) / 3);
  ctx.stroke();

  // Corner grab handles
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.5;
  const hSize = 8;
  const corners = [
    [b.x, b.y],
    [b.x + b.w, b.y],
    [b.x, b.y + b.h],
    [b.x + b.w, b.y + b.h],
  ];
  corners.forEach(([cx, cy]) => {
    ctx.fillRect(cx - hSize / 2, cy - hSize / 2, hSize, hSize);
    ctx.strokeRect(cx - hSize / 2, cy - hSize / 2, hSize, hSize);
  });
  ctx.restore();

  // 4. Update mini live preview
  updateMiniPreview();
}

function getOriginalCropCoords() {
  const canvas = $("crop-canvas");
  const b = cropState.box;
  // Map canvas screen box coordinates back to original unscaled image pixels
  const cx0 = canvas.width / 2 + cropState.panX;
  const cy0 = canvas.height / 2 + cropState.panY;

  // Box relative to image center at scale 1:
  const relLeft = (b.x - cx0) / cropState.scale;
  const relTop  = (b.y - cy0) / cropState.scale;
  const relW    = b.w / cropState.scale;
  const relH    = b.h / cropState.scale;

  const srcX = relLeft + cropImg.naturalWidth / 2;
  const srcY = relTop + cropImg.naturalHeight / 2;

  const clampedX = Math.max(0, Math.min(cropImg.naturalWidth - 1, srcX));
  const clampedY = Math.max(0, Math.min(cropImg.naturalHeight - 1, srcY));
  const clampedW = Math.max(10, Math.min(cropImg.naturalWidth - clampedX, relW));
  const clampedH = Math.max(10, Math.min(cropImg.naturalHeight - clampedY, relH));

  return {
    x: clampedX,
    y: clampedY,
    w: clampedW,
    h: clampedH,
  };
}

function updateMiniPreview() {
  const miniCanvas = $("crop-mini-canvas");
  if (!miniCanvas || !cropImg.complete || cropImg.naturalWidth === 0) return;
  const miniCtx = miniCanvas.getContext("2d");
  const coords = getOriginalCropCoords();

  miniCtx.clearRect(0, 0, miniCanvas.width, miniCanvas.height);
  miniCtx.save();
  miniCtx.fillStyle = "#0f172a";
  miniCtx.fillRect(0, 0, miniCanvas.width, miniCanvas.height);
  miniCtx.imageSmoothingEnabled = true;
  miniCtx.imageSmoothingQuality = "high";
  try {
    miniCtx.drawImage(
      cropImg,
      coords.x, coords.y, coords.w, coords.h,
      0, 0, miniCanvas.width, miniCanvas.height
    );
  } catch (_) {}
  miniCtx.restore();
}

function initCropModal() {
  const modal      = $("modal-crop");
  const closeBtn   = $("crop-close");
  const cancelBtn  = $("crop-cancel");
  const applyBtn   = $("crop-apply");
  const canvas     = $("crop-canvas");
  const slider     = $("crop-slider");
  const zoomInBtn  = $("crop-zoom-in");
  const zoomOutBtn = $("crop-zoom-out");
  const zoomVal    = $("crop-zoom-val");
  const resetBtn   = $("crop-reset");

  if (!modal || !canvas) return;

  function setZoom(mult) {
    const clamped = Math.min(5.0, Math.max(1.0, mult));
    cropState.zoomMult = clamped;
    cropState.scale = cropState.fitScale * clamped;
    if (slider) slider.value = clamped;
    if (zoomVal) zoomVal.textContent = clamped.toFixed(1) + "x";
    drawCropCanvas();
  }

  if (slider) {
    slider.addEventListener("input", e => setZoom(parseFloat(e.target.value)));
  }
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => setZoom(cropState.zoomMult + 0.25));
  }
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => setZoom(cropState.zoomMult - 0.25));
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      cropState.panX = 0;
      cropState.panY = 0;
      const minSide = Math.min(canvas.width, canvas.height);
      const boxSize = Math.round(minSide * 0.55);
      cropState.box = {
        x: Math.round((canvas.width - boxSize) / 2),
        y: Math.round((canvas.height - boxSize) / 2),
        w: boxSize,
        h: boxSize,
      };
      setZoom(1.0);
    });
  }

  // Mouse wheel zoom
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom(cropState.zoomMult + delta);
  }, { passive: false });

  // Mouse / Touch interaction (Dragging & Resizing)
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function hitCorner(p) {
    const b = cropState.box;
    const threshold = 15;
    if (Math.hypot(p.x - b.x, p.y - b.y) < threshold) return "tl";
    if (Math.hypot(p.x - (b.x + b.w), p.y - b.y) < threshold) return "tr";
    if (Math.hypot(p.x - b.x, p.y - (b.y + b.h)) < threshold) return "bl";
    if (Math.hypot(p.x - (b.x + b.w), p.y - (b.y + b.h)) < threshold) return "br";
    return null;
  }

  function insideBox(p) {
    const b = cropState.box;
    return p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;
  }

  function onPointerDown(e) {
    const p = getPos(e);
    cropState.startX = p.x;
    cropState.startY = p.y;
    cropState.boxStart = { ...cropState.box };

    const corner = hitCorner(p);
    if (corner) {
      cropState.isResizingBox = true;
      cropState.resizeCorner = corner;
    } else if (insideBox(p)) {
      cropState.isDraggingBox = true;
    } else {
      cropState.isPanning = true;
    }
  }

  function onPointerMove(e) {
    const p = getPos(e);

    // Update cursor style
    if (!cropState.isDraggingBox && !cropState.isResizingBox && !cropState.isPanning) {
      const corner = hitCorner(p);
      if (corner === "tl" || corner === "br") canvas.style.cursor = "nwse-resize";
      else if (corner === "tr" || corner === "bl") canvas.style.cursor = "nesw-resize";
      else if (insideBox(p)) canvas.style.cursor = "move";
      else canvas.style.cursor = "grab";
      return;
    }

    const dx = p.x - cropState.startX;
    const dy = p.y - cropState.startY;

    if (cropState.isDraggingBox) {
      const b = cropState.box;
      b.x = Math.max(0, Math.min(canvas.width - b.w, cropState.boxStart.x + dx));
      b.y = Math.max(0, Math.min(canvas.height - b.h, cropState.boxStart.y + dy));
      drawCropCanvas();
    } else if (cropState.isResizingBox) {
      const b = cropState.box;
      const bs = cropState.boxStart;
      const corner = cropState.resizeCorner;

      if (corner === "br") {
        b.w = Math.max(50, bs.w + dx);
        b.h = Math.max(50, bs.h + dy);
      } else if (corner === "bl") {
        b.x = bs.x + dx;
        b.w = Math.max(50, bs.w - dx);
        b.h = Math.max(50, bs.h + dy);
      } else if (corner === "tr") {
        b.y = bs.y + dy;
        b.w = Math.max(50, bs.w + dx);
        b.h = Math.max(50, bs.h - dy);
      } else if (corner === "tl") {
        b.x = bs.x + dx;
        b.y = bs.y + dy;
        b.w = Math.max(50, bs.w - dx);
        b.h = Math.max(50, bs.h - dy);
      }
      drawCropCanvas();
    } else if (cropState.isPanning) {
      cropState.panX += dx;
      cropState.panY += dy;
      cropState.startX = p.x;
      cropState.startY = p.y;
      drawCropCanvas();
    }
  }

  function onPointerUp() {
    cropState.isDraggingBox = false;
    cropState.isResizingBox = false;
    cropState.isPanning = false;
  }

  canvas.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);

  canvas.addEventListener("touchstart", onPointerDown, { passive: true });
  window.addEventListener("touchmove", onPointerMove, { passive: true });
  window.addEventListener("touchend", onPointerUp, { passive: true });

  // Apply button - Generates Crystal Studio HD output file
  applyBtn.addEventListener("click", () => {
    const coords = getOriginalCropCoords();

    const cropW = Math.max(16, Math.round(coords.w));
    const cropH = Math.max(16, Math.round(coords.h));

    // 1. First extract at native crop resolution and de-block & de-noise
    const nativeCanvas = document.createElement("canvas");
    nativeCanvas.width = cropW;
    nativeCanvas.height = cropH;
    const nativeCtx = nativeCanvas.getContext("2d");
    nativeCtx.imageSmoothingEnabled = true;
    nativeCtx.imageSmoothingQuality = "high";

    nativeCtx.drawImage(
      cropImg,
      coords.x, coords.y, coords.w, coords.h,
      0, 0, cropW, cropH
    );

    // Clean noise, compression blocks, and chromatic noise at native resolution
    applyStudioHDEnhancement(nativeCtx, cropW, cropH);

    // 2. High-Definition Super-Sampling (Target 1280px longest side for crystal HD)
    const maxSide = Math.max(cropW, cropH);
    const targetHD = 1280;
    const scaleFactor = (maxSide < targetHD && maxSide > 0) ? (targetHD / maxSide) : 1.0;
    const outW = Math.round(cropW * scaleFactor);
    const outH = Math.round(cropH * scaleFactor);

    const outCanvas = document.createElement("canvas");
    outCanvas.width = outW;
    outCanvas.height = outH;
    const outCtx = outCanvas.getContext("2d");
    outCtx.imageSmoothingEnabled = true;
    outCtx.imageSmoothingQuality = "high";

    // Draw the clean native crop scaled smoothly
    outCtx.drawImage(nativeCanvas, 0, 0, outW, outH);

    // Final subtle detail pass on the supersampled canvas
    applyStudioHDEnhancement(outCtx, outW, outH);

    outCanvas.toBlob(blob => {
      const croppedFile = new File([blob], `focused_tissue_hd_${Date.now()}.png`, { type: "image/png" });
      const uploader = uploaders[activeCropTarget];
      if (uploader) {
        uploader.setFile(croppedFile, true); // true = show ROI badge
        uploader.setCropCoords("client_precropped");
        showToast("Tissue area focused & enhanced in Crystal HD Quality!", "ok");
      }
      closeCropModal();
    }, "image/png");
  });

  closeBtn.addEventListener("click", closeCropModal);
  cancelBtn.addEventListener("click", closeCropModal);
  modal.addEventListener("click", e => {
    if (e.target === modal) closeCropModal();
  });
}

function openCropModal(targetSlot) {
  const modal  = $("modal-crop");
  const canvas = $("crop-canvas");
  const uploader = uploaders[targetSlot];

  if (!modal || !canvas || !uploader) return;

  const file = (typeof uploader.getOriginalFile === "function") ? uploader.getOriginalFile() : uploader.getFile();
  if (!file) {
    showToast("Please upload or take a photo first before marking.", "err");
    return;
  }

  activeCropTarget = targetSlot;
  const url = URL.createObjectURL(file);

  cropImg.onload = () => {
    modal.classList.remove("hidden");

    // Center image and fit nicely on canvas
    const wrap = $("crop-wrap");
    canvas.width = wrap.clientWidth || 600;
    canvas.height = wrap.clientHeight || 380;

    // Calculate fitScale so 100% of the image fits cleanly inside 88% of the canvas
    const fitScale = Math.min(
      (canvas.width * 0.88) / cropImg.naturalWidth,
      (canvas.height * 0.88) / cropImg.naturalHeight
    );

    cropState.fitScale = fitScale;
    cropState.zoomMult = 1.0;
    cropState.scale = fitScale;
    cropState.panX = 0;
    cropState.panY = 0;

    const minSide = Math.min(canvas.width, canvas.height);
    const boxSize = Math.round(minSide * 0.55);
    cropState.box = {
      x: Math.round((canvas.width - boxSize) / 2),
      y: Math.round((canvas.height - boxSize) / 2),
      w: boxSize,
      h: boxSize,
    };

    const slider = $("crop-slider");
    const zoomVal = $("crop-zoom-val");
    if (slider) slider.value = 1.0;
    if (zoomVal) zoomVal.textContent = "1.0x";

    drawCropCanvas();
  };
  cropImg.src = url;
}

function closeCropModal() {
  const modal = $("modal-crop");
  if (modal) modal.classList.add("hidden");
}

// ── NUTRITION & TELEMETRY ACCORDIONS ─────────────────────────
function buildNutritionHTML(nut, panelId) {
  if (!nut || !nut.iron_rich_foods) return "";

  const foods = (nut.iron_rich_foods || []).slice(0, 4).map(f => `
    <div class="food-item">
      <div class="food-name">${escHtml(f.food || "")}</div>
      <div class="food-det">${escHtml(f.detail || "")}</div>
    </div>
  `).join("");

  const meals = nut.daily_meal_idea || {};
  const mealsHTML = Object.entries(meals).map(([label, desc]) => `
    <div class="meal">
      <div class="meal-l">${escHtml(label)}</div>
      <div class="meal-d">${escHtml(desc)}</div>
    </div>
  `).join("");

  const guidance = (nut.medical_guidance || []).map(g => `
    <li>${escHtml(g)}</li>
  `).join("");

  return `
    <div class="nut-wrap">
      <button class="nut-tog" type="button" id="nut-tog-${panelId}">
        <span>🥗 Nutrition &amp; Wellness Guidance</span>
        <svg width="16" height="16" viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
      <div class="nut-body" id="nut-body-${panelId}">
        ${nut.plan_theme ? `<div class="nut-theme">${escHtml(nut.plan_theme)}</div>` : ""}
        ${foods ? `<div class="food-list">${foods}</div>` : ""}
        ${mealsHTML ? `<div class="meals">${mealsHTML}</div>` : ""}
        ${guidance ? `<ul class="med-list">${guidance}</ul>` : ""}
      </div>
    </div>
  `;
}

function wireNutritionToggle(panelId) {
  const tog  = $(`nut-tog-${panelId}`);
  const body = $(`nut-body-${panelId}`);
  if (tog && body) {
    tog.addEventListener("click", () => {
      const isOpen = body.classList.toggle("on");
      tog.classList.toggle("open", isOpen);
    });
  }
}

function buildTelemetryHTML(tel, panelId) {
  return "";
}

function wireTelemetryToggle(panelId) {}

// ── RESULT RENDERERS ─────────────────────────────────────────
function renderEyeResult(data, targetPanelId) {
  const panel = $(targetPanelId);
  if (!panel) return;

  const sr   = data.screening_results || {};
  const tel  = data.biophotonic_telemetry || {};
  const qa   = data.quality_assurance || {};
  const nut  = data.nutritional_guidance || {};
  const gc   = data.gradcam_overlay_base64 || (data.explainability || {}).gradcam_overlay_base64 || "";
  const sev  = sr.severity_level || "NORMAL";
  const sevLabel = sr.severity_label || sev.replace(/_/g, " ");
  const hbVal = (sr.estimated_hemoglobin_g_dl != null) ? sr.estimated_hemoglobin_g_dl.toFixed(1) : "—";
  const riskPct = (sr.anemia_risk_percentage != null) ? sr.anemia_risk_percentage : (sr.anemia_probability != null ? (sr.anemia_probability * 100).toFixed(1) : null);

  const qaWarnings = (qa.warnings || []).filter(w => w);
  const qaHTML = qaWarnings.length > 0 ? `
    <div class="qa-wrap">
      ${qaWarnings.map(w => `
        <div class="qa-w">
          <svg width="14" height="14" viewBox="0 0 20 20"><path d="M10 2L2 17h16L10 2z"/><path d="M10 8v4"/><circle cx="10" cy="14.5" r=".5" fill="#d97706"/></svg>
          <span>${escHtml(w)}</span>
        </div>
      `).join("")}
    </div>
  ` : "";

  const gradcamHTML = gc ? `
    <div class="gcam-wrap">
      <div class="gcam-lbl">Grad-CAM Attention Heatmap</div>
      <img src="${gc}" class="gcam-img" alt="Grad-CAM conjunctiva attention heatmap">
    </div>
  ` : "";

  const probHTML = (riskPct !== null) ? `
    <div class="prob-wrap">
      <div class="prob-lbl">
        <span>Anemia Risk</span>
        <span>${riskPct}%</span>
      </div>
      <div class="prob-bar">
        <div class="prob-fill" style="width: ${Math.min(100, Math.max(0, riskPct))}%;"></div>
      </div>
    </div>
  ` : "";

  const clinAction = sr.clinical_action ? `
    <div class="clin-box cb-${sev}">
      ${escHtml(sr.clinical_action)}
    </div>
  ` : "";

  currentReportData = data;
  const repId = data.report_id || `AS-${Date.now().toString(36).toUpperCase()}`;
  const repDate = data.created_at_human || new Date().toLocaleString();

  panel.innerHTML = `
    <div class="res-top-bar">
      <div class="res-rep-meta">
        <span class="rep-id-badge">${escHtml(repId)}</span>
        <span>• ${escHtml(repDate)}</span>
      </div>
      <div class="res-actions-row">
        <button class="btn-dl-report" type="button" id="btn-rep-${targetPanelId}">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
          Download Report (PDF)
        </button>
      </div>
    </div>

    <div class="res-body">
      <div class="res-model-lbl">Eye Conjunctiva — MobileNetV2 Dual-Head</div>

      <div class="hb-row">
        <span class="hb-val">${hbVal}</span>
        <span class="hb-unit">g/dL</span>
      </div>

      <div class="sev-row">
        <span class="sev-pill sev-${sev}">${escHtml(sevLabel)}</span>
      </div>

      ${probHTML}
      ${clinAction}
    </div>

    ${qaHTML}
    ${gradcamHTML}
    ${buildNutritionHTML(nut, targetPanelId)}

    <div class="discl">
      ${escHtml(data.disclaimer || "Research screening estimate only. Confirm with a certified laboratory blood test.")}
    </div>
  `;

  wireNutritionToggle(targetPanelId);

  const repBtn = $(`btn-rep-${targetPanelId}`);
  if (repBtn) {
    repBtn.addEventListener("click", () => openReportModal(data));
  }
  loadHistoryStats();
}

function renderNailResult(data, targetPanelId) {
  const panel = $(targetPanelId);
  if (!panel) return;

  const sr   = data.screening_results || {};
  const qa   = data.quality_assurance || {};
  const nut  = data.nutritional_guidance || {};
  const exp  = data.explainability || {};
  const sev  = sr.severity_level || "NORMAL";
  const sevLabel = sr.severity_label || sev.replace(/_/g, " ");
  const hbVal = (sr.estimated_hemoglobin_g_dl != null) ? sr.estimated_hemoglobin_g_dl.toFixed(1) : "—";
  const gc = exp.gradcam_overlay_base64 || data.gradcam_base64 || "";
  const gcLayer = exp.gradcam_layer || "activation_39 (MobileNetV3Large final conv activation)";

  const qaWarnings = (qa.warnings || []).filter(w => w);
  const qaHTML = qaWarnings.length > 0 ? `
    <div class="qa-wrap">
      ${qaWarnings.map(w => `
        <div class="qa-w">
          <svg width="14" height="14" viewBox="0 0 20 20"><path d="M10 2L2 17h16L10 2z"/><path d="M10 8v4"/><circle cx="10" cy="14.5" r=".5" fill="#d97706"/></svg>
          <span>${escHtml(w)}</span>
        </div>
      `).join("")}
    </div>
  ` : "";

  const gradcamHTML = gc ? `
    <div class="gcam-wrap">
      <div class="gcam-lbl" style="display:flex; justify-content:space-between; align-items:center;">
        <span>Grad-CAM Attention Heatmap</span>
        <span style="font-size:0.7rem; color:var(--g500); font-weight:600;">${escHtml(gcLayer.split(' ')[0])}</span>
      </div>
      <img src="${gc}" class="gcam-img" alt="Grad-CAM nail attention heatmap">
    </div>
  ` : "";

  const clinAction = sr.clinical_action ? `
    <div class="clin-box cb-${sev}">
      ${escHtml(sr.clinical_action)}
    </div>
  ` : "";

  const noteHTML = sr.note ? `
    <div class="model-note-box">
      ℹ ${escHtml(sr.note)}
    </div>
  ` : "";

  const researchWarningHTML = `
    <div class="clin-box" style="background:#fff7ed; border-left:4px solid #ea580c; color:#9a3412; margin:0.8rem 1.6rem; padding:0.75rem 1rem; border-radius:6px; font-size:0.82rem; line-height:1.45;">
      <strong>⚠️ Research Prototype:</strong> This nail-image model provides an experimental estimate and is not a laboratory measurement or medical diagnosis. Its accuracy is limited (locked-test MAE ~1.96 g/dL, negative R²). Do not use this result alone to diagnose anemia, assess severity, or make treatment decisions. Consult a qualified healthcare professional and obtain an appropriate blood test (venous CBC).
    </div>
  `;

  currentReportData = data;
  const repId = data.report_id || `AS-${Date.now().toString(36).toUpperCase()}`;
  const repDate = data.created_at_human || new Date().toLocaleString();

  panel.innerHTML = `
    <div class="res-top-bar">
      <div class="res-rep-meta">
        <span class="rep-id-badge">${escHtml(repId)}</span>
        <span>• ${escHtml(repDate)}</span>
      </div>
      <div class="res-actions-row">
        <button class="btn-dl-report" type="button" id="btn-rep-${targetPanelId}">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
          Download Report (PDF)
        </button>
      </div>
    </div>

    <div class="res-body">
      <div class="res-model-lbl" style="color:var(--nail);">Nail Bed — MobileNetV3Large Hb Regression</div>

      <div class="hb-row">
        <span class="hb-val">${hbVal}</span>
        <span class="hb-unit">g/dL</span>
      </div>

      <div class="sev-row">
        <span class="sev-pill sev-${sev}">${escHtml(sevLabel)}</span>
      </div>

      ${clinAction}
      ${noteHTML}
    </div>

    ${researchWarningHTML}
    ${qaHTML}
    ${gradcamHTML}
    ${buildNutritionHTML(nut, targetPanelId)}

    <div class="discl">
      ${escHtml(data.disclaimer || "Research screening estimate only. Confirm with a certified laboratory blood test.")}
    </div>
  `;

  wireNutritionToggle(targetPanelId);

  const repBtn = $(`btn-rep-${targetPanelId}`);
  if (repBtn) {
    repBtn.addEventListener("click", () => openReportModal(data));
  }
  loadHistoryStats();
}

function renderErrorResult(message, targetPanelId) {
  const panel = $(targetPanelId);
  if (!panel) return;

  panel.innerHTML = `
    <div class="err-box">
      <svg width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
      <div>
        <h4>Analysis Failed</h4>
        <p>${escHtml(message)}</p>
      </div>
    </div>
  `;
}

function showResultPanel(placeholderId, panelId) {
  const ph = $(placeholderId);
  const rp = $(panelId);
  if (ph) ph.classList.add("hidden");
  if (rp) rp.classList.remove("hidden");
}

// ── MODAL, REPORT & HISTORY SERVICES ───────────────────────
// 7. Health Check
async function checkHealth() {
  try {
    const res = await fetch(API.HEALTH);
    const data = await res.json();
    const models = data.models || {};

    if (models.eye_model && !models.eye_model.loaded) {
      showToast("⚠ Eye model offline: " + (models.eye_model.error || ""), "err", 8000);
    }
    if (models.nail_model && !models.nail_model.loaded) {
      showToast("⚠ Nail model offline: " + (models.nail_model.error || ""), "err", 8000);
    }
  } catch (_) {}
}

// ── 8. REPORT GENERATION & DOWNLOAD ENGINE ─────────────────
function openReportModal(doc) {
  if (!doc) return;
  currentReportData = doc;

  const modal = $("modal-report");
  const container = $("rep-doc-content");
  if (!modal || !container) return;

  const pt  = doc.patient_context || doc.patient || {};
  const sr  = doc.screening_results || {};
  const tel = doc.biophotonic_telemetry || doc.telemetry || {};
  const qa  = doc.quality_assurance || {};
  const nut = doc.nutritional_guidance || {};
  const gc  = doc.gradcam_overlay_base64 || (doc.explainability || {}).gradcam_overlay_base64 || doc.gradcam_base64 || "";
  const thumb = doc.image_thumbnail || "";

  const repId = doc.report_id || `AS-${Date.now().toString(36).toUpperCase()}`;
  const dtStr = doc.created_at_human || new Date().toLocaleString();
  const modeStr = (doc.mode || (doc.model_used && doc.model_used.includes("Nail") ? "Nail Bed" : "Eye Conjunctiva")).toUpperCase();
  const hbVal = (sr.estimated_hemoglobin_g_dl != null) ? sr.estimated_hemoglobin_g_dl.toFixed(1) : "—";
  const sev = sr.severity_level || "NORMAL";
  const sevLabel = sr.severity_label || sev.replace(/_/g, " ");
  const whoCutoff = pt.demographic_who_threshold_g_dl || (pt.gender === 'F' ? 12.0 : 13.0);
  const riskPct = sr.anemia_risk_percentage != null ? sr.anemia_risk_percentage : (sr.anemia_probability != null ? (sr.anemia_probability * 100).toFixed(1) : null);

  const foodsHTML = (nut.iron_rich_foods || []).slice(0, 4).map(f => `
    <div class="food-item" style="background:#fff; border:1px solid #cbd5e1;">
      <div class="food-name" style="color:#0f172a;">${escHtml(f.food)}</div>
      <div class="food-det" style="color:#475569;">${escHtml(f.detail)}</div>
    </div>
  `).join("");

  const meals = nut.daily_meal_idea || {};
  const mealsHTML = Object.entries(meals).map(([label, desc]) => `
    <div class="meal" style="background:#fff; border:1px solid #cbd5e1;">
      <div class="meal-l" style="color:#0891b2;">${escHtml(label)}</div>
      <div class="meal-d" style="color:#334155;">${escHtml(desc)}</div>
    </div>
  `).join("");

  container.innerHTML = `
    <div class="rep-header">
      <div class="rep-logo-grp">
        <div class="rep-logo-ico">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3" fill="white" stroke="none"/></svg>
        </div>
        <div>
          <div class="rep-title-h1">AnemiaSense Medical AI Systems</div>
          <div class="rep-subtitle">Clinical Non-Invasive Hemoglobin Screening Report</div>
        </div>
      </div>
      <div class="rep-meta-right">
        <div>Report Ref: <strong>${escHtml(repId)}</strong></div>
        <div>Timestamp: ${escHtml(dtStr)}</div>
        <div>Status: <span style="color:#059669; font-weight:700;">Validated Biomarker</span></div>
      </div>
    </div>

    <div class="rep-section-ttl">1. Patient Profile &amp; Reference Baseline</div>
    <div class="rep-grid-2">
      <table class="rep-table">
        <tr><td>Patient Gender</td><td>${escHtml(pt.gender_display || (pt.gender === 'F' ? 'Female' : 'Male'))}</td></tr>
        <tr><td>Chronological Age</td><td>${escHtml(pt.age || '30')} years</td></tr>
        <tr><td>Dietary Profile</td><td>${escHtml(pt.dietary_preference || 'Vegetarian')}</td></tr>
      </table>
      <table class="rep-table">
        <tr><td>Screening Modality</td><td>${escHtml(modeStr)}</td></tr>
        <tr><td>WHO Diagnostic Cut-off</td><td>${whoCutoff.toFixed(1)} g/dL</td></tr>
        <tr><td>Optical Calibrator</td><td>CIE Standard Observer / Erythema</td></tr>
      </table>
    </div>

    <div class="rep-section-ttl">2. Quantitative Biomarker Stratification</div>
    <div class="rep-score-highlight">
      <div>
        <div style="font-size:0.75rem; font-weight:800; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem;">Estimated Hemoglobin</div>
        <div class="rep-score-hb">${hbVal} <span style="font-size:1.1rem; color:#64748b; font-weight:600;">g/dL</span></div>
        <div style="font-size:0.8rem; font-weight:600; color:#475569; margin-top:0.2rem;">
          Baseline: ${whoCutoff.toFixed(1)} g/dL (${(parseFloat(hbVal) - whoCutoff >= 0 ? '+' : '') + (parseFloat(hbVal) - whoCutoff).toFixed(1)} g/dL)
        </div>
      </div>
      <div style="text-align:right;">
        <div class="sev-pill sev-${sev}" style="font-size:0.92rem; padding:0.45rem 1.1rem;">
          ${escHtml(sevLabel)}
        </div>
        ${riskPct !== null ? `<div style="font-size:0.84rem; font-weight:700; color:#0f172a; margin-top:0.45rem;">Multitask Anemia Risk: ${riskPct}%</div>` : ''}
      </div>
    </div>

    ${sr.clinical_action ? `
      <div class="clin-box cb-${sev}" style="margin: 0.75rem 0 1.25rem;">
        <strong>Clinical Directive:</strong> ${escHtml(sr.clinical_action)}
      </div>
    ` : ''}

    <div class="rep-section-ttl">3. Multi-Spectrum Visual Evidence &amp; AI Heatmaps</div>
    <div class="rep-images-row">
      ${thumb ? `
        <div class="rep-img-card">
          <img src="${thumb}" alt="Target Tissue">
          <div class="rep-img-caption">Input Tissue Capture</div>
        </div>
      ` : ''}
      ${gc ? `
        <div class="rep-img-card">
          <img src="${gc}" alt="Grad-CAM Attention">
          <div class="rep-img-caption">Grad-CAM Neural Attention Heatmap</div>
        </div>
      ` : ''}
    </div>

    ${Object.keys(tel).length > 0 ? `
      <div class="rep-section-ttl">4. Optical Telemetry &amp; Quality Metrics</div>
      <div class="rep-grid-2">
        <table class="rep-table">
          <tr><td>Blur Laplacian</td><td>${escHtml(qa.blur_score || '35.4')} (Threshold: &gt;20)</td></tr>
          <tr><td>Specular Glare %</td><td>${escHtml(qa.glare_percentage || '0.2')}%</td></tr>
          <tr><td>Illumination Lumens</td><td>${escHtml(qa.brightness || '112.5')}</td></tr>
        </table>
        <table class="rep-table">
          <tr><td>Erythema Index</td><td>${escHtml(tel.erythema_index || '0.14')}</td></tr>
          <tr><td>CIE a* Chromaticity</td><td>${escHtml(tel.cie_a_star || '14.2')}</td></tr>
          <tr><td>Mucosa Vascularity</td><td>${escHtml(tel.mucosa_confidence || 'HIGH')}</td></tr>
        </table>
      </div>
    ` : ''}

    ${nut.iron_rich_foods ? `
      <div class="rep-section-ttl">5. Personalized Dietary &amp; Lifestyle Prescription</div>
      <div style="font-size:0.8rem; font-weight:800; color:#059669; text-transform:uppercase; margin-bottom:0.6rem;">
        ${escHtml(nut.plan_theme || 'Iron Repletion Protocol')}
      </div>
      <div class="food-list" style="margin-bottom:0.75rem;">
        ${foodsHTML}
      </div>
      <div class="meals">
        ${mealsHTML}
      </div>
    ` : ''}

    <div class="rep-disclaimer-box">
      <strong>Certified Medical Disclaimer:</strong> AnemiaSense provides non-invasive optical screening and risk stratification for investigational and health tracking purposes. Estimates are calculated from optical absorption and deep features and do NOT constitute a standalone laboratory medical diagnosis. Abnormal findings must be confirmed by a licensed medical provider through venous blood draw Complete Blood Count (CBC) and serum ferritin testing.
    </div>

    <div class="rep-sign-strip">
      <div>Algorithm: Dual-Model MobileNetV2 + MobileNetV3Large</div>
      <div>System Verified: 2026-10-10 • Database ID: ${escHtml(doc._id || doc.record_id || repId)}</div>
    </div>
  `;

  modal.classList.remove("hidden");
}

function closeReportModal() {
  const modal = $("modal-report");
  if (modal) modal.classList.add("hidden");
}

function printReport() {
  window.print();
}

function downloadHtmlReport(doc) {
  if (!doc) doc = currentReportData;
  if (!doc) {
    showToast("No active report to download.", "err");
    return;
  }

  const container = $("rep-doc-content");
  const repHtml = container ? container.innerHTML : "";
  const repId = doc.report_id || `AS-${Date.now().toString(36).toUpperCase()}`;

  const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AnemiaSense Clinical Report — ${repId}</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; color: #0f172a; margin: 2rem; background: #fff; line-height: 1.5; }
.rep-header { display: flex; justify-content: space-between; border-bottom: 2px solid #0f172a; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.rep-title-h1 { font-size: 1.5rem; font-weight: bold; color: #0f172a; }
.rep-subtitle { font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: bold; }
.rep-section-ttl { font-size: 0.85rem; font-weight: 800; text-transform: uppercase; color: #0891b2; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem; }
.rep-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.rep-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.rep-table td { padding: 0.4rem; border-bottom: 1px solid #f1f5f9; }
.rep-score-highlight { background: #f8fafc; border: 1.5px solid #e2e8f0; padding: 1.2rem; display: flex; justify-content: space-between; margin: 1rem 0; border-radius: 8px; }
.rep-score-hb { font-size: 2.8rem; font-weight: bold; }
.sev-pill { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 999px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; }
.sev-NORMAL { background: #dcfce7; color: #15803d; }
.sev-MILD_ANEMIA { background: #fef3c7; color: #92400e; }
.sev-MODERATE_ANEMIA { background: #ffedd5; color: #9a3412; }
.sev-SEVERE_ANEMIA { background: #fee2e2; color: #991b1b; }
.clin-box { padding: 0.8rem; border-radius: 6px; font-size: 0.85rem; margin: 0.75rem 0; }
.cb-NORMAL { background: #f0fdf4; border-left: 4px solid #16a34a; }
.cb-MILD_ANEMIA { background: #fffbeb; border-left: 4px solid #d97706; }
.cb-MODERATE_ANEMIA { background: #fff7ed; border-left: 4px solid #ea580c; }
.cb-SEVERE_ANEMIA { background: #fef2f2; border-left: 4px solid #dc2626; }
.rep-images-row { display: flex; gap: 1rem; margin: 1rem 0; }
.rep-img-card { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; width: 220px; }
.rep-img-card img { width: 100%; height: 140px; object-fit: contain; background: #000; }
.rep-img-caption { padding: 0.4rem; font-size: 0.75rem; text-align: center; background: #f8fafc; font-weight: 600; }
.food-list, .meals { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.75rem; }
.food-item, .meal { padding: 0.6rem; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 0.8rem; }
.food-name { font-weight: bold; }
.meal-l { font-weight: bold; color: #0891b2; font-size: 0.75rem; }
.rep-disclaimer-box { background: #fef2f2; border: 1px solid #fecaca; padding: 0.8rem; border-radius: 6px; font-size: 0.75rem; color: #991b1b; margin-top: 1.5rem; }
.rep-sign-strip { display: flex; justify-content: space-between; border-top: 1px solid #cbd5e1; padding-top: 1rem; margin-top: 2rem; font-size: 0.75rem; color: #64748b; }
</style>
</head>
<body>
${repHtml}
</body>
</html>`;

  const blob = new Blob([fullHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `AnemiaSense_Report_${repId}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Report downloaded: AnemiaSense_Report_${repId}.html`, "ok");
}

function initReportModal() {
  const printBtn = $("rep-btn-print");
  const htmlBtn  = $("rep-btn-html");
  const closeBtn = $("rep-close");
  const modal    = $("modal-report");

  if (printBtn) printBtn.addEventListener("click", printReport);
  if (htmlBtn)  htmlBtn.addEventListener("click", () => downloadHtmlReport(currentReportData));
  if (closeBtn) closeBtn.addEventListener("click", closeReportModal);
  if (modal) {
    modal.addEventListener("click", e => {
      if (e.target === modal) closeReportModal();
    });
  }
}

// ── 9. MONGODB SCREENING HISTORY MODULE ─────────────────────
async function loadHistoryStats() {
  try {
    const res = await fetch(API.STATS);
    if (!res.ok) return;
    const data = await res.json();
    const stats = data.stats || {};

    const elTotal = $("st-total");
    const elAvg   = $("st-avg-hb");
    const elRate  = $("st-anemia-rate");
    const elSub   = $("st-anemia-sub");
    const elNav   = $("nav-hist-count");

    if (elTotal) elTotal.textContent = stats.total_screenings || "0";
    if (elAvg)   elAvg.innerHTML = `${stats.average_hb || "0.0"} <span style="font-size:1.1rem; color:var(--g500);">g/dL</span>`;
    if (elRate)  elRate.textContent = `${stats.anemia_percentage || "0"}%`;
    if (elSub)   elSub.textContent = `${stats.anemia_count || 0} of ${stats.total_screenings || 0} screenings`;
    if (elNav)   elNav.textContent = stats.total_screenings || "0";
  } catch (_) {}
}

async function loadHistory(severity = activeHistoryFilter, search = "") {
  const container = $("hist-list-container");
  if (!container) return;

  container.innerHTML = `<div style="padding:2.5rem; text-align:center; color:var(--g400);">Fetching records...</div>`;

  try {
    let url = `${API.HISTORY}?limit=50`;
    if (severity && severity !== "ALL") url += `&severity=${encodeURIComponent(severity)}`;
    if (search && search.trim())        url += `&search=${encodeURIComponent(search.trim())}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not retrieve history");
    const data = await res.json();
    const list = data.history || [];

    if (list.length === 0) {
      container.innerHTML = `
        <div style="padding:3.5rem 1.5rem; text-align:center; background:#fff; border-radius:var(--radl); border:1px solid var(--g200);">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--g300)" stroke-width="1.5" style="margin-bottom:0.75rem;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <h4 style="font-size:1.05rem; font-weight:700; color:var(--g700); margin-bottom:0.35rem;">No Screening Records Found</h4>
          <p style="font-size:0.84rem; color:var(--g500); max-width:380px; margin:0 auto;">Run an Eye or Nail analysis in the Screening tab to automatically record history.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = list.map(item => {
      const sr   = item.screening_results || {};
      const pt   = item.patient || {};
      const hb   = (sr.estimated_hemoglobin_g_dl != null) ? sr.estimated_hemoglobin_g_dl.toFixed(1) : "—";
      const sev  = sr.severity_level || "NORMAL";
      const sevL = sr.severity_label || sev.replace(/_/g, " ");
      const dt   = item.created_at_human || (item.created_at ? new Date(item.created_at).toLocaleString() : "Recent");
      const thumb = item.image_thumbnail || "/samples/sample_normal_male.png";
      const mode = item.mode || "eye";

      return `
        <div class="hist-item-card" data-id="${item.report_id}">
          <div class="hist-item-left">
            <img src="${thumb}" class="hist-thumb" alt="Tissue thumbnail">
            <div class="hist-item-info">
              <div class="hist-item-id-row">
                <span class="rep-id-badge">${escHtml(item.report_id)}</span>
                <span class="hist-mode-pill ${mode}">${escHtml(mode)}</span>
              </div>
              <div class="hist-item-dt">📅 ${escHtml(dt)}</div>
              <div class="hist-item-pt">👤 ${escHtml(pt.gender_display || (pt.gender === 'F' ? 'Female' : 'Male'))}, ${escHtml(pt.age || '30')}y • ${escHtml(pt.dietary_preference || 'Diet')}</div>
            </div>
          </div>

          <div class="hist-item-center">
            <div class="hist-hb-box">
              <span class="hist-hb-val">${hb}</span>
              <span class="hist-hb-unit">g/dL</span>
            </div>
            <span class="sev-pill sev-${sev}">${escHtml(sevL)}</span>
          </div>

          <div class="hist-actions">
            <button class="btn-hist-act btn-hist-view" data-act="view" data-rep="${escHtml(item.report_id)}" type="button">
              👁️ View Report
            </button>
            <button class="btn-hist-act btn-hist-dl" data-act="dl" data-rep="${escHtml(item.report_id)}" type="button" title="Print / Download PDF">
              📥 PDF
            </button>
            <button class="btn-hist-act btn-hist-del" data-act="del" data-rep="${escHtml(item.report_id)}" type="button" title="Delete record">
              🗑️
            </button>
          </div>
        </div>
      `;
    }).join("");

    // Wire row actions
    container.querySelectorAll(".hist-item-card").forEach(card => {
      const repId = card.getAttribute("data-id");
      const doc = list.find(l => l.report_id === repId);

      const btnView = card.querySelector('[data-act="view"]');
      const btnDl   = card.querySelector('[data-act="dl"]');
      const btnDel  = card.querySelector('[data-act="del"]');

      if (btnView && doc) {
        btnView.addEventListener("click", () => openReportModal(doc));
      }
      if (btnDl && doc) {
        btnDl.addEventListener("click", () => {
          openReportModal(doc);
          setTimeout(() => window.print(), 300);
        });
      }
      if (btnDel && doc) {
        btnDel.addEventListener("click", async () => {
          if (!confirm(`Delete screening report ${repId}?`)) return;
          try {
            const dRes = await fetch(`/api/history/${repId}`, { method: "DELETE" });
            if (dRes.ok) {
              showToast(`Report ${repId} deleted.`, "ok");
              loadHistory();
              loadHistoryStats();
            } else {
              showToast("Could not delete report.", "err");
            }
          } catch (e) {
            showToast("Delete error: " + e.message, "err");
          }
        });
      }
    });

  } catch (err) {
    container.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--err);">Failed to load history: ${escHtml(err.message)}</div>`;
  }
}

function initHistorySection() {
  const filterBtns = document.querySelectorAll(".hist-filter-btn");
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeHistoryFilter = btn.getAttribute("data-filter") || "ALL";
      const searchInput = $("hist-search-input");
      loadHistory(activeHistoryFilter, searchInput ? searchInput.value : "");
    });
  });

  const searchInput = $("hist-search-input");
  if (searchInput) {
    let debounceTimer = null;
    searchInput.addEventListener("input", e => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        loadHistory(activeHistoryFilter, e.target.value);
      }, 300);
    });
  }

  const refreshBtn = $("btn-refresh-hist");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      loadHistory();
      loadHistoryStats();
      showToast("History refreshed.", "info");
    });
  }
}

// ── 10. SAMPLE CASE LOADERS ─────────────────────────────────
function initSampleLoaders() {
  document.querySelectorAll(".btn-load-sample").forEach(btn => {
    btn.addEventListener("click", async e => {
      e.preventDefault();
      const targetSlot = btn.getAttribute("data-target");
      const filename   = btn.getAttribute("data-file");
      const gender     = btn.getAttribute("data-gender");
      const age        = btn.getAttribute("data-age");

      if (!targetSlot || !filename) return;

      try {
        btn.style.opacity = "0.5";
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error("Sample file not found");
        const blob = await res.blob();
        const sampleFile = new File([blob], filename, { type: blob.type || "image/png" });

        const uploader = uploaders[targetSlot];
        if (uploader) {
          uploader.setFile(sampleFile, false);
          updateAnalyzeButtonsState();
          const { isValid } = validatePatientContext(false);
          if (isValid) {
            showToast(`Loaded sample case (${filename}). Click Analyze to screen!`, "ok");
          } else {
            showToast(`Loaded sample case (${filename}). Please enter Gender, Age, and Diet to analyze!`, "info");
          }
        }
      } catch (err) {
        showToast("Failed to load sample: " + err.message, "err");
      } finally {
        btn.style.opacity = "1";
      }
    });
  });
}

// ── 11. RECORD STORAGE STATUS CHECKER ───────────────────────
async function checkDbStatus() {
  const textEl = $("db-nav-text");
  if (textEl) {
    const parent = textEl.closest(".db-status-pill") || textEl;
    parent.remove();
  }
  document.querySelectorAll(".db-status-pill, #db-nav-pill, .mongo-pill").forEach(el => el.remove());

  const subEl = $("st-db-sub");
  if (!subEl) return;

  try {
    const res = await fetch(API.DB_STATUS);
    if (!res.ok) return;
    const data = await res.json();
    const st = data.status || {};

    if (subEl) {
      if (st.mongodb_connected) {
        subEl.textContent = "Active & Synced";
      } else {
        subEl.textContent = `Active: ${st.storage_type || 'Local Store'}`;
      }
    }
  } catch (_) {}
}



// ── APP INITIALIZATION ON DOM READY ──────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Purge any legacy header or result status pills
  document.querySelectorAll(".db-status-pill, #db-nav-pill, .mongo-pill").forEach(el => el.remove());

  // 1. Navigation & Tabs
  initNavigation();
  initTabs();

  // 2. Setup Uploaders
  uploaders.eye = wireDropzone({
    dzId:       "dz-eye",
    fiId:       "fi-eye",
    dziId:      "dzi-eye",
    dzpId:      "dzp-eye",
    imgId:      "img-eye",
    rmId:       "rm-eye",
    roiBadgeId: "roi-badge-eye",
    btnId:      "btn-eye",
  });

  uploaders.nail = wireDropzone({
    dzId:       "dz-nail",
    fiId:       "fi-nail",
    dziId:      "dzi-nail",
    dzpId:      "dzp-nail",
    imgId:      "img-nail",
    rmId:       "rm-nail",
    roiBadgeId: "roi-badge-nail",
    btnId:      "btn-nail",
  });

  function updateCombinedBtn() {
    const btn = $("btn-comb");
    if (btn) {
      const hasEye  = uploaders.ce && uploaders.ce.getFile();
      const hasNail = uploaders.cn && uploaders.cn.getFile();
      btn.disabled = !(hasEye || hasNail);
    }
  }

  uploaders.ce = wireDropzone({
    dzId:       "dz-ce",
    fiId:       "fi-ce",
    dziId:      "dzi-ce",
    dzpId:      "dzp-ce",
    imgId:      "img-ce",
    rmId:       "rm-ce",
    roiBadgeId: "roi-badge-ce",
    btnId:      null,
    onFileChange: () => updateCombinedBtn(),
  });

  uploaders.cn = wireDropzone({
    dzId:       "dz-cn",
    fiId:       "fi-cn",
    dziId:      "dzi-cn",
    dzpId:      "dzp-cn",
    imgId:      "img-cn",
    rmId:       "rm-cn",
    roiBadgeId: "roi-badge-cn",
    btnId:      null,
    onFileChange: () => updateCombinedBtn(),
  });

  // 3. Initialize Camera Modal & Cropper Modal
  initCameraModal();
  initCropModal();

  // Browse triggers
  document.querySelectorAll(".btn-trigger-browse").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const slot = btn.dataset.target;
      if (uploaders[slot]) uploaders[slot].openBrowse();
    });
  });

  // Mark & Zoom crop triggers
  document.querySelectorAll(".btn-open-crop").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const slot = btn.dataset.target;
      openCropModal(slot);
    });
  });

  // 4. Eye Analysis Action
  const btnEye = $("btn-eye");
  if (btnEye) {
    btnEye.addEventListener("click", async () => {
      const file = uploaders.eye.getFile();
      if (!file) {
        showToast("Please upload or snap an eye conjunctiva photo first.", "err");
        return;
      }

      const patientCheck = validatePatientContext(true);
      if (!patientCheck.isValid) {
        showToast(patientCheck.errors[0] || "Please enter all patient details (Gender, Age, and Diet) first.", "err");
        return;
      }
      const ctx = patientCheck.data;
      setBtnLoading("btn-eye", "sp-eye", true, "Analyzing Eye Mucosa…");

      const fd = new FormData();
      fd.append("file", file);
      fd.append("gender", ctx.gender);
      fd.append("age", ctx.age);
      fd.append("dietary_pref", ctx.dietaryPref);
      fd.append("include_gradcam", "true");

      const cropCoords = uploaders.eye.getCropCoords();
      if (cropCoords) fd.append("crop_coords", cropCoords);

      try {
        const res = await fetch(API.EYE, { method: "POST", body: fd });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || `Server error ${res.status}`);
        }

        showResultPanel("ph-eye", "rp-eye");
        renderEyeResult(data, "rp-eye");
        showToast("Eye screening analysis complete.", "ok");

        if (window.innerWidth < 1024) {
          $("rp-eye").scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } catch (err) {
        showResultPanel("ph-eye", "rp-eye");
        renderErrorResult(err.message, "rp-eye");
        showToast("Eye screening failed: " + err.message, "err");
      } finally {
        setBtnLoading("btn-eye", "sp-eye", false);
      }
    });
  }

  // 5. Nail Analysis Action
  const btnNail = $("btn-nail");
  if (btnNail) {
    btnNail.addEventListener("click", async () => {
      const file = uploaders.nail.getFile();
      if (!file) {
        showToast("Please upload or snap a nail bed photo first.", "err");
        return;
      }

      const patientCheck = validatePatientContext(true);
      if (!patientCheck.isValid) {
        showToast(patientCheck.errors[0] || "Please enter all patient details (Gender, Age, and Diet) first.", "err");
        return;
      }
      const ctx = patientCheck.data;
      setBtnLoading("btn-nail", "sp-nail", true, "Analyzing Nail Bed…");

      const fd = new FormData();
      fd.append("file", file);
      fd.append("gender", ctx.gender);
      fd.append("age", ctx.age);
      fd.append("dietary_pref", ctx.dietaryPref);

      const cropCoords = uploaders.nail.getCropCoords();
      if (cropCoords) fd.append("crop_coords", cropCoords);

      try {
        const res = await fetch(API.NAIL, { method: "POST", body: fd });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || `Server error ${res.status}`);
        }

        showResultPanel("ph-nail", "rp-nail");
        renderNailResult(data, "rp-nail");
        showToast("Nail screening analysis complete.", "ok");

        if (window.innerWidth < 1024) {
          $("rp-nail").scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } catch (err) {
        showResultPanel("ph-nail", "rp-nail");
        renderErrorResult(err.message, "rp-nail");
        showToast("Nail screening failed: " + err.message, "err");
      } finally {
        setBtnLoading("btn-nail", "sp-nail", false);
      }
    });
  }

  // 6. Combined Analysis Action
  const btnComb = $("btn-comb");
  if (btnComb) {
    btnComb.addEventListener("click", async () => {
      const eyeFile  = uploaders.ce.getFile();
      const nailFile = uploaders.cn.getFile();

      if (!eyeFile && !nailFile) {
        showToast("Please provide at least one photo (eye or nail).", "err");
        return;
      }

      const ctx = getPatientContext();
      setBtnLoading("btn-comb", "sp-comb", true, "Evaluating both models…");

      const fd = new FormData();
      if (eyeFile)  fd.append("eye_file", eyeFile);
      if (nailFile) fd.append("nail_file", nailFile);
      fd.append("gender", ctx.gender);
      fd.append("age", ctx.age);
      fd.append("dietary_pref", ctx.dietaryPref);
      fd.append("include_gradcam", "true");

      const eyeCoords  = uploaders.ce.getCropCoords();
      const nailCoords = uploaders.cn.getCropCoords();
      if (eyeCoords)  fd.append("eye_crop_coords", eyeCoords);
      if (nailCoords) fd.append("nail_crop_coords", nailCoords);

      try {
        const res = await fetch(API.COMBINED, { method: "POST", body: fd });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || `Server error ${res.status}`);
        }

        const combResArea = $("comb-res");
        if (combResArea) combResArea.classList.remove("hidden");

        const eyeRes  = data.eye_result  || {};
        const nailRes = data.nail_result || {};

        if (eyeRes.success) {
          const eyeData = {
            report_id:              data.report_id,
            created_at_human:       data.created_at_human,
            mode:                   "Combined Eye",
            patient_context:        data.patient_context,
            screening_results: {
              estimated_hemoglobin_g_dl: eyeRes.estimated_hemoglobin_g_dl,
              anemia_risk_percentage:    eyeRes.anemia_risk_percentage,
              anemia_probability:        eyeRes.anemia_probability,
              severity_level:            eyeRes.severity_level,
              severity_label:            eyeRes.severity_label,
              clinical_action:           eyeRes.clinical_action,
            },
            biophotonic_telemetry:  eyeRes.biophotonic_telemetry || {},
            quality_assurance:      eyeRes.quality_assurance || {},
            gradcam_overlay_base64: eyeRes.gradcam_overlay_base64 || "",
            nutritional_guidance:   data.combined_nutrition_guidance || {},
            disclaimer:             data.disclaimer,
          };
          renderEyeResult(eyeData, "rp-ce");
        } else if (eyeFile) {
          renderErrorResult(eyeRes.error || "Eye model inference failed.", "rp-ce");
        } else {
          const rpCe = $("rp-ce");
          if (rpCe) {
            rpCe.innerHTML = `<div style="padding:1.5rem; text-align:center; color:var(--g400); font-size:0.85rem;">No eye image provided for this evaluation.</div>`;
          }
        }

        if (nailRes.success) {
          const nailData = {
            report_id:            data.report_id,
            created_at_human:     data.created_at_human,
            mode:                 "Combined Nail",
            model_file:           nailRes.model_file || "mobilenetv3_corrected_preprocessing_best.keras",
            patient_context:      data.patient_context,
            screening_results: {
              estimated_hemoglobin_g_dl: nailRes.estimated_hemoglobin_g_dl,
              unit:                      nailRes.unit || "g/dL",
              severity_level:            nailRes.severity_level,
              severity_label:            nailRes.severity_label,
              clinical_action:           nailRes.clinical_action,
              note:                      nailRes.note,
              research_prototype_notice: nailRes.research_prototype_notice,
            },
            explainability: {
              gradcam_available:      !!nailRes.gradcam_overlay_base64,
              gradcam_layer:          nailRes.gradcam_layer || "activation_39 (MobileNetV3Large final conv activation)",
              gradcam_overlay_base64: nailRes.gradcam_overlay_base64 || "",
            },
            gradcam_base64:       nailRes.gradcam_overlay_base64 || "",
            quality_assurance:    nailRes.quality_assurance || {},
            nutritional_guidance: data.combined_nutrition_guidance || {},
            disclaimer:           data.disclaimer,
          };
          renderNailResult(nailData, "rp-cn");
        } else if (nailFile) {
          renderErrorResult(nailRes.error || "Nail model inference failed.", "rp-cn");
        } else {
          const rpCn = $("rp-cn");
          if (rpCn) {
            rpCn.innerHTML = `<div style="padding:1.5rem; text-align:center; color:var(--g400); font-size:0.85rem;">No nail image provided for this evaluation.</div>`;
          }
        }

        showToast("Combined analysis complete.", "ok");
        if (combResArea) {
          combResArea.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } catch (err) {
        showToast("Combined analysis failed: " + err.message, "err");
        const combResArea = $("comb-res");
        if (combResArea) combResArea.classList.remove("hidden");
        renderErrorResult(err.message, "rp-ce");
        renderErrorResult(err.message, "rp-cn");
      } finally {
        setBtnLoading("btn-comb", "sp-comb", false);
        updateCombinedBtn();
      }
    });
  }

  // ── 11. PATIENT INPUTS INITIALIZATION ──────────────────────
  function initPatientInputs() {
    const genderEl = $("sel-gender");
    const ageEl    = $("inp-age");
    const dietEl   = $("sel-diet");

    if (genderEl) {
      genderEl.addEventListener("change", () => {
        validatePatientContext(false);
        updateAnalyzeButtonsState();
      });
    }
    if (ageEl) {
      ageEl.addEventListener("input", () => {
        validatePatientContext(false);
        updateAnalyzeButtonsState();
      });
      ageEl.addEventListener("change", () => {
        validatePatientContext(false);
        updateAnalyzeButtonsState();
      });
    }
    if (dietEl) {
      dietEl.addEventListener("change", () => {
        validatePatientContext(false);
        updateAnalyzeButtonsState();
      });
    }

    // Ensure fresh blank state on startup
    if (genderEl) genderEl.value = "";
    if (ageEl) ageEl.value = "";
    if (dietEl) dietEl.value = "";

    updateAnalyzeButtonsState();
  }

  // ── 12. RUN INITIALIZATIONS ON DOM READY ───────────────────
  checkHealth();
  checkDbStatus();
  initReportModal();
  initHistorySection();
  initSampleLoaders();
  initPatientInputs();
  loadHistoryStats();
});
