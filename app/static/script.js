/* ================================================================
   DermScan AI — Facial Skin Condition Detection
   Frontend JavaScript
   ================================================================ */

'use strict';

// ── Condition metadata (icons + accent colours) ────────────────────
const CONDITION_META = {
  'Normal Condition':        { icon: '✨', colour: '#2ECC71' },
  'Acne':                    { icon: '🔴', colour: '#E74C3C' },
  'Rosacea':                 { icon: '🟠', colour: '#E67E22' },
  'Melasma':                 { icon: '🟣', colour: '#9B59B6' },
  'Facial Vitiligo':         { icon: '🟡', colour: '#F1C40F' },
  'Seborrheic Dermatitis':   { icon: '🟢', colour: '#2ECC71' },
};

const DEFAULT_META = { icon: '🩺', colour: '#6C63FF' };

// ── Element references ─────────────────────────────────────────────
const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('file-input');
const previewWrapper   = document.getElementById('preview-wrapper');
const previewImg       = document.getElementById('preview-img');
const btnClear         = document.getElementById('btn-clear');
const btnAnalyse       = document.getElementById('btn-analyse');
const errorBanner      = document.getElementById('error-banner');
const errorText        = document.getElementById('error-text');

const uploadSection    = document.querySelector('.upload-section');
const resultSection    = document.getElementById('result-section');
const resultIcon       = document.getElementById('result-icon');
const resultHeading    = document.getElementById('result-heading');
const resultConfidence = document.getElementById('result-confidence');
const confidenceFill   = document.getElementById('confidence-bar-fill');
const descriptionText  = document.getElementById('description-text');
const symptomsList     = document.getElementById('symptoms-list');
const probaBars        = document.getElementById('proba-bars');
const disclaimerText   = document.getElementById('disclaimer-text');
const btnRetry         = document.getElementById('btn-retry');

// ── State ──────────────────────────────────────────────────────────
let selectedFile = null;

// ── Drop Zone — drag events ────────────────────────────────────────
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });
});
dropZone.addEventListener('drop', e => {
  const files = e.dataTransfer.files;
  if (files && files[0]) handleFileSelected(files[0]);
});

// ── Drop Zone — keyboard / click ───────────────────────────────────
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

// ── File selection ─────────────────────────────────────────────────
function handleFileSelected(file) {
  const ALLOWED = ['image/jpeg', 'image/png', 'image/bmp'];
  if (!ALLOWED.includes(file.type)) {
    showError('Invalid file type. Please upload a JPG, JPEG, PNG, or BMP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File too large. Maximum size is 10 MB.');
    return;
  }

  hideError();
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.alt = `Preview of ${file.name}`;
    previewWrapper.classList.add('visible');
    dropZone.style.display = 'none';
  };
  reader.readAsDataURL(file);

  btnAnalyse.disabled = false;
}

// ── Clear ──────────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  previewWrapper.classList.remove('visible');
  dropZone.style.display = '';
  btnAnalyse.disabled = true;
  hideError();
}

// ── Analyse ────────────────────────────────────────────────────────
btnAnalyse.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runPrediction(selectedFile);
});

async function runPrediction(file) {
  // Set loading state
  btnAnalyse.classList.add('loading');
  btnAnalyse.disabled = true;
  hideError();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      const errMsg = data.error || `Server error (${response.status}). Please try again.`;
      showError(errMsg);
      btnAnalyse.classList.remove('loading');
      btnAnalyse.disabled = false;
      return;
    }

    renderResult(data);

  } catch (err) {
    showError('Network error. Make sure the server is running and try again.');
    btnAnalyse.classList.remove('loading');
    btnAnalyse.disabled = false;
  }
}

// ── Render Result ──────────────────────────────────────────────────
function renderResult(data) {
  const { predicted_class, confidence, probabilities, condition_info, disclaimer } = data;

  const meta   = CONDITION_META[predicted_class] || DEFAULT_META;
  const colour = condition_info.colour || meta.colour;

  // Header
  resultIcon.textContent              = meta.icon;
  resultHeading.textContent           = predicted_class;
  resultHeading.style.color           = colour;
  resultConfidence.textContent        = `${confidence}%`;
  resultConfidence.style.color        = colour;

  // Confidence bar
  confidenceFill.style.background     = `linear-gradient(90deg, ${colour}, ${adjustColour(colour, 40)})`;

  // Description
  descriptionText.textContent = condition_info.description || '';

  // Symptoms
  symptomsList.innerHTML = '';
  const symptoms = condition_info.common_symptoms || [];
  symptoms.forEach(symptom => {
    const li = document.createElement('li');
    li.textContent = symptom;
    symptomsList.appendChild(li);
  });

  // Probability bars
  probaBars.innerHTML = '';
  const sortedEntries = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

  sortedEntries.forEach(([cls, prob]) => {
    const clsMeta  = CONDITION_META[cls] || DEFAULT_META;
    const clsColour = clsMeta.colour;
    const isTop    = cls === predicted_class;

    const row = document.createElement('div');
    row.className = 'proba-row';
    row.innerHTML = `
      <div class="proba-label-row">
        <span class="proba-name" style="${isTop ? `color:${clsColour};font-weight:700` : ''}">${cls}</span>
        <span class="proba-value">${prob.toFixed(1)}%</span>
      </div>
      <div class="proba-track">
        <div class="proba-fill" style="background:${clsColour};opacity:${isTop ? 1 : 0.45}" data-width="${prob}"></div>
      </div>
    `;
    probaBars.appendChild(row);
  });

  // Disclaimer
  disclaimerText.textContent = disclaimer;

  // Show result section, hide upload section
  uploadSection.style.display = 'none';
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Animate bars after paint
  requestAnimationFrame(() => {
    // Confidence bar
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${confidence}%`;

      // Proba fills (staggered)
      const fills = probaBars.querySelectorAll('.proba-fill');
      fills.forEach((fill, i) => {
        setTimeout(() => {
          fill.style.width = `${fill.dataset.width}%`;
        }, i * 80);
      });
    });
  });

  btnAnalyse.classList.remove('loading');
}

// ── Retry ──────────────────────────────────────────────────────────
btnRetry.addEventListener('click', () => {
  resultSection.hidden = true;
  uploadSection.style.display = '';
  resetUpload();

  // Reset result UI
  resultHeading.textContent     = '—';
  resultConfidence.textContent  = '—';
  confidenceFill.style.width    = '0%';
  descriptionText.textContent   = '';
  symptomsList.innerHTML        = '';
  probaBars.innerHTML           = '';
  disclaimerText.textContent    = '';

  uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ── Error Helpers ──────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.hidden    = false;
}
function hideError() {
  errorBanner.hidden = true;
  errorText.textContent = '';
}

// ── Colour utility ─────────────────────────────────────────────────
/**
 * Very simple: lighten a hex colour by mixing with white.
 * amount 0-100 (%)
 */
function adjustColour(hex, amount) {
  try {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const mix = v => Math.min(255, Math.round(v + (255 - v) * (amount / 100)));
    return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
  } catch {
    return hex;
  }
}
