const API_URL = "http://127.0.0.1:5000/predict";

// ========== THEME TOGGLE ==========

const themeToggle = document.getElementById("themeToggle");
const htmlElement = document.documentElement;

// Get light mode preference from localStorage
function getLightModePreference() {
  const saved = localStorage.getItem("mythFactLightMode");
  if (saved !== null) {
    return saved === "true";
  }
  // Check system preference
  return window.matchMedia("(prefers-color-scheme: light)").matches;
}

// Apply theme
function applyTheme(isLightMode) {
  if (isLightMode) {
    htmlElement.classList.add("light-mode");
    themeToggle.classList.add("active");
  } else {
    htmlElement.classList.remove("light-mode");
    themeToggle.classList.remove("active");
  }
}

// Toggle theme
function toggleTheme() {
  const isCurrentlyLight = htmlElement.classList.contains("light-mode");
  const shouldBeLightMode = !isCurrentlyLight;
  
  applyTheme(shouldBeLightMode);
  localStorage.setItem("mythFactLightMode", shouldBeLightMode);
}

// Initialize theme
function initTheme() {
  const isLightMode = getLightModePreference();
  applyTheme(isLightMode);
}

// Apply theme on page load
window.addEventListener("DOMContentLoaded", initTheme);

// Toggle button click
themeToggle.addEventListener("click", toggleTheme);

// Listen for system theme changes
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
  if (localStorage.getItem("mythFactLightMode") === null) {
    applyTheme(e.matches);
  }
});

// ========== PREDICTION FUNCTIONALITY ==========
const inputText = document.getElementById("inputText");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const tryAnotherBtn = document.getElementById("tryAnotherBtn");
const labelIcon = document.getElementById("labelIcon");
const labelOutput = document.getElementById("labelOutput");
const resultBadge = document.getElementById("resultBadge");
const confidenceBarFill = document.getElementById("confidenceBarFill");
const confidenceValue = document.getElementById("confidenceValue");
const explanationOutput = document.getElementById("explanationOutput");
const errorMsg = document.getElementById("errorMsg");
const chips = document.querySelectorAll(".chip");

let confidenceAnimationId;

// Animate confidence percentage
function animateConfidenceValue(targetPercent) {
  if (confidenceAnimationId) {
    cancelAnimationFrame(confidenceAnimationId);
  }

  const durationMs = 700;
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / durationMs, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = targetPercent * eased;

    confidenceValue.textContent = `${current.toFixed(1)}%`;

    if (progress < 1) {
      confidenceAnimationId = requestAnimationFrame(step);
    } else {
      confidenceValue.textContent = `${targetPercent.toFixed(1)}%`;
      confidenceAnimationId = undefined;
    }
  }

  confidenceAnimationId = requestAnimationFrame(step);
}

// Get emoji icon based on label
function getLabelIcon(label) {
  if (label === "Myth") return "⚠️";
  if (label === "Fact") return "✔️";
  return "❓";
}

// Render result data
function renderResult(data) {
  const confidencePercent = ((data.confidence || 0) * 100).toFixed(1);
  const label = data.label || "Uncertain";
  
  // Update badge styling
  resultBadge.classList.remove("myth", "uncertain");
  if (label === "Myth") {
    resultBadge.classList.add("myth");
    confidenceBarFill.classList.remove("myth", "uncertain");
    confidenceBarFill.classList.add("myth");
  } else if (label === "Uncertain") {
    resultBadge.classList.add("uncertain");
    confidenceBarFill.classList.remove("myth");
    confidenceBarFill.classList.add("uncertain");
  } else {
    confidenceBarFill.classList.remove("myth", "uncertain");
  }
  
  // Update label and icon
  labelIcon.textContent = getLabelIcon(label);
  labelOutput.textContent = label.toUpperCase();
  
  // Update confidence bar
  confidenceBarFill.style.width = `${confidencePercent}%`;
  animateConfidenceValue(Number(confidencePercent));

  // Update explanation
  explanationOutput.innerHTML = "";
  if (data.explanation) {
    explanationOutput.textContent = data.explanation;
  }
}

// Get color for label
function getColorForLabel(label) {
  const colors = {
    "Myth": "var(--color-myth)",
    "Fact": "var(--color-fact)",
    "Uncertain": "var(--color-uncertain)"
  };
  return colors[label] || "var(--color-fact)";
}

// Reset for another claim
function resetForAnotherClaim() {
  result.classList.add("hidden");
  inputText.value = "";
  inputText.focus();
  inputText.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Predict claim
async function predictClaim() {
  const text = inputText.value.trim();
  result.classList.add("hidden");
  errorMsg.classList.add("hidden");

  if (!text) {
    errorMsg.textContent = "Please enter a statement to analyze.";
    errorMsg.classList.remove("hidden");
    return;
  }

  predictBtn.disabled = true;
  loading.classList.remove("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Prediction failed");
    }

    const payload = await response.json();
    renderResult(payload);
    result.classList.remove("hidden");
  } catch (error) {
    errorMsg.textContent = error.message || "Unable to connect to backend API.";
    errorMsg.classList.remove("hidden");
  } finally {
    predictBtn.disabled = false;
    loading.classList.add("hidden");
  }
}

// Event listeners for predict button
predictBtn.addEventListener("click", predictClaim);
inputText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) {
    predictClaim();
  }
});

// Event listeners for try another button
tryAnotherBtn.addEventListener("click", resetForAnotherClaim);

// Event listeners for example chips
chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    inputText.value = chip.dataset.text || "";
    inputText.focus();
    inputText.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
});
