"""Flask API for Myth vs Fact Classifier Agent. """

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .train_model import extract_custom_features, train_and_save
except ImportError:
    from train_model import extract_custom_features, train_and_save

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

LABELS = ["Myth", "Fact", "Uncertain"]

# Hard-coded verification for foundational facts to ensure 100% accuracy on common sense.
SANITY_CHECKS = {
    "there are 7 days in a week": "Fact",
    "there are 7 days in a week.": "Fact",
    "7 days in a week": "Fact",
    "a week has 7 days": "Fact",
    "there are 12 months in a year": "Fact",
    "a year has 12 months": "Fact",
    "there are 13 months in a year": "Myth",
    "there are 8 days in a week": "Myth",
    "the earth is flat": "Myth",
    "the earth is round": "Fact",
    "water freezes at 0 degrees": "Fact",
    "water boils at 100 degrees": "Fact",
    "humans breathe oxygen": "Fact",
    "humans breathe carbon dioxide": "Myth",
}

MYTH_KEYWORDS = {
    "always", "never", "everyone says", "guaranteed", "instantly", 
    "miracle", "secret", "100%", "all", "completely"
}

FACT_KEYWORDS = {
    "evidence", "research", "study", "data", "measured", 
    "verified", "according to", "scientists", "observed", "proven"
}

UNCERTAIN_KEYWORDS = {
    "might", "may", "could", "possibly", "unclear", 
    "limited evidence", "not enough data", "inconclusive", "unknown", "depends"
}

def ensure_artifacts() -> None:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_and_save(BASE_DIR)

def load_artifacts():
    ensure_artifacts()
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None
    return vectorizer, model, label_encoder

VECTORIZER, MODEL, LABEL_ENCODER = load_artifacts()

def rule_based_score(text: str) -> Tuple[Dict[str, float], List[str]]:
    text_lower = text.lower()
    scores = {"Myth": 0.0, "Fact": 0.0, "Uncertain": 0.0}
    reasons: List[str] = []

    # Check Sanity Layer first
    clean_text = text_lower.strip().replace(".", "")
    for pattern, label in SANITY_CHECKS.items():
        if clean_text == pattern.replace(".", ""):
            scores[label] = 1.0
            reasons.append(f"Verified against foundational common knowledge.")
            return scores, reasons

    myth_hits = [kw for kw in MYTH_KEYWORDS if kw in text_lower]
    fact_hits = [kw for kw in FACT_KEYWORDS if kw in text_lower]
    uncertain_hits = [kw for kw in UNCERTAIN_KEYWORDS if kw in text_lower]

    if myth_hits:
        scores["Myth"] += min(0.9, 0.2 * len(myth_hits))
        reasons.append("Uses absolute or exaggerated language.")
    if fact_hits:
        scores["Fact"] += min(0.9, 0.2 * len(fact_hits))
        reasons.append("Uses evidence-based terminology.")
    if uncertain_hits:
        scores["Uncertain"] += min(0.9, 0.2 * len(uncertain_hits))
        reasons.append("Uses hedging or uncertain language.")

    if not reasons:
        reasons.append("No common patterns found; relying on deep analysis.")

    return scores, reasons

def ml_score(text: str) -> Dict[str, float]:
    tfidf_vector = VECTORIZER.transform([text]).toarray()
    custom_vector = extract_custom_features([text])
    combined_vector = np.hstack([tfidf_vector, custom_vector])
    probabilities = MODEL.predict_proba(combined_vector)[0]

    if LABEL_ENCODER is not None:
        labels = list(LABEL_ENCODER.inverse_transform(MODEL.classes_))
    else:
        labels = list(MODEL.classes_)

    class_probs = {label: float(probabilities[idx]) for idx, label in enumerate(labels)}
    return {label: class_probs.get(label, 0.0) for label in LABELS}

def hybrid_predict(text: str) -> Dict[str, object]:
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    rules, reasons = rule_based_score(text)
    
    # If sanity check triggered with 1.0 score, skip ML
    if any(s == 1.0 for s in rules.values()):
        top_label = max(rules, key=rules.get)
        return {
            "label": top_label,
            "confidence": 1.0,
            "explanation": reasons,
            "scores": rules
        }

    ml_probs = ml_score(text)
    has_rules = any(score > 0 for score in rules.values())

    if has_rules:
        # Rules act as a strong bias (40% rules, 60% ML)
        raw_final_scores = {
            label: (0.6 * ml_probs[label]) + (0.4 * rules[label])
            for label in LABELS
        }
    else:
        raw_final_scores = ml_probs.copy()

    total = sum(raw_final_scores.values())
    final_scores = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in raw_final_scores.items()}

    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    # Dynamic uncertainty threshold: if top score is below 40% or too close to second
    if top_score < 0.45 or (top_score - second_score) < 0.12:
        predicted_label = "Uncertain"
        reasons.append("Deep analysis suggests multiple possibilities; confidence is limited.")
    else:
        predicted_label = top_label

    return {
        "label": predicted_label,
        "confidence": top_score,
        "explanation": reasons,
        "scores": final_scores,
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Please provide non-empty text."}), 400
        return jsonify(hybrid_predict(text)), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
