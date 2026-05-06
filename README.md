# Myth vs Fact Classifier Agent

A beginner-friendly AI mini project that classifies a sentence into:
- Myth
- Fact
- Uncertain

It uses a **hybrid approach**:
1. Rule-based keyword detection
2. Machine Learning (TF-IDF + Logistic Regression)

## Project Structure

```text
myth-fact-ui/
  backend/
    app.py
    train_model.py
    model.pkl
    vectorizer.pkl
  frontend/
    index.html
    style.css
    script.js
  requirements.txt
  README.md
```

## How to Run

1. Open terminal in project folder:
   ```bash
   cd myth-fact-ui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train model artifacts (creates `model.pkl` and `vectorizer.pkl`):
   ```bash
   python backend/train_model.py
   ```

4. Start Flask backend:
   ```bash
   python backend/app.py
   ```

5. Open frontend:
   - Open `frontend/index.html` in your browser.
   - Keep Flask running at `http://127.0.0.1:5000`.

## API

### POST `/predict`
Request body:
```json
{
  "text": "Everyone says detox drinks always cure flu"
}
```

Sample response:
```json
{
  "label": "Myth",
  "confidence": 0.84,
  "explanation": [
    "Contains exaggerated or absolute wording (e.g., always/never)."
  ],
  "scores": {
    "Myth": 0.84,
    "Fact": 0.1,
    "Uncertain": 0.06
  }
}
```

## Notes
- If model files are missing, backend auto-trains them on startup.
- Frontend includes loading animation, error handling, and explanation rendering.
- Designed for college mini-project demonstration.
