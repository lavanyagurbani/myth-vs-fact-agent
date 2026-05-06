"""Train a text classifier for Myth vs Fact classification using XGBoost and custom features."""

import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Import the dataset from data.py
try:
    from data import DATASET
except ImportError:
    from .data import DATASET

def extract_custom_features(texts):
    """Extract structural and keyword-based features from text."""
    features = []
    
    # Keyword sets
    myth_words = {"always", "never", "everyone", "guaranteed", "secret", "miracle"}
    fact_words = {"research", "study", "data", "scientists", "evidence", "proven"}
    uncertain_words = {"might", "may", "could", "perhaps", "study suggests", "limited"}

    for text in texts:
        text_lower = text.lower()
        words = text_lower.split()
        
        # 1. Text length features
        char_count = len(text)
        word_count = len(words)
        
        # 2. Numerical features
        has_number = 1 if any(char.isdigit() for char in text) else 0
        num_digits = sum(c.isdigit() for c in text)
        
        # 3. Keyword ratio features
        myth_count = sum(1 for w in myth_words if w in text_lower)
        fact_count = sum(1 for w in fact_words if w in text_lower)
        uncertain_count = sum(1 for w in uncertain_words if w in text_lower)
        
        # 4. Punctuation
        has_period = 1 if "." in text else 0
        
        features.append([
            char_count, 
            word_count, 
            has_number, 
            num_digits, 
            myth_count, 
            fact_count, 
            uncertain_count,
            has_period
        ])
        
    return np.array(features)

def train_and_save(output_dir: Path) -> None:
    """Train an XGBoost model with TF-IDF and custom features."""
    texts = [text for text, _ in DATASET]
    labels = [label for _, label in DATASET]

    # Encode labels to integers (required by XGBoost)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # Map back labels to understand order
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"Label Mapping: {label_mapping}")

    # 1. Feature Extraction: TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()

    # 2. Feature Extraction: Custom
    X_custom = extract_custom_features(texts)

    # 3. Combine Features
    X = np.hstack([X_tfidf, X_custom])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 4. Train XGBoost model
    # We use a slightly deeper tree and lower learning rate for robustness on small data
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='multi:softprob',
        num_class=3
    )

    # Simple grid search for key parameters
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Re-train on full dataset for production
    best_model.fit(X, y)

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, output_dir / "vectorizer.pkl")
    joblib.dump(best_model, output_dir / "model.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    
    print(f"Model and artifacts saved to {output_dir}")

if __name__ == "__main__":
    train_and_save(Path(__file__).resolve().parent)
