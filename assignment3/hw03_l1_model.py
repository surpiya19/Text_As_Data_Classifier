"""
hw03_l1_model.py
IDS 570 - Homework 3
L1 (Lasso) Logistic Regression — same pipeline as L2 baseline, penalty changed to L1.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ── 1. Load the saved train/test JSON files (from step9_prepare_datasets.py) ──
DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Data format: list of [text, label] pairs (as saved by step9)
X_train_texts = [t for (t, y) in train_data]
y_train       = [y for (t, y) in train_data]

X_test_texts  = [t for (t, y) in test_data]
y_test        = [y for (t, y) in test_data]

# ── 2. TF-IDF vectorisation — identical to L2 baseline ──────────────────────
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9,
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

# ── 3. Train L1 logistic regression ─────────────────────────────────────────
# Key change from baseline: penalty="l1", solver="liblinear"
# liblinear is required for L1 in scikit-learn
clf = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=2000,
)
clf.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("=" * 60)
print("L1 MODEL — EVALUATION")
print("=" * 60)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {roc_auc:.4f}")

# ── 5. Sparsity diagnostic ───────────────────────────────────────────────────
coef      = clf.coef_[0]
n_nonzero = int(np.sum(coef != 0))
n_total   = len(coef)
print(f"\nModel sparsity — non-zero coefficients: {n_nonzero} / {n_total} "
      f"({100 * n_nonzero / n_total:.1f}%)")
print("(L1 should be dramatically sparser than L2)")

# ── 6. Top 15 positive and negative words ────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
sorted_idx    = np.argsort(coef)

print("\n--- Top 15 Positive-Weight Words (predictive of CORE = 1) ---")
for idx in sorted_idx[-15:][::-1]:
    print(f"  {feature_names[idx]:30s}  {coef[idx]:+.4f}")

print("\n--- Top 15 Negative-Weight Words (predictive of NEG = 0) ---")
for idx in sorted_idx[:15]:
    print(f"  {feature_names[idx]:30s}  {coef[idx]:+.4f}")