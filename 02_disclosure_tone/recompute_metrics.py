"""Recompute the three-method metrics from the public CSV (no course data)."""
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

df = pd.read_csv("outputs_public/annotation_150_public.csv")
df["label"] = df["label"].astype(int)
te = df[df["split"] == "test"]

def report(name, y, p):
    print(f"{name:28s} acc={accuracy_score(y, p):.3f}  prec={precision_score(y, p, zero_division=0):.3f}  "
          f"rec={recall_score(y, p, zero_division=0):.3f}  F1={f1_score(y, p, zero_division=0):.3f}  "
          f"kappa={cohen_kappa_score(y, p):.3f}")

y = te["label"]
report("Dictionary (2011)", y, te["dict_pred"])
report("TF-IDF + LR (trained)", y, te["tfidf_pred"].astype(int))
report("FinBERT (2023)", y, te["finbert_pred"])
