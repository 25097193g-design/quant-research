# Disclosure-Tone Measurement: Dictionary vs TF-IDF+LR vs FinBERT

Public (portfolio) version of a sentence-level sentiment-measurement study on
US 10-K MD&A text (2016 filings, SEC EDGAR — public data). 150 sentences were
hand-labelled (1 = genuinely negative in context) in a stratified design:
60 dictionary-flagged (precision), 60 unflagged (recall), 30 hard cases
(digit-dense / negation / hedging).

Three sentiment extractors are benchmarked on the same held-out test split:

| Method | Type | Era |
|---|---|---|
| Loughran-McDonald dictionary | unsupervised word list | 2011 |
| TF-IDF + Logistic Regression | supervised, trained on 105 labelled sentences | traditional ML |
| FinBERT (ProsusAI) | pretrained financial transformer | 2023 |

Contents (`outputs_public/`):
- `annotation_150_public.csv` — sentences, labels, per-method scores/predictions
- `table_v3_methods_test.csv` / `..._full.csv` — accuracy / precision / recall / F1 / kappa
- `error_taxonomy.csv` — where the dictionary's false positives come from
- `disagreement_examples.csv` — sentences FinBERT got right and the dictionary missed

Reproduce: `python3 recompute_metrics.py`

The measurement was also applied in a disclosure-return study (see the
accompanying course report); per course data-use rules the CRSP/COMPUSTAT
variables and regression results are not part of this public repository.

**Label note:** labels were assigned by one annotator following a written
codebook (1 = conveys negative information about the firm's realized
performance, conditions, or firm-specific risk; 0 = neutral, positive,
procedural or safe-harbor boilerplate). A second-annotator column (`label_2`)
is included for inter-rater kappa reproduction.
