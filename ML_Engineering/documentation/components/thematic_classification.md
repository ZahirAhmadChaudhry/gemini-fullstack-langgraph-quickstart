# Thematic Classification Component

## Overview

The thematic-classification stage predicts whether a text segment belongs to the
**Performance** or **Légitimité** sustainability theme.  After several design
iterations we converged on a *light-weight, fast-to-train* baseline that runs
comfortably on a Windows laptop while still achieving competitive accuracy.

## Finalised Model Line-up

| Flag | Model | Purpose | Train Time (CPU) |
|------|-------|---------|------------------|
| *(default)* | **Logistic-Regression** (TF-IDF) | Solid lexical baseline | < 10 s |
| *(default)* | **Naïve-Bayes** (TF-IDF) | Probabilistic baseline | < 10 s |
| *(default)* | **MiniLM + Linear SVM** | Semantic sentence embeddings | ≈ 1 min |
| `--deep` | Sentence-BERT fine-tune | Heavier transformer fine-tune | 15-30 min (❌) |
| `--deep` | CamemBERT fine-tune | French transformer fine-tune | 1-2 h (❌) |

*The "deep" models remain disabled by default (`--deep` opt-in) because they
still break on the current Windows environment—see Known Issues below.*

## RankScore

Model selection and Optuna hyper-parameter optimisation use a custom ranking
metric:

```text
RankScore = 0.7 × Macro-F1  +  0.3 × Accuracy
```

## Key Engineering Decisions

1. **Drop heavyweight transformers on Windows**  
   Transformer fine-tuning (SBERT / CamemBERT) consumes >8 GB RAM & >1 h CPU,
   and repeatedly fails due to HuggingFace / `accelerate` edge-cases.  We made
   them optional (`--deep`) and focus on small models for day-to-day runs.
2. **Light MiniLM sentence embeddings**  
   Implemented `ThematicMiniLMSVMModel` — encodes text with
   `sentence-transformers/all-MiniLM-L6-v2` and trains a Calibrated Linear-SVM
   (~90 MB download, <1 min training).
3. **Remove custom cache folder**  ✅  
   Attempting a project-local cache (`cache_folder=…`) triggered HuggingFace's
   *new* `models--*` directory layout, which the installed
   `sentence-transformers 2.2` cannot parse.  We reverted to the default cache
   path to regain compatibility.
4. **Optuna tuple-to-string workaround**  ✅  
   Converted tuple hyper-parameters to strings to silence "tuple not JSON
   serialisable" warnings.
5. **Class-imbalance handling**  ✅  
   Class weights computed from the training split are passed to models that
   support them.

## Known Issues (2025-06-13)

| Area | Symptom | Root Cause / Notes |
|------|---------|--------------------|
| **HF Cache on Windows** | `ValueError: Unrecognised model in … sentence-transformers_all-MiniLM-L6-v2` | The old `sentence-transformers` expects *legacy* cache layout.  If symlinks are disabled the Hub creates a degraded copy inside `models--…` which breaks the library.  **Fix**: rely on the default cache path and *avoid* `cache_folder=` until we upgrade to `sentence-transformers ≥ 2.6`. |
| **Deep models** | `EarlyStopping requires the chosen metric to exist`, `accelerate not installed` | Multiple upstream breaks; revisit once GPU resources are available. |
| **Excessive downloads** | Large ONNX / OpenVINO files (~470 MB) pulled even though we use PyTorch | These artefacts live in the same repo release tag.  No official filter in HF Hub yet; acceptable once-off cost. |

## Typical Workflow

```powershell
# single table quick-test
ecd ML_Models_Plan
python train_thematic_models.py --table Table_A          # light models only
python train_thematic_models.py --table Table_A --deep   # incl. heavy models
```

## Next Steps

1. **Upgrade HuggingFace stack** → `transformers 4.41+`,
   `sentence-transformers 2.6+`, `huggingface_hub 0.24+` to regain cache
   compatibility.
2. **Evaluate out-of-the-box MiniLM variations** for potential accuracy gains.
3. **Re-enable GPU fine-tuning** once Linux/WSL-based runners are available.
4. **Export calibrated probabilities** to facilitate ensemble stacking with
   tension-detection outputs.

---
*Last updated: 2025-06-13* 