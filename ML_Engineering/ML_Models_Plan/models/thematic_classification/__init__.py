"""
Thematic Classification Models Package

This package contains machine learning models for improving thematic classification
from limited scope (only "Performance") to full coverage with 85-95% accuracy.

Models included:
1. CamemBERTThematicClassifier - French language understanding
2. LogisticRegressionThematicClassifier - Lightweight baseline
3. SentenceBERTThematicClassifier - Semantic embedding approach
4. NaiveBayesThematicClassifier - Probabilistic text classification

Current Problem:
- Only detects "Performance" theme, misses "Légitimité" nuances
- Limited semantic understanding of French sustainability discourse

Target Performance:
- Accuracy: 85-95% with full theme coverage
- Classes: ["Performance", "Légitimité"] 
- F1-Score: 0.85+ for both classes
- Confidence Calibration: 80%+ reliability
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Compatibility patch: some versions of `sentence_transformers` expect
# `huggingface_hub.cached_download`, which was renamed to `hf_hub_download` in
# newer releases (>=0.15). We add a shim if it is missing so imports succeed
# even with the latest hub library.
# ---------------------------------------------------------------------------
try:
    import huggingface_hub as _hf
    if not hasattr(_hf, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download
        from urllib.parse import urlparse

        def _cached_download(*args, **kwargs):
            """Backwards-compat shim for the removed cached_download().

            The historical API allowed both positional ``url`` as the first
            argument **and** a keyword argument ``url=...``.

            Newer ``hf_hub_download`` instead takes the decomposed pieces
            ``repo_id`` / ``filename`` / ``revision``.  We therefore
            1) detect a full HTTPS URL and parse it into these components;
            2) otherwise, fall through to the new function directly.
            """
            if args:
                # Original form: cached_download(url, cache_dir=..., force=...)
                url = args[0]
                remaining_args = args[1:]
            else:
                url = kwargs.pop("url", None)
                remaining_args = ()

            if url and url.startswith("http"):
                parsed = urlparse(url)
                # Expected path: /<repo_id>/resolve/<revision>/<filepath>
                path_parts = parsed.path.lstrip("/").split("/")
                if "resolve" in path_parts:
                    idx = path_parts.index("resolve")
                    repo_id = "/".join(path_parts[:idx])
                    revision = path_parts[idx + 1]
                    filename = "/".join(path_parts[idx + 2 :])
                else:
                    # Fallback – treat first token as repo, rest as filename
                    repo_id = path_parts[0]
                    filename = "/".join(path_parts[1:])
                    revision = kwargs.get("revision", None)

                return _hf.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    **kwargs,
                )

            # Non-URL path: directly delegate (keeps HF semantics)
            positional = ([url] if url else list(remaining_args))
            return _hf.hf_hub_download(*positional, **kwargs)

        _hf.cached_download = _cached_download  # type: ignore[attr-defined]
except Exception:
    # If patching fails we continue; SentenceTransformer may not be used.
    pass

# ---------------------------------------------------------------------------
# Patch ``hf_hub_download`` itself so that extra kwargs coming from newer
# libraries (e.g., ``legacy_cache_layout`` from sentence-transformers >=2.6)
# are silently discarded when the local huggingface_hub version is older and
# does not recognise them.
# ---------------------------------------------------------------------------
try:
    import inspect as _inspect
    import functools as _functools

    _orig_hf_download = getattr(_hf, "hf_hub_download", None)

    if _orig_hf_download is not None and not hasattr(_hf, "_patched_legacy_kw"):  # type: ignore[attr-defined]

        _sig = _inspect.signature(_orig_hf_download)

        @_functools.wraps(_orig_hf_download)
        def _hf_download_compat(*args, **kwargs):  # noqa: D401
            # Drop kwargs that the current huggingface_hub version does not accept
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in _sig.parameters}
            return _orig_hf_download(*args, **filtered_kwargs)

        _hf.hf_hub_download = _hf_download_compat  # type: ignore[attr-defined]
        _hf._patched_legacy_kw = True  # flag so we don't double-patch
except Exception:
    # Non-fatal: continue without the patch if something goes wrong
    pass

# Import implemented models
from .camembert_model import ThematicCamemBERTModel
from .logistic_regression import ThematicLogisticRegressionModel
from .sentence_bert import ThematicSentenceBERTModel
from .naive_bayes import ThematicNaiveBayesModel
from .miniLM_svm import ThematicMiniLMSVMModel

__all__ = [
    'ThematicCamemBERTModel',
    'ThematicLogisticRegressionModel',
    'ThematicSentenceBERTModel',
    'ThematicNaiveBayesModel',
    'ThematicMiniLMSVMModel'
]

THEME_CLASSES = [
    "Performance",
    "Légitimité"
]

FEATURE_GROUPS = {
    "text_embeddings": "Semantic representation of text",
    "thematic_indicators": "Performance/legitimacy density scores",
    "sustainability_terms": "Domain-specific vocabulary",
    "discourse_types": "Argumentation patterns",
    "temporal_confidence": "Time-based context",
    "entity_types": "Named entity patterns",
    "sentiment_scores": "Opinion indicators"
}

PERFORMANCE_INDICATORS = [
    "performance", "efficacité", "croissance", "résultats", 
    "productivité", "optimisation", "rendement", "profit"
]

LEGITIMACY_INDICATORS = [
    "légitimité", "éthique", "responsabilité", "durabilité",
    "justice", "équité", "transparence", "intégrité"
]
