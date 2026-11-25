# Preload PyArrow/Pandas/Scikit-learn before torch/transformers/spacy (llm-guard)
# to prevent Windows C++ DLL initialization access violations (0xC0000005).
try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import pyarrow.dataset  # noqa: F401
    import sklearn  # noqa: F401
except Exception:
    pass
