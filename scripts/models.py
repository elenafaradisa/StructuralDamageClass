"""Daftar model non-NN, pembungkus strategi multi-kelas, dan ruang pencarian tuning.

Strategi multi-kelas yang didukung:
    - ``native``: penanganan multi-kelas bawaan model.
    - ``ovr``: One-vs-All (satu model per kelas melawan semua kelas lain).
    - ``ovo``: One-vs-One (satu model per pasangan kelas).
"""
from scipy.stats import loguniform, randint
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

SEED = 42

# nama -> (pembuat estimator, jenis preprocessor untuk build_preprocessor)
REGISTRY = {
    "logreg": (lambda: LogisticRegression(max_iter=1000, class_weight="balanced"), "linear"),
    "rf": (lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                          n_jobs=-1, random_state=SEED), "tree"),
    "et": (lambda: ExtraTreesClassifier(n_estimators=200, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=SEED), "tree"),
    "hgb": (lambda: HistGradientBoostingClassifier(class_weight="balanced",
                                                   random_state=SEED), "tree"),
}

# Ruang pencarian hyperparameter untuk RandomizedSearchCV (tanpa prefix pipeline)
SEARCH_SPACE = {
    "logreg": {"C": loguniform(1e-2, 1e2)},
    "rf": {"n_estimators": randint(200, 500), "max_depth": randint(10, 40),
           "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", "log2", 0.5]},
    "et": {"n_estimators": randint(200, 500), "max_depth": randint(10, 40),
           "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", "log2", 0.5]},
    "hgb": {"learning_rate": loguniform(0.03, 0.3), "max_iter": randint(200, 600),
            "max_leaf_nodes": randint(15, 127), "min_samples_leaf": randint(20, 200),
            "l2_regularization": loguniform(1e-3, 10)},
}


def get_estimator(name, strategy="native"):
    """Membuat estimator ``name`` dengan strategi multi-kelas tertentu.

    Args:
        name (str): Kunci di ``REGISTRY`` (``logreg``, ``rf``, ``et``, ``hgb``).
        strategy (str): ``"native"``, ``"ovr"`` (One-vs-All), atau ``"ovo"`` (One-vs-One).

    Returns:
        tuple: ``(estimator, model_type)``; ``model_type`` (``"linear"``/``"tree"``)
        dipakai oleh ``build_preprocessor``.

    Raises:
        ValueError: Jika ``strategy`` tidak dikenal.
    """
    make, model_type = REGISTRY[name]
    est = make()
    if strategy == "ovr":
        est = OneVsRestClassifier(est)
    elif strategy == "ovo":
        est = OneVsOneClassifier(est)
    elif strategy != "native":
        raise ValueError("strategy harus 'native', 'ovr', atau 'ovo'")
    return est, model_type


def prefixed_space(pipe, name, strategy="native"):
    """Memberi prefix ruang pencarian agar cocok dengan nama parameter di Pipeline.

    Args:
        pipe (Pipeline): Pipeline yang langkah terakhirnya adalah model.
        name (str): Kunci model di ``SEARCH_SPACE``.
        strategy (str): Strategi multi-kelas; pembungkus OvR/OvO menambah ``estimator__``.

    Returns:
        dict: Ruang pencarian dengan kunci seperti ``model__estimator__learning_rate``.
    """
    step = pipe.steps[-1][0]
    prefix = f"{step}__" + ("" if strategy == "native" else "estimator__")
    return {prefix + k: v for k, v in SEARCH_SPACE[name].items()}
