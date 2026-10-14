"""Pelatihan model: CV Macro F1, tuning opsional, lalu simpan Pipeline terlatih.

Contoh (dijalankan dari root repo):
    python -m scripts.train --model hgb --strategy ovo --tune --n-iter 20
    python -m scripts.train --model hgb --strategy ovo --params models/best_params.json --skip-cv
"""
import argparse
import json
import os
import time

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from scripts.models import SEED, get_estimator, prefixed_space
from scripts.preprocess import build_pipeline, split_features_target


def parse_args():
    """Membaca argumen command line.

    Returns:
        argparse.Namespace: Argumen yang sudah di-parse.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.csv", help="path train.csv")
    p.add_argument("--model", default="hgb", choices=["logreg", "rf", "et", "hgb"])
    p.add_argument("--strategy", default="native", choices=["native", "ovr", "ovo"])
    p.add_argument("--tune", action="store_true", help="RandomizedSearchCV sebelum fit akhir")
    p.add_argument("--n-iter", type=int, default=20, help="jumlah kandidat tuning")
    p.add_argument("--params", default=None, help="JSON hyperparameter (models/best_params.json)")
    p.add_argument("--skip-cv", action="store_true", help="lewati CV, langsung fit data penuh")
    p.add_argument("--out", default="models/model.joblib", help="path model keluaran")
    return p.parse_args()


def train(args):
    """Melatih Pipeline sesuai argumen dan mengembalikan model beserta skor CV.

    Args:
        args (argparse.Namespace): Hasil ``parse_args``.

    Returns:
        tuple: ``(pipeline_terlatih, skor_cv_macro_f1 atau None)``.
    """
    X, y = split_features_target(pd.read_csv(args.data))
    est, model_type = get_estimator(args.model, args.strategy)
    pipe = build_pipeline(est, model_type)
    if args.params:
        with open(args.params) as f:
            pipe.set_params(**json.load(f))
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    if args.tune:
        search = RandomizedSearchCV(pipe, prefixed_space(pipe, args.model, args.strategy),
                                    n_iter=args.n_iter, scoring="f1_macro", cv=cv,
                                    random_state=SEED, n_jobs=-1, verbose=1)
        search.fit(X, y)
        print("Best params:", search.best_params_)
        return search.best_estimator_, search.best_score_

    cv_score = None
    if not args.skip_cv:
        cv_score = cross_val_score(pipe, X, y, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    return pipe.fit(X, y), cv_score


def main():
    """Menjalankan training, mencetak ringkasan, dan menyimpan model + metrik."""
    args = parse_args()
    t0 = time.time()
    pipe, cv_score = train(args)
    print(f"CV Macro F1 = {cv_score} | waktu = {time.time() - t0:.0f} dtk")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipe, args.out)
    metrics = {"model": args.model, "strategy": args.strategy,
               "tuned": bool(args.tune or args.params),
               "cv_macro_f1": None if cv_score is None else float(cv_score)}
    with open(args.out.replace(".joblib", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Tersimpan:", args.out)


if __name__ == "__main__":
    main()
