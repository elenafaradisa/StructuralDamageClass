"""Prediksi data test dan pembuatan file submission Kaggle.

Contoh (dijalankan dari root repo):
    python -m scripts.predict --model models/model.joblib --out submission.csv
"""
import argparse

import joblib
import pandas as pd

from scripts.preprocess import ID_COL, TARGET, decode_target


def make_submission(model_path="models/model.joblib", test_path="data/test.csv",
                    out_path="submission.csv"):
    """Memuat Pipeline terlatih, memprediksi data test, dan menulis CSV submission.

    Args:
        model_path (str): Path file model ``.joblib`` hasil ``scripts.train``.
        test_path (str): Path ``test.csv``.
        out_path (str): Path CSV keluaran.

    Returns:
        pd.DataFrame: Submission dengan kolom ``building_id`` dan ``damage_grade`` (1..5).
    """
    pipe = joblib.load(model_path)
    test = pd.read_csv(test_path)
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: decode_target(pipe.predict(test))})
    sub.to_csv(out_path, index=False)
    return sub


def main():
    """Menjalankan prediksi dari command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/model.joblib", help="path model .joblib")
    p.add_argument("--data", default="data/test.csv", help="path test.csv")
    p.add_argument("--out", default="submission.csv", help="path file submission")
    a = p.parse_args()
    sub = make_submission(a.model, a.data, a.out)
    print(f"Submission tersimpan: {a.out} ({len(sub)} baris)")


if __name__ == "__main__":
    main()
