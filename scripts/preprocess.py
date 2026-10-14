"""Preprocessing: pemuatan data, encoding target, dan pembuatan Pipeline.

Seluruh langkah (feature engineering, imputasi, encoding, scaling, model)
digabung dalam satu ``Pipeline`` + ``ColumnTransformer`` supaya hanya
mempelajari statistik dari data latih pada setiap fold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from scripts.feature_engineering import FeatureEngineer

TARGET = "damage_grade"
ID_COL = "building_id"

# Grade 1..5 -> 0..4 (dibutuhkan oleh LightGBM/XGBoost); dikembalikan saat submission
GRADE_TO_INT = {f"Grade {i}": i - 1 for i in range(1, 6)}

NUM_COLS = [
    "count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
    "age_unknown", "height_per_floor", "volume_proxy",
]
CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
    "district_id", "vdcmun_id",
]
HIGH_CARD_COLS = ["ward_id"]  # ratusan/ribuan kategori


def load_data(data_dir="data"):
    """Membaca train.csv dan test.csv.

    Args:
        data_dir (str | Path): Folder berisi train.csv dan test.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame train dan test.
    """
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def split_features_target(train):
    """Memisahkan fitur dan target, lalu meng-encode target menjadi 0..4.

    Args:
        train (pd.DataFrame): Data train lengkap dengan kolom ``damage_grade``.

    Returns:
        tuple[pd.DataFrame, pd.Series]: ``X`` (fitur) dan ``y`` (target 0..4).
    """
    X = train.drop(columns=[TARGET])
    y = train[TARGET].map(GRADE_TO_INT)
    return X, y


def decode_target(pred):
    """Mengubah prediksi menjadi damage_grade 1..5 untuk submission.

    Args:
        pred (array-like): Prediksi berupa kode 0..4, atau label teks
            ``"Grade 1"`` .. ``"Grade 5"``.

    Returns:
        np.ndarray: Prediksi bertipe integer dengan nilai 1..5.
    """
    pred = np.asarray(pred)
    if pred.dtype.kind in "OUS":  # label teks, ambil angka di belakang kata "Grade"
        return np.array([int(str(p).split()[-1]) for p in pred])
    return pred.astype(int) + 1


def build_preprocessor(model_type="linear"):
    """Membuat ColumnTransformer sesuai jenis model.

    Args:
        model_type (str): ``"linear"`` untuk model linear (imputasi median,
            StandardScaler, One-Hot Encoding; ``ward_id`` tidak dipakai).
            ``"tree"`` untuk model berbasis pohon (imputasi median, Ordinal
            Encoding termasuk ``ward_id``).

    Returns:
        ColumnTransformer: Transformer yang siap dimasukkan ke Pipeline.

    Raises:
        ValueError: Jika ``model_type`` bukan "linear" atau "tree".
    """
    if model_type == "linear":
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
        return ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])

    if model_type == "tree":
        num_pipe = SimpleImputer(strategy="median")
        cat_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return ColumnTransformer([
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS + HIGH_CARD_COLS),
        ])

    raise ValueError("model_type harus 'linear' atau 'tree'")


def build_pipeline(model, model_type="linear"):
    """Menggabungkan feature engineering, preprocessing, dan model dalam satu Pipeline.

    Args:
        model: Estimator scikit-learn (atau yang kompatibel, mis. LightGBM).
        model_type (str): ``"linear"`` atau ``"tree"``, lihat ``build_preprocessor``.

    Returns:
        Pipeline: Pipeline dengan langkah ``features`` -> ``preprocess`` -> ``model``.
    """
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(model_type)),
        ("model", model),
    ])
