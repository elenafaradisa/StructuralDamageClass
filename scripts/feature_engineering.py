"""Feature engineering untuk data kerusakan bangunan (gempa Gorkha, Nepal).

Semua transformasi dibuat sebagai transformer scikit-learn agar bisa dipakai
langsung di dalam Pipeline (tanpa data leakage saat cross-validation).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Kode "umur tidak diketahui" yang ditemukan saat EDA
UNKNOWN_AGE_CODE = 999

# Kolom ID wilayah diperlakukan sebagai kategori, bukan angka
LOCATION_COLS = ["district_id", "vdcmun_id", "ward_id"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Membersihkan nilai tidak wajar dan menambah fitur turunan.

    Transformasi yang dilakukan:
        - ``age_building`` bernilai 999 diganti NaN, lalu dibuat flag ``age_unknown``.
        - ``height_per_floor``: tinggi bangunan dibagi jumlah lantai.
        - ``volume_proxy``: luas dasar dikali tinggi (perkiraan volume bangunan).
        - Kolom ID wilayah diubah menjadi string agar dibaca sebagai kategori.
        - Kolom ``building_id`` dibuang karena hanya identitas.

    Transformer ini tidak mempelajari apa pun dari data (stateless), sehingga
    aman dipakai di dalam cross-validation.
    """

    def fit(self, X, y=None):
        """Tidak ada parameter yang dipelajari.

        Args:
            X (pd.DataFrame): Data fitur.
            y: Diabaikan, ada demi kompatibilitas Pipeline.

        Returns:
            FeatureEngineer: Objek itu sendiri.
        """
        return self

    def transform(self, X):
        """Menerapkan pembersihan dan penambahan fitur.

        Args:
            X (pd.DataFrame): Data fitur mentah.

        Returns:
            pd.DataFrame: Data dengan fitur baru; ``building_id`` sudah dibuang.
        """
        df = X.copy()

        df["age_unknown"] = (df["age_building"] == UNKNOWN_AGE_CODE).astype(int)
        df["age_building"] = df["age_building"].replace(UNKNOWN_AGE_CODE, np.nan)

        df["height_per_floor"] = df["height_ft_pre_eq"] / df["count_floors_pre_eq"].clip(lower=1)
        df["volume_proxy"] = df["plinth_area_sq_ft"] * df["height_ft_pre_eq"]

        for col in LOCATION_COLS:
            df[col] = df[col].astype(str)

        return df.drop(columns=["building_id"], errors="ignore")
