"""Definisi skema label growth.

Modul ini adalah satu-satunya sumber kebenaran untuk cara label "growth"
dibentuk. Semua konsumen (step10_13 untuk audit label, step14_21 untuk
modeling) memakai fungsi di sini sehingga definisi label tidak pernah
berbeda antara audit dan training.

Kontrol leakage yang berlaku dan tidak diubah oleh modul ini:
1. threshold hanya boleh di-fit pada baris split training (`fit_threshold`
   menerima frame training saja dan menolak frame yang memuat split lain);
2. kolom skor (`growth_score`, `burst_score`) dan `future_volume` adalah
   sisi label, bukan predictor;
3. seleksi populasi hanya boleh memakai informasi yang tersedia pada origin
   date (di sini: `lag_1`, volume pada t-1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


MAD_SCALE = 1.4826
PRIMARY_GROWTH_QUANTILE = 0.80

# Skala robust minimum untuk definisi burst. Pada data ini 91,6% observasi
# eligible memiliki MAD historis = 0, sehingga median + k*MAD akan kolaps
# menjadi "future_volume > median". Floor 1.0 artikel menyatakan konvensi
# eksplisit: perubahan terkecil yang bermakna pada volume harian adalah satu
# artikel. Floor ini dipilih sebelum melihat metrik model.
BURST_SCALE_FLOOR = 1.0
BURST_K = 2.0

POPULATION_ALL = "all"
POPULATION_ACTIVE_BASELINE = "active_baseline"

THRESHOLD_TRAIN_QUANTILE = "train_quantile"
THRESHOLD_TRAIN_QUANTILE_POSITIVE = "train_quantile_positive"
THRESHOLD_FIXED = "fixed"


def growth_score_values(
    future_volume: pd.Series | np.ndarray,
    historical_median: pd.Series | np.ndarray,
    historical_mad: pd.Series | np.ndarray,
) -> np.ndarray:
    """Skor lama: (future - median) / (1 + 1.4826 * MAD)."""
    future_volume = np.asarray(future_volume, dtype=float)
    historical_median = np.asarray(historical_median, dtype=float)
    historical_mad = np.asarray(historical_mad, dtype=float)
    return (future_volume - historical_median) / (1.0 + MAD_SCALE * historical_mad)


def burst_score_values(
    future_volume: pd.Series | np.ndarray,
    historical_median: pd.Series | np.ndarray,
    historical_mad: pd.Series | np.ndarray,
) -> np.ndarray:
    """Skor burst: (future - median) / max(1.4826 * MAD, BURST_SCALE_FLOOR).

    `burst_score > k` ekuivalen dengan
    `future_volume > median + k * max(1.4826 * MAD, 1.0)`.
    """
    future_volume = np.asarray(future_volume, dtype=float)
    historical_median = np.asarray(historical_median, dtype=float)
    historical_mad = np.asarray(historical_mad, dtype=float)
    scale = np.maximum(MAD_SCALE * historical_mad, BURST_SCALE_FLOOR)
    return (future_volume - historical_median) / scale


@dataclass(frozen=True)
class LabelScheme:
    key: str
    name: str
    score_column: str
    population: str
    threshold_kind: str
    quantile: float | None
    fixed_threshold: float | None
    definition: str
    rationale: str

    @property
    def is_fitted(self) -> bool:
        return self.threshold_kind != THRESHOLD_FIXED


SCHEMES: dict[str, LabelScheme] = {
    "baseline_quantile_p80": LabelScheme(
        key="baseline_quantile_p80",
        name="S0 baseline kuantil p80",
        score_column="growth_score",
        population=POPULATION_ALL,
        threshold_kind=THRESHOLD_TRAIN_QUANTILE,
        quantile=PRIMARY_GROWTH_QUANTILE,
        fixed_threshold=None,
        definition="growth_score > kuantil p80 growth_score training",
        rationale=(
            "Skema yang berlaku sebelumnya. Karena zero-inflation, p80 training "
            "bernilai 0,0 sehingga aturan kolaps menjadi future_volume > median "
            "historis, dan untuk 91,5% observasi eligible median historis = 0. "
            "Label 'growth' praktis berarti 'besok ada artikel'."
        ),
    ),
    "active_baseline_p80": LabelScheme(
        key="active_baseline_p80",
        name="S1 baseline aktif (lag_1 > 0), kuantil p80",
        score_column="growth_score",
        population=POPULATION_ACTIVE_BASELINE,
        threshold_kind=THRESHOLD_TRAIN_QUANTILE,
        quantile=PRIMARY_GROWTH_QUANTILE,
        fixed_threshold=None,
        definition=(
            "hanya topic-day dengan lag_1 > 0; growth_score > kuantil p80 "
            "growth_score training pada populasi tersebut"
        ),
        rationale=(
            "Memperbaiki populasi, bukan threshold: growth hanya didefinisikan "
            "ketika ada baseline aktif untuk tumbuh dari. lag_1 diketahui pada "
            "origin date sehingga seleksi ini tidak melihat masa depan."
        ),
    ),
    "positive_only_p80": LabelScheme(
        key="positive_only_p80",
        name="S2 kuantil p80 pada growth_score > 0 saja",
        score_column="growth_score",
        population=POPULATION_ALL,
        threshold_kind=THRESHOLD_TRAIN_QUANTILE_POSITIVE,
        quantile=PRIMARY_GROWTH_QUANTILE,
        fixed_threshold=None,
        definition=(
            "growth_score > kuantil p80 dari growth_score training yang > 0"
        ),
        rationale=(
            "Memperbaiki threshold, bukan populasi: kuantil di-fit pada ekor "
            "positif sehingga tidak lagi jatuh di 0. Referensi kuantilnya "
            "dipilih berdasarkan outcome, dan untuk baris bermedian 0 aturannya "
            "tetap menjadi pernyataan volume absolut."
        ),
    ),
    "burst_mad_k2": LabelScheme(
        key="burst_mad_k2",
        name="S3 burst MAD k=2 (threshold tetap)",
        score_column="burst_score",
        population=POPULATION_ALL,
        threshold_kind=THRESHOLD_FIXED,
        quantile=None,
        fixed_threshold=BURST_K,
        definition=(
            "future_volume > median historis + 2 * max(1,4826 * MAD historis, 1,0)"
        ),
        rationale=(
            "Threshold tetap, tidak di-fit dari distribusi, sehingga arti label "
            "tidak berubah ketika distribusi volume bergeser antar split. Floor "
            "skala 1,0 artikel dinyatakan eksplisit untuk MAD = 0."
        ),
    ),
}

DEFAULT_SCHEME_KEY = "burst_mad_k2"
COMPARISON_ORDER = [
    "baseline_quantile_p80",
    "active_baseline_p80",
    "positive_only_p80",
    "burst_mad_k2",
]


def get_scheme(key: str | None = None) -> LabelScheme:
    """Skema aktif. Urutan prioritas: argumen, env LABEL_SCHEME, default."""
    resolved = key or os.environ.get("LABEL_SCHEME") or DEFAULT_SCHEME_KEY
    if resolved not in SCHEMES:
        raise ValueError(
            f"Skema label tidak dikenal: {resolved!r}. Pilihan: {sorted(SCHEMES)}"
        )
    return SCHEMES[resolved]


def _baseline_column(frame: pd.DataFrame) -> str:
    for column in ["lag_1", "baseline_volume"]:
        if column in frame.columns:
            return column
    raise RuntimeError(
        "Seleksi populasi baseline aktif membutuhkan kolom lag_1 atau baseline_volume."
    )


def population_mask(frame: pd.DataFrame, scheme: LabelScheme) -> pd.Series:
    """Baris yang termasuk populasi skema. Hanya memakai info pada origin date."""
    if scheme.population == POPULATION_ALL:
        return pd.Series(True, index=frame.index)
    if scheme.population == POPULATION_ACTIVE_BASELINE:
        baseline = frame[_baseline_column(frame)]
        return baseline.notna() & baseline.gt(0)
    raise ValueError(scheme.population)


def fit_threshold(
    train_frame: pd.DataFrame,
    scheme: LabelScheme,
    require_train_split: bool = False,
) -> float:
    """Threshold label, di-fit hanya dari baris yang diberikan pemanggil.

    Pemanggil bertanggung jawab mengirim partisi training saja: split training
    resmi pada audit label, atau training window fold pada temporal CV (di
    fold ekspansif, training window boleh memuat tanggal yang pada split resmi
    berlabel validation; itu adalah kontrol yang sudah berlaku dan tidak
    diubah di sini). `require_train_split=True` menegakkan pemeriksaan kolom
    `split` untuk pemanggil yang memang memakai split resmi.

    Untuk skema fixed, nilai dikembalikan tanpa melihat data sama sekali.
    """
    if scheme.threshold_kind == THRESHOLD_FIXED:
        return float(scheme.fixed_threshold)
    if require_train_split and "split" in train_frame.columns and len(train_frame):
        splits = set(pd.unique(train_frame["split"].dropna()))
        if not splits.issubset({"train"}):
            raise RuntimeError(
                f"Threshold hanya boleh di-fit pada split training; ditemukan {sorted(splits)}"
            )
    scores = train_frame.loc[population_mask(train_frame, scheme), scheme.score_column]
    scores = scores.loc[np.isfinite(scores)]
    if scheme.threshold_kind == THRESHOLD_TRAIN_QUANTILE_POSITIVE:
        scores = scores.loc[scores > 0]
    if scores.empty:
        raise RuntimeError(f"Tidak ada skor training untuk skema {scheme.key}")
    return float(scores.quantile(scheme.quantile))


def apply_labels(frame: pd.DataFrame, scheme: LabelScheme, threshold: float) -> np.ndarray:
    return frame[scheme.score_column].gt(threshold).astype("int8").to_numpy()


def class_balance(
    frame: pd.DataFrame,
    scheme: LabelScheme,
    threshold: float,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> list[dict]:
    rows = []
    included = frame.loc[population_mask(frame, scheme)]
    for split in splits:
        subset = included.loc[included["split"].eq(split)]
        labels = subset[scheme.score_column].gt(threshold)
        rows.append(
            {
                "scheme": scheme.key,
                "scheme_name": scheme.name,
                "score_column": scheme.score_column,
                "population": scheme.population,
                "threshold_kind": scheme.threshold_kind,
                "threshold_value": threshold,
                "split": split,
                "observations": int(len(subset)),
                "growth_count": int(labels.sum()),
                "normal_count": int((~labels).sum()),
                "growth_rate": float(labels.mean()) if len(labels) else np.nan,
                "active_topics": int(subset["topic_id"].nunique()),
            }
        )
    return rows
