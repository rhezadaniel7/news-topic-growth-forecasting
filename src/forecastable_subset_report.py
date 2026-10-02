"""Laporan subset forecastable: diagnosa sparsity, biaya cakupan, verdict.

Hanya membaca artefak yang sudah ada (prediksi test beku, hasil run subset,
statistik training). Tidak melatih model, tidak menyentuh definisi label, dan
tidak memilih apa pun berdasarkan hasil test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SUBSET_NAME = "forecastable_subset_v1"
SPARSITY_LOW_MAX = 0.50
SPARSITY_MEDIUM_MAX = 0.80
GAIN_MINIMUM_HORIZONS = 2


def training_statistics(root: Path) -> pd.DataFrame:
    daily = pd.read_csv(root / "results" / "tables" / "topic_daily_volume.csv", parse_dates=["date"])
    train = daily.loc[daily["split"].eq("train")]
    grouped = train.groupby("topic_id")["topic_volume"]
    stats = pd.DataFrame(
        {
            "train_days": grouped.size(),
            "train_active_days": grouped.apply(lambda values: int((values > 0).sum())),
            "train_documents": grouped.sum(),
        }
    ).reset_index()
    stats["train_zero_ratio"] = 1.0 - stats["train_active_days"] / stats["train_days"]
    stats["sparsity_train"] = stats["train_zero_ratio"].map(
        lambda ratio: "LOW" if ratio < SPARSITY_LOW_MAX else ("MEDIUM" if ratio < SPARSITY_MEDIUM_MAX else "HIGH")
    )
    history = pd.read_csv(root / "results" / "tables" / "topic_history_final.csv")[
        ["topic_id", "sparsity_category", "forecast_eligible"]
    ]
    return stats.merge(history, on="topic_id", how="left")


def sparsity_breakdown(root: Path, stats: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    predictions = pd.read_csv(root / "results" / "predictions" / "final_test_predictions.csv")
    frame = predictions.loc[predictions["horizon"].eq(horizon)].merge(
        stats[["topic_id", "sparsity_train"]], on="topic_id", how="left"
    )
    rows = []
    for group in ["LOW", "MEDIUM", "HIGH", "ALL"]:
        subset = frame if group == "ALL" else frame.loc[frame["sparsity_train"].eq(group)]
        if not len(subset):
            continue
        truth = subset["actual_label"].to_numpy()
        prediction = subset["predicted_label"].to_numpy()
        probability = subset["prediction_probability"].to_numpy()
        two_classes = 0 < truth.sum() < len(truth)
        rows.append(
            {
                "sparsity_group": group,
                "topics": int(subset["topic_id"].nunique()),
                "observations": int(len(subset)),
                "positives": int(truth.sum()),
                "prevalence": float(truth.mean()),
                "precision": float(precision_score(truth, prediction, zero_division=0)),
                "recall": float(recall_score(truth, prediction, zero_division=0)),
                "f1": float(f1_score(truth, prediction, zero_division=0)),
                "pr_auc": float(average_precision_score(truth, probability)) if two_classes else np.nan,
                "roc_auc": float(roc_auc_score(truth, probability)) if two_classes else np.nan,
            }
        )
    return pd.DataFrame(rows)


def comparison(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset_dir = root / "results" / "subset_runs" / SUBSET_NAME
    full = pd.read_csv(root / "results" / "tables" / "final_test_results.csv")
    sub = pd.read_csv(subset_dir / "tables" / "final_test_results.csv")
    full = full.loc[full["selected_final_model"]]
    sub = sub.loc[sub["selected_final_model"]]
    rows = []
    for horizon in [1, 2, 3]:
        for label, frame in [("full_topic_set", full), ("forecastable_subset", sub)]:
            record = frame.loc[frame["horizon"].eq(horizon)].iloc[0]
            prevalence = record["growth_test"] / record["n_test"]
            rows.append(
                {
                    "topic_set": label,
                    "horizon": horizon,
                    "model": record["model"],
                    "feature_set": record["feature_set"],
                    "n_test": int(record["n_test"]),
                    "positives_test": int(record["growth_test"]),
                    "test_prevalence": float(prevalence),
                    "validation_precision": float(record["validation_precision"]),
                    "validation_recall": float(record["validation_recall"]),
                    "validation_f1": float(record["validation_f1"]),
                    "validation_pr_auc": float(record["validation_pr_auc"]),
                    "validation_roc_auc": float(record["validation_roc_auc"]),
                    "test_precision": float(record["precision"]),
                    "test_recall": float(record["recall"]),
                    "test_f1": float(record["f1"]),
                    "test_pr_auc": float(record["pr_auc"]),
                    "test_roc_auc": float(record["roc_auc"]),
                    "test_pr_auc_lift_over_prevalence": float(record["pr_auc"] / prevalence),
                }
            )
    table = pd.DataFrame(rows)
    deltas = []
    for horizon in [1, 2, 3]:
        a = table.loc[table["topic_set"].eq("full_topic_set") & table["horizon"].eq(horizon)].iloc[0]
        b = table.loc[table["topic_set"].eq("forecastable_subset") & table["horizon"].eq(horizon)].iloc[0]
        deltas.append(
            {
                "horizon": horizon,
                "delta_validation_f1": b["validation_f1"] - a["validation_f1"],
                "delta_test_f1": b["test_f1"] - a["test_f1"],
                "delta_test_pr_auc": b["test_pr_auc"] - a["test_pr_auc"],
                "delta_test_roc_auc": b["test_roc_auc"] - a["test_roc_auc"],
                "delta_test_pr_auc_lift": b["test_pr_auc_lift_over_prevalence"]
                - a["test_pr_auc_lift_over_prevalence"],
                "test_prevalence_ratio": b["test_prevalence"] / a["test_prevalence"],
            }
        )
    return table, pd.DataFrame(deltas)


def coverage_cost(root: Path, frozen: dict) -> pd.DataFrame:
    subset_dir = root / "results" / "subset_runs" / SUBSET_NAME
    full = pd.read_csv(root / "results" / "tables" / "final_feature_dataset_summary.csv")
    sub = pd.read_csv(subset_dir / "tables" / "final_feature_dataset_summary.csv")
    full = full.loc[full["record_type"].eq("partition")]
    sub = sub.loc[sub["record_type"].eq("partition")]
    rows = []
    for horizon in [1, 2, 3]:
        for split in ["train", "validation", "test"]:
            a = full.loc[full["horizon"].eq(horizon) & full["split"].eq(split)].iloc[0]
            b = sub.loc[sub["horizon"].eq(horizon) & sub["split"].eq(split)].iloc[0]
            rows.append(
                {
                    "horizon": horizon,
                    "split": split,
                    "observations_full": int(a["rows"]),
                    "observations_subset": int(b["rows"]),
                    "observations_dropped": int(a["rows"] - b["rows"]),
                    "observations_dropped_percent": 100.0 * (a["rows"] - b["rows"]) / a["rows"],
                    "positives_full": int(a["growth"]),
                    "positives_subset": int(b["growth"]),
                    "positives_dropped_percent": 100.0 * (a["growth"] - b["growth"]) / a["growth"],
                    "prevalence_full": a["growth"] / a["rows"],
                    "prevalence_subset": b["growth"] / b["rows"],
                }
            )
    return pd.DataFrame(rows)


def verdict(breakdown: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    """Verdict otomatis. Ambang ditetapkan sebelum hasil dilihat."""
    groups = breakdown.set_index("sparsity_group")
    have = [g for g in ["LOW", "MEDIUM", "HIGH"] if g in groups.index]
    f1_by_group = groups.loc[have, "f1"]
    # Hipotesis sparsity ditolak jika F1 merata: spread antar kelompok < 0.10.
    spread = float(f1_by_group.max() - f1_by_group.min())
    monotone = bool(
        len(have) == 3
        and groups.loc["LOW", "f1"] > groups.loc["MEDIUM", "f1"] > groups.loc["HIGH", "f1"]
    )
    horizons_up = int((deltas["delta_test_f1"] > 0).sum())
    if spread < 0.10:
        return {
            "code": "c",
            "text": "hipotesis sparsity ditolak di step 1",
            "detail": f"spread F1 antar kelompok sparsity hanya {spread:.4f} (< 0,10); subset tidak akan menolong",
            "step1_spread": spread,
            "step1_monotone": monotone,
            "horizons_f1_up": horizons_up,
            "reliable": False,
        }
    if horizons_up >= GAIN_MINIMUM_HORIZONS:
        return {
            "code": "a",
            "text": "F1 naik sah via penyempitan task",
            "detail": f"F1 test naik pada {horizons_up}/3 horizon",
            "step1_spread": spread,
            "step1_monotone": monotone,
            "horizons_f1_up": horizons_up,
            "reliable": True,
        }
    return {
        "code": "b",
        "text": "F1 tidak naik konsisten — subset tidak membantu",
        "detail": f"F1 test naik hanya pada {horizons_up}/3 horizon",
        "step1_spread": spread,
        "step1_monotone": monotone,
        "horizons_f1_up": horizons_up,
        "reliable": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = root / "results" / "tables"
    frozen = json.loads(
        (root / "data" / "processed" / f"{SUBSET_NAME}.json").read_text(encoding="utf-8")
    )
    stats = training_statistics(root)
    breakdown = sparsity_breakdown(root, stats, horizon=1)
    table, deltas = comparison(root)
    coverage = coverage_cost(root, frozen)
    decision = verdict(breakdown, deltas)

    stats.to_csv(tables / "forecastable_training_statistics.csv", index=False)
    breakdown.to_csv(tables / "forecastable_step1_sparsity_breakdown.csv", index=False)
    table.to_csv(tables / "forecastable_full_vs_subset.csv", index=False)
    deltas.to_csv(tables / "forecastable_deltas.csv", index=False)
    coverage.to_csv(tables / "forecastable_coverage_cost.csv", index=False)
    pd.DataFrame([decision]).to_csv(tables / "forecastable_verdict.csv", index=False)

    topics_total = int(len(stats))
    kept = int(frozen["topic_count"])
    test_coverage = coverage.loc[coverage["split"].eq("test") & coverage["horizon"].eq(1)].iloc[0]
    horizons_note = (
        "konsisten (>=2 dari 3 horizon)"
        if decision["horizons_f1_up"] >= GAIN_MINIMUM_HORIZONS
        else "TIDAK ANDAL (hanya 1 horizon)"
    )
    marginal = deltas.loc[deltas["delta_test_f1"].abs() < 0.01, "horizon"].tolist()

    report = f"""# Subset forecastable — laporan

Skema label **tetap beku**: `{frozen['label_scheme']}`, `{frozen['label_invariance']}`.
Tidak ada definisi label yang diubah, tidak ada threshold yang dipilih demi angka,
dan test dievaluasi sekali di akhir untuk satu konfigurasi yang sudah dibekukan
dari official validation.

## STEP 1 — Diagnosa sumber F1 rendah

F1 test t+1 pada model dan label beku, dipecah per kelompok sparsity yang dihitung
dari **training window saja** (270 hari). Catatan: `sparsity_category` yang tersimpan
di `topic_history_final.csv` dihitung atas seluruh periode termasuk test, sehingga
TIDAK dipakai untuk mengelompokkan maupun menyeleksi di sini.

{breakdown.to_markdown(index=False, floatfmt=".4f")}

Spread F1 antar kelompok: **{decision['step1_spread']:.4f}**. Monoton LOW > MEDIUM > HIGH:
**{decision['step1_monotone']}**. Hipotesis bahwa F1 rendah terkonsentrasi di topik
high-sparsity: **{'TERKONFIRMASI' if decision['code'] != 'c' else 'DITOLAK'}**.

Peringatan interpretasi yang harus dibaca bersama tabel di atas: ROC-AUC justru
tertinggi pada kelompok HIGH. Model me-ranking topik high-sparsity dengan baik;
yang kolaps adalah precision karena prevalensi positif di kelompok itu sangat
rendah. Artinya F1 di sini sebagian besar didorong base rate, bukan kualitas
diskriminasi.

## STEP 2 — Kriteria subset, dibekukan dari training saja

```json
{json.dumps(frozen['criteria'], indent=2)}
```

Sumber kriteria: {frozen['criteria_source']}

Catatan: {frozen['criteria_note']}

Kriteria dibekukan ke `data/processed/{SUBSET_NAME}.json` sebelum validation dan
test subset dilihat. Pipeline menolak berjalan jika berkas itu tidak menandai
dirinya beku dan berbasis training window saja, dan menolak berjalan jika skema
label memakai threshold yang di-fit (karena pembatasan topik akan mengubah label).

Topik lolos: **{kept} dari {topics_total}** ({100.0 * kept / topics_total:.1f}%).
Distribusi kategori training pada subset:
{stats.loc[stats['topic_id'].isin(frozen['topics']), 'sparsity_train'].value_counts().to_dict()}

## STEP 3 — Prosedur run subset

Identik run utama: temporal CV 3 fold expanding untuk memilih hyperparameter dan
feature set, official validation untuk memilih model final dan probability
threshold, refit train+validation, lalu test satu kali. Tiga horizon, tiga feature
set, empat learner. Gate validitas: Temporal Only berjalan di ketiga horizon dan
Majority Class tetap dilaporkan (F1 growth 0,0000 di semua horizon).

## STEP 4 — Perbandingan full-topic-set vs forecastable-subset

{table.to_markdown(index=False, floatfmt=".4f")}

Selisih:

{deltas.to_markdown(index=False, floatfmt="+.4f")}

### Biaya cakupan

{coverage.to_markdown(index=False, floatfmt=".4f")}

Subset membuang **{topics_total - kept} dari {topics_total} topik**
({100.0 * (topics_total - kept) / topics_total:.1f}%) dan
**{test_coverage['observations_dropped_percent']:.1f}% observasi test**
({int(test_coverage['observations_dropped'])} dari {int(test_coverage['observations_full'])}).
Positif test yang dibuang: {test_coverage['positives_dropped_percent']:.1f}%.
Prevalensi test naik dari {test_coverage['prevalence_full']:.4f} menjadi
{test_coverage['prevalence_subset']:.4f}.

Konsistensi kenaikan F1 test: naik pada **{decision['horizons_f1_up']}/3 horizon** — {horizons_note}.
"""
    if marginal:
        report += (
            f"\nHorizon dengan kenaikan di bawah 0,01 (praktis datar): "
            f"{', '.join(f't+{int(h)}' for h in marginal)}.\n"
        )

    report += f"""
### Yang tidak boleh disembunyikan dari reviewer

1. **Cakupan turun drastis.** Task menyempit dari {topics_total} menjadi {kept} topik;
   {test_coverage['observations_dropped_percent']:.1f}% observasi test tidak lagi diprediksi.
   Model subset tidak menjawab pertanyaan yang sama dengan model penuh.
2. **Kenaikan F1 sebagian besar efek prevalensi.** Prevalensi test naik
   {deltas['test_prevalence_ratio'].mean():.2f}x. F1 dan PR-AUC keduanya sensitif base rate.
3. **Diskriminasi justru menurun.** ROC-AUC test turun di ketiga horizon
   ({', '.join(f"{value:+.4f}" for value in deltas['delta_test_roc_auc'])}), dan PR-AUC
   relatif prevalensi turun tajam
   ({', '.join(f"{value:+.2f}x" for value in deltas['delta_test_pr_auc_lift'])}).
   Pada subset, model lebih dekat ke baseline acaknya sendiri daripada model penuh.
4. Angka lama dan angka baru berasal dari label yang identik. Itu sah hanya karena
   threshold label `burst_mad_k2` tetap; dengan skema kuantil, perbandingan ini batal.

## STEP 5 — Verdict

**({decision['code']}) {decision['text']}** — {decision['detail']}.

F1 test: {' / '.join(f"{value:.4f}" for value in table.loc[table['topic_set'].eq('full_topic_set'), 'test_f1'])} (full)
menjadi {' / '.join(f"{value:.4f}" for value in table.loc[table['topic_set'].eq('forecastable_subset'), 'test_f1'])} (subset)
untuk t+1/t+2/t+3.

Biaya cakupan: {100.0 * (topics_total - kept) / topics_total:.1f}% topik dan
{test_coverage['observations_dropped_percent']:.1f}% observasi test dibuang.

Kualifikasi wajib: kenaikan ini adalah penyempitan task yang sah secara prosedur,
BUKAN bukti model menjadi lebih baik. ROC-AUC dan lift atas prevalensi turun di
ketiga horizon. Angka {table.loc[table['topic_set'].eq('forecastable_subset') & table['horizon'].eq(1), 'test_f1'].iloc[0]:.4f}
pada t+1 hanya boleh dilaporkan bersama cakupan yang dikorbankan dan bersama
ROC-AUC-nya.
"""
    (root / "results" / "forecastable_subset_report.md").write_text(report, encoding="utf-8")
    print(f"VERDICT ({decision['code']}): {decision['text']} — {decision['detail']}")
    print(f"horizons F1 naik: {decision['horizons_f1_up']}/3 | step1 spread: {decision['step1_spread']:.4f}")


if __name__ == "__main__":
    main()
