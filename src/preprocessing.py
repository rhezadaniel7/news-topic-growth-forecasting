from __future__ import annotations

import hashlib
import html
import re

import pandas as pd


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_data(frame: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    initial = len(out)
    missing_before = out.isna().sum().astype(int).to_dict()
    out["title"] = out[mapping["title"]].map(normalize_text)
    out["content"] = out[mapping["text"]].map(normalize_text)
    out["published_at"] = pd.to_datetime(out[mapping["date"]], errors="coerce", utc=True)
    out["source"] = out[mapping["source"]].map(normalize_text) if mapping.get("source") else "unknown"
    out["url"] = out[mapping["url"]].map(normalize_text) if mapping.get("url") else ""
    out["article_key"] = [hashlib.sha256(f"{u}|{t}".casefold().encode("utf-8")).hexdigest()[:20] for u, t in zip(out["url"], out["title"])]
    invalid_date = int(out["published_at"].isna().sum())
    empty_text = int((out["title"].eq("") & out["content"].eq("")).sum())
    out = out.dropna(subset=["published_at"])
    out = out.loc[~(out["title"].eq("") & out["content"].eq(""))].copy()
    before = len(out)
    if mapping.get("url"):
        out = out.loc[~(out["url"].ne("") & out["url"].duplicated(keep="first"))]
    duplicate_url = before - len(out)
    before = len(out)
    out = out.drop_duplicates(subset=["title"], keep="first")
    duplicate_title = before - len(out)
    out["document"] = (out["title"] + ". " + out["content"]).str.strip(". ")
    out["date"] = out["published_at"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None).dt.floor("D")
    out = out.sort_values("published_at").reset_index(drop=True)
    report = {
        "initial_rows": initial, "cleaned_rows": len(out), "invalid_dates_removed": invalid_date,
        "empty_text_removed": empty_text, "duplicate_url_removed": duplicate_url,
        "duplicate_title_removed": duplicate_title, "missing_before": missing_before,
        "date_min": out["published_at"].min(), "date_max": out["published_at"].max(),
        "sources": out["source"].value_counts().to_dict(),
    }
    return out, report

