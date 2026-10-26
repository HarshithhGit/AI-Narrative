"""
dashboard_report.py
===================
Builds a landscape, dashboard-style PDF that showcases EVERY feature of the
Enterprise AI Data Analyst Suite PRO and embeds the LLM-written narrative.

Pipeline:
    compute_report_facts()  ->  facts (JSON-safe, sent to LLM) + artifacts (plot data)
    LLMNarrator.generate()  ->  narrative (dict)             [llm_narrator.py]
    DashboardPDFReport.build() -> BytesIO with an 8-page dashboard PDF
"""

import datetime
import io
import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from llm_narrator import key_findings

warnings.filterwarnings("ignore")
logger = logging.getLogger("DashboardReport")

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
NAVY, BLUE, TEAL, AMBER, RED = "#0F172A", "#2563EB", "#14B8A6", "#F59E0B", "#EF4444"
SLATE, LIGHT, BORDER, INDIGO, TEXT = "#64748B", "#F1F5F9", "#E2E8F0", "#6366F1", "#334155"
PALETTE = [BLUE, TEAL, AMBER, INDIGO, RED, "#8B5CF6", "#10B981", "#F97316"]

W, H = landscape(letter)  # 792 x 612
MARGIN, TOP, BOT, GAP = 30, H - 60, 34, 10
TOTAL_PAGES = 8

plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.titlesize": 8,
        "figure.facecolor": "white",
    }
)


# ============================================================================
# 1. FACTS ENGINE
# ============================================================================
def _f(x: Any, nd: int = 4) -> Optional[float]:
    try:
        x = float(x)
        return round(x, nd) if np.isfinite(x) else None
    except Exception:
        return None


def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    s = df[col]
    n = len(s)
    if n == 0:
        return False
    name = col.lower()
    ratio = s.nunique(dropna=True) / n
    if name in ("id", "index") or name.endswith("_id") or name.startswith("id_"):
        return ratio > 0.9
    return bool(pd.api.types.is_integer_dtype(s) and ratio > 0.98 and n > 20)


def _model_pool(df: pd.DataFrame, num_cols: List[str]) -> List[str]:
    return [c for c in num_cols if c in df.columns and not _is_id_like(df, c) and df[c].nunique() > 1]


def _run_regression(df: pd.DataFrame, num_cols: List[str], art: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pool = _model_pool(df, num_cols)
    if len(pool) < 2:
        return None
    try:
        corr = df[pool].corr().abs()
        corr = corr.where(~np.eye(len(corr), dtype=bool), 0)
        target = corr.mean().idxmax()
        feats = [c for c in pool if c != target][:12]
        data = df[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
        if len(data) > 20000:
            data = data.sample(20000, random_state=42)
        if len(data) < 30:
            return None
        X = data[feats].fillna(data[feats].median())
        y = data[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        }
        results, preds = {}, {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            preds[name] = p
            results[name] = {
                "r2": _f(r2_score(yte, p)),
                "rmse": _f(np.sqrt(mean_squared_error(yte, p))),
                "mae": _f(mean_absolute_error(yte, p)),
            }
        best = max(results, key=lambda k: results[k]["r2"] if results[k]["r2"] is not None else -9)
        rf = models["Random Forest"]
        imp = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])[:6]
        art["reg_y_test"], art["reg_pred"] = yte.values, preds[best]
        art["reg_importance"] = [(a, float(b)) for a, b in imp]
        return {
            "target": target,
            "features": feats,
            "train_rows": int(len(Xtr)),
            "test_rows": int(len(Xte)),
            "best_model": best,
            "r2": results[best]["r2"],
            "rmse": results[best]["rmse"],
            "mae": results[best]["mae"],
            "models": results,
            "top_features": [{"feature": a, "importance": _f(b)} for a, b in imp],
        }
    except Exception as exc:
        logger.warning("Regression skipped: %s", exc)
        return None


def _run_classification(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], art: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pool = _model_pool(df, num_cols)
    cands = [c for c in cat_cols if c in df.columns and 2 <= df[c].nunique() <= 10]
    if not pool or not cands:
        return None
    try:
        target = cands[0]
        feats = pool[:12]
        data = df[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
        data[target] = data[target].astype(str)
        vc = data[target].value_counts()
        data = data[data[target].isin(vc[vc >= 5].index)]
        if data[target].nunique() < 2:
            return None
        if len(data) > 20000:
            data = data.sample(20000, random_state=42)
        if len(data) < 30:
            return None
        X = data[feats].fillna(data[feats].median())
        le = LabelEncoder()
        y = le.fit_transform(data[target])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        models = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        }
        results, preds = {}, {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            preds[name] = p
            results[name] = {
                "accuracy": _f(accuracy_score(yte, p)),
                "f1": _f(f1_score(yte, p, average="weighted", zero_division=0)),
            }
        best = max(results, key=lambda k: results[k]["f1"] or -1)
        art["cls_cm"] = confusion_matrix(yte, preds[best])
        art["cls_labels"] = [str(c) for c in le.classes_]
        return {
            "target": target,
            "classes": [str(c) for c in le.classes_],
            "features": feats,
            "best_model": best,
            "accuracy": results[best]["accuracy"],
            "f1": results[best]["f1"],
            "models": results,
        }
    except Exception as exc:
        logger.warning("Classification skipped: %s", exc)
        return None


def compute_report_facts(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (facts, artifacts). `facts` is small + JSON-safe (goes to the LLM)."""
    df = df.copy()
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    orig_num_n = len(num_cols)
    num_cols = [c for c in num_cols if not _is_id_like(df, c)] or num_cols  # drop ID-like columns from analysis
    art: Dict[str, Any] = {}
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)
    miss_cells = int(df.isnull().sum().sum())

    facts: Dict[str, Any] = {
        "dataset": {
            "rows": int(n_rows),
            "cols": int(n_cols),
            "numeric_cols": orig_num_n,
            "categorical_cols": len(cat_cols),
            "missing_cells": miss_cells,
            "completeness_pct": round(100 * (1 - miss_cells / total_cells), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_names": [str(c) for c in df.columns[:30]],
        }
    }

    # --- missing values
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    facts["missing_top"] = [{"col": str(c), "pct": round(float(p), 2)} for c, p in miss.items() if p > 0][:8]

    # --- numeric summary + outliers
    numeric, outliers_all = [], []
    for col in num_cols[:12]:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        mean = float(s.mean())
        row = {
            "col": str(col),
            "mean": _f(mean),
            "median": _f(s.median()),
            "std": _f(s.std()),
            "min": _f(s.min()),
            "max": _f(s.max()),
            "skew": _f(s.skew()),
            "cv": _f(abs(s.std() / mean)) if mean else None,
            "outliers_iqr": out,
        }
        outliers_all.append((str(col), out))
        if len(numeric) < 8:
            numeric.append(row)
    facts["numeric"] = numeric
    art["outliers_all"] = outliers_all

    # --- categorical summary
    categorical = []
    for col in cat_cols[:6]:
        vc = df[col].value_counts(dropna=True)
        if vc.empty:
            continue
        categorical.append(
            {
                "col": str(col),
                "unique": int(df[col].nunique()),
                "top_value": str(vc.index[0]),
                "top_share_pct": round(100 * float(vc.iloc[0]) / max(int(vc.sum()), 1), 1),
            }
        )
    facts["categorical"] = categorical

    # --- correlations
    pool = _model_pool(df, num_cols)[:10]
    facts["top_correlations"] = []
    if len(pool) >= 2:
        corr = df[pool].corr()
        art["corr"] = corr
        pairs = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                r = corr.iloc[i, j]
                if np.isfinite(r):
                    pairs.append((pool[i], pool[j], float(r)))
        pairs.sort(key=lambda t: -abs(t[2]))
        facts["top_correlations"] = [{"a": a, "b": b, "r": round(r, 3)} for a, b, r in pairs[:5]]

    # --- Shapiro-Wilk
    normality = []
    for col in num_cols[:4]:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) >= 3 and s.nunique() > 1:
            try:
                sample = s.sample(5000, random_state=42) if len(s) > 5000 else s
                w_stat, p = stats.shapiro(sample)
                normality.append({"col": str(col), "W": _f(w_stat), "p": _f(p, 6), "normal": bool(p > 0.05)})
            except Exception:
                pass
    facts["normality"] = normality

    # --- ANOVA
    anova = []
    for cat in [c for c in cat_cols if 2 <= df[c].nunique() <= 10][:3]:
        for num in num_cols[:5]:
            try:
                groups = [g[num].dropna().values for _, g in df.groupby(cat) if g[num].notna().sum() >= 2]
                if len(groups) >= 2:
                    F, p = stats.f_oneway(*groups)
                    if np.isfinite(F):
                        anova.append({"num": str(num), "cat": str(cat), "F": _f(F, 3), "p": _f(p, 6), "significant": bool(p < 0.05)})
            except Exception:
                pass
    anova.sort(key=lambda r: (r["p"] if r["p"] is not None else 1))
    facts["anova"] = anova[:4]

    # --- Chi-square
    chi = []
    ccats = [c for c in cat_cols if 2 <= df[c].nunique() <= 15][:4]
    for i in range(len(ccats)):
        for j in range(i + 1, len(ccats)):
            try:
                ct = pd.crosstab(df[ccats[i]], df[ccats[j]])
                if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                    c2, p, _, _ = stats.chi2_contingency(ct)
                    v = np.sqrt(c2 / (ct.values.sum() * (min(ct.shape) - 1)))
                    chi.append({"a": str(ccats[i]), "b": str(ccats[j]), "chi2": _f(c2, 2), "p": _f(p, 6), "cramers_v": _f(v, 3), "significant": bool(p < 0.05)})
            except Exception:
                pass
    chi.sort(key=lambda r: (r["p"] if r["p"] is not None else 1))
    facts["chi_square"] = chi[:3]

    # --- ML
    facts["ml"] = {
        "regression": _run_regression(df, num_cols, art),
        "classification": _run_classification(df, num_cols, cat_cols, art),
    }
    return facts, art


# ============================================================================
# 2. CHARTS (matplotlib -> ImageReader). Sizes are in PDF points.
# ============================================================================
def _new(w: float, h: float):
    fig, ax = plt.subplots(figsize=(max(w, 40) / 72, max(h, 40) / 72), constrained_layout=True)
    return fig, ax


def _img(fig) -> ImageReader:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def _placeholder(w: float, h: float, msg: str) -> ImageReader:
    fig, ax = _new(w, h)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=SLATE, fontsize=8, wrap=True, transform=ax.transAxes)
    return _img(fig)


def chart_hist(s: pd.Series, w, h, color=BLUE) -> ImageReader:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return _placeholder(w, h, "No numeric data")
    fig, ax = _new(w, h)
    ax.hist(s, bins=min(30, max(8, int(np.sqrt(len(s))))), color=color, alpha=0.85, edgecolor="white")
    ax.axvline(s.mean(), color=RED, ls="--", lw=1)
    ax.text(0.98, 0.95, f"mean {s.mean():,.2f}", transform=ax.transAxes, ha="right", va="top", color=RED, fontsize=7)
    ax.set_xlabel(str(s.name))
    ax.set_ylabel("Frequency")
    return _img(fig)


def chart_missing(df: pd.DataFrame, w, h) -> ImageReader:
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(10)
    if miss.empty:
        return _placeholder(w, h, "100% complete: no missing values")
    fig, ax = _new(w, h)
    ax.barh(miss.index.astype(str)[::-1], miss.values[::-1], color=AMBER)
    ax.set_xlabel("% missing")
    for i, v in enumerate(miss.values[::-1]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=6.5)
    return _img(fig)


def chart_dtype_donut(n_num, n_cat, n_other, w, h) -> ImageReader:
    vals = [(n_num, "Numeric", BLUE), (n_cat, "Categorical", TEAL), (n_other, "Other", AMBER)]
    vals = [v for v in vals if v[0] > 0]
    fig, ax = _new(w, h)
    ax.pie([v[0] for v in vals], labels=[f"{v[1]} ({v[0]})" for v in vals], colors=[v[2] for v in vals],
           startangle=90, wedgeprops=dict(width=0.38, edgecolor="white"), textprops=dict(fontsize=7))
    return _img(fig)


def chart_outliers(outliers: List[Tuple[str, int]], w, h) -> ImageReader:
    outliers = sorted(outliers, key=lambda t: -t[1])[:10]
    if not outliers:
        return _placeholder(w, h, "No numeric columns")
    fig, ax = _new(w, h)
    names, vals = [o[0] for o in outliers][::-1], [o[1] for o in outliers][::-1]
    ax.barh(names, vals, color=[RED if v > 0 else TEAL for v in vals])
    ax.set_xlabel("IQR outlier count (1.5 x IQR rule)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=6.5)
    return _img(fig)


def chart_cat_bar(df: pd.DataFrame, col: str, w, h) -> ImageReader:
    vc = df[col].astype(str).value_counts().head(8)
    fig, ax = _new(w, h)
    ax.barh(vc.index[::-1], vc.values[::-1], color=BLUE)
    ax.set_xlabel("Records")
    ax.set_title(f"Top values: {col}", loc="left", fontweight="bold")
    for i, v in enumerate(vc.values[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=6.5)
    return _img(fig)


def chart_donut_cat(df: pd.DataFrame, col: str, w, h) -> ImageReader:
    vc = df[col].astype(str).value_counts()
    top = vc.head(5)
    if len(vc) > 5:
        top = pd.concat([top, pd.Series({"Other": vc.iloc[5:].sum()})])
    fig, ax = _new(w, h)
    ax.pie(top.values, labels=top.index, colors=PALETTE[: len(top)], startangle=90,
           autopct="%1.0f%%", pctdistance=0.78, wedgeprops=dict(width=0.4, edgecolor="white"),
           textprops=dict(fontsize=6.5))
    return _img(fig)


def chart_group_bar(df: pd.DataFrame, num: str, cat: str, w, h) -> ImageReader:
    g = df.groupby(cat)[num].mean().sort_values(ascending=False).head(8)
    fig, ax = _new(w, h)
    ax.bar(g.index.astype(str), g.values, color=PALETTE[: len(g)])
    ax.set_ylabel(f"Mean {num}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6.5)
    return _img(fig)


def chart_line(df: pd.DataFrame, col: str, w, h) -> ImageReader:
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(s) > 3000:
        s = s.iloc[:: max(1, len(s) // 3000)].reset_index(drop=True)
    fig, ax = _new(w, h)
    ax.plot(s.index, s.values, color=BLUE, alpha=0.3, lw=0.7)
    ax.plot(s.index, s.rolling(max(3, len(s) // 25), min_periods=1).mean(), color=BLUE, lw=1.6)
    ax.set_xlabel("Record order")
    ax.set_ylabel(col)
    return _img(fig)


def chart_scatter(df: pd.DataFrame, x: str, y: str, w, h) -> ImageReader:
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) > 3000:
        d = d.sample(3000, random_state=1)
    fig, ax = _new(w, h)
    ax.scatter(d[x], d[y], s=6, color=TEAL, alpha=0.5)
    if len(d) > 2 and d[x].nunique() > 1:
        m, b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xs, m * xs + b, color=RED, lw=1.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return _img(fig)


def chart_gauge(value: float, label: str, w, h) -> ImageReader:
    fig, ax = _new(w, h)
    ax.set_aspect("equal")
    ax.axis("off")
    col = TEAL if value >= 90 else AMBER if value >= 70 else RED
    ax.add_patch(Wedge((0, 0), 1, 0, 180, width=0.28, color=BORDER))
    ax.add_patch(Wedge((0, 0), 1, 180 - 1.8 * max(0, min(value, 100)), 180, width=0.28, color=col))
    ax.text(0, 0.18, f"{value:.1f}%", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0, -0.08, label, ha="center", fontsize=7, color=SLATE)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.2, 1.1)
    return _img(fig)


def chart_waterfall(df: pd.DataFrame, num: str, cat: str, w, h) -> ImageReader:
    g = df.groupby(cat)[num].sum().sort_values(ascending=False).head(5)
    vals = g.values.astype(float)
    base = np.concatenate([[0], np.cumsum(vals)[:-1]])
    fig, ax = _new(w, h)
    ax.bar(range(len(vals)), vals, bottom=base, color=[TEAL if v >= 0 else RED for v in vals])
    ax.bar(len(vals), vals.sum(), color=NAVY)
    ax.set_xticks(range(len(vals) + 1))
    ax.set_xticklabels(list(g.index.astype(str)) + ["Total"], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel(f"Sum of {num}")
    return _img(fig)


def chart_corr(corr: pd.DataFrame, w, h) -> ImageReader:
    fig, ax = _new(w, h)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(c)[:12] for c in corr.columns], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels([str(c)[:12] for c in corr.index], fontsize=6.5)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.55 else NAVY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    return _img(fig)


def chart_box(df: pd.DataFrame, num: str, cat: str, w, h) -> ImageReader:
    cats = df[cat].astype(str).value_counts().head(6).index.tolist()
    data = [df.loc[df[cat].astype(str) == c, num].dropna().values for c in cats]
    fig, ax = _new(w, h)
    bp = ax.boxplot(data, labels=cats, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_ylabel(num)
    ax.set_title(f"{num} by {cat}", loc="left", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=6.5)
    return _img(fig)


def chart_pvalues(facts: Dict[str, Any], w, h) -> ImageReader:
    items = []
    for a in facts.get("anova", []):
        items.append((f"ANOVA {a['num']}~{a['cat']}", a["p"]))
    for c in facts.get("chi_square", []):
        items.append((f"Chi2 {c['a']}x{c['b']}", c["p"]))
    for n in facts.get("normality", []):
        items.append((f"Shapiro {n['col']}", n["p"]))
    items = [(n[:26], max(p, 1e-12)) for n, p in items if p is not None][:9]
    if not items:
        return _placeholder(w, h, "No tests could be run")
    fig, ax = _new(w, h)
    vals = [min(-np.log10(p), 10.0) for _, p in items]
    ax.barh([n for n, _ in items][::-1], vals[::-1], color=[RED if v > 1.301 else SLATE for v in vals[::-1]])
    ax.axvline(1.301, color=NAVY, ls="--", lw=1)
    ax.text(1.4, 0.99, "p = 0.05", fontsize=6.5, color=NAVY, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("-log10(p-value)   (longer bar = stronger evidence)")
    return _img(fig)


def chart_pred_actual(y, p, w, h) -> ImageReader:
    fig, ax = _new(w, h)
    ax.scatter(y, p, s=7, color=BLUE, alpha=0.5)
    lo, hi = float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))
    ax.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return _img(fig)


def chart_importance(imp: List[Tuple[str, float]], w, h) -> ImageReader:
    fig, ax = _new(w, h)
    ax.barh([a[:18] for a, _ in imp][::-1], [b for _, b in imp][::-1], color=TEAL)
    ax.set_xlabel("Random Forest importance")
    return _img(fig)


def chart_confusion(cm: np.ndarray, labels: List[str], w, h) -> ImageReader:
    fig, ax = _new(w, h)
    ax.imshow(cm, cmap="Blues")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l[:10] for l in labels], rotation=30, ha="right", fontsize=6.5)
    ax.set_yticklabels([l[:10] for l in labels], fontsize=6.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else NAVY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return _img(fig)


# ============================================================================
# 3. PDF DRAWING HELPERS
# ============================================================================
def _clean(text: Any) -> str:
    """Make text safe for Helvetica (drops emojis / unsupported glyphs, strips markdown)."""
    t = str(text if text is not None else "")
    t = re.sub(r"\*{1,3}|`+|^#+\s*", "", t, flags=re.M)
    return t.encode("cp1252", "ignore").decode("cp1252")


def _hex(c: str):
    return colors.HexColor(c)


def _frame(c: rl_canvas.Canvas, title: str, subtitle: str, page: int, doc_title: str, stamp: str, total: Optional[int] = None):
    c.setFillColor(_hex(LIGHT))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(_hex(NAVY))
    c.rect(0, H - 50, W, 50, fill=1, stroke=0)
    c.setFillColor(_hex(BLUE))
    c.rect(0, H - 53, W, 3, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN, H - 29, _clean(title))
    c.setFont("Helvetica", 8)
    c.setFillColor(_hex("#94A3B8"))
    c.drawString(MARGIN, H - 43, _clean(subtitle))
    c.drawRightString(W - MARGIN, H - 29, _clean(doc_title)[:70])
    c.drawRightString(W - MARGIN, H - 43, stamp)
    c.setFont("Helvetica", 7)
    c.setFillColor(_hex(SLATE))
    c.drawString(MARGIN, 16, "Enterprise AI Data Analyst Suite PRO  |  AI-narrated dashboard report")
    c.drawRightString(W - MARGIN, 16, f"Page {page} of {total or TOTAL_PAGES}")


def _panel(c, x, y, w, h, title: str = ""):
    """White rounded card with a title. Returns inner (x, y, w, h)."""
    c.setFillColor(colors.white)
    c.setStrokeColor(_hex(BORDER))
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    inner_top = h - 8
    if title:
        c.setFillColor(_hex(BLUE))
        c.rect(x, y + h - 22, 3, 14, fill=1, stroke=0)
        c.setFillColor(_hex(NAVY))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(x + 10, y + h - 17, _clean(title))
        inner_top = h - 28
    return x + 8, y + 8, w - 16, inner_top - 8


def _para(c, text: str, x, y_top, w, h_max, size=8.6, color=TEXT, bold=False):
    """Draw wrapped text, shrinking the font until it fits h_max. Returns height used."""
    text = escape(_clean(text)).replace("\n", "<br/>")
    s = size
    while True:
        st = ParagraphStyle("p", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=s,
                            leading=s * 1.36, textColor=_hex(color))
        p = Paragraph(text, st)
        _, ph = p.wrap(w, 10000)
        if ph <= h_max or s <= 6.2:
            break
        s -= 0.4
    p.drawOn(c, x, y_top - ph)
    return ph


def _kpi(c, x, y, w, h, label, value, sub="", color=BLUE):
    c.setFillColor(colors.white)
    c.setStrokeColor(_hex(BORDER))
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.setFillColor(_hex(color))
    c.roundRect(x, y + 6, 3.5, h - 12, 1.5, fill=1, stroke=0)
    c.setFillColor(_hex(SLATE))
    c.setFont("Helvetica-Bold", 6.6)
    c.drawString(x + 12, y + h - 15, _clean(label).upper()[:30])
    c.setFillColor(_hex(NAVY))
    fs = 19 if len(str(value)) < 10 else 14 if len(str(value)) < 16 else 10.5
    c.setFont("Helvetica-Bold", fs)
    c.drawString(x + 12, y + h - 38, _clean(value)[:26])
    if sub:
        c.setFont("Helvetica", 6.8)
        c.setFillColor(_hex(SLATE))
        c.drawString(x + 12, y + 8, _clean(sub)[:40])


def _image(c, reader: ImageReader, ix, iy, iw, ih):
    c.drawImage(reader, ix, iy, iw, ih, preserveAspectRatio=True, anchor="c", mask="auto")


def _table(c, data: List[List[Any]], col_w: List[float], x, y_top, max_h, font=7.8, head=NAVY):
    if len(data) < 2:
        data = data + [["-"] * len(data[0])]
    sh = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=font, leading=font + 2, textColor=colors.white)
    sd = ParagraphStyle("td", fontName="Helvetica", fontSize=font, leading=font + 2, textColor=_hex(NAVY))

    def build(rows):
        cells = [[Paragraph(escape(_clean(v)), sh if i == 0 else sd) for v in r] for i, r in enumerate(rows)]
        t = Table(cells, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _hex(head)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _hex("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.4, _hex(BORDER)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    rows = list(data)
    t = build(rows)
    _, th = t.wrap(0, 0)
    while th > max_h and len(rows) > 2:
        rows.pop()
        t = build(rows)
        _, th = t.wrap(0, 0)
    t.drawOn(c, x, y_top - th)
    return th


def _fmt(v: Any, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


# ============================================================================
# 4. THE REPORT
# ============================================================================
class DashboardPDFReport:
    """Create the 8-page dashboard PDF."""

    @staticmethod
    def build(
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        title: str,
        narrative: Dict[str, Any],
        facts: Dict[str, Any],
        artifacts: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> io.BytesIO:
        meta = meta or {}
        num_cols = [c for c in num_cols if c in df.columns]
        num_cols = [c for c in num_cols if not _is_id_like(df, c)] or num_cols
        cat_cols = [c for c in cat_cols if c in df.columns]
        stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=(W, H))
        c.setTitle(_clean(title))
        c.setAuthor("Enterprise AI Data Analyst Suite PRO")
        ctx = dict(df=df, num=num_cols, cat=cat_cols, title=title, n=narrative, f=facts, a=artifacts, m=meta, stamp=stamp)

        for page_no, fn in enumerate(
            [_p1_exec, _p2_quality, _p3_bi, _p4_relations, _p5_stats, _p6_ml, _p7_insights, _p8_features], start=1
        ):
            try:
                fn(c, ctx, page_no)
            except Exception as exc:  # one broken page must never kill the whole PDF
                logger.exception("Page %s failed: %s", page_no, exc)
                _frame(c, "Section unavailable", "", page_no, title, stamp)
                _para(c, f"This section could not be rendered: {exc}", MARGIN, TOP - 10, W - 2 * MARGIN, 100)
            c.showPage()
        c.save()
        buf.seek(0)
        return buf


# ---------------------------------------------------------------- page 1
def _p1_exec(c, x, no):
    df, f, n = x["df"], x["f"], x["n"]
    d = f["dataset"]
    _frame(c, "Executive Dashboard", "Operations overview, KPIs and AI executive summary", no, x["title"], x["stamp"])
    kw = (W - 2 * MARGIN - 5 * GAP) / 6
    kh = 60
    ky = TOP - kh
    cards = [
        ("Observations", f"{d['rows']:,}", "rows analysed", BLUE),
        ("Features", f"{d['cols']}", f"{d['numeric_cols']} numeric / {d['categorical_cols']} categorical", TEAL),
        ("Completeness", f"{d['completeness_pct']}%", f"{d['missing_cells']:,} missing cells", TEAL if d["completeness_pct"] >= 95 else AMBER),
        ("Duplicates", f"{d['duplicate_rows']:,}", "duplicate rows", RED if d["duplicate_rows"] else TEAL),
    ]
    reg = (f.get("ml") or {}).get("regression")
    cards.append(("Best model R2", _fmt(reg["r2"]) if reg else "n/a", reg["best_model"] if reg else "no regression", INDIGO))
    sig = sum(1 for a in f.get("anova", []) if a.get("significant")) + sum(1 for a in f.get("chi_square", []) if a.get("significant"))
    cards.append(("Significant tests", f"{sig}", "ANOVA + chi-square (p<0.05)", AMBER))
    for i, (lab, val, sub, col) in enumerate(cards):
        _kpi(c, MARGIN + i * (kw + GAP), ky, kw, kh, lab, val, sub, col)

    r2h = 200
    r2y = ky - GAP - r2h
    lw = 372
    ix, iy, iw, ih = _panel(c, MARGIN, r2y, lw, r2h, "AI Executive Summary")
    _para(c, n["executive_summary"], ix, iy + ih, iw, ih, size=9.2)
    ix, iy, iw, ih = _panel(c, MARGIN + lw + GAP, r2y, W - 2 * MARGIN - lw - GAP, r2h,
                            f"Distribution: {x['num'][0]}" if x["num"] else "Distribution")
    _image(c, chart_hist(df[x["num"][0]], iw, ih) if x["num"] else _placeholder(iw, ih, "No numeric column"), ix, iy, iw, ih)

    r3h = r2y - GAP - BOT
    ix, iy, iw, ih = _panel(c, MARGIN, BOT, lw, r3h, "Data Completeness by Column")
    _image(c, chart_missing(df, iw, ih), ix, iy, iw, ih)
    ix, iy, iw, ih = _panel(c, MARGIN + lw + GAP, BOT, W - 2 * MARGIN - lw - GAP, r3h, "Category Composition")
    _image(c, chart_cat_bar(df, x["cat"][0], iw, ih) if x["cat"] else _placeholder(iw, ih, "No categorical column"), ix, iy, iw, ih)


# ---------------------------------------------------------------- page 2
def _p2_quality(c, x, no):
    df, f, n = x["df"], x["f"], x["n"]
    d = f["dataset"]
    _frame(c, "Data Quality & ETL Pipeline Studio", "Completeness, schema composition and outlier scan", no, x["title"], x["stamp"])
    lw = 250
    ph = TOP - BOT
    ix, iy, iw, ih = _panel(c, MARGIN, BOT, lw, ph, "AI Data-Quality Narrative")
    used = _para(c, n["data_quality"], ix, iy + ih, iw, ih * 0.45, size=8.8)
    _para(c, "ETL actions available in the app: mean/mode or drop imputation, 5 scalers "
             "(Standard, MinMax, Robust, Quantile, Power), de-duplication, rare-category pooling.",
          ix, iy + ih - used - 8, iw, 70, size=7.6, color=SLATE)
    _image(c, chart_gauge(d["completeness_pct"], "Data completeness score", iw, 150), ix, iy, iw, 150)

    rx = MARGIN + lw + GAP
    rw = W - MARGIN - rx
    hh = (ph - GAP) / 2
    pw = (rw - GAP) / 2
    ix, iy, iw, ih = _panel(c, rx, BOT + hh + GAP, pw, hh, "Missing Values (% by column)")
    _image(c, chart_missing(df, iw, ih), ix, iy, iw, ih)
    n_other = d["cols"] - d["numeric_cols"] - d["categorical_cols"]
    ix, iy, iw, ih = _panel(c, rx + pw + GAP, BOT + hh + GAP, pw, hh, "Schema Composition")
    _image(c, chart_dtype_donut(d["numeric_cols"], d["categorical_cols"], max(n_other, 0), iw, ih), ix, iy, iw, ih)
    ix, iy, iw, ih = _panel(c, rx, BOT, rw, hh, "Outlier Scan (IQR rule) per numeric column")
    _image(c, chart_outliers(x["a"].get("outliers_all", []), iw, ih), ix, iy, iw, ih)


# ---------------------------------------------------------------- page 3
def _p3_bi(c, x, no):
    df, f, n = x["df"], x["f"], x["n"]
    num, cat = x["num"], x["cat"]
    _frame(c, "Native BI Canvas Showcase", "Six of the 15 Power BI-style visuals, generated from your data", no, x["title"], x["stamp"])
    sh = 78
    ix, iy, iw, ih = _panel(c, MARGIN, TOP - sh, W - 2 * MARGIN, sh, "AI Narrative")
    _para(c, n["bi_showcase"], ix, iy + ih, iw, ih, size=8.6)

    th = (TOP - sh - GAP - BOT - GAP) / 2
    tw = (W - 2 * MARGIN - 2 * GAP) / 3
    ys = [TOP - sh - GAP - th, BOT]
    xs = [MARGIN + i * (tw + GAP) for i in range(3)]
    n0 = num[0] if num else None
    c0 = cat[0] if cat else None
    n1 = num[1] if len(num) > 1 else None
    tiles = [
        ("Donut Chart", lambda w, h: chart_donut_cat(df, c0, w, h) if c0 else _placeholder(w, h, "Needs a categorical column")),
        ("Clustered Bar Chart", lambda w, h: chart_group_bar(df, n0, c0, w, h) if (c0 and n0) else _placeholder(w, h, "Needs numeric + categorical")),
        ("Line Chart (trend)", lambda w, h: chart_line(df, n0, w, h) if n0 else _placeholder(w, h, "Needs a numeric column")),
        ("Scatter Plot", lambda w, h: chart_scatter(df, n0, n1, w, h) if n1 else _placeholder(w, h, "Needs 2 numeric columns")),
        ("Gauge (data completeness)", lambda w, h: chart_gauge(f["dataset"]["completeness_pct"], "completeness", w, h)),
        ("Waterfall Chart", lambda w, h: chart_waterfall(df, n0, c0, w, h) if (c0 and n0) else _placeholder(w, h, "Needs numeric + categorical")),
    ]
    for k, (name, fn) in enumerate(tiles):
        row, col = divmod(k, 3)
        ix, iy, iw, ih = _panel(c, xs[col], ys[row], tw, th, name)
        _image(c, fn(iw, ih), ix, iy, iw, ih)


# ---------------------------------------------------------------- page 4
def _p4_relations(c, x, no):
    f, n, art = x["f"], x["n"], x["a"]
    _frame(c, "Relationships & Descriptive Statistics", "Correlation matrix, strongest pairs and numeric profile", no, x["title"], x["stamp"])
    sh = 78
    ix, iy, iw, ih = _panel(c, MARGIN, TOP - sh, W - 2 * MARGIN, sh, "AI Narrative")
    _para(c, n["relationships"], ix, iy + ih, iw, ih, size=8.6)

    bh = TOP - sh - GAP - BOT
    lw = 360
    ix, iy, iw, ih = _panel(c, MARGIN, BOT, lw, bh, "Correlation Heatmap (Pearson)")
    _image(c, chart_corr(art["corr"], iw, ih) if "corr" in art else _placeholder(iw, ih, "Needs 2+ numeric columns"), ix, iy, iw, ih)

    rx = MARGIN + lw + GAP
    rw = W - MARGIN - rx
    t1h = 170
    ix, iy, iw, ih = _panel(c, rx, BOT + bh - t1h, rw, t1h, "Strongest Correlations")
    rows = [["Variable A", "Variable B", "r", "Strength"]]
    for p in f.get("top_correlations", []):
        a = abs(p["r"])
        rows.append([p["a"], p["b"], f"{p['r']:+.3f}", "Strong" if a >= 0.7 else "Moderate" if a >= 0.4 else "Weak"])
    _table(c, rows, [iw * 0.33, iw * 0.33, iw * 0.15, iw * 0.19], ix, iy + ih, ih)

    t2h = bh - t1h - GAP
    ix, iy, iw, ih = _panel(c, rx, BOT, rw, t2h, "Numeric Profile")
    rows = [["Column", "Mean", "Std", "Skew", "Outliers"]]
    for r in f.get("numeric", []):
        rows.append([r["col"], _fmt(r["mean"], 2), _fmt(r["std"], 2), _fmt(r["skew"], 2), _fmt(r["outliers_iqr"])])
    _table(c, rows, [iw * 0.30, iw * 0.2, iw * 0.2, iw * 0.15, iw * 0.15], ix, iy + ih, ih)


# ---------------------------------------------------------------- page 5
def _p5_stats(c, x, no):
    df, f, n = x["df"], x["f"], x["n"]
    _frame(c, "Diagnostic Statistics Lab", "Shapiro-Wilk normality, one-way ANOVA and chi-square association", no, x["title"], x["stamp"])
    sh = 78
    ix, iy, iw, ih = _panel(c, MARGIN, TOP - sh, W - 2 * MARGIN, sh, "AI Narrative")
    _para(c, n["statistics"], ix, iy + ih, iw, ih, size=8.6)

    rh = (TOP - sh - GAP - BOT - GAP) / 2
    y1, y2 = TOP - sh - GAP - rh, BOT
    widths = [240, 250, W - 2 * MARGIN - 240 - 250 - 2 * GAP]
    xs = [MARGIN, MARGIN + widths[0] + GAP, MARGIN + widths[0] + widths[1] + 2 * GAP]

    ix, iy, iw, ih = _panel(c, xs[0], y1, widths[0], rh, "Normality (Shapiro-Wilk)")
    rows = [["Column", "W", "p", "Normal?"]] + [[r["col"], _fmt(r["W"]), _fmt(r["p"], 4), "Yes" if r["normal"] else "No"] for r in f.get("normality", [])]
    _table(c, rows, [iw * 0.38, iw * 0.2, iw * 0.24, iw * 0.18], ix, iy + ih, ih)

    ix, iy, iw, ih = _panel(c, xs[1], y1, widths[1], rh, "One-way ANOVA (numeric ~ category)")
    rows = [["Numeric", "Category", "F", "p", "Sig."]] + [[r["num"], r["cat"], _fmt(r["F"], 2), _fmt(r["p"], 4), "Yes" if r["significant"] else "No"] for r in f.get("anova", [])]
    _table(c, rows, [iw * 0.28, iw * 0.27, iw * 0.15, iw * 0.18, iw * 0.12], ix, iy + ih, ih)

    ix, iy, iw, ih = _panel(c, xs[2], y1, widths[2], rh, "Chi-square Association")
    rows = [["A", "B", "V", "p"]] + [[r["a"], r["b"], _fmt(r["cramers_v"]), _fmt(r["p"], 4)] for r in f.get("chi_square", [])]
    _table(c, rows, [iw * 0.32, iw * 0.32, iw * 0.15, iw * 0.21], ix, iy + ih, ih)

    lw = 372
    best = (f.get("anova") or [None])[0]
    ix, iy, iw, ih = _panel(c, MARGIN, y2, lw, rh, "Group Comparison (top ANOVA pair)")
    if best and best["num"] in df.columns and best["cat"] in df.columns:
        _image(c, chart_box(df, best["num"], best["cat"], iw, ih), ix, iy, iw, ih)
    else:
        _image(c, _placeholder(iw, ih, "No categorical grouping available"), ix, iy, iw, ih)
    ix, iy, iw, ih = _panel(c, MARGIN + lw + GAP, y2, W - 2 * MARGIN - lw - GAP, rh, "Evidence Strength Across All Tests")
    _image(c, chart_pvalues(f, iw, ih), ix, iy, iw, ih)


# ---------------------------------------------------------------- page 6
def _p6_ml(c, x, no):
    f, n, art = x["f"], x["n"], x["a"]
    reg = (f.get("ml") or {}).get("regression")
    cls = (f.get("ml") or {}).get("classification")
    _frame(c, "Predictive ML Lab", "Auto-trained regression and classification baselines (80/20 hold-out)", no, x["title"], x["stamp"])
    sh = 70
    ix, iy, iw, ih = _panel(c, MARGIN, TOP - sh, W - 2 * MARGIN, sh, "AI Narrative")
    _para(c, n["ml"], ix, iy + ih, iw, ih, size=8.4)

    kh = 56
    ky = TOP - sh - GAP - kh
    kw = (W - 2 * MARGIN - 3 * GAP) / 4
    cards = [
        ("Regression target", reg["target"] if reg else "n/a", reg["best_model"] if reg else "", BLUE),
        ("R2 / RMSE", f"{_fmt(reg['r2'])} / {_fmt(reg['rmse'], 2)}" if reg else "n/a", "hold-out set", TEAL),
        ("Classification target", cls["target"] if cls else "n/a", cls["best_model"] if cls else "", INDIGO),
        ("Accuracy / F1", f"{_fmt(cls['accuracy'])} / {_fmt(cls['f1'])}" if cls else "n/a", "weighted F1", AMBER),
    ]
    for i, (lab, val, sub, col) in enumerate(cards):
        _kpi(c, MARGIN + i * (kw + GAP), ky, kw, kh, lab, val, sub, col)

    ch = 200
    cy = ky - GAP - ch
    cw = (W - 2 * MARGIN - 2 * GAP) / 3
    ix, iy, iw, ih = _panel(c, MARGIN, cy, cw, ch, "Predicted vs Actual")
    _image(c, chart_pred_actual(art["reg_y_test"], art["reg_pred"], iw, ih) if reg and "reg_y_test" in art else _placeholder(iw, ih, "No regression model"), ix, iy, iw, ih)
    ix, iy, iw, ih = _panel(c, MARGIN + cw + GAP, cy, cw, ch, "Feature Importance")
    _image(c, chart_importance(art["reg_importance"], iw, ih) if reg and "reg_importance" in art else _placeholder(iw, ih, "No regression model"), ix, iy, iw, ih)
    ix, iy, iw, ih = _panel(c, MARGIN + 2 * (cw + GAP), cy, cw, ch, "Confusion Matrix")
    _image(c, chart_confusion(art["cls_cm"], art["cls_labels"], iw, ih) if cls and "cls_cm" in art else _placeholder(iw, ih, "No categorical target (2-10 classes)"), ix, iy, iw, ih)

    th = cy - GAP - BOT
    hw = (W - 2 * MARGIN - GAP) / 2
    ix, iy, iw, ih = _panel(c, MARGIN, BOT, hw, th, "Regression Model Comparison")
    rows = [["Model", "R2", "RMSE", "MAE"]]
    if reg:
        rows += [[k + (" *" if k == reg["best_model"] else ""), _fmt(v["r2"]), _fmt(v["rmse"], 2), _fmt(v["mae"], 2)] for k, v in reg["models"].items()]
    _table(c, rows, [iw * 0.4, iw * 0.2, iw * 0.2, iw * 0.2], ix, iy + ih, ih)
    ix, iy, iw, ih = _panel(c, MARGIN + hw + GAP, BOT, hw, th, "Classification Model Comparison")
    rows = [["Model", "Accuracy", "Weighted F1"]]
    if cls:
        rows += [[k + (" *" if k == cls["best_model"] else ""), _fmt(v["accuracy"]), _fmt(v["f1"])] for k, v in cls["models"].items()]
    _table(c, rows, [iw * 0.5, iw * 0.25, iw * 0.25], ix, iy + ih, ih)


# ---------------------------------------------------------------- page 7
def _p7_insights(c, x, no):
    f, n, m = x["f"], x["n"], x["m"]
    _frame(c, "AI Insights & Recommendations", "LLM-written narrative grounded in the computed statistics", no, x["title"], x["stamp"])
    ph = TOP - BOT
    lw = 372
    ix, iy, iw, ih = _panel(c, MARGIN, BOT, lw, ph, "Recommended Actions")
    y = iy + ih
    for i, rec in enumerate(n.get("recommendations", [])[:6], start=1):
        c.setFillColor(_hex(BLUE))
        c.circle(ix + 9, y - 9, 9, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(ix + 9, y - 12, str(i))
        used = _para(c, rec, ix + 26, y - 2, iw - 26, 70, size=9)
        y -= max(used, 20) + 16

    rx = MARGIN + lw + GAP
    rw = W - MARGIN - rx
    mh = 150
    ix, iy, iw, ih = _panel(c, rx, BOT + ph - (ph - mh - GAP), rw, ph - mh - GAP, "Key Findings (computed)")
    y = iy + ih
    for finding in key_findings(f)[:9]:
        c.setFillColor(_hex(TEAL))
        c.circle(ix + 4, y - 7, 2.2, fill=1, stroke=0)
        used = _para(c, finding, ix + 14, y - 1, iw - 14, 40, size=8)
        y -= used + 6
        if y < iy + 6:
            break

    ix, iy, iw, ih = _panel(c, rx, BOT, rw, mh, "About this Narrative")
    mode = "Generated by an LLM" if m.get("used_llm") else "Generated by the offline rule engine"
    txt = (f"{mode}\nProvider: {m.get('provider', '-')}\nModel: {m.get('model', '-')}\n"
           "Privacy: only aggregated statistics (counts, means, test results) are sent to the LLM; raw rows never leave the app.")
    if m.get("error"):
        txt += f"\nNote: {m['error']}"
    _para(c, txt, ix, iy + ih, iw, ih, size=7.8)


# ---------------------------------------------------------------- page 8
def _p8_features(c, x, no):
    df, f, m = x["df"], x["f"], x["m"]
    d = f["dataset"]
    _frame(c, "Platform Feature Map", "Every capability of the suite and what it found in this dataset", no, x["title"], x["stamp"])
    num, cat = len(x["num"]), len(x["cat"])
    has_geo = any("lat" in str(c_).lower() for c_ in df.columns) and any(("lon" in str(c_).lower() or "lng" in str(c_).lower()) for c_ in df.columns)
    usable = 4 + (7 if (num and cat) else 0) + (2 if num else 0) + (1 if num >= 2 else 0) + (1 if has_geo else 0)
    usable = min(usable, 15)
    tests = len(f.get("normality", [])) + len(f.get("anova", [])) + len(f.get("chi_square", []))
    sig = sum(1 for a in f.get("anova", []) + f.get("chi_square", []) if a.get("significant"))
    reg = (f.get("ml") or {}).get("regression")
    cls = (f.get("ml") or {}).get("classification")

    rows = [
        ["Feature", "Where in the app", "What it does", "Result on this dataset"],
        ["CSV ingestion & cross-filtering", "Sidebar", "Upload a CSV, filter by any category, slice any numeric range; all tabs update.", f"{d['rows']:,} rows x {d['cols']} columns loaded"],
        ["Operations Overview", "Tab 1", "KPI cards, live data monitor and algorithmic insight engine.", f"{d['completeness_pct']}% complete, {d['duplicate_rows']:,} duplicates"],
        ["Data Pipeline Studio (ETL)", "Tab 2", "Imputation, 5 scalers, de-duplication, noise pooling, schema profiling.", f"{d['missing_cells']:,} missing cells flagged for imputation"],
        ["Native BI Canvas", "Tab 3", "15 visuals: bar, column, line, pie, donut, treemap, scatter, table, matrix, card, KPI, gauge, waterfall, funnel, map.", f"About {usable} of 15 visuals applicable to this schema"],
        ["Diagnostic Statistics", "Tab 4", "Shapiro-Wilk, ANOVA / Kruskal-Wallis, chi-square association tests.", f"{tests} tests run, {sig} significant association(s)"],
        ["Predictive ML Lab", "Tab 5", "Regression (OLS, Ridge, Lasso, RF) and classification (LogReg, RF, GB) with K-fold CV.",
         (f"R2 {_fmt(reg['r2'])} on '{reg['target']}'" if reg else "Regression n/a") + (f"; F1 {_fmt(cls['f1'])} on '{cls['target']}'" if cls else "")],
        ["Multimodal CV Auditor", "Sidebar mode 2", "Image upload with luminosity / dark-field diagnostics.", "Image-based; not applicable to CSV data"],
        ["LLM Narrator", "Tab 6", "Provider-agnostic narrator (Claude, OpenAI, Groq, Gemini, offline) writing every section's commentary.", f"{m.get('provider', '-')} / {m.get('model', '-')}"],
        ["Dashboard PDF Publisher", "Tab 6", "Compiles all of the above into this 8-page landscape dashboard.", "You are reading it"],
    ]
    bh = 118
    ty = BOT + bh + GAP
    ix, iy, iw, ih = _panel(c, MARGIN, ty, W - 2 * MARGIN, TOP - ty, "Capabilities")
    _table(c, rows, [iw * 0.19, iw * 0.10, iw * 0.45, iw * 0.26], ix, iy + ih, ih, font=7.2)

    ix, iy, iw, ih = _panel(c, MARGIN, BOT, W - 2 * MARGIN, bh, "Methodology & Notes")
    txt = ("Outliers use the 1.5 x IQR rule. Correlations are Pearson (ID-like columns excluded). Shapiro-Wilk uses up to 5,000 sampled values. "
           "ANOVA is run for categorical columns with 2-10 levels; chi-square for pairs with up to 15 levels (Cramer's V = effect size). "
           "ML baselines use an 80/20 random split (seed 42); the regression target is the numeric column most correlated with the others and "
           "the classification target is the first categorical column with 2-10 classes. These are automatic baselines for orientation; use the "
           "Predictive ML Lab tab for tuned, user-selected models.")
    _para(c, txt, ix, iy + ih, iw, ih, size=8)
