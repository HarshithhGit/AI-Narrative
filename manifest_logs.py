"""
manifest_logs.py
================
Enterprise Pipeline Manifest & Audit Trail.

* `log_event()`  - call from anywhere in the app to record an auditable event
                   (dataset load, ETL run, report generated, image audit ...).
* `render_manifest_workspace()` - the "Enterprise Pipeline Manifest Logs" page:
      1. System Health      - uptime, memory, environment, integrations
      2. Data Lineage       - raw -> production -> filtered, per-column change report
      3. Activity Timeline  - filterable audit trail + charts + CSV/JSON export
      4. Session Inspector  - what is held in memory (secrets are never shown)
      5. Export Manifest    - one JSON file describing the whole run

Secrets (API keys) are NEVER stored in the log and never displayed.
"""

import datetime
import json
import os
import platform
import sys
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LOG_KEY = "activity_log"
START_KEY = "session_started_at"
SENSITIVE_HINTS = ("key", "token", "secret", "password")

CATEGORY_COLORS = {
    "Data": "#2563EB", "ETL": "#14B8A6", "Report": "#6366F1",
    "Vision": "#F59E0B", "System": "#64748B", "Error": "#EF4444",
}


# ============================================================================
# 1. EVENT LOGGING
# ============================================================================
def _now() -> datetime.datetime:
    return datetime.datetime.now()


def ensure_log() -> None:
    if LOG_KEY not in st.session_state:
        st.session_state[LOG_KEY] = []
    if START_KEY not in st.session_state:
        st.session_state[START_KEY] = _now().isoformat(timespec="seconds")


def log_event(
    category: str,
    action: str,
    detail: str = "",
    shape_before: Optional[Tuple[int, int]] = None,
    shape_after: Optional[Tuple[int, int]] = None,
    status: str = "success",
) -> None:
    """Append one audit event. Safe to call from anywhere (never raises)."""
    try:
        ensure_log()
        entry = {
            "time": _now().isoformat(timespec="seconds"),
            "category": category,
            "action": action,
            "status": status,
            "detail": str(detail)[:300],
            "rows_before": shape_before[0] if shape_before else None,
            "rows_after": shape_after[0] if shape_after else None,
            "cols_before": shape_before[1] if shape_before else None,
            "cols_after": shape_after[1] if shape_after else None,
        }
        st.session_state[LOG_KEY].append(entry)
    except Exception:
        pass


# ============================================================================
# 2. HELPERS
# ============================================================================
def _mb(n_bytes: float) -> float:
    return round(n_bytes / (1024 * 1024), 2)


def _df_stats(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        return {"rows": 0, "cols": 0, "missing": 0, "dupes": 0, "mem_mb": 0.0}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "dupes": int(df.duplicated().sum()),
        "mem_mb": _mb(df.memory_usage(deep=True).sum()),
    }


def _versions() -> List[Dict[str, str]]:
    rows = []
    for pkg in ["streamlit", "pandas", "numpy", "scikit-learn", "scipy", "plotly",
                "matplotlib", "seaborn", "statsmodels", "reportlab", "pillow", "requests"]:
        try:
            rows.append({"Package": pkg, "Version": metadata.version(pkg), "Status": "OK"})
        except Exception:
            rows.append({"Package": pkg, "Version": "-", "Status": "Not installed"})
    return rows


def _key_available(env_name: str) -> bool:
    if os.environ.get(env_name):
        return True
    try:
        return bool(st.secrets.get(env_name, ""))
    except Exception:
        return False


def _integrations() -> List[Dict[str, str]]:
    out = []
    for label, env in [("Anthropic Claude", "ANTHROPIC_API_KEY"), ("OpenAI", "OPENAI_API_KEY"),
                       ("Groq", "GROQ_API_KEY"), ("Google Gemini", "GEMINI_API_KEY")]:
        out.append({"Integration": label, "Preconfigured key": "Yes" if _key_available(env) else "No (paste in app)"})
    return out


def _describe_value(key: str, value: Any) -> Tuple[str, str, float]:
    """(type, summary, approx_mb). Never reveals sensitive values."""
    t = type(value).__name__
    if any(h in key.lower() for h in SENSITIVE_HINTS):
        return t, "hidden (sensitive)", 0.0
    if isinstance(value, pd.DataFrame):
        return t, f"{value.shape[0]:,} rows x {value.shape[1]} cols", _mb(value.memory_usage(deep=True).sum())
    if isinstance(value, (bytes, bytearray)):
        return t, f"{len(value) / 1024:,.1f} KB", _mb(len(value))
    if isinstance(value, (list, tuple, set, dict)):
        return t, f"{len(value)} items", _mb(sys.getsizeof(value))
    if isinstance(value, (int, float, bool)) or value is None:
        return t, str(value), 0.0
    if isinstance(value, str):
        return t, f"{len(value)} chars", _mb(sys.getsizeof(value))
    return t, "object", _mb(sys.getsizeof(value))


def _uptime(start_iso: str) -> str:
    try:
        secs = int((_now() - datetime.datetime.fromisoformat(start_iso)).total_seconds())
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m" if h else f"{m}m {s}s"
    except Exception:
        return "-"


def build_manifest() -> Dict[str, Any]:
    """Everything about this run as a JSON-safe dict (no secrets, no raw data)."""
    ensure_log()
    raw, prod, filt = (st.session_state.get(k) for k in ("raw_data", "production_pipeline", "filtered_pipeline"))
    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "session_started_at": st.session_state[START_KEY],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {r["Package"]: r["Version"] for r in _versions()}},
        "dataset_lineage": {"raw": _df_stats(raw), "production": _df_stats(prod), "filtered": _df_stats(filt),
                            "source_file": st.session_state.get("dataset_name")},
        "reports_generated": st.session_state.get("reports_generated", 0),
        "images_audited": st.session_state.get("images_audited", 0),
        "activity_log": st.session_state[LOG_KEY],
    }


# ============================================================================
# 3. PAGE
# ============================================================================
def render_manifest_workspace() -> None:
    ensure_log()
    st.subheader("📋 Enterprise Pipeline Manifest & Audit Trail")
    st.caption("A live record of what happened to your data in this session: system health, data lineage, "
               "every auditable action, and a downloadable run manifest. API keys are never logged or displayed.")

    raw = st.session_state.get("raw_data")
    prod = st.session_state.get("production_pipeline")
    filt = st.session_state.get("filtered_pipeline")
    log = st.session_state[LOG_KEY]
    s_raw, s_prod, s_filt = _df_stats(raw), _df_stats(prod), _df_stats(filt)

    t_health, t_lineage, t_time, t_inspect, t_export = st.tabs(
        ["🩺 System Health", "🧬 Data Lineage", "📜 Activity Timeline", "🔍 Session Inspector", "📦 Export Manifest"])

    # ------------------------------------------------------------ HEALTH
    with t_health:
        errors = sum(1 for e in log if e["status"] != "success")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Application status", "ACTIVE", "running")
        k2.metric("Session uptime", _uptime(st.session_state[START_KEY]))
        k3.metric("Events logged", len(log), f"{errors} error(s)" if errors else "no errors")
        k4.metric("Dataset", "Loaded" if raw is not None else "None",
                  f"{s_prod['rows']:,} x {s_prod['cols']}" if raw is not None else None)
        k5.metric("Memory in session", f"{s_raw['mem_mb'] + s_prod['mem_mb'] + s_filt['mem_mb']:.1f} MB")

        if raw is not None:
            total_cells = max(s_prod["rows"] * s_prod["cols"], 1)
            completeness = 100 * (1 - s_prod["missing"] / total_cells)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=completeness, number={"suffix": "%"},
                title={"text": "Production data completeness"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#14B8A6" if completeness >= 95 else "#F59E0B" if completeness >= 80 else "#EF4444"}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=0), template="plotly_white")
            g1, g2 = st.columns([1, 2])
            g1.plotly_chart(fig, use_container_width=True)
            with g2:
                st.markdown("**Data health checks**")
                checks = [
                    ("Missing values", s_prod["missing"] == 0, f"{s_prod['missing']:,} missing cells"),
                    ("Duplicate rows", s_prod["dupes"] == 0, f"{s_prod['dupes']:,} duplicate rows"),
                    ("ETL applied", any(e["category"] == "ETL" for e in log), "No ETL run yet" if not any(e["category"] == "ETL" for e in log) else "ETL executed this session"),
                    ("Report generated", st.session_state.get("reports_generated", 0) > 0, f"{st.session_state.get('reports_generated', 0)} PDF report(s)"),
                ]
                for name, ok, msg in checks:
                    (st.success if ok else st.warning)(f"**{name}:** {msg}")
        else:
            st.info("No dataset loaded yet. Upload a CSV in 'Analytical Engine Core' to see data health checks.")

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Runtime environment**")
            st.write(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
            st.dataframe(pd.DataFrame(_versions()), use_container_width=True, hide_index=True)
        with e2:
            st.markdown("**LLM integrations** (keys are never displayed)")
            st.dataframe(pd.DataFrame(_integrations()), use_container_width=True, hide_index=True)
            st.markdown("**Module usage this session**")
            st.dataframe(pd.DataFrame([
                {"Module": "AI dashboard reports", "Count": st.session_state.get("reports_generated", 0)},
                {"Module": "Images audited", "Count": st.session_state.get("images_audited", 0)},
                {"Module": "ETL runs", "Count": sum(1 for e in log if e["category"] == "ETL")},
            ]), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------ LINEAGE
    with t_lineage:
        if raw is None:
            st.info("Upload a dataset to see its lineage.")
        else:
            st.markdown("#### Dataset lineage: Raw → Production (after ETL) → Filtered (after sidebar filters)")
            lin = pd.DataFrame([
                {"Stage": "Raw upload", **{k: v for k, v in s_raw.items()}},
                {"Stage": "Production (ETL)", **{k: v for k, v in s_prod.items()}},
                {"Stage": "Filtered view", **{k: v for k, v in s_filt.items()}},
            ]).rename(columns={"rows": "Rows", "cols": "Columns", "missing": "Missing cells", "dupes": "Duplicate rows", "mem_mb": "Memory (MB)"})
            st.dataframe(lin, use_container_width=True, hide_index=True)

            fig = go.Figure(go.Funnel(y=lin["Stage"], x=lin["Rows"], textinfo="value+percent initial",
                                      marker={"color": ["#2563EB", "#14B8A6", "#6366F1"]}))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
            l1, l2 = st.columns([1, 1])
            l1.plotly_chart(fig, use_container_width=True)
            with l2:
                removed = s_raw["rows"] - s_prod["rows"]
                st.metric("Rows removed by ETL", f"{removed:,}", f"{(removed / max(s_raw['rows'], 1)) * 100:.1f}% of raw")
                st.metric("Missing cells fixed", f"{s_raw['missing'] - s_prod['missing']:,}", f"{s_raw['missing']:,} → {s_prod['missing']:,}")
                st.metric("Rows hidden by filters", f"{s_prod['rows'] - s_filt['rows']:,}")

            st.markdown("#### Per-column change report (raw vs production)")
            common = [c for c in raw.columns if c in prod.columns]
            added = [c for c in prod.columns if c not in raw.columns]
            dropped = [c for c in raw.columns if c not in prod.columns]
            if added or dropped:
                st.warning(f"Columns added: {added or 'none'} | Columns removed/renamed: {dropped or 'none'}")
            rows = []
            for c in common:
                r_s, p_s = raw[c], prod[c]
                row = {"Column": c, "Type (raw → now)": f"{r_s.dtype} → {p_s.dtype}",
                       "Missing (raw → now)": f"{int(r_s.isnull().sum())} → {int(p_s.isnull().sum())}",
                       "Unique (raw → now)": f"{r_s.nunique()} → {p_s.nunique()}"}
                if pd.api.types.is_numeric_dtype(r_s) and pd.api.types.is_numeric_dtype(p_s):
                    row["Mean (raw → now)"] = f"{r_s.mean():,.3f} → {p_s.mean():,.3f}"
                    changed = not np.isclose(r_s.mean(), p_s.mean(), rtol=1e-3, equal_nan=True)
                else:
                    row["Mean (raw → now)"] = "-"
                    changed = r_s.nunique() != p_s.nunique()
                row["Changed?"] = "Yes" if (changed or str(r_s.dtype) != str(p_s.dtype) or int(r_s.isnull().sum()) != int(p_s.isnull().sum())) else "No"
                rows.append(row)
            report = pd.DataFrame(rows)
            only_changed = st.checkbox("Show only columns that changed", value=False)
            st.dataframe(report[report["Changed?"] == "Yes"] if only_changed else report, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------ TIMELINE
    with t_time:
        if not log:
            st.info("No events yet. Load a dataset, run the ETL, generate a report or audit an image and events will appear here.")
        else:
            df = pd.DataFrame(log)
            df["time"] = pd.to_datetime(df["time"])
            cats = sorted(df["category"].unique())
            chosen = st.multiselect("Filter by category", cats, default=cats)
            view = df[df["category"].isin(chosen)].sort_values("time", ascending=False)

            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                for cat in chosen:
                    sub = df[df["category"] == cat]
                    fig.add_trace(go.Scatter(x=sub["time"], y=[cat] * len(sub), mode="markers", name=cat,
                                             marker=dict(size=13, color=CATEGORY_COLORS.get(cat, "#64748B")),
                                             text=sub["action"], hovertemplate="%{text}<br>%{x}<extra></extra>"))
                fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                counts = df["category"].value_counts()
                fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                       marker_color=[CATEGORY_COLORS.get(c, "#64748B") for c in counts.index]))
                fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", title="Events by category")
                st.plotly_chart(fig, use_container_width=True)

            etl = df[df["category"] == "ETL"].dropna(subset=["rows_before", "rows_after"])
            if not etl.empty:
                st.markdown("**ETL row-count impact per run**")
                etl = etl.assign(run=range(1, len(etl) + 1), delta=etl["rows_after"] - etl["rows_before"])
                fig = go.Figure(go.Bar(x=etl["run"].astype(str), y=etl["delta"],
                                       marker_color=["#EF4444" if d < 0 else "#14B8A6" for d in etl["delta"]]))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white",
                                  xaxis_title="ETL run #", yaxis_title="Row delta")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(view.assign(time=view["time"].dt.strftime("%H:%M:%S")).rename(columns={"time": "Time"}),
                         use_container_width=True, hide_index=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Download log (CSV)", view.to_csv(index=False).encode(), "activity_log.csv", "text/csv", use_container_width=True)
            d2.download_button("⬇️ Download log (JSON)", json.dumps(log, indent=2).encode(), "activity_log.json", "application/json", use_container_width=True)
            if d3.button("🗑️ Clear log", use_container_width=True):
                st.session_state[LOG_KEY] = []
                st.rerun()

    # ------------------------------------------------------------ INSPECTOR
    with t_inspect:
        st.markdown("Everything currently held in memory for this session. Sensitive values are hidden.")
        rows = []
        for k in list(st.session_state.keys()):
            if k.startswith("_") or k.startswith("FormSubmitter") or k.startswith("$$"):
                continue
            try:
                t, summary, mb = _describe_value(k, st.session_state[k])
            except Exception:
                t, summary, mb = "?", "unreadable", 0.0
            rows.append({"Key": k, "Type": t, "Summary": summary, "Approx MB": mb})
        ins = pd.DataFrame(rows).sort_values("Approx MB", ascending=False)
        i1, i2 = st.columns([3, 2])
        i1.dataframe(ins, use_container_width=True, hide_index=True)
        top = ins[ins["Approx MB"] > 0].head(8)
        if not top.empty:
            fig = go.Figure(go.Bar(x=top["Approx MB"], y=top["Key"], orientation="h", marker_color="#2563EB"))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", title="Largest items (MB)",
                              yaxis=dict(autorange="reversed"))
            i2.plotly_chart(fig, use_container_width=True)
        total = ins["Approx MB"].sum()
        st.caption(f"{len(ins)} session keys, about {total:.1f} MB tracked.")

    # ------------------------------------------------------------ EXPORT
    with t_export:
        st.markdown("Download one JSON file describing this whole run (environment, lineage, activity). "
                    "It contains no raw data and no API keys, so it is safe to attach to your project submission.")
        manifest = build_manifest()
        st.download_button("💾 Download run manifest (JSON)", json.dumps(manifest, indent=2, default=str).encode(),
                           "pipeline_manifest.json", "application/json", use_container_width=True)
        with st.expander("Preview manifest"):
            st.json(manifest)
