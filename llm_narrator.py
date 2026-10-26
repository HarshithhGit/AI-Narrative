"""
llm_narrator.py
===============
LLM Narrator for the Enterprise AI Data Analyst Suite PRO.

What it does
------------
1. Receives a compact dictionary of *aggregated facts* about the dataset
   (counts, means, correlations, test results, model scores). Raw rows are
   NEVER sent to the LLM.
2. Asks an LLM to write a structured business narrative (JSON) for every
   section of the dashboard PDF.
3. If no API key is given, or the API call fails for any reason, it falls back
   to a deterministic rule-based narrator so the project ALWAYS produces a PDF.

Supported providers (all use plain HTTPS via `requests`, no extra SDKs):
    - Anthropic Claude
    - OpenAI (and any OpenAI-compatible API)
    - Groq (free tier, OpenAI-compatible)
    - Google Gemini (free tier)
    - Offline rule-based (no key needed)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("LLMNarrator")

OFFLINE = "Offline (rule-based, no API key)"

PROVIDERS: Dict[str, Dict[str, Any]] = {
    OFFLINE: {"kind": "offline", "default_model": "rule-engine-v1", "env": ""},
    "Anthropic Claude": {
        "kind": "anthropic",
        "default_model": "claude-sonnet-4-5",
        "env": "ANTHROPIC_API_KEY",
    },
    "OpenAI": {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
    },
    "Groq (free tier)": {
        "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "env": "GROQ_API_KEY",
    },
    "Google Gemini": {
        "kind": "gemini",
        "default_model": "gemini-2.0-flash",
        "env": "GEMINI_API_KEY",
    },
}

NARRATIVE_KEYS = [
    "executive_summary",
    "data_quality",
    "bi_showcase",
    "relationships",
    "statistics",
    "ml",
    "recommendations",
]

SYSTEM_PROMPT = (
    "You are a senior data analyst writing the narrative for an executive PDF dashboard. "
    "You receive a JSON object of aggregated dataset facts. Rules: "
    "(1) Use ONLY numbers present in the facts; never invent figures. "
    "(2) Write plain text only: no markdown, no bullet symbols, no emojis. "
    "(3) Be specific: name columns and cite values. Explain what the number MEANS for the business. "
    "(4) If a section has no data (null/empty), say briefly that it could not be assessed. "
    "(5) Return ONLY one valid JSON object with exactly these keys: "
    "executive_summary (3-4 sentences), data_quality (2-3 sentences), bi_showcase (2-3 sentences describing "
    "what the charts show), relationships (2-3 sentences), statistics (2-3 sentences interpreting the "
    "normality, ANOVA and chi-square results), ml (2-3 sentences on model performance and top drivers), "
    "recommendations (a JSON array of 4 to 5 short, actionable strings)."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def provider_names() -> List[str]:
    return list(PROVIDERS.keys())


def default_model(provider: str) -> str:
    return PROVIDERS.get(provider, PROVIDERS[OFFLINE])["default_model"]


def env_key_for(provider: str) -> str:
    """Return an API key from environment variables if present."""
    env = PROVIDERS.get(provider, {}).get("env", "")
    return os.environ.get(env, "") if env else ""


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (handles ```json fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM reply did not contain a JSON object.")
    return json.loads(text[start:end + 1])


def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee every expected key exists with the right type."""
    out: Dict[str, Any] = {}
    for key in NARRATIVE_KEYS:
        val = raw.get(key, "")
        if key == "recommendations":
            if isinstance(val, str):
                val = [v.strip(" -•\t") for v in val.split("\n") if v.strip()]
            out[key] = [str(v) for v in val][:6] if isinstance(val, list) else []
        else:
            out[key] = str(val).strip() if val else ""
    return out


# ----------------------------------------------------------------------------
# Rule-based (offline) narrator + key findings
# ----------------------------------------------------------------------------
def key_findings(facts: Dict[str, Any]) -> List[str]:
    """Deterministic bullet findings derived straight from the facts."""
    out: List[str] = []
    d = facts.get("dataset", {})
    if d:
        out.append(
            f"The dataset contains {d['rows']:,} rows and {d['cols']} columns "
            f"({d['numeric_cols']} numeric, {d['categorical_cols']} categorical)."
        )
        out.append(
            f"Data completeness is {d['completeness_pct']}% with {d['missing_cells']:,} missing cells "
            f"and {d['duplicate_rows']:,} duplicate rows."
        )
    miss = facts.get("missing_top") or []
    if miss:
        out.append(f"Most incomplete column: '{miss[0]['col']}' ({miss[0]['pct']}% missing).")
    num = facts.get("numeric") or []
    if num:
        vol = max(num, key=lambda r: abs(r.get("cv") or 0))
        out.append(f"Most volatile metric: '{vol['col']}' (coefficient of variation {vol['cv']}).")
        outl = max(num, key=lambda r: r.get("outliers_iqr", 0))
        if outl.get("outliers_iqr", 0) > 0:
            out.append(f"'{outl['col']}' has the most IQR outliers ({outl['outliers_iqr']:,}).")
    cor = facts.get("top_correlations") or []
    if cor:
        c = cor[0]
        out.append(f"Strongest relationship: '{c['a']}' vs '{c['b']}' (r = {c['r']}).")
    an = [a for a in (facts.get("anova") or []) if a.get("significant")]
    if an:
        a = an[0]
        out.append(f"'{a['cat']}' significantly affects '{a['num']}' (ANOVA p = {a['p']}).")
    chi = [x for x in (facts.get("chi_square") or []) if x.get("significant")]
    if chi:
        x = chi[0]
        out.append(f"'{x['a']}' and '{x['b']}' are associated (Cramer's V = {x['cramers_v']}).")
    reg = (facts.get("ml") or {}).get("regression")
    if reg:
        out.append(
            f"Best regression model for '{reg['target']}': {reg['best_model']} "
            f"(R2 = {reg['r2']}, RMSE = {reg['rmse']})."
        )
    cls = (facts.get("ml") or {}).get("classification")
    if cls:
        out.append(
            f"Best classifier for '{cls['target']}': {cls['best_model']} "
            f"(accuracy = {cls['accuracy']}, weighted F1 = {cls['f1']})."
        )
    return out


def rule_based_narrative(facts: Dict[str, Any]) -> Dict[str, Any]:
    d = facts.get("dataset", {})
    miss = facts.get("missing_top") or []
    num = facts.get("numeric") or []
    cor = facts.get("top_correlations") or []
    norm = facts.get("normality") or []
    an = facts.get("anova") or []
    chi = facts.get("chi_square") or []
    ml = facts.get("ml") or {}
    reg, cls = ml.get("regression"), ml.get("classification")

    exec_s = (
        f"This report analyses {d.get('rows', 0):,} records across {d.get('cols', 0)} fields. "
        f"Data completeness stands at {d.get('completeness_pct', 0)}% and "
        f"{d.get('duplicate_rows', 0):,} duplicate rows were detected. "
    )
    if reg:
        exec_s += (
            f"A {reg['best_model']} model explains {round(reg['r2'] * 100, 1)}% of the variance in "
            f"'{reg['target']}', indicating {'strong' if reg['r2'] >= 0.7 else 'moderate' if reg['r2'] >= 0.4 else 'limited'} "
            f"predictability."
        )

    dq = (
        f"{d.get('missing_cells', 0):,} cells are missing in total. "
        + (f"The most affected column is '{miss[0]['col']}' at {miss[0]['pct']}% missing. " if miss else "No column has missing values. ")
        + "Imputation and de-duplication in the Data Pipeline Studio are recommended before modelling."
    )

    bi = (
        "The BI Canvas showcase combines a category share donut, comparative bars, a trend line, a scatter plot, "
        "a data-completeness gauge and a waterfall contribution chart so stakeholders can read composition, "
        "trend and relationship patterns at a glance."
    )

    if cor:
        c = cor[0]
        rel = (
            f"The strongest linear relationship is between '{c['a']}' and '{c['b']}' (r = {c['r']}), "
            f"a {'positive' if c['r'] > 0 else 'negative'} association. "
            "Highly correlated pairs may carry redundant information for modelling."
        )
    else:
        rel = "Fewer than two usable numeric columns were available, so no correlation analysis was possible."

    n_norm = sum(1 for n in norm if n.get("normal"))
    stat = (
        f"Shapiro-Wilk was run on {len(norm)} numeric columns and {n_norm} look approximately normal. "
        f"{sum(1 for a in an if a.get('significant'))} of {len(an)} ANOVA tests and "
        f"{sum(1 for x in chi if x.get('significant'))} of {len(chi)} chi-square tests were significant at the 5% level."
    )

    if reg and reg.get("top_features"):
        tf = reg["top_features"][0]
        ml_s = (
            f"For regression on '{reg['target']}', {reg['best_model']} achieved R2 = {reg['r2']} and RMSE = {reg['rmse']}; "
            f"the most influential feature is '{tf['feature']}'."
        )
    else:
        ml_s = "No regression model could be trained on this dataset."
    if cls:
        ml_s += f" Classification of '{cls['target']}' reached accuracy {cls['accuracy']} (weighted F1 {cls['f1']})."

    recs = []
    if d.get("duplicate_rows", 0) > 0:
        recs.append("Remove duplicate rows before reporting to avoid double counting.")
    if miss:
        recs.append(f"Investigate the source of missing values in '{miss[0]['col']}' and apply an imputation policy.")
    if num:
        o = max(num, key=lambda r: r.get("outliers_iqr", 0))
        if o.get("outliers_iqr", 0) > 0:
            recs.append(f"Review the {o['outliers_iqr']:,} outliers in '{o['col']}' to separate errors from genuine extremes.")
    if reg and reg.get("top_features"):
        recs.append(f"Focus business levers on '{reg['top_features'][0]['feature']}', the strongest driver of '{reg['target']}'.")
    recs.append("Re-run this report after each data refresh to track quality and model drift.")

    return _normalise(
        {
            "executive_summary": exec_s,
            "data_quality": dq,
            "bi_showcase": bi,
            "relationships": rel,
            "statistics": stat,
            "ml": ml_s,
            "recommendations": recs,
        }
    )


# ----------------------------------------------------------------------------
# LLM narrator
# ----------------------------------------------------------------------------
class LLMNarrator:
    """Generate the dashboard narrative using the chosen provider."""

    def __init__(self, provider: str = OFFLINE, api_key: str = "", model: str = "", timeout: int = 60):
        self.provider = provider if provider in PROVIDERS else OFFLINE
        self.cfg = PROVIDERS[self.provider]
        self.api_key = (api_key or env_key_for(self.provider)).strip()
        self.model = (model or self.cfg["default_model"]).strip()
        self.timeout = timeout

    # ---- provider calls --------------------------------------------------
    def _call_anthropic(self, user_msg: str) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1800,
                "temperature": 0.3,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")

    def _call_openai_compatible(self, user_msg: str) -> str:
        r = requests.post(
            f"{self.cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, user_msg: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        r = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    # ---- public API ------------------------------------------------------
    def generate(self, facts: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (narrative_dict, meta_dict). Never raises."""
        meta = {"provider": self.provider, "model": self.model, "used_llm": False, "error": ""}

        if self.cfg["kind"] == "offline":
            meta["model"] = "rule-engine-v1"
            return rule_based_narrative(facts), meta

        if not self.api_key:
            meta["error"] = "No API key supplied - used offline rule-based narrator."
            meta["provider"], meta["model"] = OFFLINE, "rule-engine-v1"
            return rule_based_narrative(facts), meta

        user_msg = "Dataset facts (JSON):\n" + json.dumps(facts, default=str)[:24000]
        try:
            kind = self.cfg["kind"]
            if kind == "anthropic":
                text = self._call_anthropic(user_msg)
            elif kind == "openai":
                text = self._call_openai_compatible(user_msg)
            else:
                text = self._call_gemini(user_msg)

            narrative = _normalise(_extract_json(text))
            # fill any empty section from the rule-based engine so the PDF is never blank
            fallback = rule_based_narrative(facts)
            for k in NARRATIVE_KEYS:
                if not narrative[k]:
                    narrative[k] = fallback[k]
            meta["used_llm"] = True
            return narrative, meta
        except Exception as exc:  # network, auth, quota, bad JSON ...
            logger.warning("LLM call failed: %s", exc)
            msg = str(exc)
            # never leak the key into logs/UI
            if self.api_key:
                msg = msg.replace(self.api_key, "***")
            meta["error"] = f"LLM call failed ({msg[:180]}) - used offline rule-based narrator."
            meta["provider"], meta["model"] = OFFLINE, "rule-engine-v1"
            return rule_based_narrative(facts), meta
