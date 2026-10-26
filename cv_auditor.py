"""
cv_auditor.py
=============
Image Quality & Vision Auditor for the Enterprise AI Data Analyst Suite PRO.

Features
--------
* Multi-image upload with a side-by-side comparison table
* Quality metrics: brightness, contrast, sharpness (blur), exposure clipping,
  entropy, edge density, colourfulness, saturation  ->  0-100 quality score
* Dominant colour palette (K-Means), RGB histogram, edge map, grayscale view
* AI vision description (Claude / OpenAI / Groq / Gemini) with offline fallback
* Downloadable PDF audit report in the same dashboard style as the data report

Only numpy / scipy / scikit-learn / Pillow are used (no OpenCV needed).
"""

import base64
import datetime
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from scipy import ndimage
from sklearn.cluster import KMeans

import dashboard_report as dr
from llm_narrator import OFFLINE, PROVIDERS, env_key_for
from manifest_logs import log_event

logger = logging.getLogger("CVAuditor")

VISION_MODELS = {
    "Anthropic Claude": "claude-sonnet-4-5",
    "OpenAI": "gpt-4o-mini",
    "Groq (free tier)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Google Gemini": "gemini-2.0-flash",
}

VISION_PROMPT = (
    "You are auditing an image for a data-quality report. In 3 to 4 plain-text sentences "
    "(no markdown, no bullet points) describe: the main subject and scene, notable objects or visible text, "
    "and any visible quality problems such as blur, noise, poor lighting or low resolution."
)


# ============================================================================
# 1. ANALYSIS
# ============================================================================
def analyze_image(raw: bytes) -> Dict[str, Any]:
    """Return a dict with `metrics`, `flags`, `palette` and numpy/PIL `views`."""
    img = Image.open(io.BytesIO(raw))
    fmt = img.format or "unknown"          # read BEFORE exif_transpose (it drops .format)
    img = ImageOps.exif_transpose(img)
    mode = img.mode
    width, height = img.size
    rgb = img.convert("RGB")

    work = rgb.copy()
    work.thumbnail((800, 800))
    arr = np.asarray(work).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    brightness = float(gray.mean() / 255)
    contrast = float(gray.std())
    sharpness = float(ndimage.laplace(gray).var())

    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    edge_density = float((mag > 100).mean())
    edge_img = Image.fromarray((mag / max(mag.max(), 1e-6) * 255).astype(np.uint8), "L")

    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    dark_clip = float((gray < 10).mean())
    bright_clip = float((gray > 245).mean())

    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    saturation = float(np.asarray(work.convert("HSV"))[..., 1].mean() / 255)

    # dominant colours
    pixels = arr.reshape(-1, 3)
    if len(pixels) > 6000:
        pixels = pixels[np.random.default_rng(42).choice(len(pixels), 6000, replace=False)]
    k = int(min(5, max(1, len(np.unique(pixels, axis=0)))))
    palette: List[Dict[str, Any]] = []
    try:
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(pixels)
        shares = np.bincount(km.labels_, minlength=k) / len(km.labels_)
        for idx in np.argsort(-shares):
            rr, gg, bb = [int(round(v)) for v in km.cluster_centers_[idx]]
            palette.append({"hex": f"#{rr:02X}{gg:02X}{bb:02X}", "share": round(float(shares[idx]) * 100, 1)})
    except Exception as exc:
        logger.warning("Palette failed: %s", exc)

    # quality score (heuristic, 0-100)
    exposure = 1 - min(1.0, abs(brightness - 0.5) / 0.5)
    score = (
        30 * exposure
        + 25 * min(1.0, contrast / 60)
        + 30 * min(1.0, sharpness / 150)
        + 15 * (1 - min(1.0, (dark_clip + bright_clip) / 0.3))
    )

    flags: List[Tuple[str, str]] = []  # (severity, message)
    if brightness < 0.25:
        flags.append(("bad", f"Underexposed: mean brightness {brightness:.2f} (dark image)."))
    elif brightness > 0.75:
        flags.append(("bad", f"Overexposed: mean brightness {brightness:.2f} (very bright image)."))
    else:
        flags.append(("ok", f"Exposure is balanced (brightness {brightness:.2f})."))
    if contrast < 30:
        flags.append(("warn", f"Low contrast (std {contrast:.1f}); image looks flat."))
    else:
        flags.append(("ok", f"Contrast is healthy (std {contrast:.1f})."))
    if sharpness < 40:
        flags.append(("bad", f"Blurry: Laplacian variance {sharpness:.0f} is very low."))
    elif sharpness < 120:
        flags.append(("warn", f"Slightly soft: Laplacian variance {sharpness:.0f}."))
    else:
        flags.append(("ok", f"Sharp detail (Laplacian variance {sharpness:.0f})."))
    if dark_clip + bright_clip > 0.10:
        flags.append(("warn", f"Clipping: {dark_clip * 100:.1f}% crushed blacks, {bright_clip * 100:.1f}% blown highlights."))
    if min(width, height) < 224:
        flags.append(("warn", f"Low resolution ({width}x{height}); many vision models expect at least 224 px."))

    return {
        "metrics": {
            "width": width, "height": height, "format": fmt, "mode": mode,
            "megapixels": round(width * height / 1e6, 2), "file_kb": round(len(raw) / 1024, 1),
            "brightness": round(brightness, 3), "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1), "entropy": round(entropy, 2),
            "edge_density": round(edge_density * 100, 1), "colorfulness": round(colorfulness, 1),
            "saturation": round(saturation, 3), "dark_clip_pct": round(dark_clip * 100, 1),
            "bright_clip_pct": round(bright_clip * 100, 1), "quality_score": round(score, 1),
            "verdict": "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor",
        },
        "flags": flags,
        "palette": palette,
        "hist": {"r": np.histogram(r, bins=64, range=(0, 255))[0],
                 "g": np.histogram(g, bins=64, range=(0, 255))[0],
                 "b": np.histogram(b, bins=64, range=(0, 255))[0]},
        "rgb": rgb, "gray": Image.fromarray(gray.astype(np.uint8), "L"), "edges": edge_img,
    }


def rule_based_description(a: Dict[str, Any]) -> str:
    m = a["metrics"]
    tone = "dark" if m["brightness"] < 0.35 else "bright" if m["brightness"] > 0.65 else "mid-toned"
    detail = "sharp" if m["sharpness"] >= 120 else "slightly soft" if m["sharpness"] >= 40 else "blurry"
    pal = ", ".join(f"{p['hex']} ({p['share']}%)" for p in a["palette"][:3]) or "n/a"
    return (
        f"This {m['width']}x{m['height']} {m['format']} image is {tone} and {detail}, with "
        f"{'strong' if m['contrast'] >= 50 else 'moderate' if m['contrast'] >= 30 else 'low'} contrast and "
        f"{'rich' if m['colorfulness'] >= 60 else 'muted'} colour (colourfulness {m['colorfulness']}). "
        f"Dominant colours: {pal}. Overall quality score is {m['quality_score']}/100 ({m['verdict']}). "
        "Note: this description is computed from pixel statistics; use AI vision for a content description."
    )


# ============================================================================
# 2. AI VISION
# ============================================================================
def _jpeg_b64(img: Image.Image, max_side: int = 1024) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def describe_with_vision(a: Dict[str, Any], provider: str, api_key: str, model: str = "") -> Tuple[str, Dict[str, Any]]:
    """Return (description, meta). Falls back to the rule-based text; never raises."""
    meta = {"used_llm": False, "provider": provider, "model": model, "error": ""}
    cfg = PROVIDERS.get(provider, PROVIDERS[OFFLINE])
    key = (api_key or env_key_for(provider)).strip()
    model = model or VISION_MODELS.get(provider, "")
    meta["model"] = model
    if cfg["kind"] == "offline" or not key:
        if cfg["kind"] != "offline":
            meta["error"] = "No API key - showing pixel-statistics description instead."
        return rule_based_description(a), meta
    try:
        b64 = _jpeg_b64(a["rgb"])
        if cfg["kind"] == "anthropic":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": VISION_PROMPT}]}]},
                timeout=90)
            r.raise_for_status()
            text = "".join(x.get("text", "") for x in r.json()["content"] if x.get("type") == "text")
        elif cfg["kind"] == "openai":
            r = requests.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]},
                timeout=90)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        else:  # gemini
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [
                    {"text": VISION_PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]},
                timeout=90)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        meta["used_llm"] = True
        return text.strip(), meta
    except Exception as exc:
        msg = str(exc).replace(key, "***")
        meta["error"] = f"Vision call failed ({msg[:160]}) - showing pixel-statistics description instead."
        return rule_based_description(a), meta


# ============================================================================
# 3. PDF REPORT
# ============================================================================
def _rgb_hist_chart(a: Dict[str, Any], w: float, h: float) -> ImageReader:
    fig, ax = dr._new(w, h)
    x = np.linspace(0, 255, 64)
    for ch, col in (("r", "#EF4444"), ("g", "#10B981"), ("b", "#2563EB")):
        ax.fill_between(x, a["hist"][ch], color=col, alpha=0.35)
        ax.plot(x, a["hist"][ch], color=col, lw=1)
    ax.set_xlabel("Pixel value (0-255)")
    ax.set_ylabel("Pixels")
    return dr._img(fig)


def _quality_bars(items, w: float, h: float) -> ImageReader:
    fig, ax = dr._new(w, h)
    names = [n[:22] for n, _, _ in items][::-1]
    vals = [a["metrics"]["quality_score"] for _, a, _ in items][::-1]
    cols = [dr.TEAL if v >= 60 else dr.AMBER if v >= 40 else dr.RED for v in vals]
    ax.barh(names, vals, color=cols)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Quality score (0-100)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=7)
    return dr._img(fig)


def build_audit_pdf(items: List[Tuple[str, Dict[str, Any], str]], title: str, meta: Optional[Dict[str, Any]] = None) -> io.BytesIO:
    """items = [(filename, analysis, description), ...] -> landscape dashboard PDF."""
    W, H, M, TOP, BOT, G = dr.W, dr.H, dr.MARGIN, dr.TOP, dr.BOT, dr.GAP
    stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    total = len(items) + (1 if len(items) > 1 else 0)
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(W, H))
    c.setTitle(dr._clean(title))
    page = 0

    if len(items) > 1:  # ---- summary page
        page += 1
        dr._frame(c, "Image Audit Summary", f"{len(items)} images compared", page, title, stamp, total)
        ix, iy, iw, ih = dr._panel(c, M, BOT, 420, TOP - BOT, "Quality Score by Image")
        dr._image(c, _quality_bars(items, iw, ih), ix, iy, iw, ih)
        ix, iy, iw, ih = dr._panel(c, M + 430, BOT, W - 2 * M - 430, TOP - BOT, "Comparison Table")
        rows = [["Image", "Size", "Bright.", "Sharp.", "Score", "Verdict"]]
        for n, a, _ in items[:18]:
            m = a["metrics"]
            rows.append([n[:20], f"{m['width']}x{m['height']}", f"{m['brightness']:.2f}", f"{m['sharpness']:.0f}", m["quality_score"], m["verdict"]])
        dr._table(c, rows, [iw * .27, iw * .19, iw * .13, iw * .13, iw * .13, iw * .15], ix, iy + ih, ih, font=7.2)
        c.showPage()

    for name, a, desc in items:
        page += 1
        m = a["metrics"]
        dr._frame(c, "Image Quality & Vision Audit", dr._clean(name), page, title, stamp, total)
        kw = (W - 2 * M - 4 * G) / 5
        kh = 60
        ky = TOP - kh
        cards = [
            ("Resolution", f"{m['width']}x{m['height']}", f"{m['megapixels']} MP, {m['format']}", dr.BLUE),
            ("Quality score", f"{m['quality_score']}", m["verdict"], dr.TEAL if m["quality_score"] >= 60 else dr.AMBER),
            ("Brightness", f"{m['brightness']:.2f}", "0 dark - 1 bright", dr.INDIGO),
            ("Sharpness", f"{m['sharpness']:.0f}", "Laplacian variance", dr.AMBER),
            ("Contrast", f"{m['contrast']:.1f}", f"entropy {m['entropy']}", dr.TEAL),
        ]
        for i, (lab, val, sub, col) in enumerate(cards):
            dr._kpi(c, M + i * (kw + G), ky, kw, kh, lab, val, sub, col)

        body_top = ky - G
        body_h = body_top - BOT
        w1, w2 = 250, 250
        w3 = W - 2 * M - w1 - w2 - 2 * G
        x1, x2, x3 = M, M + w1 + G, M + w1 + w2 + 2 * G

        # col 1: original + findings
        h_img = 250
        ix, iy, iw, ih = dr._panel(c, x1, body_top - h_img, w1, h_img, "Original Image")
        pil = a["rgb"].copy()
        pil.thumbnail((900, 900))
        dr._image(c, ImageReader(pil), ix, iy, iw, ih)
        h_f = body_h - h_img - G
        ix, iy, iw, ih = dr._panel(c, x1, BOT, w1, h_f, "Quality Findings")
        y = iy + ih
        sev_col = {"ok": dr.TEAL, "warn": dr.AMBER, "bad": dr.RED}
        for sev, msg in a["flags"]:
            c.setFillColor(dr._hex(sev_col[sev]))
            c.circle(ix + 4, y - 7, 2.6, fill=1, stroke=0)
            used = dr._para(c, msg, ix + 14, y - 1, iw - 14, 40, size=7.6)
            y -= used + 5
            if y < iy:
                break

        # col 2: edge map + histogram
        hh = (body_h - G) / 2
        ix, iy, iw, ih = dr._panel(c, x2, body_top - hh, w2, hh, f"Edge Map (edge density {m['edge_density']}%)")
        dr._image(c, ImageReader(a["edges"]), ix, iy, iw, ih)
        ix, iy, iw, ih = dr._panel(c, x2, BOT, w2, hh, "RGB Histogram")
        dr._image(c, _rgb_hist_chart(a, iw, ih), ix, iy, iw, ih)

        # col 3: palette + description
        h_p = 170
        ix, iy, iw, ih = dr._panel(c, x3, body_top - h_p, w3, h_p, "Dominant Colours")
        y = iy + ih - 4
        for p in a["palette"][:5]:
            c.setFillColor(dr._hex(p["hex"]))
            c.setStrokeColor(dr._hex(dr.BORDER))
            c.roundRect(ix, y - 18, 28, 18, 3, fill=1, stroke=1)
            c.setFillColor(dr._hex(dr.NAVY))
            c.setFont("Helvetica", 8)
            c.drawString(ix + 36, y - 12, f"{p['hex']}   {p['share']}%")
            y -= 26
        ix, iy, iw, ih = dr._panel(c, x3, BOT, w3, body_h - h_p - G, "Description")
        mode = "AI vision" if (meta and meta.get("used_llm")) else "Pixel statistics"
        txt = f"[{mode}] {desc}"
        dr._para(c, txt, ix, iy + ih, iw, ih, size=7.8)
        c.showPage()

    c.save()
    buf.seek(0)
    return buf


# ============================================================================
# 4. STREAMLIT WORKSPACE
# ============================================================================
def render_cv_workspace():
    import streamlit as st  # imported lazily so the module is usable without a UI

    @st.cache_data(show_spinner=False)
    def _cached(raw: bytes):
        return analyze_image(raw)

    st.subheader("🖼️ Image Quality & Vision Auditor")
    st.caption("Upload one or more images. The auditor measures exposure, blur, contrast, colour and edges, "
               "scores overall quality, and can ask an AI vision model to describe the content.")

    files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "webp", "bmp"],
                             accept_multiple_files=True, key="cv_files")
    if not files:
        st.info("💡 Upload one or more images to start the audit.")
        return

    analyses: List[Tuple[str, Dict[str, Any]]] = []
    for f in files:
        try:
            raw_bytes = f.getvalue()
            analyses.append((f.name, _cached(raw_bytes)))
            sig = f"cv_logged_{f.name}_{len(raw_bytes)}"
            if not st.session_state.get(sig):
                st.session_state[sig] = True
                st.session_state["images_audited"] = st.session_state.get("images_audited", 0) + 1
                mm = analyses[-1][1]["metrics"]
                log_event("Vision", "Image analysed", f"{f.name}: score {mm['quality_score']} ({mm['verdict']}), {mm['width']}x{mm['height']}")
        except Exception as exc:
            st.error(f"Could not read `{f.name}`: {exc}")
    if not analyses:
        return

    # ---- comparison
    if len(analyses) > 1:
        st.markdown("#### 📊 Batch Comparison")
        df = pd.DataFrame([{"Image": n, "Resolution": f"{a['metrics']['width']}x{a['metrics']['height']}",
                            "Brightness": a["metrics"]["brightness"], "Contrast": a["metrics"]["contrast"],
                            "Sharpness": a["metrics"]["sharpness"], "Quality score": a["metrics"]["quality_score"],
                            "Verdict": a["metrics"]["verdict"]} for n, a in analyses])
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with cc2:
            fig = go.Figure(go.Bar(x=df["Quality score"], y=df["Image"], orientation="h",
                                   marker_color=["#14B8A6" if v >= 60 else "#F59E0B" if v >= 40 else "#EF4444" for v in df["Quality score"]]))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis_range=[0, 100], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        names = [n for n, _ in analyses]
        chosen = st.selectbox("Inspect image in detail", names)
    else:
        chosen = analyses[0][0]
    a = dict(analyses)[chosen]
    m = a["metrics"]

    st.markdown(f"#### 🔎 Detail: `{chosen}`")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Quality score", f"{m['quality_score']}/100", m["verdict"])
    k2.metric("Resolution", f"{m['width']}x{m['height']}", f"{m['megapixels']} MP")
    k3.metric("Brightness", f"{m['brightness']:.2f}")
    k4.metric("Sharpness", f"{m['sharpness']:.0f}")
    k5.metric("Contrast", f"{m['contrast']:.1f}")

    for sev, msg in a["flags"]:
        (st.success if sev == "ok" else st.warning if sev == "warn" else st.error)(msg)

    t_view, t_color, t_edge, t_ai, t_pdf = st.tabs(["🖼️ Views", "🎨 Colour & Histogram", "📐 Edges & Detail", "🤖 AI Vision", "📥 PDF Audit"])

    with t_view:
        v1, v2, v3 = st.columns(3)
        v1.image(a["rgb"], caption="Original", use_container_width=True)
        v2.image(a["gray"], caption="Grayscale", use_container_width=True)
        v3.image(a["edges"], caption="Edge map (Sobel)", use_container_width=True)

    with t_color:
        c1, c2 = st.columns(2)
        with c1:
            xs = np.linspace(0, 255, 64)
            fig = go.Figure()
            for ch, col in (("r", "#EF4444"), ("g", "#10B981"), ("b", "#2563EB")):
                fig.add_trace(go.Scatter(x=xs, y=a["hist"][ch], name=ch.upper(), fill="tozeroy", line=dict(color=col)))
            fig.update_layout(title="RGB Histogram", height=320, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pal = a["palette"]
            fig = go.Figure(go.Pie(labels=[p["hex"] for p in pal], values=[p["share"] for p in pal],
                                   marker=dict(colors=[p["hex"] for p in pal]), hole=0.45, textinfo="label+percent"))
            fig.update_layout(title="Dominant Colours", height=320, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Colourfulness", m["colorfulness"])
        s2.metric("Mean saturation", m["saturation"])
        s3.metric("Entropy (bits)", m["entropy"])

    with t_edge:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Edge density", f"{m['edge_density']}%")
        e2.metric("Crushed blacks", f"{m['dark_clip_pct']}%")
        e3.metric("Blown highlights", f"{m['bright_clip_pct']}%")
        e4.metric("File size", f"{m['file_kb']} KB")
        st.caption("Sharpness = variance of the Laplacian (below ~40 is blurry, above ~120 is sharp). "
                   "Edge density = share of pixels with a strong Sobel gradient.")

    with t_ai:
        st.markdown("Ask a vision-capable LLM to describe the image. **The (resized) image is sent to the chosen provider.**")
        ac1, ac2, ac3 = st.columns([1.2, 1.4, 1.6])
        with ac1:
            prov = st.selectbox("Provider", [p for p in PROVIDERS if p != OFFLINE], key="cv_provider")
        with ac2:
            mdl = st.text_input("Vision model", VISION_MODELS.get(prov, ""), key=f"cv_model_{prov}")
        with ac3:
            key = st.text_input("API Key", type="password", value=env_key_for(prov), key=f"cv_key_{prov}")
        if st.button("✨ Describe image with AI", key="cv_describe"):
            with st.spinner("Asking the vision model..."):
                desc, meta = describe_with_vision(a, prov, key, mdl)
            st.session_state[f"cv_desc_{chosen}"] = (desc, meta)
            log_event("Vision", "AI image description", f"{chosen}: provider={meta['provider']}; llm_used={meta['used_llm']}",
                      status="success" if meta["used_llm"] or not meta["error"] else "warning")
        res = st.session_state.get(f"cv_desc_{chosen}")
        if res:
            desc, meta = res
            (st.success if meta["used_llm"] else st.warning)(
                f"Described by {meta['provider']} ({meta['model']})" if meta["used_llm"] else (meta["error"] or "Offline description"))
            st.write(desc)

    with t_pdf:
        st.markdown("Create a dashboard-style PDF for all uploaded images (uses AI descriptions where you generated them).")
        title = st.text_input("Report title", "Image Quality & Vision Audit Report", key="cv_pdf_title")
        if st.button("🖨️ Build PDF audit report", key="cv_build_pdf"):
            with st.spinner("Rendering PDF..."):
                items, last_meta = [], None
                for n, an in analyses:
                    cached = st.session_state.get(f"cv_desc_{n}")
                    if cached:
                        desc, last_meta = cached
                    else:
                        desc = rule_based_description(an)
                    items.append((n, an, desc))
                st.session_state["cv_pdf_bytes"] = build_audit_pdf(items[:12], title, last_meta).getvalue()
                log_event("Report", "Image audit PDF generated", f"{min(len(items), 12)} image(s)")
        if st.session_state.get("cv_pdf_bytes"):
            st.download_button("💾 Download Image Audit PDF", st.session_state["cv_pdf_bytes"],
                               file_name="Image_Audit_Report.pdf", mime="application/pdf", use_container_width=True)
