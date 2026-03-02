import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from risk_engine import (
    RiskConfig,
    build_feature_table,
    fit_ai_models,
    score_candidates,
    batch_statistics,
    split_candidates_by_acceptance,
)

# Plotly for 3D
PLOTLY_OK = False
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


st.set_page_config(page_title="Protein Reliability Risk Screening", layout="wide")

# -----------------------------
# Sidebar: logo + controls + reference
# -----------------------------
with st.sidebar:
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.markdown("### Controls")

    use_builtin = st.checkbox(
        "Use built-in demo file (data/demo_100.csv)",
        value=True,
        help="Loads a built-in demo dataset (id + sequence). Upload is optional for custom datasets."
    )
    uploaded = None
    if not use_builtin:
        uploaded = st.file_uploader(
            "Upload CSV (columns: id, sequence)",
            type=["csv"],
            help="Upload a CSV containing at least: id, sequence. Additional columns are ignored."
        )

    st.divider()

    cfg = RiskConfig()

    c1, c2 = st.columns(2)
    with c1:
        w_agg = st.slider(
            "w(Agg)",
            0.0, 1.0, float(cfg.w_agg), 0.05,
            help="Weight for Aggregation risk in overall score. Recommended start: 0.40."
        )
        w_scale = st.slider(
            "w(Scale)",
            0.0, 1.0, float(cfg.w_scale), 0.05,
            help="Weight for Scale-up sensitivity in overall score. Recommended start: 0.35."
        )
    with c2:
        w_stab = st.slider(
            "w(Stab)",
            0.0, 1.0, float(cfg.w_stab), 0.05,
            help="Weight for Stability risk in overall score. Recommended start: 0.25."
        )
        normalize = st.checkbox(
            "Normalize weights",
            value=True,
            help="If enabled, weights are normalized to sum to 1."
        )

    st.caption("Thresholds")
    overall_threshold = st.slider(
        "Overall threshold (reject ≥)",
        0.0, 100.0, float(cfg.overall_review_threshold), 1.0,
        help="Candidates with overall risk score ≥ this value are filtered out. Recommended default: 70."
    )
    effort_threshold = st.slider(
        "Effort threshold (reject >)",
        10.0, 1500.0, float(cfg.effort_accept_threshold), 5.0,
        help="Candidates requiring wet-lab effort index > this value are filtered out. Recommended default: 100."
    )

    st.caption("Flags")
    cfg.critical_subrisk_threshold = st.slider(
        "Critical sub-risk flag (≥)",
        0.0, 100.0, float(cfg.critical_subrisk_threshold), 1.0,
        help="If any sub-risk score ≥ this value, candidate is marked Critical (overrides accept). Recommended: 92."
    )
    cfg.major_risk_threshold = st.slider(
        "Major axis flag (≥)",
        0.0, 100.0, float(cfg.major_risk_threshold), 1.0,
        help="If any major axis (Agg/Scale/Stab) ≥ this value, candidate is flagged as major-risk. Recommended: 80."
    )

    st.divider()
    st.subheader("View")
    enable_3d = st.checkbox(
        "Enable 3D portfolio plot (Agg/Scale/Stab)",
        value=True,
        help="Requires plotly. If disabled or plotly not installed, a 2D fallback plot is shown."
    )
    if enable_3d and not PLOTLY_OK:
        st.warning("Plotly not installed → 3D disabled. Run: `pip install plotly`")

    run_btn = st.button("Evaluate", type="primary", help="Compute risk profiles for the current dataset.")


# -----------------------------
# Main title row + mini logo
# -----------------------------
tcol1, tcol2 = st.columns([0.86, 0.14], gap="small")
with tcol1:
    st.markdown("## AI-Driven Early-Stage Protein Reliability Risk Screening")
    st.caption("Public sequences → interpretable features → AI-weighted risk mapping → decision support for wet-lab resource allocation")
with tcol2:
    mini = Path("logomini.png")
    if mini.exists():
        st.image(str(mini), use_container_width=True)


# -----------------------------
# Load data
# -----------------------------
df = None
if use_builtin:
    try:
        df_raw = pd.read_csv("data/demo_100.csv")
        df = df_raw[["id", "sequence"]].copy()
    except Exception as e:
        st.error(f"Failed to load data/demo_100.csv: {e}")
else:
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            df = df_raw[["id", "sequence"]].copy()
        except Exception as e:
            st.error(f"Failed to read uploaded CSV or missing required columns: {e}")

if df is not None:
    # keep this subtle; not a big banner
    st.caption(f"Dataset ready: **{len(df)}** candidates (id + sequence).")


# -----------------------------
# Helpers
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rgb(r, g, b) -> str:
    return f"rgb({int(r)},{int(g)},{int(b)})"

def acceptance_status(row, overall_th, effort_th):
    if bool(row["critical_subrisk"]):
        return "Critical"
    if float(row["wetlab_effort_index"]) > effort_th:
        return "Reject"
    if float(row["overall"]) >= overall_th:
        return "Reject"
    return "Accept"

def color_for_point(status: str, overall: float, accept_split: float = 45.0) -> str:
    if status in ("Reject", "Critical"):
        return rgb(160, 160, 160)

    if overall < accept_split:
        t = clamp01(overall / max(accept_split, 1e-6))
        r = 30 + 110 * t
        g = 190 + 40 * (1 - t)
        b = 30 + 110 * t
        return rgb(r, g, b)
    else:
        t = clamp01((overall - accept_split) / max(100.0 - accept_split, 1e-6))
        r = 210 + 45 * t
        g = 90 + 70 * (1 - t)
        b = 90 + 70 * (1 - t)
        return rgb(r, g, b)

def symbol_for_point(status: str) -> str:
    return "x" if status in ("Reject", "Critical") else "circle"

def plot_mw_distribution_sidebar(mw_kda: np.ndarray, bins: int = 14):
    mw = mw_kda[~np.isnan(mw_kda)]
    fig = plt.figure(figsize=(4.8, 2.6))
    ax = plt.gca()

    if len(mw) < 5:
        ax.text(0.1, 0.5, "MW distribution not available", fontsize=10)
        ax.axis("off")
        return fig

    mu = float(np.mean(mw))
    sigma = float(np.std(mw, ddof=1)) if len(mw) > 1 else 1.0

    counts, bin_edges, _ = ax.hist(mw, bins=bins, alpha=0.9)
    ax.set_xlabel("MW (kDa)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("MW distribution", fontsize=10)

    # fitted normal curve (scaled to histogram)
    x = np.linspace(bin_edges[0], bin_edges[-1], 250)
    bin_w = bin_edges[1] - bin_edges[0]
    pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y = pdf * len(mw) * bin_w
    ax.plot(x, y, linewidth=2)

    ax.axvline(mu, linestyle="--", linewidth=1)
    ax.tick_params(axis="both", labelsize=8)
    return fig


# -----------------------------
# Run evaluation (store results)
# -----------------------------
if run_btn:
    if df is None:
        st.warning("Please load a dataset first.")
        st.stop()

    cfg.w_agg, cfg.w_scale, cfg.w_stab = float(w_agg), float(w_scale), float(w_stab)
    if normalize:
        s = cfg.w_agg + cfg.w_scale + cfg.w_stab
        if s <= 1e-12:
            st.error("Weights sum to zero; set at least one > 0.")
            st.stop()
        cfg.w_agg /= s
        cfg.w_scale /= s
        cfg.w_stab /= s

    with st.spinner("Computing features and AI-weighted risk profiles..."):
        X = build_feature_table(df)
        models = fit_ai_models(X, cfg=cfg)
        scored = score_candidates(X, models, cfg=cfg)

    scored = scored.copy()
    scored["status"] = scored.apply(lambda r: acceptance_status(r, overall_threshold, effort_threshold), axis=1)
    stats = batch_statistics(scored)

    st.session_state["scored"] = scored
    st.session_state["stats"] = stats
    st.session_state["df_input"] = df


# -----------------------------
# Sidebar reference block (after evaluation)
# -----------------------------
with st.sidebar:
    if "stats" in st.session_state:
        stats = st.session_state["stats"]

        st.divider()
        st.markdown("### Reference (batch context)")

        r1, r2 = st.columns(2)
        r1.metric("N", int(stats["n"]))
        r2.metric("Effort med.", f"{stats['effort_median']:.1f}")

        r3, r4 = st.columns(2)
        r3.metric("MW mean", f"{stats['mw_kda_mean']:.2f}")
        r4.metric("MW med.", f"{stats['mw_kda_median']:.2f}")

        fig = plot_mw_distribution_sidebar(stats["dist_mw_kda"], bins=14)
        st.pyplot(fig, use_container_width=True)

        st.caption("Reference only. Core decision is driven by risk axes + effort gating.")


# -----------------------------
# Main core outputs
# -----------------------------
if "scored" not in st.session_state:
    st.info("Click **Evaluate** to compute portfolio results.")
    st.stop()

scored = st.session_state["scored"]
df = st.session_state["df_input"]
stats = st.session_state["stats"]

st.markdown("### Decision summary (core output)")

accept_n = int((scored["status"] == "Accept").sum())
reject_n = int((scored["status"] == "Reject").sum())
critical_n = int((scored["status"] == "Critical").sum())

c1, c2, c3, c4 = st.columns([1, 1, 1, 2], gap="large")
c1.metric("Accept", accept_n)
c2.metric("Reject", reject_n)
c3.metric("Critical", critical_n)
c4.markdown(
    f"""
**Operational thresholds**  
- Reject if **overall ≥ {overall_threshold:.0f}** or **effort > {effort_threshold:.0f}**  
- Critical if **any sub-risk ≥ {cfg.critical_subrisk_threshold:.0f}**
"""
)

st.divider()

st.markdown("#### Portfolio map (Agg / Scale-up / Stability)")

accept_split = cfg.overall_accept_threshold
colors = [color_for_point(s, o, accept_split=accept_split) for s, o in zip(scored["status"], scored["overall"])]
symbols = [symbol_for_point(s) for s in scored["status"]]
sizes = np.where(scored["status"].isin(["Reject", "Critical"]), 7, 8).astype(int)

hover = []
for _, r in scored.iterrows():
    hover.append(
        f"{r['id']}<br>"
        f"status={r['status']} | overall={r['overall']}<br>"
        f"Agg={r['Aggregation']} | Scale={r['ScaleUpSensitivity']} | Stab={r['Stability']}<br>"
        f"effort={r['wetlab_effort_index']} ({r['effort_class']})<br>"
        f"critical={bool(r['critical_subrisk'])} | major_flag={bool(r['major_risk_flag'])}"
    )

if enable_3d and PLOTLY_OK:
    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=scored["Aggregation"],
            y=scored["ScaleUpSensitivity"],
            z=scored["Stability"],
            mode="markers",
            marker=dict(color=colors, symbol=symbols, size=sizes, opacity=0.92, line=dict(width=0)),
            text=hover,
            hoverinfo="text",
            customdata=np.column_stack([scored["id"].values]),
        )
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Agg (0–100)",
            yaxis_title="Scale (0–100)",
            zaxis_title="Stab (0–100)",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        title="3D portfolio map (green→red = accepted intensity; gray X = filtered out)"
    )
    event = st.plotly_chart(
    fig3d,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
)

    # If a point is selected, store its ID
    try:
        if event and event.selection and event.selection.points:
            # We put id into customdata[0] below; adjust if your customdata is different
            selected_id = event.selection.points[0]["customdata"][0]
            st.session_state["selected_id"] = selected_id
    except Exception:
        pass
else:
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    accept = scored[scored["status"] == "Accept"]
    reject = scored[scored["status"] == "Reject"]
    critical = scored[scored["status"] == "Critical"]
    ax.scatter(accept["overall"], accept["wetlab_effort_index"], alpha=0.9)
    ax.scatter(reject["overall"], reject["wetlab_effort_index"], marker="x", alpha=0.9)
    ax.scatter(critical["overall"], critical["wetlab_effort_index"], marker="x", alpha=1.0)
    ax.axvline(overall_threshold)
    ax.axhline(effort_threshold)
    ax.set_xlabel("Overall (0–100)")
    ax.set_ylabel("Effort (proxy)")
    ax.set_title("2D fallback: overall vs effort (threshold gating)")
    ax.legend(["Accept", "Reject", "Critical"])
    st.pyplot(fig, use_container_width=True)

st.caption("Accepted candidates are green→red by overall score. Rejected/Critical are gray X.")

st.divider()

# -----------------------------
# Outputs table: compact headers + progress bars + hover help
# -----------------------------
st.markdown("#### Wet-lab planning outputs (exportable)")

groups = split_candidates_by_acceptance(
    df_input=df,
    scored=scored,
    cfg=cfg,
    overall_threshold=overall_threshold,
    effort_threshold=effort_threshold
)

def to_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")

def compact_table(d: pd.DataFrame) -> pd.DataFrame:
    # rename to short headers
    out = d.copy()
    out = out.rename(columns={
        "id": "ID",
        "overall": "Overall",
        "wetlab_effort_index": "Effort",
        "Aggregation": "Agg",
        "ScaleUpSensitivity": "Scale",
        "Stability": "Stab",
        "risk_class": "Class",
        "effort_class": "EffCls",
        "critical_subrisk": "Critical",
        "major_risk_flag": "MajorFlag",
    })
    # keep only key columns for display (sequence stays in CSV export)
    keep = ["ID", "Overall", "Effort", "Agg", "Scale", "Stab", "Class", "EffCls", "Critical", "MajorFlag"]
    keep = [c for c in keep if c in out.columns]
    return out[keep]

col_cfg = {
    "ID": st.column_config.TextColumn("ID", help="Candidate identifier."),
    "Overall": st.column_config.ProgressColumn(
        "Overall",
        help="Overall risk score (0–100). Lower is better.",
        min_value=0.0, max_value=100.0, format="%.1f"
    ),
    "Effort": st.column_config.ProgressColumn(
        "Effort",
        help="Wet-lab effort index (proxy). Lower means fewer iteration loops expected.",
        min_value=0.0, max_value=1500.0, format="%.1f"
    ),
    "Agg": st.column_config.ProgressColumn(
        "Agg",
        help="Aggregation risk axis (0–100).",
        min_value=0.0, max_value=100.0, format="%.1f"
    ),
    "Scale": st.column_config.ProgressColumn(
        "Scale",
        help="Scale-up sensitivity axis (0–100).",
        min_value=0.0, max_value=100.0, format="%.1f"
    ),
    "Stab": st.column_config.ProgressColumn(
        "Stab",
        help="Stability risk axis (0–100).",
        min_value=0.0, max_value=100.0, format="%.1f"
    ),
    "Class": st.column_config.TextColumn("Class", help="Risk class derived from overall thresholds."),
    "EffCls": st.column_config.TextColumn("EffCls", help="Effort class (Low/Medium/High)."),
    "Critical": st.column_config.CheckboxColumn("Crit", help="Critical if any sub-risk exceeds the critical threshold."),
    "MajorFlag": st.column_config.CheckboxColumn("Maj", help="Flag if any major axis exceeds the major-axis threshold."),
}

g1, g2, g3 = st.columns(3, gap="large")
with g1:
    st.write("✅ Accept (low-risk)")
    view = compact_table(groups["accept_low_risk"]).head(15)
    st.dataframe(view, use_container_width=True, column_config=col_cfg, hide_index=True)
    st.download_button("Download accept_low_risk.csv", data=to_csv_bytes(groups["accept_low_risk"]),
                       file_name="accept_low_risk.csv", mime="text/csv")
with g2:
    st.write("🟨 Accept (review)")
    view = compact_table(groups["accept_high_risk"]).head(15)
    st.dataframe(view, use_container_width=True, column_config=col_cfg, hide_index=True)
    st.download_button("Download accept_high_risk.csv", data=to_csv_bytes(groups["accept_high_risk"]),
                       file_name="accept_high_risk.csv", mime="text/csv")
with g3:
    st.write("⛔ Reject / Critical")
    view = compact_table(groups["reject"]).head(15)
    st.dataframe(view, use_container_width=True, column_config=col_cfg, hide_index=True)
    st.download_button("Download reject.csv", data=to_csv_bytes(groups["reject"]),
                       file_name="reject.csv", mime="text/csv")

st.divider()

# Single-candidate radar (kept)
st.markdown("### Single-candidate explainability (6 sub-metrics)")

# pick default: worst overall
ids = scored["id"].tolist()

# default: use selected from portfolio if exists; else worst overall
fallback_id = scored.sort_values("overall", ascending=False)["id"].iloc[0]
default_id = st.session_state.get("selected_id", fallback_id)

# find index safely
default_idx = ids.index(default_id) if default_id in ids else 0

# (optional) anchor
st.markdown("<div id='radar'></div>", unsafe_allow_html=True)

pick = st.selectbox("Select candidate", ids, index=default_idx)

# show a small indicator if selection came from portfolio
if "selected_id" in st.session_state:
    st.caption(f"Selected from portfolio: **{st.session_state['selected_id']}**")
    
row = scored[scored["id"] == pick].iloc[0]

# --- pull the original sequence (from input df)
seq_map = dict(zip(df["id"], df["sequence"]))
seq = seq_map.get(pick, "")
seq_len = len(seq)

# --- basic inferred MW from scored table if present 
mw_kda = float(row["mw_kda"]) if "mw_kda" in row.index else float("nan")

metrics6 = [
    ("Self-assoc", "agg_self_association"),
    ("Hydro patch", "agg_hydrophobic_patch"),
    ("Viscosity", "scale_viscosity"),
    ("Process", "scale_processability"),
    ("Conform", "stab_conformational"),
    ("Chem liab", "stab_chemical_liabilities"),
]
labels = [m[0] for m in metrics6]
keys = [m[1] for m in metrics6]

cand_vals = scored.loc[scored["id"] == pick, keys].iloc[0].values.astype(float)

mean_vals = scored[keys].mean().values.astype(float)
q25 = scored[keys].quantile(0.25).values.astype(float)
q75 = scored[keys].quantile(0.75).values.astype(float)

# -----------------------------
# Layout: left info + right radar
# -----------------------------
left, right = st.columns([1.05, 1.2], gap="large")

with left:
    st.markdown("#### Candidate overview")

    # compact key KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Status", str(row.get("status", "")))
    k2.metric("Overall", f"{float(row['overall']):.1f}")
    k3.metric("Effort", f"{float(row['wetlab_effort_index']):.1f}")

    k4, k5, k6 = st.columns(3)
    k4.metric("Agg", f"{float(row['Aggregation']):.1f}")
    k5.metric("Scale", f"{float(row['ScaleUpSensitivity']):.1f}")
    k6.metric("Stab", f"{float(row['Stability']):.1f}")

    # extra basic info
    b1, b2, b3 = st.columns(3)
    b1.metric("Seq length", f"{seq_len}")
    if np.isnan(mw_kda):
        b2.metric("MW (kDa)", "—")
    else:
        b2.metric("MW (kDa)", f"{mw_kda:.2f}")

    b3.metric("Critical", "Yes" if bool(row.get("critical_subrisk", False)) else "No")

    # show sub-metrics table (more technical)
    st.markdown("#### 6-metric signals")
    df_view = pd.DataFrame({
        "Metric": labels,
        "Candidate": np.round(cand_vals, 1),
        "Batch mean": np.round(mean_vals, 1),
        "IQR25": np.round(q25, 1),
        "IQR75": np.round(q75, 1),
    })
    st.dataframe(df_view, use_container_width=True, hide_index=True)


with right:
    st.markdown("#### Risk radar (candidate vs batch)")

    # --- Radar plot (dark-friendly, professional)
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles_c = np.concatenate([angles, angles[:1]])

    def close(v):
        return np.concatenate([v, v[:1]])

    fig = plt.figure(figsize=(7.2, 5.9))
    fig.patch.set_alpha(0.0)  # transparent canvas
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor((0, 0, 0, 0))  # transparent

    # soft grid / spines for dark background
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.20))
    ax.grid(color=(1, 1, 1, 0.16))
    ax.tick_params(colors=(1, 1, 1, 0.78))
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=10, color=(1, 1, 1, 0.90))
    ax.set_ylim(0, 100)

    # IQR band (subtle)
    ax.fill_between(
        angles_c,
        close(q25),
        close(q75),
        color=(0.85, 0.90, 1.00, 0.10),
        alpha=0.25
    )

    # mean (dashed, cool tone)
    ax.plot(
        angles_c, close(mean_vals),
        linestyle="--",
        linewidth=2.0,
        color=(0.75, 0.82, 1.00, 0.65)
    )

    # candidate (primary accent)
    ax.plot(
        angles_c, close(cand_vals),
        linestyle="-",
        linewidth=2.8,
        color=(0.40, 0.85, 0.95, 0.95)
    )
    ax.fill(
        angles_c, close(cand_vals),
        color=(0.40, 0.85, 0.95, 0.15),
        alpha=0.22
    )

    # warning badge if high risk
    vmax = float(np.max(cand_vals))
    warn_threshold = 80.0
    if vmax >= warn_threshold:
        idx = int(np.argmax(cand_vals))
        ax.scatter(
            [angles[idx]],
            [cand_vals[idx]],
            s=90,
            marker="^",
            color=(1.0, 0.35, 0.35, 0.95),
            edgecolors=(1, 1, 1, 0.35),
            linewidths=0.8
        )
        ax.text(
            0.02, 1.06,
            "⚠ High-risk signal",
            transform=ax.transAxes,
            color=(1.0, 0.40, 0.40, 0.95),
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title("Candidate (solid) vs batch mean (dashed) + IQR band", fontsize=12, color=(1, 1, 1, 0.90), pad=14)
    st.pyplot(fig, use_container_width=True)

    # --- Amino-acid sequence (direct display)
    st.markdown("#### Amino-acid sequence")

    if seq:
        st.code(
            seq,
            language="text"
        )
        st.download_button(
            "Download FASTA",
            data=f">{pick}\n{seq}\n",
            file_name=f"{pick}.fasta",
            mime="text/plain"
        )
    else:
        st.caption("Sequence not available in input dataset.")


st.caption(
    "This panel provides interpretable early signals aligned with aggregation, scale-up sensitivity, and stability risks. "
    "It supports prioritization and resource allocation (not a substitute for experimental validation)."
)