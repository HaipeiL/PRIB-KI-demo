import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import Ridge


# -----------------------------
# Amino acid sets (rough)
# -----------------------------
AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_HYDRO = set("AILMFWVY")          # hydrophobic (rough)
AA_AROM = set("FWY")                # aromatic
AA_POS = set("KRH")                 # positive at ~neutral (rough)
AA_NEG = set("DE")                  # negative at ~neutral (rough)

# Chemical liability proxies (motif heuristics; demo-safe)
DEAMID_MOTIFS = ("NG", "NS", "NT")  # deamidation-prone motifs (proxy)
ISOM_MOTIFS = ("DG",)               # Asp isomerization proxy
OXID_AA = set("MW")                 # oxidation-prone residues (proxy)
CLIP_MOTIFS = ("DP", "DG")          # clipping hotspots (very rough)


# -----------------------------
# AA molecular weights (Da) - average residue masses (approx, no water)
# For demo: good enough to give a meaningful MW distribution.
# -----------------------------
AA_MASS_DA = {
    "A": 71.08, "C": 103.14, "D": 115.09, "E": 129.12, "F": 147.18,
    "G": 57.05, "H": 137.14, "I": 113.16, "K": 128.17, "L": 113.16,
    "M": 131.20, "N": 114.11, "P": 97.12,  "Q": 128.13, "R": 156.19,
    "S": 87.08, "T": 101.11, "V": 99.13,  "W": 186.21, "Y": 163.18
}
WATER_DA = 18.015  # for peptide bond correction (rough)

# Kyte-Doolittle hydropathy scale (common)
KD = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3
}

# pKa values (rough, widely used approximations)
PKA = {
    "Cterm": 3.1,
    "Nterm": 8.0,
    "C": 8.3,
    "D": 3.9,
    "E": 4.3,
    "H": 6.0,
    "K": 10.5,
    "R": 12.5,
    "Y": 10.1,
}

def gravy(seq: str) -> float:
    s = seq.strip().upper()
    vals = [KD[a] for a in s if a in KD]
    return float(np.mean(vals)) if vals else float("nan")


def net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    """
    Net charge at given pH using Henderson–Hasselbalch with rough pKa values.
    This is an approximation but good enough for overview plots.
    """
    s = seq.strip().upper()
    if not s:
        return float("nan")

    # counts of ionizable side chains
    nD = s.count("D")
    nE = s.count("E")
    nC = s.count("C")
    nY = s.count("Y")
    nH = s.count("H")
    nK = s.count("K")
    nR = s.count("R")

    # positive groups: N-terminus, K, R, H
    pos = 0.0
    pos += 1.0 / (1.0 + 10.0 ** (ph - PKA["Nterm"]))
    pos += nK * (1.0 / (1.0 + 10.0 ** (ph - PKA["K"])))
    pos += nR * (1.0 / (1.0 + 10.0 ** (ph - PKA["R"])))
    pos += nH * (1.0 / (1.0 + 10.0 ** (ph - PKA["H"])))

    # negative groups: C-terminus, D, E, C, Y
    neg = 0.0
    neg += 1.0 / (1.0 + 10.0 ** (PKA["Cterm"] - ph))
    neg += nD * (1.0 / (1.0 + 10.0 ** (PKA["D"] - ph)))
    neg += nE * (1.0 / (1.0 + 10.0 ** (PKA["E"] - ph)))
    neg += nC * (1.0 / (1.0 + 10.0 ** (PKA["C"] - ph)))
    neg += nY * (1.0 / (1.0 + 10.0 ** (PKA["Y"] - ph)))

    return float(pos - neg)


def estimate_pi(seq: str, ph_low: float = 0.0, ph_high: float = 14.0, steps: int = 60) -> float:
    """
    Find pH where net charge crosses 0 using simple bisection.
    """
    s = seq.strip().upper()
    if not s:
        return float("nan")

    lo, hi = ph_low, ph_high
    clo = net_charge_at_ph(s, lo)
    chi = net_charge_at_ph(s, hi)

    # If no sign change (rare with rough pKa), fallback to scanning minimum abs charge
    if clo * chi > 0:
        grid = np.linspace(lo, hi, steps)
        charges = np.array([net_charge_at_ph(s, p) for p in grid])
        return float(grid[np.argmin(np.abs(charges))])

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        cmid = net_charge_at_ph(s, mid)
        if clo * cmid <= 0:
            hi, chi = mid, cmid
        else:
            lo, clo = mid, cmid
    return float(0.5 * (lo + hi))

@dataclass
class RiskConfig:
    seed: int = 42
    ridge_alpha: float = 1.0

    # overall weighting
    w_agg: float = 0.40
    w_scale: float = 0.35
    w_stab: float = 0.25

    # "major risk" thresholds (for "major risk flag")
    major_risk_threshold: float = 80.0
    # any sub-risk above this triggers a "critical flag"
    critical_subrisk_threshold: float = 92.0

    # Default acceptance thresholds (used by group/export helpers)
    overall_accept_threshold: float = 45.0     # below => accept_low_risk (unless critical)
    overall_review_threshold: float = 70.0     # 45-70 => accept_high_risk (review) (unless critical)
    effort_accept_threshold: float = 100.0     # below => acceptable wetlab effort
    effort_review_threshold: float = 250.0     # above => tends to reject unless user chooses otherwise


# -----------------------------
# Validation helpers
# -----------------------------
def _is_seq(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    s = seq.strip().upper()
    if len(s) < 60:
        return False
    return sum(ch in AA for ch in s) / len(s) > 0.95


def _count_motifs(s: str, motifs: Tuple[str, ...]) -> int:
    c = 0
    for m in motifs:
        start = 0
        while True:
            idx = s.find(m, start)
            if idx == -1:
                break
            c += 1
            start = idx + 1
    return c


def estimate_mw_kda(seq: str) -> float:
    """
    Approximate molecular weight in kDa from sequence alone.
    - Uses average residue masses (no PTMs)
    - Applies a simple water correction for peptide bonds
    This is intended for demo-level size distribution and rough batching, not analytical QC.
    """
    s = seq.strip().upper()
    masses = [AA_MASS_DA.get(a, 0.0) for a in s if a in AA_MASS_DA]
    if not masses:
        return float("nan")
    total = float(np.sum(masses))
    # peptide bond formation loses water: (n-1)*H2O
    total -= WATER_DA * max(len(masses) - 1, 0)
    return round(total / 1000.0, 3)


# -----------------------------
# Feature extraction (sequence-only proxies)
# -----------------------------
def sequence_features(seq: str) -> Dict[str, float]:
    """
    Low-cost, interpretable sequence proxies.
    These are not claims of experimental truth; they are aligned early risk signals.
    """
    s = seq.strip().upper()
    L = len(s)

    hydro_frac = sum(a in AA_HYDRO for a in s) / L
    arom_frac = sum(a in AA_AROM for a in s) / L
    pos_frac = sum(a in AA_POS for a in s) / L
    neg_frac = sum(a in AA_NEG for a in s) / L
    net_charge = (sum(a in AA_POS for a in s) - sum(a in AA_NEG for a in s)) / max(L, 1)

    # Hydrophobic peak: max windowed hydrophobic density
    w = 9
    if L >= w:
        peak = max(sum(a in AA_HYDRO for a in s[i:i+w]) / w for i in range(L - w + 1))
    else:
        peak = hydro_frac

    # Glycosylation motif proxy: N{P}[ST] (counts per length)
    ngly = 0
    for i in range(L - 2):
        if s[i] == "N" and s[i+1] != "P" and s[i+2] in ("S", "T"):
            ngly += 1
    ngly_rate = ngly / max(L, 1)

    # Chemical liability proxies (counts per length)
    deamid_rate = _count_motifs(s, DEAMID_MOTIFS) / max(L, 1)
    isom_rate = _count_motifs(s, ISOM_MOTIFS) / max(L, 1)
    oxid_rate = sum(a in OXID_AA for a in s) / L
    clip_rate = _count_motifs(s, CLIP_MOTIFS) / max(L, 1)

    # Simple low-complexity proxy: runs of same residue >=4
    runs = 0
    cur = 1
    for i in range(1, L):
        if s[i] == s[i-1]:
            cur += 1
        else:
            if cur >= 4:
                runs += 1
            cur = 1
    if cur >= 4:
        runs += 1
    low_complex_rate = runs / max(L, 1)

    mw_kda = estimate_mw_kda(s)

    pi = estimate_pi(s)
    gravy_score = gravy(s)
    charge74 = net_charge_at_ph(s, 7.4)

    return {
        "length": float(L),
        "mw_kda": float(mw_kda),
        "hydro_frac": float(hydro_frac),
        "hydro_peak": float(peak),
        "arom_frac": float(arom_frac),
        "pos_frac": float(pos_frac),
        "neg_frac": float(neg_frac),
        "net_charge": float(net_charge),
        "ngly_rate": float(ngly_rate),
        "deamid_rate": float(deamid_rate),
        "isom_rate": float(isom_rate),
        "oxid_rate": float(oxid_rate),
        "clip_rate": float(clip_rate),
        "low_complex_rate": float(low_complex_rate),
        "pi": float(pi),
        "gravy": float(gravy_score),
        "charge_7p4": float(charge74),        
    }


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Input dataframe must contain columns: id, sequence")

    ok = df["sequence"].astype(str).apply(_is_seq)
    if not ok.all():
        bad = df.loc[~ok, "id"].tolist()[:10]
        raise ValueError(f"Some sequences are invalid (showing up to 10 ids): {bad}")

    feats = [sequence_features(s) for s in df["sequence"].astype(str)]
    X = pd.DataFrame(feats)
    X.insert(0, "id", df["id"].values)
    return X


# -----------------------------
# Pseudo-targets (DEMO ONLY)
# Replace with wet-lab labels later in EXIST phase.
# -----------------------------
def _pseudo_subrisks_raw(X: pd.DataFrame) -> pd.DataFrame:
    """
    9 sub-risk raw scores before normalization.
    Each is a weighted combination of interpretable proxies.
    """
    hydro = X["hydro_frac"]
    peak = X["hydro_peak"]
    arom = X["arom_frac"]
    ngly = X["ngly_rate"]
    Ls = X["length"] / 150.0  # rough scaling
    mw = X["mw_kda"].fillna(X["mw_kda"].median()) / 150.0  # used softly, not dominating

    charge_abs = np.abs(X["net_charge"])
    charge_neutrality = 1.0 - np.tanh(charge_abs * 3.0)  # close to 1 near-neutral

    # Aggregation (sub)
    agg_self_assoc = 1.4*peak + 1.0*hydro + 0.8*charge_neutrality + 0.4*arom
    agg_hydro_patch = 1.8*peak + 0.8*hydro + 0.3*Ls
    agg_polyspec = 0.9*arom + 0.8*hydro + 0.9*charge_neutrality

    # Scale-up sensitivity (sub)
    scale_viscosity = 1.6*charge_neutrality + 1.0*peak + 0.6*hydro + 0.2*mw
    scale_processability = 1.2*peak + 0.6*ngly + 0.4*Ls + 0.3*X["low_complex_rate"] + 0.15*mw
    scale_charge_behavior = 1.3*charge_neutrality + 0.6*(X["pos_frac"] + X["neg_frac"]) + 0.2*Ls

    # Stability (sub)
    stab_conformational = 1.1*charge_neutrality + 0.7*X["low_complex_rate"] + 0.5*peak
    stab_chemical = 1.4*X["deamid_rate"] + 1.0*X["isom_rate"] + 0.8*X["oxid_rate"]
    stab_fragmentation = 1.2*X["clip_rate"] + 0.4*X["isom_rate"] + 0.3*X["deamid_rate"]

    return pd.DataFrame({
        "agg_self_association_raw": agg_self_assoc,
        "agg_hydrophobic_patch_raw": agg_hydro_patch,
        "agg_polyspecificity_raw": agg_polyspec,
        "scale_viscosity_raw": scale_viscosity,
        "scale_processability_raw": scale_processability,
        "scale_charge_behavior_raw": scale_charge_behavior,
        "stab_conformational_raw": stab_conformational,
        "stab_chemical_liabilities_raw": stab_chemical,
        "stab_fragmentation_raw": stab_fragmentation,
    })


# -----------------------------
# AI-weighted mapping layer (demo: trained on pseudo targets)
# -----------------------------
def fit_ai_models(X: pd.DataFrame, cfg: RiskConfig = RiskConfig()) -> Dict[str, Tuple[Ridge, List[str]]]:
    np.random.seed(cfg.seed)

    feats = [
        "length", "mw_kda",
        "hydro_frac", "hydro_peak", "arom_frac",
        "pos_frac", "neg_frac", "net_charge",
        "ngly_rate", "deamid_rate", "isom_rate",
        "oxid_rate", "clip_rate", "low_complex_rate"
    ]
    Xmat = X[feats].values.astype(float)

    Y = _pseudo_subrisks_raw(X)

    models: Dict[str, Tuple[Ridge, List[str]]] = {}
    for target in Y.columns:
        m = Ridge(alpha=cfg.ridge_alpha)
        m.fit(Xmat, Y[target].values.astype(float))
        models[target] = (m, feats)

    return models


def _minmax_to_0_100(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx - mn < 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn) * 100.0


# -----------------------------
# Wet-lab effort/cost index (3rd dimension)
# -----------------------------
def estimate_wetlab_effort_index(
    agg: np.ndarray,
    scale: np.ndarray,
    stab: np.ndarray,
    mw_kda: np.ndarray
) -> np.ndarray:
    """
    A pragmatic 'effort index' (>= ~10 to ~1000+) for demo visualization:
    - higher developability risk => larger expected iteration loops => more time/cost
    - larger molecules / more complex (mw) => slightly higher baseline effort
    This is NOT a financial promise; it is a decision-support proxy for demo.
    """
    # Normalize risks 0-1
    r = (0.45*agg + 0.35*scale + 0.20*stab) / 100.0
    r = np.clip(r, 0.0, 1.0)

    # Baseline effort grows gently with size (kDa)
    size = np.clip(mw_kda / 150.0, 0.3, 2.0)  # typical IgG ~150kDa => ~1.0

    # Exponential-like escalation: small risk => ~10-40; high risk => hundreds+
    effort = 10.0 * (10 ** (2.2 * r)) * size

    # cap for display stability
    effort = np.clip(effort, 8.0, 1500.0)
    return effort


# -----------------------------
# Scoring
# -----------------------------
SUBRISK_KEYS = [
    "agg_self_association",
    "agg_hydrophobic_patch",
    "agg_polyspecificity",
    "scale_viscosity",
    "scale_processability",
    "scale_charge_behavior",
    "stab_conformational",
    "stab_chemical_liabilities",
    "stab_fragmentation",
]


def score_candidates(
    X: pd.DataFrame,
    models: Dict[str, Tuple[Ridge, List[str]]],
    cfg: RiskConfig = RiskConfig()
) -> pd.DataFrame:
    """
    Output includes:
      - basic stats: length, mw_kda
      - 9 sub risks (0-100)
      - 3 major risks (0-100)
      - overall score + risk_class
      - critical flags: critical_subrisk, major_risk_flag
      - wetlab_effort_index (3rd dimension) + effort_class
    """
    out = pd.DataFrame({
        "id": X["id"].values,
        "length": X["length"].values,
        "mw_kda": X["mw_kda"].values,
    })

    # Predict raw sub-scores via AI models
    for target, (m, feats) in models.items():
        y_raw = m.predict(X[feats].values.astype(float))
        out[target.replace("_raw", "")] = y_raw

    # Normalize each sub-risk to 0-100
    for c in SUBRISK_KEYS:
        out[c] = _minmax_to_0_100(out[c].values)
        out[c] = np.round(out[c], 1)

    # Aggregate to 3 major axes
    out["Aggregation"] = np.round(
        (out["agg_self_association"] + out["agg_hydrophobic_patch"] + out["agg_polyspecificity"]) / 3.0, 1
    )
    out["ScaleUpSensitivity"] = np.round(
        (out["scale_viscosity"] + out["scale_processability"] + out["scale_charge_behavior"]) / 3.0, 1
    )
    out["Stability"] = np.round(
        (out["stab_conformational"] + out["stab_chemical_liabilities"] + out["stab_fragmentation"]) / 3.0, 1
    )

    # Overall (weighted)
    s = cfg.w_agg + cfg.w_scale + cfg.w_stab
    if s <= 1e-12:
        raise ValueError("Sum of weights is zero; set non-zero weights.")
    w_agg, w_scale, w_stab = cfg.w_agg / s, cfg.w_scale / s, cfg.w_stab / s

    out["overall"] = np.round(
        w_agg*out["Aggregation"] + w_scale*out["ScaleUpSensitivity"] + w_stab*out["Stability"], 1
    )

    # Risk class by overall (can be overridden by critical flags)
    def risk_class(x: float) -> str:
        if x >= cfg.overall_review_threshold:
            return "High-risk"
        if x >= cfg.overall_accept_threshold:
            return "Uncertain"
        return "Robust"

    out["risk_class"] = out["overall"].apply(risk_class)

    # Critical flags
    out["critical_subrisk"] = (out[SUBRISK_KEYS].max(axis=1) >= cfg.critical_subrisk_threshold)
    out["major_risk_flag"] = (
        (out["Aggregation"] >= cfg.major_risk_threshold) |
        (out["ScaleUpSensitivity"] >= cfg.major_risk_threshold) |
        (out["Stability"] >= cfg.major_risk_threshold)
    )

    # Wet-lab effort index (3rd dimension)
    effort = estimate_wetlab_effort_index(
        out["Aggregation"].values,
        out["ScaleUpSensitivity"].values,
        out["Stability"].values,
        out["mw_kda"].fillna(np.nanmedian(out["mw_kda"].values)).values
    )
    out["wetlab_effort_index"] = np.round(effort, 1)

    def effort_class(v: float) -> str:
        if v <= cfg.effort_accept_threshold:
            return "Low"
        if v <= cfg.effort_review_threshold:
            return "Medium"
        return "High"

    out["effort_class"] = out["wetlab_effort_index"].apply(effort_class)

    return out


# -----------------------------
# Explainability: coefficients
# -----------------------------
def model_coefficients(models: Dict[str, Tuple[Ridge, List[str]]]) -> pd.DataFrame:
    rows = []
    for target, (m, feats) in models.items():
        for f, w in zip(feats, m.coef_):
            rows.append({"target": target.replace("_raw", ""), "feature": f, "weight": float(w)})
    df = pd.DataFrame(rows)
    df["abs_weight"] = df["weight"].abs()
    return df.sort_values(["target", "abs_weight"], ascending=[True, False]).drop(columns=["abs_weight"])


# -----------------------------
# Batch statistics for UI (mean MW, distributions, etc.)
# -----------------------------
def batch_statistics(scored: pd.DataFrame) -> Dict[str, object]:
    """
    Returns a dict for UI:
      - counts by risk_class / effort_class
      - mean/median length & mw_kda
      - arrays ready for plotting distributions (overall, mw_kda, effort)
    """
    stats = {}

    stats["n"] = int(len(scored))
    stats["risk_counts"] = scored["risk_class"].value_counts().to_dict()
    stats["effort_counts"] = scored["effort_class"].value_counts().to_dict()

    stats["length_mean"] = float(np.nanmean(scored["length"].values))
    stats["length_median"] = float(np.nanmedian(scored["length"].values))

    stats["mw_kda_mean"] = float(np.nanmean(scored["mw_kda"].values))
    stats["mw_kda_median"] = float(np.nanmedian(scored["mw_kda"].values))

    stats["overall_mean"] = float(np.nanmean(scored["overall"].values))
    stats["overall_median"] = float(np.nanmedian(scored["overall"].values))

    stats["effort_mean"] = float(np.nanmean(scored["wetlab_effort_index"].values))
    stats["effort_median"] = float(np.nanmedian(scored["wetlab_effort_index"].values))

    # Distributions (for plotting)
    stats["dist_overall"] = scored["overall"].values.astype(float)
    stats["dist_mw_kda"] = scored["mw_kda"].values.astype(float)
    stats["dist_effort"] = scored["wetlab_effort_index"].values.astype(float)

    return stats


# -----------------------------
# Grouping / Export helper
# -----------------------------
def split_candidates_by_acceptance(
    df_input: pd.DataFrame,
    scored: pd.DataFrame,
    cfg: RiskConfig = RiskConfig(),
    overall_threshold: Optional[float] = None,
    effort_threshold: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Produces three groups and returns sequences for export:
      - accept_low_risk
      - accept_high_risk (review)
      - reject

    Rules (demo-pragmatic):
      - reject if critical_subrisk OR effort too high OR overall too high
      - accept_low_risk: overall < overall_accept_threshold AND effort acceptable AND not critical
      - accept_high_risk: between accept_threshold and review_threshold (or user-provided threshold), still within effort threshold, not critical
    """
    merged = scored.merge(df_input[["id", "sequence"]], on="id", how="left")

    ot = float(overall_threshold) if overall_threshold is not None else cfg.overall_review_threshold
    et = float(effort_threshold) if effort_threshold is not None else cfg.effort_accept_threshold

    # Reject conditions
    reject_mask = (
        (merged["critical_subrisk"]) |
        (merged["wetlab_effort_index"] > et) |
        (merged["overall"] >= ot)
    )

    # Low-risk accept
    low_mask = (
        (~reject_mask) &
        (merged["overall"] < cfg.overall_accept_threshold) &
        (merged["wetlab_effort_index"] <= et) &
        (~merged["critical_subrisk"])
    )

    # High-risk accept (review): not rejected, but above low threshold
    high_mask = (
        (~reject_mask) &
        (~low_mask)
    )

    accept_low = merged[low_mask].copy().sort_values("overall", ascending=True)
    accept_high = merged[high_mask].copy().sort_values("overall", ascending=True)
    reject = merged[reject_mask].copy().sort_values("overall", ascending=False)

    # Keep export-friendly columns
    cols = [
        "id", "sequence",
        "overall", "risk_class",
        "wetlab_effort_index", "effort_class",
        "Aggregation", "ScaleUpSensitivity", "Stability",
        "critical_subrisk", "major_risk_flag",
    ] + SUBRISK_KEYS

    return {
        "accept_low_risk": accept_low[cols],
        "accept_high_risk": accept_high[cols],
        "reject": reject[cols],
    }


# -----------------------------
# Quick test helper
# -----------------------------
def run_quick_test(csv_path: str = "data/demo_100.csv", cfg: RiskConfig = RiskConfig()) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    X = build_feature_table(df)
    models = fit_ai_models(X, cfg=cfg)
    scored = score_candidates(X, models, cfg=cfg)
    return scored.sort_values("overall", ascending=False).reset_index(drop=True)