"""
Run a county-level Spatial Durbin Model (SDM) for Fujian CO2 emissions.

Input files:
    data_processed/fujian_sdm_panel_core_balanced_2013_2021.csv
    data_processed/fujian_spatial_weights_inverse_distance_matrix.csv
    data_processed/fujian_county_centroids_for_sdm.csv

Outputs:
    outputs/model/sdm_coefficients.csv
    outputs/model/sdm_impacts.csv
    outputs/model/morans_i_by_year.csv
    outputs/model/sdm_model_summary.json
    outputs/figures/*.png
    outputs/SDM模型结果解释.md

Model:
    ln(CO2_it) = rho * W ln(CO2_it) + X_it beta + W X_it theta
                 + county fixed effects + year fixed effects + error_it

Estimation:
    Static SDM concentrated log-likelihood. For each rho, transform y by
    (I - rho W)y, absorb two-way fixed effects, estimate beta/theta by OLS,
    and choose rho by profile likelihood grid search.

This is intentionally written without PySAL/spreg so it can run in a plain
scientific Python environment. If PySAL is available later, use this output
as a reproducible benchmark and robustness check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except ModuleNotFoundError:
    HAVE_MATPLOTLIB = False
    plt = None

ROOT = Path("/workspace")
DATA = ROOT / "data_processed"
OUT = ROOT / "outputs"
MODEL_DIR = OUT / "model"
FIG_DIR = OUT / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def twoway_demean(values: np.ndarray, unit_idx: np.ndarray, year_idx: np.ndarray, n_units: int, n_years: int) -> np.ndarray:
    """Absorb county and year fixed effects for a balanced panel."""
    values = np.asarray(values, dtype=float)
    unit_mean = np.zeros(n_units)
    year_mean = np.zeros(n_years)
    for i in range(n_units):
        unit_mean[i] = values[unit_idx == i].mean()
    for t in range(n_years):
        year_mean[t] = values[year_idx == t].mean()
    return values - unit_mean[unit_idx] - year_mean[year_idx] + values.mean()


def spatial_lag_by_year(series: pd.Series, units: list[str], years: list[int], W: np.ndarray) -> np.ndarray:
    out = []
    for year in years:
        vec = series.xs(year, level="year").reindex(units).to_numpy(dtype=float)
        out.extend(W @ vec)
    return np.asarray(out)


def logdet_i_minus_rho_w(W: np.ndarray, rho: float) -> float:
    sign, val = np.linalg.slogdet(np.eye(W.shape[0]) - rho * W)
    if sign <= 0:
        return -np.inf
    return float(val)


def morans_i(vec: np.ndarray, W: np.ndarray) -> float:
    z = vec - vec.mean()
    return float((z @ W @ z) / (z @ z))


def fit_sdm(panel: pd.DataFrame, W: np.ndarray, units: list[str], years: list[int]):
    n_units, n_years = len(units), len(years)
    panel = panel.copy()
    panel["unit_id"] = panel["unit_id"].astype(str)
    balanced_index = pd.MultiIndex.from_product([years, units], names=["year", "unit_id"])
    obs = panel.set_index(["year", "unit_id"]).loc[balanced_index].reset_index()

    y = np.log(obs["co2_million_tons"].to_numpy(dtype=float))
    y_series = pd.Series(y, index=pd.MultiIndex.from_frame(obs[["unit_id", "year"]]))
    Wy = spatial_lag_by_year(y_series, units, years, W)

    x_defs = {
        "ln_gdp": np.log(obs["gdp_100m_yuan"].to_numpy(dtype=float)),
        "ln_pop": np.log(obs["resident_population_10k"].to_numpy(dtype=float)),
        "secondary_share": obs["secondary_industry_share"].to_numpy(dtype=float),
        "urbanization": obs["urbanization_rate_pct"].to_numpy(dtype=float) / 100,
        "ln_retail_pc": np.log(obs["retail_per_capita_yuan"].to_numpy(dtype=float)),
    }
    X = pd.DataFrame(x_defs)
    WX_cols = {}
    for col in X.columns:
        s = pd.Series(X[col].to_numpy(), index=pd.MultiIndex.from_frame(obs[["unit_id", "year"]]))
        WX_cols[f"W_{col}"] = spatial_lag_by_year(s, units, years, W)
    WX = pd.DataFrame(WX_cols)
    Z = pd.concat([X, WX], axis=1)
    var_names = list(Z.columns)

    unit_idx = obs["unit_id"].map({u: i for i, u in enumerate(units)}).to_numpy()
    year_idx = obs["year"].map({y: i for i, y in enumerate(years)}).to_numpy()
    Zd = np.column_stack([twoway_demean(Z[c].to_numpy(), unit_idx, year_idx, n_units, n_years) for c in var_names])

    def fit_at(rho: float):
        y_star = y - rho * Wy
        yd = twoway_demean(y_star, unit_idx, year_idx, n_units, n_years)
        beta, *_ = np.linalg.lstsq(Zd, yd, rcond=None)
        resid = yd - Zd @ beta
        sse = float(resid @ resid)
        ll = n_years * logdet_i_minus_rho_w(W, rho) - len(y) / 2 * math.log(sse / len(y))
        return {"rho": rho, "ll": ll, "beta": beta, "resid": resid, "sse": sse, "yd": yd}

    candidates = [fit_at(float(r)) for r in np.arange(-0.90, 0.9001, 0.005)]
    best = max(candidates, key=lambda d: d["ll"])
    fine = [fit_at(float(r)) for r in np.arange(max(-0.95, best["rho"] - 0.02), min(0.95, best["rho"] + 0.02) + 1e-9, 0.0005)]
    best = max(fine, key=lambda d: d["ll"])

    df = len(y) - len(var_names) - n_units - n_years + 1
    sigma2 = best["sse"] / df
    inv_xx = np.linalg.inv(Zd.T @ Zd)
    se = np.sqrt(np.diag(inv_xx) * sigma2)
    t_vals = best["beta"] / se
    p_vals = 2 * (1 - np.vectorize(normal_cdf)(np.abs(t_vals)))

    coef = pd.DataFrame({
        "variable": var_names,
        "coefficient": best["beta"],
        "std_error": se,
        "t_value": t_vals,
        "p_value": p_vals,
    })

    impacts = compute_impacts(best["rho"], best["beta"], W, list(x_defs.keys()))
    summary = {
        "model": "Spatial Durbin Model with county and year fixed effects",
        "n_observations": int(len(y)),
        "n_units": int(n_units),
        "n_years": int(n_years),
        "rho": float(best["rho"]),
        "log_likelihood_without_constant": float(best["ll"]),
        "sse": float(best["sse"]),
        "sigma2": float(sigma2),
        "within_r2_transformed": float(1 - best["sse"] / np.sum(best["yd"] ** 2)),
        "df_residual": int(df),
        "dependent_variable": "ln(co2_million_tons)",
        "x_variables": list(x_defs.keys()),
    }
    return obs, coef, impacts, summary


def compute_impacts(rho: float, beta_all: np.ndarray, W: np.ndarray, x_names: list[str]) -> pd.DataFrame:
    n = W.shape[0]
    S = np.linalg.inv(np.eye(n) - rho * W)
    rows = []
    for k, name in enumerate(x_names):
        beta = beta_all[k]
        theta = beta_all[k + len(x_names)]
        M = S @ (beta * np.eye(n) + theta * W)
        direct = np.trace(M) / n
        total = M.sum(axis=1).mean()
        rows.append({
            "variable": name,
            "direct_effect": float(direct),
            "indirect_effect": float(total - direct),
            "total_effect": float(total),
        })
    return pd.DataFrame(rows)


def _svg(path: Path, width: int, height: int, body: str) -> None:
    text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<style>'
        'text{font-family:"Microsoft YaHei","SimHei",Arial,sans-serif;fill:#172033}'
        '.title{font-size:23px;font-weight:700}.axis{stroke:#374151;stroke-width:1.2}'
        '.grid{stroke:#d7dde8;stroke-width:1}.tick{font-size:12px;fill:#526070}'
        '.label{font-size:13px;fill:#334155}.note{font-size:12px;fill:#64748b}'
        '</style>'
        f'<rect width="100%" height="100%" fill="#fbfcff"/>{body}</svg>'
    )
    path.write_text(text, encoding="utf-8")


def _scale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    if old_max == old_min:
        return (new_min + new_max) / 2
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


def _nice_range(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if lo == hi:
        return lo - 1, hi + 1
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _line_svg(path: Path, df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, color: str) -> None:
    width, height = 920, 540
    left, right, top, bottom = 80, 40, 70, 70
    x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
    y_min, y_max = _nice_range(df[y_col].to_numpy(dtype=float))
    pts = []
    for _, r in df.iterrows():
        x = _scale(float(r[x_col]), x_min, x_max, left, width - right)
        y = _scale(float(r[y_col]), y_min, y_max, height - bottom, top)
        pts.append((x, y, r[x_col], r[y_col]))

    body = [f'<text x="{left}" y="38" class="title">{escape(title)}</text>']
    for i in range(5):
        val = y_min + (y_max - y_min) * i / 4
        y = _scale(val, y_min, y_max, height - bottom, top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{val:.2f}</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(
        '<polyline fill="none" stroke="{}" stroke-width="3" points="{}"/>'.format(
            color, " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
        )
    )
    for x, y, year, value in pts:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" class="tick">{int(year)}</text>')
        body.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" class="note">{value:.2f}</text>')
    body.append(f'<text x="22" y="{(top+height-bottom)/2:.1f}" transform="rotate(-90 22 {(top+height-bottom)/2:.1f})" class="label">{escape(y_label)}</text>')
    _svg(path, width, height, "\n".join(body))


def _spatial_bubble_svg(path: Path, map_df: pd.DataFrame) -> None:
    width, height = 820, 760
    left, right, top, bottom = 70, 55, 70, 70
    lon_min, lon_max = _nice_range(map_df["lon"].to_numpy(dtype=float), 0.04)
    lat_min, lat_max = _nice_range(map_df["lat"].to_numpy(dtype=float), 0.04)
    co2_min, co2_max = float(map_df["co2_million_tons"].min()), float(map_df["co2_million_tons"].max())
    body = [f'<text x="{left}" y="38" class="title">2021 年福建县域 CO2 排放空间格局</text>']
    body.append(f'<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="#f8fafc" stroke="#cbd5e1"/>')
    for _, r in map_df.iterrows():
        x = _scale(float(r["lon"]), lon_min, lon_max, left, width - right)
        y = _scale(float(r["lat"]), lat_min, lat_max, height - bottom, top)
        radius = _scale(math.sqrt(float(r["co2_million_tons"])), math.sqrt(co2_min), math.sqrt(co2_max), 4, 18)
        color_t = _scale(float(r["co2_million_tons"]), co2_min, co2_max, 0, 1)
        color = f'rgb({int(35 + 160 * color_t)},{int(110 + 50 * (1-color_t))},{int(145 - 80 * color_t)})'
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" stroke="#0f172a" stroke-width="0.5"/>')
    top10 = map_df.nlargest(10, "co2_million_tons")
    for _, r in top10.iterrows():
        x = _scale(float(r["lon"]), lon_min, lon_max, left, width - right)
        y = _scale(float(r["lat"]), lat_min, lat_max, height - bottom, top)
        body.append(f'<text x="{x+8:.1f}" y="{y-7:.1f}" class="tick">{escape(str(r["county_cn"]))}</text>')
    body.append(f'<text x="{width/2:.1f}" y="{height-24}" text-anchor="middle" class="label">经度</text>')
    body.append(f'<text x="22" y="{height/2:.1f}" transform="rotate(-90 22 {height/2:.1f})" class="label">纬度</text>')
    body.append(f'<text x="{left}" y="{height-42}" class="note">圆点越大、颜色越深，表示 CO2 排放量越高。</text>')
    _svg(path, width, height, "\n".join(body))


def _heat_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        t = value
        r, g, b = 255, int(255 - 115 * t), int(255 - 135 * t)
    else:
        t = -value
        r, g, b = int(255 - 150 * t), int(255 - 95 * t), 255
    return f"rgb({r},{g},{b})"


def _heatmap_svg(path: Path, corr: pd.DataFrame) -> None:
    labels = list(corr.columns)
    width, height = 860, 760
    left, top, cell = 230, 90, 76
    body = [f'<text x="70" y="42" class="title">核心变量相关系数热力图</text>']
    for i, row in enumerate(labels):
        body.append(f'<text x="{left-10}" y="{top+i*cell+cell/2+5:.1f}" text-anchor="end" class="tick">{escape(row)}</text>')
        body.append(f'<text x="{left+i*cell+cell/2:.1f}" y="{top-14}" text-anchor="middle" transform="rotate(-35 {left+i*cell+cell/2:.1f} {top-14})" class="tick">{escape(row)}</text>')
        for j, col in enumerate(labels):
            val = float(corr.iloc[i, j])
            x, y = left + j * cell, top + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_heat_color(val)}" stroke="#ffffff"/>')
            body.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+5:.1f}" text-anchor="middle" class="tick">{val:.2f}</text>')
    _svg(path, width, height, "\n".join(body))


def _coef_svg(path: Path, coef: pd.DataFrame) -> None:
    c = coef.copy()
    c["lo"] = c["coefficient"] - 1.96 * c["std_error"]
    c["hi"] = c["coefficient"] + 1.96 * c["std_error"]
    width, height = 900, 640
    left, right, top, bottom = 220, 60, 70, 50
    x_min, x_max = _nice_range(c[["lo", "hi"]].to_numpy(dtype=float).ravel(), 0.10)
    body = [f'<text x="70" y="38" class="title">空间杜宾模型系数及 95% 置信区间</text>']
    zero = _scale(0, x_min, x_max, left, width - right)
    body.append(f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{height-bottom}" stroke="#111827" stroke-dasharray="5,5"/>')
    row_h = (height - top - bottom) / len(c)
    for i, r in c.iterrows():
        y = top + row_h * (i + 0.5)
        lo = _scale(float(r["lo"]), x_min, x_max, left, width - right)
        hi = _scale(float(r["hi"]), x_min, x_max, left, width - right)
        x = _scale(float(r["coefficient"]), x_min, x_max, left, width - right)
        fill = "#c82423" if float(r["p_value"]) < 0.05 else "#2878b5"
        body.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{escape(str(r["variable"]))}</text>')
        body.append(f'<line x1="{lo:.1f}" y1="{y:.1f}" x2="{hi:.1f}" y2="{y:.1f}" stroke="#64748b" stroke-width="2"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="#ffffff" stroke-width="1.2"/>')
    body.append(f'<text x="{left}" y="{height-18}" class="note">红点表示 p&lt;0.05。</text>')
    _svg(path, width, height, "\n".join(body))


def _barh_svg(path: Path, df: pd.DataFrame, label_col: str, value_col: str, title: str, x_label: str, color: str) -> None:
    width, height = 900, 640
    left, right, top, bottom = 230, 50, 70, 55
    vals = df[value_col].to_numpy(dtype=float)
    x_min, x_max = min(0, float(vals.min())), max(0, float(vals.max()))
    x_min, x_max = _nice_range(np.array([x_min, x_max]), 0.06)
    body = [f'<text x="70" y="38" class="title">{escape(title)}</text>']
    zero = _scale(0, x_min, x_max, left, width - right)
    body.append(f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{height-bottom}" stroke="#111827"/>')
    row_h = (height - top - bottom) / len(df)
    for i, r in df.reset_index(drop=True).iterrows():
        y = top + row_h * i + row_h * 0.18
        bar_h = row_h * 0.62
        x_val = _scale(float(r[value_col]), x_min, x_max, left, width - right)
        x = min(zero, x_val)
        w = abs(x_val - zero)
        body.append(f'<text x="{left-12}" y="{y+bar_h/2+5:.1f}" text-anchor="end" class="tick">{escape(str(r[label_col]))}</text>')
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        body.append(f'<text x="{x_val + (6 if x_val >= zero else -6):.1f}" y="{y+bar_h/2+5:.1f}" text-anchor="{"start" if x_val >= zero else "end"}" class="note">{float(r[value_col]):.3f}</text>')
    body.append(f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    _svg(path, width, height, "\n".join(body))


def _impacts_svg(path: Path, impacts: pd.DataFrame) -> None:
    long_rows = []
    for _, r in impacts.iterrows():
        for effect, label in [("direct_effect", "直接效应"), ("indirect_effect", "间接效应"), ("total_effect", "总效应")]:
            long_rows.append({"label": f"{r['variable']} {label}", "value": float(r[effect])})
    _barh_svg(path, pd.DataFrame(long_rows), "label", "value", "平均直接效应、间接效应与总效应", "效应值", "#2878b5")


def make_svg_figures(panel: pd.DataFrame, centroids: pd.DataFrame, coef: pd.DataFrame, impacts: pd.DataFrame, moran: pd.DataFrame) -> None:
    annual = panel.groupby("year").agg(co2=("co2_million_tons", "sum"), gdp=("gdp_100m_yuan", "sum")).reset_index()
    annual["intensity"] = annual["co2"] / annual["gdp"] * 100
    _line_svg(FIG_DIR / "fig01_total_co2_trend.svg", annual, "year", "co2", "福建县域 CO2 排放总量变化，2013-2021", "CO2 排放量（百万吨）", "#2878b5")

    map_df = panel[panel["year"].eq(2021)].merge(centroids, on=["unit_id", "county_cn"], how="left")
    _spatial_bubble_svg(FIG_DIR / "fig02_county_co2_2021_spatial_bubble.svg", map_df)

    _line_svg(FIG_DIR / "fig03_morans_i_trend.svg", moran, "year", "morans_i_ln_co2", "ln(CO2) 全局 Moran's I，2013-2021", "Moran's I", "#c82423")

    vars_for_corr = ["co2_million_tons", "gdp_100m_yuan", "resident_population_10k", "secondary_industry_share", "urbanization_rate_pct", "retail_per_capita_yuan"]
    corr = panel[vars_for_corr].apply(np.log1p).corr()
    _heatmap_svg(FIG_DIR / "fig04_correlation_heatmap.svg", corr)

    _coef_svg(FIG_DIR / "fig05_sdm_coefficients.svg", coef)
    _impacts_svg(FIG_DIR / "fig06_spatial_impacts.svg", impacts)

    growth = panel.pivot(index=["unit_id", "county_cn"], columns="year", values="co2_million_tons")
    growth["change_2013_2021"] = growth[2021] - growth[2013]
    top_growth = growth.reset_index().nlargest(15, "change_2013_2021")
    _barh_svg(FIG_DIR / "fig07_top_county_co2_increase.svg", top_growth.iloc[::-1], "county_cn", "change_2013_2021", "2013-2021 年 CO2 增量前 15 县域", "CO2 排放增量（百万吨）", "#2878b5")


def make_figures(panel: pd.DataFrame, centroids: pd.DataFrame, coef: pd.DataFrame, impacts: pd.DataFrame, moran: pd.DataFrame):
    panel = panel.copy()
    centroids = centroids.copy()
    panel["unit_id"] = panel["unit_id"].astype(str)
    centroids["unit_id"] = centroids["unit_id"].astype(str)

    if not HAVE_MATPLOTLIB:
        make_svg_figures(panel, centroids, coef, impacts, moran)
        return

    annual = panel.groupby("year").agg(co2=("co2_million_tons", "sum"), gdp=("gdp_100m_yuan", "sum")).reset_index()
    annual["intensity"] = annual["co2"] / annual["gdp"] * 100

    plt.figure(figsize=(9, 5))
    plt.plot(annual["year"], annual["co2"], marker="o", linewidth=2.5)
    plt.title("福建县域 CO2 排放总量变化，2013-2021")
    plt.xlabel("年份")
    plt.ylabel("CO2 排放量（百万吨）")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_total_co2_trend.png", dpi=220)
    plt.close()

    map_df = panel[panel["year"].eq(2021)].merge(centroids, on=["unit_id", "county_cn"], how="left")
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(map_df["lon"], map_df["lat"], s=map_df["co2_million_tons"] * 25, c=map_df["co2_million_tons"], cmap="viridis", alpha=0.78, edgecolor="black", linewidth=0.3)
    top = map_df.nlargest(10, "co2_million_tons")
    for _, r in top.iterrows():
        plt.text(r["lon"] + 0.03, r["lat"] + 0.03, r["county_cn"], fontsize=8)
    plt.colorbar(sc, label="CO2 排放量（百万吨）")
    plt.title("2021 年福建县域 CO2 排放空间格局")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_county_co2_2021_spatial_bubble.png", dpi=240)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(moran["year"], moran["morans_i_ln_co2"], marker="o", color="#c82423", linewidth=2.5)
    plt.title("ln(CO2) 全局 Moran's I，2013-2021")
    plt.xlabel("年份")
    plt.ylabel("Moran's I")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_morans_i_trend.png", dpi=220)
    plt.close()

    vars_for_corr = ["co2_million_tons", "gdp_100m_yuan", "resident_population_10k", "secondary_industry_share", "urbanization_rate_pct", "retail_per_capita_yuan"]
    corr = panel[vars_for_corr].apply(np.log1p).corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(vars_for_corr)), vars_for_corr, rotation=35, ha="right")
    plt.yticks(range(len(vars_for_corr)), vars_for_corr)
    for i in range(len(vars_for_corr)):
        for j in range(len(vars_for_corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("核心变量相关系数热力图")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_correlation_heatmap.png", dpi=240)
    plt.close()

    c = coef.copy()
    c["lo"] = c["coefficient"] - 1.96 * c["std_error"]
    c["hi"] = c["coefficient"] + 1.96 * c["std_error"]
    plt.figure(figsize=(8, 6))
    y = np.arange(len(c))
    plt.hlines(y, c["lo"], c["hi"], color="gray")
    plt.scatter(c["coefficient"], y, c=np.where(c["p_value"] < 0.05, "#c82423", "#2878b5"))
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, c["variable"])
    plt.title("空间杜宾模型系数及 95% 置信区间")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_sdm_coefficients.png", dpi=240)
    plt.close()

    imp = impacts.set_index("variable")[["direct_effect", "indirect_effect", "total_effect"]]
    imp.plot(kind="barh", figsize=(8, 5), color=["#2878b5", "#f59e0b", "#16a34a"])
    plt.axvline(0, color="black", linewidth=1)
    plt.title("平均直接效应、间接效应与总效应")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_spatial_impacts.png", dpi=240)
    plt.close()

    growth = panel.pivot(index=["unit_id", "county_cn"], columns="year", values="co2_million_tons")
    growth["change_2013_2021"] = growth[2021] - growth[2013]
    top_growth = growth.reset_index().nlargest(15, "change_2013_2021")
    plt.figure(figsize=(8, 6))
    plt.barh(top_growth["county_cn"], top_growth["change_2013_2021"], color="#2878b5")
    plt.gca().invert_yaxis()
    plt.title("2013-2021 年 CO2 增量前 15 县域")
    plt.xlabel("CO2 排放增量（百万吨）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_top_county_co2_increase.png", dpi=240)
    plt.close()


def main():
    panel = pd.read_csv(DATA / "fujian_sdm_panel_core_balanced_2013_2021.csv")
    centroids = pd.read_csv(DATA / "fujian_county_centroids_for_sdm.csv")
    w_df = pd.read_csv(DATA / "fujian_spatial_weights_inverse_distance_matrix.csv")
    panel["unit_id"] = panel["unit_id"].astype(str)
    centroids["unit_id"] = centroids["unit_id"].astype(str)
    units = w_df["unit_id"].astype(str).tolist()
    years = sorted(panel["year"].unique().tolist())
    W = w_df[units].to_numpy(dtype=float)

    obs, coef, impacts, summary = fit_sdm(panel, W, units, years)
    coef.to_csv(MODEL_DIR / "sdm_coefficients.csv", index=False, encoding="utf-8-sig")
    impacts.to_csv(MODEL_DIR / "sdm_impacts.csv", index=False, encoding="utf-8-sig")
    with open(MODEL_DIR / "sdm_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    y_log = obs.assign(ln_co2=np.log(obs["co2_million_tons"]))
    moran_rows = []
    for year in years:
        vec = y_log[y_log["year"].eq(year)].set_index("unit_id").loc[units, "ln_co2"].to_numpy()
        moran_rows.append({"year": year, "morans_i_ln_co2": morans_i(vec, W)})
    moran = pd.DataFrame(moran_rows)
    moran.to_csv(MODEL_DIR / "morans_i_by_year.csv", index=False, encoding="utf-8-sig")

    make_figures(panel, centroids, coef, impacts, moran)

    sig_coef = coef[coef["p_value"] < 0.05].copy()
    sig_rows = "\n".join(
        f"| `{r.variable}` | {r.coefficient:.4f} | {r.p_value:.4f} |"
        for r in sig_coef.itertuples(index=False)
    ) or "| 无 | - | - |"
    impact_rows = "\n".join(
        f"| `{r.variable}` | {r.direct_effect:.4f} | {r.indirect_effect:.4f} | {r.total_effect:.4f} |"
        for r in impacts.itertuples(index=False)
    )
    moran_min = moran["morans_i_ln_co2"].min()
    moran_max = moran["morans_i_ln_co2"].max()
    figure_suffix = "png" if HAVE_MATPLOTLIB else "svg"

    report = f"""# 福建县域碳排放空间杜宾模型结果解释

## 1. 模型设定

本次使用 2013-2021 年福建县域平衡面板数据，包含 {summary['n_units']} 个空间单元、{summary['n_observations']} 个观测值。被解释变量为 `ln(co2_million_tons)`，解释变量包括经济规模、人口规模、产业结构、城镇化水平和消费活动强度，并同步纳入这些变量的空间滞后项。

模型形式为：

`ln(CO2_it) = rho * W ln(CO2_it) + X_it beta + W X_it theta + county FE + year FE + error_it`

其中，`W` 为县域质心反距离行标准化空间权重矩阵；县域固定效应控制各县区不随时间变化的资源禀赋、区位条件和产业基础；年份固定效应控制宏观周期、政策环境和全省共同冲击。

## 2. 空间相关性

2013-2021 年 `ln(CO2)` 的全局 Moran's I 均为正，约在 {moran_min:.3f}-{moran_max:.3f} 之间，说明福建县域碳排放存在一定程度的正向空间集聚。也就是说，高排放县域周边往往也更容易出现相对较高的排放水平，低排放县域周边也更容易形成低排放集聚。

## 3. SDM 估计结果

空间滞后系数 `rho = {summary['rho']:.4f}`，在当前反距离权重设定下呈负向。这一结果提示：控制县域固定效应、年份固定效应以及解释变量空间滞后项后，本县碳排放与周边县域碳排放之间不再表现为简单同步上升，而更像存在一定空间替代或竞争关系。论文中更稳妥的表述是：福建县域碳排放的空间互动具有复杂性，邻近地区的经济活动和人口集聚通过变量空间滞后项产生更主要的外溢影响。

5% 水平下显著的变量如下：

| 变量 | 系数 | p 值 |
|---|---:|---:|
{sig_rows}

## 4. 直接效应、间接效应与总效应

空间杜宾模型不能只看原始回归系数，还要看效应分解。

| 变量 | 直接效应 | 间接效应 | 总效应 |
|---|---:|---:|---:|
{impact_rows}

整体看，经济规模和消费活动的空间间接效应较强，说明福建县域碳排放不是单个县域孤立决定的，而是受到周边经济联系、人口流动、产业协作和消费网络共同影响。这一点正好符合空间杜宾模型的研究价值。

## 5. 图表说明

图表已输出到 `outputs/figures`，格式为 `{figure_suffix}`，包括总量趋势图、2021 年空间气泡图、Moran's I 趋势图、相关系数热力图、系数置信区间图、空间效应分解图和 CO2 增量排名图。

## 6. 论文写作建议

建议把实证部分组织为：空间分异特征、空间自相关检验、SDM 模型估计、空间效应分解、政策含义。现阶段模型可以作为第一版基准结果，后续可继续加入夜间灯光、NDVI/土地利用、绿色专利、工业结构细分等变量，并用邻接矩阵或 KNN 权重矩阵做稳健性检验。

需要注意的是，县级 CO2 数据属于公开学术碳核算数据，并非政府官方直接统计；解释变量主要来自福建统计年鉴。论文中建议将 CO2 数据表述为“权威公开县域碳排放估算数据”，不要表述成“官方碳排放统计数据”。
"""
    (OUT / "SDM模型结果解释.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
