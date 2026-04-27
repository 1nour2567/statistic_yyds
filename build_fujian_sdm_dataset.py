"""
Build the Fujian county-level SDM dataset used in this project.

The script documents the full reproducible route:
1. Download public CO2, carbon-sink and boundary data from Figshare.
2. Download official county-level tables from Fujian Statistical Yearbooks.
3. Parse and harmonize county names and administrative changes.
4. Aggregate monthly CO2 to annual data.
5. Build a balanced 2013-2021 panel and inverse-distance spatial weights.

Run from the project root:
    python code/build_fujian_sdm_dataset.py

Note: the local Codex runtime did not have Python available, so the delivered
CSV/XLSX files were generated with equivalent PowerShell/Node routines. This
Python file is the clean reproducible version for later reruns.
"""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data_processed"
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


FIGSHARE_FILES = {
    "county_monthly_co2_2013_2021": (
        "https://ndownloader.figshare.com/files/51684707",
        RAW / "figshare_county_monthly_co2_2013_2021_updated_dataset.xlsx",
    ),
    "county_co2_1997_2017": (
        "https://ndownloader.figshare.com/files/25359740",
        RAW / "figshare_county_co2_1997_2017_co2.xlsx",
    ),
    "county_sequestration_2000_2017": (
        "https://ndownloader.figshare.com/files/25359689",
        RAW / "figshare_county_co2_sequestration_2000_2017.xlsx",
    ),
    "county_boundary": (
        "https://ndownloader.figshare.com/files/24848030",
        RAW / "figshare_china_county_boundary_vector_map.zip",
    ),
}

YEARBOOK_BASE = {
    2014: "https://tjj.fujian.gov.cn/tongjinianjian/dz2014/",
    2015: "https://tjj.fujian.gov.cn/tongjinianjian/dz2015/",
    2016: "https://tjj.fujian.gov.cn/tongjinianjian/dz2016/",
    2017: "https://tjj.fujian.gov.cn/tongjinianjian/dz2017/",
    2018: "http://tjj.fujian.gov.cn/tongjinianjian/dz2018/",
    2019: "http://tjj.fujian.gov.cn/tongjinianjian/dz2019/",
    2020: "http://tjj.fujian.gov.cn/tongjinianjian/dz2020/",
    2021: "http://tjj.fujian.gov.cn/tongjinianjian/dz2021/",
    2022: "https://tjj.fujian.gov.cn/tongjinianjian/dz2022/",
}

COUNTY_ALIASES = {
    "建阳区": "建阳市",
    "永定区": "永定县",
    "长乐区": "长乐市",
    "龙海区": "龙海市",
    "长泰区": "长泰县",
    "沙县区": "沙县",
}


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def norm_name(x: str) -> str:
    x = re.sub(r"[\s\u3000]+", "", str(x or ""))
    return COUNTY_ALIASES.get(x, x)


def read_yearbook_html(path: Path) -> str:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
        if re.search(r"地区|全\s*省|鼓楼|福州", text):
            return text
    except UnicodeDecodeError:
        pass
    return data.decode("gb18030", errors="ignore")


def extract_cells(tr) -> list[str]:
    cells = []
    for td in tr.find_all("td"):
        text = td.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
        cells.append(text)
    return cells


def to_num(x):
    if x is None:
        return np.nan
    x = str(x).strip().replace(",", "")
    if x in {"", "—", "--", "…"}:
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def value_offset(cells: list[str]) -> int:
    return 1 if len(cells) > 1 and not math.isnan(to_num(cells[1])) else 2


def parse_official_table(path: Path, title: str, data_year: int, allowed_counties: set[str]) -> pd.DataFrame:
    html = read_yearbook_html(path)
    soup = BeautifulSoup(html, "lxml")
    rows = []
    if "地区生产总值（" in title:
        group = "gdp"
    elif "年末常住人口数（" in title:
        group = "population"
    elif "固定资产投资" in title:
        group = "fixed_asset_investment"
    elif "地方一般公共预算收入" in title:
        group = "budget_revenue"
    elif "一般公共预算支出" in title:
        group = "budget_expenditure"
    elif "规模以上工业增加值" in title:
        group = "industrial_va_growth"
    elif "社会消费品零售总额" in title:
        group = "retail_sales"
    else:
        return pd.DataFrame()

    for tr in soup.find_all("tr"):
        cells = extract_cells(tr)
        if len(cells) < 3:
            continue
        blocks = [cells]
        if group == "retail_sales" and len(cells) >= 9 and math.isnan(to_num(cells[1])):
            blocks = [cells[0:4], cells[5:9]]
        elif group == "retail_sales" and len(cells) >= 7 and not math.isnan(to_num(cells[1])):
            blocks = [cells[0:3], cells[4:7]]
        for block in blocks:
            county = norm_name(block[0])
            if county not in allowed_counties:
                continue
            off = value_offset(block)
            rec = {"year": data_year, "county_cn": county, "variable_group": group}
            if group == "gdp":
                rec.update(
                    gdp_100m_yuan=to_num(block[off]),
                    primary_industry_100m_yuan=to_num(block[off + 1]),
                    secondary_industry_100m_yuan=to_num(block[off + 2]),
                    tertiary_industry_100m_yuan=to_num(block[off + 3]),
                    industry_100m_yuan=to_num(block[off + 4]),
                    construction_100m_yuan=to_num(block[off + 5]),
                    per_capita_gdp_yuan=to_num(block[off + 6]),
                )
            elif group == "population":
                rec.update(
                    resident_population_10k=to_num(block[off]),
                    urban_population_10k=to_num(block[off + 1]),
                    rural_population_10k=to_num(block[off + 2]),
                    urbanization_rate_pct=to_num(block[off + 3]),
                )
            elif group == "retail_sales":
                rec.update(retail_sales_100m_yuan=to_num(block[off]), retail_sales_growth_pct=to_num(block[off + 1]))
            rows.append(rec)
    return pd.DataFrame(rows)


def build_main_co2() -> pd.DataFrame:
    path = FIGSHARE_FILES["county_monthly_co2_2013_2021"][1]
    df = pd.read_excel(path)
    df = df[df["PROVNAME"].eq("福建省")].copy()
    df["year"] = df["YEAR"].astype(int)
    df["month"] = df["YEAR_MON"].astype(str).str[-2:].astype(int)
    df = df.rename(columns={"DISTCODE": "distcode", "County Name": "county_cn", "CO2": "co2_million_tons"})
    df.to_csv(OUT / "fujian_county_monthly_co2_2013_2021_long.csv", index=False, encoding="utf-8-sig")
    annual = (
        df.groupby(["distcode", "county_cn", "year"], as_index=False)
        .agg(month_count=("month", "count"), co2_million_tons=("co2_million_tons", "sum"))
    )
    annual["co2_10000_tons"] = annual["co2_million_tons"] * 100
    annual.to_csv(OUT / "fujian_county_annual_co2_2013_2021_from_monthly.csv", index=False, encoding="utf-8-sig")
    return annual


def unit_id(county: str, code_lookup: dict[str, str]) -> str:
    if county in {"三元区", "梅列区"}:
        return "350402_350403"
    return code_lookup[county]


def unit_name(county: str) -> str:
    return "三元区（含原梅列区）" if county in {"三元区", "梅列区"} else county


def build_spatial_weights(panel: pd.DataFrame) -> None:
    zip_path = FIGSHARE_FILES["county_boundary"][1]
    extract_dir = RAW / "china_county_boundary_vector_map"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    shp = extract_dir / "China county-level vector map" / "China_county_vector_map.shp"
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y
    rows = []
    for unit, county in panel[["unit_id", "county_cn"]].drop_duplicates().itertuples(index=False):
        names = ["三元区", "梅列区"] if county == "三元区（含原梅列区）" else [county]
        m = gdf[gdf["DISTNAME"].isin(names) & gdf["lon"].between(115, 121) & gdf["lat"].between(23, 29)]
        rows.append({"unit_id": unit, "county_cn": county, "lon": m["lon"].mean(), "lat": m["lat"].mean()})
    centroids = pd.DataFrame(rows)
    centroids.to_csv(OUT / "fujian_county_centroids_for_sdm.csv", index=False, encoding="utf-8-sig")

    def haversine(a, b):
        lon1, lat1, lon2, lat2 = map(np.radians, [a.lon, a.lat, b.lon, b.lat])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0088 * 2 * np.arcsin(np.sqrt(h))

    long_rows = []
    for i in centroids.itertuples(index=False):
        tmp = []
        for j in centroids.itertuples(index=False):
            if i.unit_id == j.unit_id:
                continue
            d = haversine(i, j)
            tmp.append([i.unit_id, i.county_cn, j.unit_id, j.county_cn, d, 1 / d])
        s = sum(x[-1] for x in tmp)
        for r in tmp:
            long_rows.append(r[:-1] + [r[-1] / s])
    w_long = pd.DataFrame(
        long_rows,
        columns=["unit_i", "county_i", "unit_j", "county_j", "distance_km", "weight_inverse_distance"],
    )
    w_long.to_csv(OUT / "fujian_spatial_weights_inverse_distance_long.csv", index=False, encoding="utf-8-sig")
    w_mat = w_long.pivot(index="unit_i", columns="unit_j", values="weight_inverse_distance").fillna(0)
    np.fill_diagonal(w_mat.values, 0)
    w_mat.to_csv(OUT / "fujian_spatial_weights_inverse_distance_matrix.csv", encoding="utf-8-sig")


def main() -> None:
    for url, path in FIGSHARE_FILES.values():
        download(url, path)

    annual = build_main_co2()
    code_lookup = annual.drop_duplicates("county_cn").set_index("county_cn")["distcode"].astype(str).to_dict()
    allowed = set(annual["county_cn"])

    table_dir = RAW / "fj_stat_yearbook_tables"
    table_dir.mkdir(exist_ok=True)
    parsed = []
    link_rows = []
    patterns = re.compile(
        r"地区生产总值（\d{4}年）|年末常住人口数（\d{4}年）|固定资产投资|地方一般公共预算收入（\d{4}年）|"
        r"一般公共预算支出（\d{4}年）|规模以上工业增加值|社会消费品零售总额（\d{4}年）"
    )
    for pub_year, base in YEARBOOK_BASE.items():
        contents_url = urljoin(base, "contents-cn.htm")
        contents = requests.get(contents_url, timeout=60).content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(contents, "lxml")
        for a in soup.find_all("a", href=True):
            title = a.get_text(" ", strip=True)
            if not patterns.search(title):
                continue
            year_match = re.search(r"20\d{2}", title)
            if not year_match:
                continue
            data_year = int(year_match.group())
            if not 2013 <= data_year <= 2021:
                continue
            url = urljoin(base, a["href"].replace("./", ""))
            local = table_dir / f"fjyb_{pub_year}_{re.sub(r'[^0-9]+', '_', title).strip('_')}.htm"
            download(url, local)
            link_rows.append({"pub_year": pub_year, "data_year": data_year, "title": title, "url": url, "local_file": str(local)})
            parsed.append(parse_official_table(local, title, data_year, allowed))
    pd.DataFrame(link_rows).to_csv(RAW / "fj_stat_yearbook_official_table_links.csv", index=False, encoding="utf-8-sig")
    official = pd.concat(parsed, ignore_index=True)
    official.to_csv(OUT / "fujian_county_official_yearbook_variables_long_2013_2021.csv", index=False, encoding="utf-8-sig")

    wide = official.groupby(["year", "county_cn"], as_index=False).first()
    wide["unit_id"] = wide["county_cn"].map(lambda x: unit_id(x, code_lookup))
    wide["county_cn"] = wide["county_cn"].map(unit_name)
    co2 = annual[annual["county_cn"].ne("金门县")].copy()
    co2["unit_id"] = co2["county_cn"].map(lambda x: unit_id(x, code_lookup))
    co2["county_cn"] = co2["county_cn"].map(unit_name)
    co2 = co2.groupby(["unit_id", "county_cn", "year"], as_index=False)["co2_million_tons"].sum()
    panel = co2.merge(wide, on=["unit_id", "county_cn", "year"], how="left")
    panel["per_capita_gdp_yuan"] = panel["gdp_100m_yuan"] / panel["resident_population_10k"] * 10000
    panel["urbanization_rate_pct"] = panel["urban_population_10k"] / panel["resident_population_10k"] * 100
    panel["secondary_industry_share"] = panel["secondary_industry_100m_yuan"] / panel["gdp_100m_yuan"]
    panel["tertiary_industry_share"] = panel["tertiary_industry_100m_yuan"] / panel["gdp_100m_yuan"]
    panel["industry_share"] = panel["industry_100m_yuan"] / panel["gdp_100m_yuan"]
    panel["co2_ton_per_10000_yuan"] = panel["co2_million_tons"] / panel["gdp_100m_yuan"] * 100
    panel["co2_per_capita_ton"] = panel["co2_million_tons"] / panel["resident_population_10k"] * 100
    panel.to_csv(OUT / "fujian_sdm_panel_extended_2013_2021.csv", index=False, encoding="utf-8-sig")
    core_cols = [
        "year", "unit_id", "county_cn", "co2_million_tons", "gdp_100m_yuan",
        "resident_population_10k", "per_capita_gdp_yuan", "urbanization_rate_pct",
        "secondary_industry_share", "tertiary_industry_share", "industry_share",
        "retail_sales_100m_yuan", "co2_ton_per_10000_yuan", "co2_per_capita_ton",
    ]
    panel[core_cols].to_csv(OUT / "fujian_sdm_panel_core_balanced_2013_2021.csv", index=False, encoding="utf-8-sig")
    build_spatial_weights(panel[["unit_id", "county_cn"]].drop_duplicates())


if __name__ == "__main__":
    main()
