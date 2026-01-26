# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import math
import re
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import streamlit as st

# Geo libs (não adicionar dependências novas)
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    from shapely.geometry import shape  # type: ignore
    from shapely import wkb, wkt  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    Element = None  # type: ignore
    st_folium = None  # type: ignore
    wkb = None  # type: ignore
    wkt = None  # type: ignore

# =============================================================================
# Config da página e paleta (identidade PlanBairros)
# =============================================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",
    "teal": "#6FA097",
    "navy": "#14407D",
}

# Gradiente alto contraste (sem azul/verde)
ORANGE_RED_GRAD = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]
JENKS_COLORS_6 = ["#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]

SIMPLIFY_TOL = 0.0005   # ~55m (aprox) em graus — bom compromisso p/ Leaflet
MAX_FEATURES_DEFAULT = 15000  # segurança: evita “sumir” por payload gigante


# =============================================================================
# CSS (identidade + tooltips + LayerControl no canto inferior direito)
# =============================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            :root {{
                --pb-amarelo: {PB_COLORS['amarelo']};
                --pb-verde:   {PB_COLORS['verde']};
                --pb-laranja: {PB_COLORS['laranja']};
                --pb-telha:   {PB_COLORS['telha']};
                --pb-teal:    {PB_COLORS['teal']};
                --pb-navy:    {PB_COLORS['navy']};
            }}
            html, body, .stApp {{
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background: var(--pb-navy); color:#fff; border-radius: 14px;
                padding: 16px 20px; display:flex; align-items:center; gap:16px;
            }}
            .pb-title {{ font-size: 1.9rem; font-weight: 700; letter-spacing: .2px; }}
            .pb-subtitle {{ opacity:.95; margin-top:2px; font-size: .95rem; }}

            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.10);
                box-shadow:0 1px 2px rgba(0,0,0,.04);
                border-radius:14px; padding:12px;
            }}
            .pb-card h4 {{ margin: 0 0 .6rem 0; }}
            .main .block-container {{ padding-top:.6rem; padding-bottom:.8rem; }}

            /* Tooltips grandes */
            .leaflet-tooltip.pb-big-tooltip,
            .leaflet-tooltip.pb-big-tooltip * {{
                font-size: 22px !important;
                font-weight: 800 !important;
                color: #111 !important;
                line-height: 1.05 !important;
            }}
            .leaflet-tooltip.pb-big-tooltip {{
                background:#fff !important;
                border: 2px solid #222 !important; border-radius: 10px !important;
                padding: 10px 14px !important; white-space: nowrap !important;
                pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.2) !important;
                z-index: 200000 !important;
            }}

            /* deixa o mapa “respirar” melhor dentro do Streamlit */
            iframe[title="streamlit_folium.st_folium"] {{
                border-radius: 14px !important;
                border: 1px solid rgba(20,64,125,.10) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Paths e utilidades
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = (REPO_ROOT / "limites_administrativos") if (REPO_ROOT / "limites_administrativos").exists() else REPO_ROOT

# Arquivos (conforme a sua pasta no GitHub / screenshot)
DATA_FILES = {
    "Distritos": ["Distritos.parquet"],
    "Subprefeitura": ["subprefeitura.parquet", "Subprefeitura.parquet"],
    "ZonasOD2023": ["ZonasOD2023.parquet", "ZonasOD2023.parquet"],
    "SetoresCensitarios2023": ["SetoresCensitarios2023.parquet"],
    "IDCenso2023": ["IDCenso2023.parquet"],
    "Isocronas": ["isocronas.parquet"],
    # referências (na sua pasta aparecem como geojson)
    "AreasVerdes": ["area_verde.geojson", "area_verde.parquet"],
    "Rios": ["rios.geojson", "rios.parquet"],
    "LinhasMetro": ["linhas_metro.geojson", "linhas_metro.parquet"],
    "LinhasTrem": ["linhas_trem.geojson", "linhas_trem.parquet"],
}

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _find_existing_file(folder: Path, candidates: List[str]) -> Optional[Path]:
    """Tenta por nome exato; se falhar, tenta por stem (slug) dentro do folder."""
    if not folder.exists():
        return None

    # 1) tentativa por nome exato
    for name in candidates:
        fp = folder / name
        if fp.exists():
            return fp

    # 2) tentativa por stem “compatível”
    wanted = {_slug(Path(n).stem) for n in candidates}
    for fp in folder.iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in (".parquet", ".geojson", ".json"):
            continue
        if _slug(fp.stem) in wanted:
            return fp
    return None

def center_from_bounds(gdf) -> Tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2.0, (minx + maxx) / 2.0)

def find_col(cols, *cands) -> Optional[str]:
    """Match robusto: case-insensitive + remove separadores."""
    low = {c.lower(): c for c in cols}
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        if not c:
            continue
        if c.lower() in low:
            return low[c.lower()]
        key = re.sub(r"[^a-z0-9]", "", c.lower())
        if key in norm:
            return norm[key]
    return None

def to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        def _cast(x):
            if isinstance(x, Decimal):
                return float(x)
            return x
        s = s.apply(_cast)
    return pd.to_numeric(s, errors="coerce")

def _safe_force_2d(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Remove Z se existir (evita GeoJSON quebrado)."""
    try:
        from shapely import force_2d  # type: ignore
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(lambda geom: force_2d(geom) if geom is not None else geom)
        return gdf
    except Exception:
        return gdf

def _valid_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Evita geometrias vazias/invalidas -> reduz chance de “mapa sumir” no front."""
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass

    # conserta inválidas
    try:
        inv = ~gdf.is_valid
        if inv.any():
            try:
                from shapely import make_valid  # type: ignore
                gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].apply(make_valid)
            except Exception:
                # fallback clássico
                gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].buffer(0)
    except Exception:
        pass

    gdf = _safe_force_2d(gdf)
    return gdf

def _simplify(gdf: "gpd.GeoDataFrame", tol: float = SIMPLIFY_TOL) -> "gpd.GeoDataFrame":
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    return gdf


# =============================================================================
# Leitura on-demand (cache leve)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_layer(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        if path.suffix.lower() in (".geojson", ".json"):
            gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_parquet(path)
        if gdf is None or len(gdf) == 0:
            return gdf

        # CRS -> 4326 (Folium/Leaflet)
        gdf = gdf if gdf.crs is not None else gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf = _valid_geoms(gdf)
        return gdf
    except Exception:
        # fallback: parquet via pandas + WKB/WKT
        try:
            pdf = pd.read_parquet(path)
            if gpd is None:
                return None
            geom_col = find_col(pdf.columns, "geometry", "geom", "wkb", "wkt")
            if geom_col is None:
                return None

            vals = pdf[geom_col]
            vals_nonnull = vals.dropna()
            if vals_nonnull.empty:
                return None

            # heurística: WKT vs WKB
            as_str = vals_nonnull.astype(str)
            if as_str.str.startswith(("POLYGON", "MULTIPOLYGON", "LINESTRING", "MULTILINESTRING", "POINT", "MULTIPOINT")).any():
                geoms = vals.apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
            else:
                def _wkb_load(x):
                    if x is None or (isinstance(x, float) and math.isnan(x)):
                        return None
                    if isinstance(x, (bytes, bytearray, memoryview)):
                        return wkb.loads(bytes(x))
                    if isinstance(x, str):
                        return wkb.loads(x, hex=True)
                    return None
                geoms = vals.apply(_wkb_load)

            gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geoms, crs=4326)
            gdf = _valid_geoms(gdf)
            return gdf
        except Exception:
            return None


def load_named(key: str) -> Optional["gpd.GeoDataFrame"]:
    cand = DATA_FILES.get(key, [])
    fp = _find_existing_file(DATA_DIR, cand)
    return read_layer(fp) if fp else None


# =============================================================================
# Join SetoresCensitarios2023 (variáveis) + IDCenso2023 (geometria) via fid
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def load_setores_joined_by_fid() -> Optional["gpd.GeoDataFrame"]:
    """IDCenso fornece geometria. SetoresCensitarios2023 fornece atributos. Join por fid."""
    idc = load_named("IDCenso2023")
    sc = load_named("SetoresCensitarios2023")
    if idc is None or sc is None or len(idc) == 0 or len(sc) == 0:
        return None

    idc_fid = find_col(idc.columns, "fid", "FID")
    sc_fid = find_col(sc.columns, "fid", "FID")

    if idc_fid is None or sc_fid is None:
        return None

    # padroniza fid (evita "não encontra variáveis")
    idc = idc.copy()
    sc = sc.copy()
    idc[idc_fid] = pd.to_numeric(idc[idc_fid], errors="coerce").astype("Int64")
    sc[sc_fid] = pd.to_numeric(sc[sc_fid], errors="coerce").astype("Int64")

    # remove geometry de sc para não conflitar
    sc_attrs = sc.drop(columns=["geometry"], errors="ignore")

    gdf = idc.merge(sc_attrs, left_on=idc_fid, right_on=sc_fid, how="left")
    # garante nome canônico "fid" (para tooltip/legenda)
    if idc_fid != "fid":
        gdf = gdf.rename(columns={idc_fid: "fid"})
    gdf = _valid_geoms(gdf)
    return gdf


# =============================================================================
# Folium — mapa base Carto + panes + rótulos
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11) -> "folium.Map":
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)

    # Base Carto Positron (SEM ESRI)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="Carto Positron",
        overlay=False,
        control=False,
    ).add_to(m)

    # panes (ordem importa: maior = mais acima)
    try:
        folium.map.CustomPane("choropleth", z_index=450).add_to(m)
        folium.map.CustomPane("outlines", z_index=650).add_to(m)
        folium.map.CustomPane("lines", z_index=760).add_to(m)
        folium.map.CustomPane("green", z_index=900).add_to(m)
        folium.map.CustomPane("labels", z_index=950).add_to(m)
    except Exception:
        pass

    return m


def inject_label_scaler(m: "folium.Map", min_px=14, max_px=26, min_zoom=9, max_zoom=18) -> None:
    if Element is None:
        return
    map_var = m.get_name()
    js = f"""
    <script>
      (function() {{
        var minZ = {min_zoom}, maxZ = {max_zoom};
        var minPx = {min_px}, maxPx = {max_px};
        function scaleFont(z) {{
          if (z < minZ) z = minZ;
          if (z > maxZ) z = maxZ;
          var t = (z - minZ) / (maxZ - minZ);
          return Math.round(minPx + t*(maxPx - minPx));
        }}
        function updatePBLabels() {{
          var z = {map_var}.getZoom();
          var px = scaleFont(z);
          var els = document.querySelectorAll('.pb-static-label');
          els.forEach(function(el) {{ el.style.fontSize = px + 'px'; }});
        }}
        {map_var}.on('zoomend', updatePBLabels);
        {map_var}.whenReady(updatePBLabels);
      }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))


def add_centroid_labels(m: "folium.Map", gdf: "gpd.GeoDataFrame") -> None:
    if folium is None:
        return
    cols = {c.lower(): c for c in gdf.columns}
    name_col = cols.get("ds_nome") or cols.get("sp_nome") or cols.get("nome")
    if not name_col:
        return

    try:
        reps = gdf.to_crs(4326).representative_point()
    except Exception:
        return

    for name, pt in zip(gdf[name_col].astype(str), reps):
        if pt is None or getattr(pt, "is_empty", False):
            continue
        html = (
            "<div class='pb-static-label' "
            "style=\"font: 600 12px/1 Roboto, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;"
            "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
            f"{name}</div>"
        )
        folium.Marker(
            location=[pt.y, pt.x],
            icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
            z_index_offset=1000,
        ).add_to(m)


# =============================================================================
# Legendas
# =============================================================================
def add_categorical_legend(m: "folium.Map", title: str, items: List[Tuple[str, str]], pos: str = "bottomleft") -> None:
    if Element is None:
        return
    if pos == "bottomleft":
        csspos = "left:16px; bottom:24px;"
    elif pos == "bottomright":
        csspos = "right:16px; bottom:24px;"
    elif pos == "topright":
        csspos = "right:16px; top:24px;"
    else:
        csspos = "left:16px; top:24px;"

    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
        f'<span style="width:16px;height:16px;border-radius:3px;background:{color};display:inline-block"></span>'
        f'<span>{label}</span></div>'
        for label, color in items
    )
    html = f"""
    <div style="position:absolute; {csspos} z-index:9999; background:rgba(255,255,255,.95);
      padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25);
      font: 500 12px Roboto, sans-serif; max-width: 520px;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(Element(html))


# =============================================================================
# Render — overlays (admin outlines + referências)
# =============================================================================
def _outline_as_boundary(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Converte qualquer geometria em contorno (LineString/MultiLineString)."""
    g = gdf[["geometry"]].copy()
    g = _valid_geoms(g)
    try:
        g["geometry"] = g.geometry.boundary
    except Exception:
        pass
    g = _valid_geoms(g)
    g = _simplify(g)
    return g

def add_outline_layer(
    m: "folium.Map",
    gdf: "gpd.GeoDataFrame",
    name: str,
    color: str = "#222222",
    weight: float = 1.0,
    dashed: bool = False,
    pane: str = "outlines",
) -> None:
    if folium is None or gdf is None or len(gdf) == 0:
        return
    g = _outline_as_boundary(gdf)

    dash = "6,6" if dashed else None

    folium.GeoJson(
        data=g.to_json(),
        name=name,
        pane=pane,
        style_function=lambda f: {
            "fillOpacity": 0.0,
            "opacity": 1.0,
            "color": color,
            "weight": weight,
            "dashArray": dash,
        },
        control=False,  # fica “fixo”; LayerControl usamos para outras coisas
    ).add_to(m)

def add_reference_layers(m: "folium.Map") -> None:
    """Camadas de referência sempre visíveis: áreas verdes, rios, linhas metro/trem."""
    if folium is None:
        return

    # Áreas verdes (acima de tudo)
    green = load_named("AreasVerdes")
    if green is not None and len(green) > 0:
        green = _valid_geoms(green)
        green = _simplify(green)
        folium.GeoJson(
            data=green[["geometry"]].to_json(),
            name="Áreas verdes",
            pane="green",
            style_function=lambda f: {
                "fillOpacity": 1.0,              # 100%
                "opacity": 1.0,
                "color": PB_COLORS["verde"],
                "weight": 0.6,
                "fillColor": PB_COLORS["verde"],
            },
            control=False,
        ).add_to(m)

    # Rios (azul)
    rios = load_named("Rios")
    if rios is not None and len(rios) > 0:
        rios = _valid_geoms(rios)
        rios = _simplify(rios)
        folium.GeoJson(
            data=_outline_as_boundary(rios).to_json(),
            name="Rios",
            pane="lines",
            style_function=lambda f: {
                "fillOpacity": 0.0,
                "opacity": 1.0,
                "color": "#1f77b4",
                "weight": 2.0,
            },
            control=False,
        ).add_to(m)

    # Linhas metro/trem (preto)
    for key, label in [("LinhasMetro", "Linhas de metrô"), ("LinhasTrem", "Linhas de trem")]:
        ldf = load_named(key)
        if ldf is None or len(ldf) == 0:
            continue
        ldf = _valid_geoms(ldf)
        ldf = _simplify(ldf)
        folium.GeoJson(
            data=_outline_as_boundary(ldf).to_json(),
            name=label,
            pane="lines",
            style_function=lambda f: {
                "fillOpacity": 0.0,
                "opacity": 1.0,
                "color": "#000000",
                "weight": 2.2,
            },
            control=False,
        ).add_to(m)


# =============================================================================
# Jenks (Fisher–Jenks) — com amostragem (evita travar em 27k+)
# =============================================================================
def _jenks_breaks_fisher(values: List[float], k: int) -> List[float]:
    """
    Fisher–Jenks natural breaks (exato) — custo O(k*n^2).
    Use apenas em amostra (n ~ 1000–2000) para ser viável em Python puro.
    """
    data = sorted([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))])
    n = len(data)
    if n == 0:
        return []
    if k <= 1:
        return [data[0], data[-1]]

    mat1 = [[0] * (k + 1) for _ in range(n + 1)]
    mat2 = [[0.0] * (k + 1) for _ in range(n + 1)]

    for i in range(1, k + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0.0
        for j in range(2, n + 1):
            mat2[j][i] = float("inf")

    v = 0.0
    for l in range(2, n + 1):
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = data[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1.0
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v

    kclass = [0.0] * (k + 1)
    kclass[k] = data[-1]
    count = k
    idx = n
    while count >= 2:
        id_ = int(mat1[idx][count] - 2)
        kclass[count - 1] = data[id_]
        idx = int(mat1[idx][count] - 1)
        count -= 1
    kclass[0] = data[0]
    return kclass

def jenks_breaks(values: pd.Series, k: int = 6, sample_size: int = 1500, seed: int = 42) -> List[float]:
    vals = to_numeric_series(values).dropna().astype(float).tolist()
    if len(vals) == 0:
        return []
    if len(vals) <= sample_size:
        sample = vals
    else:
        rnd = random.Random(seed)
        sample = rnd.sample(vals, sample_size)
    brks = _jenks_breaks_fisher(sample, k)
    # garante monotonicidade e extremos
    if not brks:
        return []
    brks[0] = float(min(vals))
    brks[-1] = float(max(vals))
    # remove duplicatas que podem ocorrer em dados com muitos valores iguais
    out = [brks[0]]
    for b in brks[1:]:
        if b > out[-1]:
            out.append(b)
    # se faltou classe por repetição, fallback para quantis (último recurso)
    if len(out) < 2:
        qs = np.quantile(np.array(vals), np.linspace(0, 1, k + 1)).tolist()
        return [float(qs[0])] + [float(x) for x in qs[1:]]
    return out

def _classify_by_breaks(v: float, breaks: List[float]) -> int:
    # breaks = [min, b1, b2, ..., max]
    if v is None or (isinstance(v, float) and math.isnan(v)) or not breaks:
        return -1
    for i in range(1, len(breaks)):
        if v <= breaks[i]:
            return i - 1
    return len(breaks) - 2


# =============================================================================
# Pintura — Setores (via IDCenso geom + atributos SC) e Isócronas
# =============================================================================
def paint_setores_jenks(
    m: "folium.Map",
    gdf: "gpd.GeoDataFrame",
    value_col: str,
    title: str,
    show_fid: bool = True,
    max_features: int = MAX_FEATURES_DEFAULT,
    render_all: bool = False,
) -> None:
    if folium is None or gdf is None or len(gdf) == 0:
        return

    # proteção — payload gigante costuma “sumir” no browser
    if (not render_all) and len(gdf) > max_features:
        gdf = gdf.sample(max_features, random_state=42)

    s = to_numeric_series(gdf[value_col])
    brks = jenks_breaks(s, k=6, sample_size=1500, seed=42)

    df = gdf[["geometry"]].copy()
    df["__v__"] = s.astype(float)
    df["__c__"] = df["__v__"].apply(lambda x: _classify_by_breaks(x, brks)).astype(int)
    if show_fid and "fid" in gdf.columns:
        df["fid"] = gdf["fid"]

    def style_fn(feat):
        c = feat["properties"].get("__c__", -1)
        if c is None:
            c = -1
        if c < 0 or c >= 6:
            fill = "#c8c8c8"
        else:
            fill = JENKS_COLORS_6[int(c)]
        return {
            "fillOpacity": 0.80,
            "opacity": 0.0,
            "weight": 0.0,
            "color": "#00000000",
            "fillColor": fill,
        }

    fields = ["__v__"]
    aliases = [f"{title}: "]
    if show_fid and "fid" in df.columns:
        fields = ["fid", "__v__"]
        aliases = ["fid: ", f"{title}: "]

    folium.GeoJson(
        data=df[["geometry"] + fields].to_json(),
        name=title,
        pane="choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            sticky=True,
            labels=False,
            class_name="pb-big-tooltip",
        ),
        control=False,
    ).add_to(m)

    # legenda Jenks 6 classes
    items = []
    # brks pode ter menos pontos se houver repetição; ajusta classes possíveis
    n_classes = min(6, max(1, len(brks) - 1))
    for i in range(n_classes):
        lo = brks[i]
        hi = brks[i + 1]
        items.append((f"{lo:,.2f} — {hi:,.2f}", JENKS_COLORS_6[min(i, 5)]))
    add_categorical_legend(m, f"{title} (Jenks — {n_classes} classes)", items, pos="bottomleft")


def paint_setores_cluster(m: "folium.Map", gdf: "gpd.GeoDataFrame", cluster_col: str) -> None:
    if folium is None or gdf is None or len(gdf) == 0:
        return

    color_map = {0:"#bf7db2", 1:"#f7bd6a", 2:"#cf651f", 3:"#ede4e6", 4:"#793393"}
    label_map = {
        0:"1 - Periférico com predominância residencial de alta densidade construtiva",
        1:"2 - Uso misto de média densidade construtiva",
        2:"3 - Periférico com predominância residencial de média densidade construtiva",
        3:"4 - Verticalizado de uso-misto",
        4:"5 - Predominância de uso comercial e serviços",
    }
    default_color = "#c8c8c8"

    df = gdf[["geometry"]].copy()
    c = to_numeric_series(gdf[cluster_col]).astype("Int64")
    df["__c__"] = c
    if "fid" in gdf.columns:
        df["fid"] = gdf["fid"]

    def style_fn(feat):
        v = feat["properties"].get("__c__")
        col = color_map.get(int(v) if v is not None and not pd.isna(v) else -1, default_color)
        return {"fillOpacity": 0.75, "weight": 0.0, "opacity": 0.0, "color": "#00000000", "fillColor": col}

    folium.GeoJson(
        data=df[["geometry", "__c__"] + (["fid"] if "fid" in df.columns else [])].to_json(),
        name="Cluster (perfil urbano)",
        pane="choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=(["fid", "__c__"] if "fid" in df.columns else ["__c__"]),
            aliases=(["fid: ", "cluster: "] if "fid" in df.columns else ["cluster: "]),
            sticky=True, labels=False, class_name="pb-big-tooltip"
        ),
        control=False,
    ).add_to(m)

    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
    items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items, pos="bottomleft")


def paint_isocronas_nova_class(m: "folium.Map", iso: "gpd.GeoDataFrame") -> None:
    if folium is None or iso is None or len(iso) == 0:
        return
    cls = find_col(iso.columns, "nova_class")
    if not cls:
        st.info("A coluna 'nova_class' não foi encontrada em isócronas.")
        return

    lut = {
        0: ("Predominância uso misto", "#542788"),
        1: ("Zona de transição local", "#f7f7f7"),
        2: ("Periférico residencial de média densidade", "#d8daeb"),
        3: ("Transição central verticalizada", "#b35806"),
        4: ("Periférico adensado em transição", "#b2abd2"),
        5: ("Centralidade comercial e de serviços", "#8073ac"),
        6: ("Predominância residencial média densidade", "#fdb863"),
        7: ("Áreas íngremes e de encosta", "#7f3b08"),
        8: ("Alta densidade residencial", "#e08214"),
        9: ("Central verticalizado", "#fee0b6"),
    }

    df = iso[["geometry"]].copy()
    k = to_numeric_series(iso[cls]).astype("Int64")
    df["__k__"] = k

    def style_fn(feat):
        v = feat["properties"].get("__k__")
        color = lut.get(int(v) if v is not None and not pd.isna(v) else -1, ("Outros", "#c8c8c8"))[1]
        return {"fillOpacity": 0.65, "weight": 0.0, "opacity": 0.0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        data=df[["geometry", "__k__"]].to_json(),
        name="Isócronas (nova_class)",
        pane="choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(fields=["__k__"], aliases=["classe: "], sticky=True, labels=False, class_name="pb-big-tooltip"),
        control=False,
    ).add_to(m)

    items = [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()]
    add_categorical_legend(m, "Isócronas (nova_class)", items, pos="bottomleft")


# =============================================================================
# UI — controles (esquerda)
# =============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores — join por `fid`)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "densidade_demografica",
            "Populacao",
            "Diferenca_elevacao",
            "elevacao",
            "raio_maximo_caminhada",
            "Cluster",
            "Isócronas (nova_class)",
        ],
        index=0,
        key="pb_var",
        help="Setores são exibidos com Jenks (6 classes) usando geometria do IDCenso + atributos por fid.",
    )

    st.markdown("### Limites administrativos (contorno)")
    highlight = st.selectbox(
        "Destaque principal",
        ["Distritos", "Subprefeitura", "ZonasOD2023"],
        index=0,
        key="pb_highlight",
        help="Todos os limites aparecem; este fica com traço mais forte.",
    )

    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)",
        value=False,
        key="pb_labels_on",
        help="Rótulos centróides do limite em destaque.",
    )

    st.markdown("### Performance / estabilidade")
    render_all = st.checkbox(
        "Renderizar TODOS os setores (pode ser pesado)",
        value=False,
        key="pb_render_all",
        help="Se o navegador 'sumir' o mapa, desative e use a amostra.",
    )

    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo. Recarregue as camadas.")

    with st.expander("Diagnóstico rápido (caminhos / arquivos)", expanded=False):
        st.caption(f"REPO_ROOT: `{REPO_ROOT}`")
        st.caption(f"DATA_DIR: `{DATA_DIR}`")
        if DATA_DIR.exists():
            files = sorted([p.name for p in DATA_DIR.iterdir() if p.is_file()])
            st.write(f"Arquivos encontrados ({len(files)}):")
            st.code("\n".join(files)[:6000])
        else:
            st.error("DATA_DIR não existe — confira o path no GitHub.")

    return {"var": var, "highlight": highlight, "labels_on": labels_on, "render_all": render_all}


# =============================================================================
# App
# =============================================================================
def main() -> None:
    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium` instalados.")
        return

    inject_css()

    # Cabeçalho (identidade)
    with st.container():
        c1, c2 = st.columns([1, 8])
        with c1:
            st.image(
                "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png",
                width=64,
                caption="",
            )
        with c2:
            st.markdown(
                """
                <div class="pb-header">
                    <div style="display:flex;flex-direction:column">
                        <div class="pb-title">PlanBairros</div>
                        <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Layout: filtros (esq) / mapa (dir)
    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        # Carregamentos principais
        # 1) limites
        distritos = load_named("Distritos")
        subpref = load_named("Subprefeitura")
        zonasod = load_named("ZonasOD2023")

        # 2) setores (join fid)
        setores = load_setores_joined_by_fid()

        # 3) isócronas
        isocronas = load_named("Isocronas")

        # Centro do mapa (prioridade: setores, senão highlight, senão SP)
        center = (-23.55, -46.63)
        if setores is not None and len(setores) > 0:
            center = center_from_bounds(setores)
        else:
            hl = {"Distritos": distritos, "Subprefeitura": subpref, "ZonasOD2023": zonasod}.get(ui["highlight"])
            if hl is not None and len(hl) > 0:
                center = center_from_bounds(hl)

        fmap = make_carto_map(center=center, zoom=11)
        inject_label_scaler(fmap, min_px=14, max_px=26, min_zoom=9, max_zoom=18)

        # Camadas de referência sempre ativas (acima)
        add_reference_layers(fmap)

        # Limites administrativos: todos como contorno fino + destaque forte
        if distritos is not None and len(distritos) > 0:
            add_outline_layer(fmap, distritos, "Distritos", color="#3b3b3b", weight=0.9, dashed=True)
        if subpref is not None and len(subpref) > 0:
            add_outline_layer(fmap, subpref, "Subprefeitura", color="#3b3b3b", weight=0.9, dashed=True)
        if zonasod is not None and len(zonasod) > 0:
            add_outline_layer(fmap, zonasod, "Zonas OD 2023", color="#3b3b3b", weight=0.9, dashed=True)

        # Destaque principal (traço mais forte)
        hl = ui["highlight"]
        hl_gdf = {"Distritos": distritos, "Subprefeitura": subpref, "ZonasOD2023": zonasod}.get(hl)
        if hl_gdf is not None and len(hl_gdf) > 0:
            add_outline_layer(fmap, hl_gdf, f"{hl} (destaque)", color="#000000", weight=2.0, dashed=False)
            if ui["labels_on"]:
                add_centroid_labels(fmap, hl_gdf)
        else:
            st.warning(f"Não encontrei o arquivo do limite em destaque: {hl}.")

        # Variável selecionada
        var = ui["var"]

        if var == "Isócronas (nova_class)":
            if isocronas is None or len(isocronas) == 0:
                st.warning("Isócronas não encontradas em limites_administrativos/.")
            else:
                paint_isocronas_nova_class(fmap, _simplify(_valid_geoms(isocronas)))
        else:
            if setores is None or len(setores) == 0:
                st.warning("Não consegui montar Setores (join por fid). Verifique IDCenso2023 e SetoresCensitarios2023.")
            else:
                # garante nomes exatamente como seu dado (imagem do QGIS)
                # possíveis chaves reais: fid, Area_km2, Populacao, area_hectare, densidade_demografica,
                # raio_maximo_caminhada, Diferenca_elevacao, Isocrona, elevacao, Cluster
                if var == "Cluster":
                    ccol = find_col(setores.columns, "Cluster", "cluster")
                    if not ccol:
                        st.warning("Coluna 'Cluster' não encontrada nos setores (após join por fid).")
                    else:
                        paint_setores_cluster(fmap, setores, ccol)
                else:
                    # numéricas
                    ccol = find_col(
                        setores.columns,
                        var,  # exato
                        var.lower(),
                        var.upper(),
                    )
                    if not ccol:
                        st.warning(f"Coluna '{var}' não encontrada. Colunas disponíveis (amostra): {list(setores.columns)[:35]}")
                    else:
                        paint_setores_jenks(
                            fmap,
                            _simplify(setores),
                            ccol,
                            title=var,
                            show_fid=True,
                            render_all=bool(ui["render_all"]),
                        )

        # LayerControl no canto inferior direito (como você pediu anteriormente)
        try:
            folium.LayerControl(collapsed=False, position="bottomright").add_to(fmap)
        except Exception:
            pass

        # Render sem capturar eventos (reduz rerun e “mapa some” ao interagir)
        st_folium(
            fmap,
            height=780,
            use_container_width=True,
            returned_objects=[],
            key="pb_map",
        )


if __name__ == "__main__":
    main()
