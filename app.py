# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from unicodedata import normalize as _ud_norm
import math
import os
import re
import time

import pandas as pd
import streamlit as st

# Geo libs (opcionais – o app funciona sem, só não mostra o mapa)
try:
    import geopandas as gpd  # type: ignore
    import shapely  # type: ignore
    from shapely import set_precision  # type: ignore
    from shapely.geometry import box  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    shapely = None  # type: ignore
    set_precision = None  # type: ignore
    folium = None  # type: ignore
    Element = None  # type: ignore
    st_folium = None  # type: ignore


# ============================================================================
# Config da página e paleta
# ============================================================================
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

# Gradiente contínuo pedido (sem azul/verde)
PB_GRADIENT = ["#F4DD63", "#D58243", "#C65534"]  # amarelo -> laranja -> telha


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
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background: var(--pb-navy); color: #fff; border-radius: 18px;
                padding: 20px 24px; min-height: 110px; display: flex; align-items: center; gap: 16px;
            }}
            .pb-title {{ font-size: 2.2rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.05rem; }}

            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04);
                border-radius:16px; padding:16px;
            }}
            .pb-card h4 {{ margin: 0 0 .6rem 0; }}
            .main .block-container {{ padding-top: .6rem; padding-bottom: .6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_path() -> Optional[str]:
    for p in [
        "assets/logo_todos.jpg",
        "assets/logo_paleta.jpg",
        "logo_todos.jpg",
        "logo_paleta.jpg",
        "/mnt/data/logo_todos.jpg",
        "/mnt/data/logo_paleta.jpg",
    ]:
        if Path(p).exists():
            return p
    return None


# ============================================================================
# Caminhos / leitura com cache
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_dir(subdir: str) -> Path:
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / subdir
        if p.exists():
            return p
    return REPO_ROOT / subdir


ADM_DIR = _resolve_dir("limites_administrativos")  # Setores + Isócronas ficam aqui


def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _first_parquet_matching(folder: Path, names: list[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in names}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None


def _dir_mtime_key(folder: Path) -> float:
    """Sinaliza mudanças em disco para invalidar caches derivados."""
    try:
        return max((p.stat().st_mtime for p in folder.glob("*.parquet")), default=0.0)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def read_gdf_from_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        return gdf
    except Exception:
        try:
            pdf = pd.read_parquet(path)
        except Exception:
            return None
        geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
        if geom_col is None:
            return None
        try:
            from shapely import wkb, wkt
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith("POLY").any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
            return gdf
        except Exception:
            return None


# nomes de arquivos
SETORES_FILE = _first_parquet_matching(ADM_DIR, ["SetoresCensitarios2023"])
ISO_FILE = _first_parquet_matching(ADM_DIR, ["isocronas", "isócronas"])


@st.cache_data(show_spinner=False)
def load_setores(mtime_key: float) -> Optional["gpd.GeoDataFrame"]:
    return read_gdf_from_parquet(SETORES_FILE) if SETORES_FILE else None


@st.cache_data(show_spinner=False)
def load_isocronas(mtime_key: float) -> Optional["gpd.GeoDataFrame"]:
    return read_gdf_from_parquet(ISO_FILE) if ISO_FILE else None


@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str, mtime_key: float) -> Optional["gpd.GeoDataFrame"]:
    """Limites administrativos (sem Setores)."""
    filename = {
        "Distritos": "Distritos",
        "ZonasOD2023": "ZonasOD2023",
        "Subprefeitura": "subprefeitura",
        "Isócronas": "isocronas",
    }.get(layer_name, layer_name)
    path = _first_parquet_matching(ADM_DIR, [filename])
    return read_gdf_from_parquet(path) if path else None


# ============================================================================
# Utilidades geom / performance
# ============================================================================
def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
    for c in cands:
        if c and c.lower() in low:
            return low[c.lower()]
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        key = re.sub(r"[^a-z0-9]", "", (c or "").lower())
        if key in norm:
            return norm[key]
    return None


def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)


def bounds_tuple(gdf) -> Optional[tuple[float, float, float, float]]:
    if gdf is None or gdf.empty:
        return None
    minx, miny, maxx, maxy = gdf.total_bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def pick_tol_by_extent(limite_gdf, fallback_gdf) -> int:
    """Tolerância (m) para simplificação conforme extensão do recorte."""
    g = limite_gdf if (limite_gdf is not None and len(limite_gdf) > 0) else fallback_gdf
    if g is None or g.empty:
        return 60
    minx, miny, maxx, maxy = g.total_bounds
    largura_km = (maxx - minx) * 111  # aprox.
    if largura_km > 100:
        return 160
    if largura_km > 50:
        return 120
    if largura_km > 20:
        return 60
    if largura_km > 8:
        return 30
    return 15


def simplify_3857(gdf: "gpd.GeoDataFrame", tol_m: int = 60, precision_m: float = 1.0) -> "gpd.GeoDataFrame":
    """Simplifica em metros (EPSG:3857) e reduz precisão das coordenadas."""
    if gdf is None or gdf.empty or gpd is None:
        return gdf
    g = gdf.to_crs(3857).copy()
    try:
        g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
        if set_precision is not None:
            g["geometry"] = set_precision(g.geometry, grid_size=precision_m)
    except Exception:
        pass
    return g.to_crs(4326)


def filter_bbox(gdf: "gpd.GeoDataFrame", b: Optional[tuple[float, float, float, float]]) -> "gpd.GeoDataFrame":
    """Filtra por interseção com a caixa do limite (barato e efetivo)."""
    if gdf is None or gdf.empty or b is None or gpd is None:
        return gdf
    try:
        bb = box(*b)
        return gdf[gdf.intersects(bb)]
    except Exception:
        return gdf


# ============================================================================
# UI – filtros à esquerda
# ============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    with st.container():
        var = st.selectbox(
            "Selecione a variável",
            [
                "População (Pessoa/ha)",
                "Densidade demográfica (hab/ha)",
                "Variação de elevação média",
                "Elevação média",
                "Cluster (perfil urbano)",
                "Área de influência de bairro",
            ],
            index=0,
            key="pb_var",
            help="Variáveis numéricas são pintadas por setor (sem linhas). 'Área de influência' usa isócronas.",
        )
        st.markdown("")

    st.markdown("### Configurações")
    with st.container():
        limite = st.selectbox(
            "Limites Administrativos",
            ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
            index=0,
            key="pb_limite",
            help="Exibe contorno do limite selecionado. (Setores foram movidos para 'Variáveis')",
        )
        labels_on = st.checkbox(
            "Rótulos permanentes (dinâmicos por zoom)",
            value=False,
            key="pb_labels_on",
            help="Rotula centróides com fonte proporcional ao nível de zoom.",
        )
    return {"variavel": var, "limite": limite, "labels_on": labels_on}


# ============================================================================
# Folium – mapa + helpers (CSS/JS e desenho)
# ============================================================================
SIMPLIFY_TOL_DEFAULT = 60  # fallback


def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    # Esri Imagery (sem rótulos)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=False,
        opacity=tiles_opacity,
    ).add_to(m)
    return m


def inject_tooltip_css(m, font_px: int = 40):
    if Element is None:
        return
    css = f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip,
      .leaflet-tooltip.pb-big-tooltip * {{
        font-size: {font_px}px !important;
        font-weight: 700 !important;
        color: #111 !important;
        line-height: 1 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background: #fff !important;
        border: 2px solid #222 !important; border-radius: 10px !important;
        padding: 6px 10px !important; white-space: nowrap !important;
        pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.2) !important;
        z-index: 200000 !important;
      }}
    </style>
    """
    m.get_root().html.add_child(Element(css))


def inject_label_scaler(m, min_px=14, max_px=24, min_zoom=9, max_zoom=18):
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
          return Math.round(minPx + t * (maxPx - minPx));
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


def add_admin_outline(m, gdf, layer_name: str, tol_m: int):
    if gdf is None or folium is None:
        return
    # Linha (contorno) com simplificação forte
    gf_line = simplify_3857(gdf[["geometry"]], tol_m=max(30, tol_m))
    gf_line["geometry"] = gf_line.geometry.boundary
    folium.GeoJson(
        data=gf_line.to_json(),
        name=f"{layer_name} (contorno)",
        pane="vectors",
        style_function=lambda f: {"fillOpacity": 0, "color": "#000000", "weight": 1.1},
        smooth_factor=1.2,
    ).add_to(m)

    # Regra especial — isócronas como “área de transição” nos limites
    lname = layer_name.lower()
    if lname.startswith("isócron") or lname.startswith("isocron"):
        cls = find_col(gdf.columns, "nova_class")
        if cls:
            trans = gdf[gdf[cls].isin([1, 3, 4, 6])][["geometry"]].copy()
            if not trans.empty:
                trans = simplify_3857(trans, tol_m=max(30, tol_m))
                folium.GeoJson(
                    data=trans.to_json(),
                    name="Área de transição",
                    pane="vectors",
                    style_function=lambda f: {"fillOpacity": 0.2, "color": "#836e60", "weight": 0.8},
                    smooth_factor=1.2,
                ).add_to(m)
                add_categorical_legend(
                    m,
                    "Limites — Isócronas",
                    [("Área de transição (classes 1,3,4,6)", "#836e60")],
                    topright=True,
                )


def add_centroid_labels(m, gdf):
    if folium is None or gdf is None or gdf.empty:
        return
    cols_lower = {c.lower(): c for c in gdf.columns}
    name_col = cols_lower.get("ds_nome") or cols_lower.get("sp_nome") or cols_lower.get("nome")
    if name_col is None:
        return
    try:
        centers = gdf.copy().to_crs(4326).representative_point()
    except Exception:
        return
    for name, pt in zip(gdf[name_col].astype(str), centers):
        if pt is None or pt.is_empty:
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


def add_gradient_legend(m, title: str, vmin: float, vmax: float, colors: list[str]):
    if Element is None:
        return
    gradient_css = f"background: linear-gradient(to top, {', '.join(colors)});"
    html = f"""
    <div style="
      position: absolute; bottom: 24px; left: 16px; z-index: 9999;
      background: rgba(255,255,255,.95); padding: 10px 12px; border-radius: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,.25); font: 500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      <div style="display:flex; align-items:stretch; gap:10px;">
        <div style="width:18px; height:120px; {gradient_css}; border-radius:4px;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:120px;">
          <div>{vmax:,.0f}</div>
          <div>{(vmin+vmax)/2:,.0f}</div>
          <div>{vmin:,.0f}</div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(html))


def add_categorical_legend(m, title: str, items: list[tuple[str, str]], topright: bool = False):
    if Element is None:
        return
    pos = "right: 16px; top: 24px;" if topright else "left: 16px; bottom: 24px;"
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
        f'<span style="width:16px;height:16px;border-radius:3px;background:{color};display:inline-block"></span>'
        f'<span>{label}</span></div>'
        for label, color in items
    )
    html = f"""
    <div style="position:absolute; {pos} z-index:9999; background:rgba(255,255,255,.95);
      padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25);
      font: 500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(Element(html))


# ============================================================================
# Construção de camadas *com cache* (clip+bbox + simplifica)
# ============================================================================

@st.cache_data(show_spinner=False)
def build_setores_numeric_layer(
    value_col: str,
    limit_bbox: Optional[tuple[float, float, float, float]],
    tol_m: int,
    mtime_key: float,
) -> Optional["gpd.GeoDataFrame"]:
    setores = load_setores(mtime_key)
    if setores is None or setores.empty:
        return None
    if value_col not in setores.columns:
        col = find_col(setores.columns, value_col)
        if col is None:
            return None
        value_col = col
    g = setores[[value_col, "geometry"]].copy()
    g = filter_bbox(g, limit_bbox)
    g = simplify_3857(g, tol_m=tol_m, precision_m=1.0)
    # estatística simples para legenda
    s = pd.to_numeric(g[value_col], errors="coerce")
    g["_vmin"] = float(s.min()) if s.notna().any() else 0.0
    g["_vmax"] = float(s.max()) if s.notna().any() else 0.0
    return g


@st.cache_data(show_spinner=False)
def build_setores_cluster_layer(
    cluster_col: str,
    limit_bbox: Optional[tuple[float, float, float, float]],
    tol_m: int,
    mtime_key: float,
) -> Optional["gpd.GeoDataFrame"]:
    setores = load_setores(mtime_key)
    if setores is None or setores.empty:
        return None
    if cluster_col not in setores.columns:
        col = find_col(setores.columns, cluster_col, "cluster", "cluster_label", "label", "classe")
        if col is None:
            return None
        cluster_col = col
    g = setores[[cluster_col, "geometry"]].copy()
    g = filter_bbox(g, limit_bbox)
    g = simplify_3857(g, tol_m=tol_m, precision_m=1.0)
    return g


@st.cache_data(show_spinner=False)
def build_isocronas_layer(
    limit_bbox: Optional[tuple[float, float, float, float]],
    tol_m: int,
    mtime_key: float,
) -> Optional["gpd.GeoDataFrame"]:
    iso = load_isocronas(mtime_key)
    if iso is None or iso.empty:
        return None
    g = iso[["nova_class", "geometry"]].copy() if "nova_class" in iso.columns else iso.copy()
    g = filter_bbox(g, limit_bbox)
    g = simplify_3857(g, tol_m=tol_m, precision_m=1.0)
    return g


# ============================================================================
# Pintura por setor/isócronas
# ============================================================================
def ramp_color(value: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if value is None or math.isnan(value):
        return "#cccccc"
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    n = len(colors) - 1
    i = min(int(t * n), n - 1)
    frac = (t * n) - i

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[k:k+2], 16) for k in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    c1 = hex_to_rgb(colors[i])
    c2 = hex_to_rgb(colors[i + 1])
    rgb = tuple(int(c1[k] + frac * (c2[k] - c1[k])) for k in range(3))
    return rgb_to_hex(rgb)


def draw_setores_numeric(m, g: "gpd.GeoDataFrame", value_col: str, label: str):
    if g is None or g.empty or folium is None:
        return
    vmin = float(g["_vmin"].iloc[0]) if "_vmin" in g.columns else float(pd.to_numeric(g[value_col], errors="coerce").min())
    vmax = float(g["_vmax"].iloc[0]) if "_vmax" in g.columns else float(pd.to_numeric(g[value_col], errors="coerce").max())

    def style_fn(feat):
        v = feat["properties"].get(value_col)
        v = float(v) if v is not None else float("nan")
        return {"fillOpacity": 0.75, "weight": 0.0, "color": "#00000000", "fillColor": ramp_color(v, vmin, vmax, PB_GRADIENT)}

    folium.GeoJson(
        data=g.to_json(),
        name=label,
        pane="vectors",
        style_function=style_fn,
        smooth_factor=1.2,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[value_col],
            aliases=[label + ": "],
            sticky=True,
            labels=False,
            class_name="pb-big-tooltip",
        ),
    ).add_to(m)
    add_gradient_legend(m, label, vmin, vmax, PB_GRADIENT)


def draw_setores_cluster(m, g: "gpd.GeoDataFrame", cluster_col: str):
    if g is None or g.empty or folium is None:
        return
    color_map = {0: "#bf7db2", 1: "#f7bd6a", 2: "#cf651f", 3: "#ede4e6", 4: "#793393"}
    label_map = {
        0: "1 - Periférico com predominância residencial de alta densidade construtiva",
        1: "2 - Uso misto de média densidade construtiva",
        2: "3 - Periférico com predominância residencial de média densidade construtiva",
        3: "4 - Verticalizado de uso-misto",
        4: "5 - Predominância de uso comercial e serviços",
    }
    default_color = "#c8c8c8"

    def style_fn(feat):
        v = feat["properties"].get(cluster_col)
        try:
            vi = int(v)
        except Exception:
            vi = -1
        col = color_map.get(vi, default_color)
        return {"fillOpacity": 0.75, "weight": 0, "color": "#00000000", "fillColor": col}

    folium.GeoJson(
        data=g.to_json(),
        name="Cluster (perfil urbano)",
        pane="vectors",
        style_function=style_fn,
        smooth_factor=1.2,
    ).add_to(m)
    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
    items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items)


def draw_isocronas_area(m, g: "gpd.GeoDataFrame"):
    if g is None or g.empty or folium is None:
        return
    cls = find_col(g.columns, "nova_class")
    if cls is None:
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

    def style_fn(feat):
        v = feat["properties"].get(cls)
        try:
            vi = int(v)
        except Exception:
            vi = -1
        color = lut.get(vi, ("Outros", "#c8c8c8"))[1]
        return {"fillOpacity": 0.65, "weight": 0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        data=g.to_json(),
        name="Área de influência de bairro",
        pane="vectors",
        style_function=style_fn,
        smooth_factor=1.2,
    ).add_to(m)
    legend_items = [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()]
    add_categorical_legend(m, "Área de influência de bairro (nova_class)", legend_items)


# ============================================================================
# App
# ============================================================================
def main() -> None:
    inject_css()

    # Header
    with st.container():
        c1, c2 = st.columns([1, 7])
        with c1:
            lp = get_logo_path()
            if lp:
                st.image(lp, width=140)
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

    a1, a2, a3, a4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])

    with a1:
        left, map_col = st.columns([1, 5], gap="small")

        with left:
            st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
            ui = left_controls()
            st.markdown("</div>", unsafe_allow_html=True)

        with map_col:
            if folium is None or st_folium is None or gpd is None:
                st.error("Instale `geopandas`, `folium` e `streamlit-folium`.")
                return

            # chave de invalidação de cache quando arquivos mudam
            mtime_key = _dir_mtime_key(ADM_DIR)

            # Carregamentos básicos (apenas limites para centralizar)
            limite_gdf = load_admin_layer(ui["limite"], mtime_key)
            setores_all = load_setores(mtime_key)  # para fallback de centro/escala

            # Centro do mapa
            center = (-23.55, -46.63)
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)
            elif setores_all is not None and len(setores_all) > 0:
                center = center_from_bounds(setores_all)

            fmap = make_satellite_map(center=center, zoom=11, tiles_opacity=0.5)
            inject_tooltip_css(fmap, font_px=36)
            inject_label_scaler(fmap, 14, 24, 9, 18)

            # Tolerância dinâmica e bbox do limite (para filtrar)
            tol_m = pick_tol_by_extent(limite_gdf, setores_all) if (limite_gdf is not None or setores_all is not None) else SIMPLIFY_TOL_DEFAULT
            bbox = bounds_tuple(limite_gdf)

            # Limites (contorno)
            if limite_gdf is not None:
                add_admin_outline(fmap, limite_gdf, ui["limite"], tol_m=tol_m)
                if ui["labels_on"]:
                    add_centroid_labels(fmap, limite_gdf)

            # Variável
            var = ui["variavel"]
            if var == "Área de influência de bairro":
                g = build_isocronas_layer(bbox, tol_m, mtime_key)
                if g is None:
                    st.info("Arquivo de isócronas não encontrado em 'limites_administrativos'.")
                else:
                    draw_isocronas_area(fmap, g)
            elif var == "Cluster (perfil urbano)":
                g = build_setores_cluster_layer("cluster", bbox, tol_m, mtime_key)
                if g is None:
                    st.info("Coluna de cluster não encontrada nos setores.")
                else:
                    draw_setores_cluster(fmap, g, find_col(g.columns, "cluster", "cluster_label", "label", "classe") or "cluster")
            else:
                mapping = {
                    "População (Pessoa/ha)": "populacao",
                    "Densidade demográfica (hab/ha)": "densidade_demografica",
                    "Variação de elevação média": "diferenca_elevacao",
                    "Elevação média": "elevacao",
                }
                col = mapping[var]
                g = build_setores_numeric_layer(col, bbox, tol_m, mtime_key)
                if g is None:
                    st.info(f"A coluna para '{var}' não foi encontrada nos setores.")
                else:
                    draw_setores_numeric(fmap, g, find_col(g.columns, col) or col, var)

            st_folium(fmap, use_container_width=True, height=850)

    with a2:
        st.info("Conteúdo a definir.")
    with a3:
        st.info("Conteúdo a definir.")
    with a4:
        st.info("Conteúdo a definir.")


if __name__ == "__main__":
    main()
