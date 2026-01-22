# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from unicodedata import normalize as _ud_norm
import base64
import math
import re
from decimal import Decimal

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Fail-fast: dependências geoespaciais obrigatórias
# ---------------------------------------------------------------------------
def _import_geostack():
    try:
        import geopandas as gpd  # type: ignore
        import folium            # type: ignore
        from branca.element import Element  # type: ignore
        from streamlit_folium import st_folium  # type: ignore
        return gpd, folium, Element, st_folium
    except ImportError as e:
        st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")
        st.error(
            f"Dependência geoespacial ausente: **{e.__class__.__name__}: {e}**.\n\n"
            "Instale os pacotes listados em `requirements.txt` e reinicie o app."
        )
        st.stop()

gpd, folium, Element, st_folium = _import_geostack()

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
ORANGE_RED_GRAD = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]
SIMPLIFY_TOL = 0.0005
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

# ============================================================================
# CSS
# ============================================================================
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
            html, body, .stApp {{ font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important; }}

            /* Header fixo e limpo */
            .pb-header {{
                background: var(--pb-navy);
                color:#fff;
                border-radius: 14px;
                padding: 14px 18px;
                display:flex; align-items:center; gap:16px;
                width: 100%;
                box-sizing: border-box;
            }}
            .pb-logo {{ height: 52px; width:auto; display:block; }}
            .pb-title {{ font-size: 1.9rem; font-weight: 700; letter-spacing: .2px; }}
            .pb-subtitle {{ opacity:.95; margin-top:2px; font-size: .95rem; }}

            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.10);
                box-shadow:0 1px 2px rgba(0,0,0,.04);
                border-radius:14px; padding:12px;
            }}
            .pb-card h4 {{ margin: 0 0 .6rem 0; }}
            .main .block-container {{ padding-top:.6rem; padding-bottom:.8rem; }}

            /* Tooltips do mapa */
            .leaflet-tooltip.pb-big-tooltip,
            .leaflet-tooltip.pb-big-tooltip * {{
                font-size: 28px !important;
                font-weight: 800 !important;
                color: #111 !important;
                line-height: 1 !important;
            }}
            .leaflet-tooltip.pb-big-tooltip {{
                background:#fff !important; border: 2px solid #222 !important; border-radius: 10px !important;
                padding: 10px 14px !important; white-space: nowrap !important; pointer-events: none !important;
                box-shadow: 0 2px 6px rgba(0,0,0,.2) !important; z-index: 200000 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# Paths e utilidades
# ============================================================================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (REPO_ROOT / "limites_administrativos") if (REPO_ROOT / "limites_administrativos").exists() else REPO_ROOT

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _first_parquet_by_stems(folder: Path, stems: List[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in stems}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)

def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
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

def to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        return s.apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype("Float64")
    return pd.to_numeric(s, errors="coerce").astype("Float64")

# ============================================================================
# Logo: lê de assets e retorna data URI (funciona local/deploy)
# ============================================================================
def get_logo_data_uri() -> str:
    names = [
        "logo.png", "logo.svg", "logo.jpg", "logo.jpeg", "logo.webp",
        "planbairros.png", "planbairros.svg", "plan_bairros.png", "brand.png"
    ]
    assets = REPO_ROOT / "assets"
    if assets.exists():
        for n in names:
            p = assets / n
            if p.exists():
                mime = "image/" + p.suffix.lower().lstrip(".")
                b64 = base64.b64encode(p.read_bytes()).decode()
                return f"data:{mime};base64,{b64}"
    # fallback seguro
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"

# ============================================================================
# Cache de leitura
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_gdf(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if path is None:
        return None
    try:
        gdf = gpd.read_parquet(path)
        gdf = gdf if gdf.crs is not None else gdf.set_crs(4326)
        return gdf.to_crs(4326)
    except Exception:
        # fallback pandas + WKB/WKT
        try:
            pdf = pd.read_parquet(path)
            geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
            if geom_col is None:
                return None
            from shapely import wkb, wkt
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith("POLY").any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            return gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
        except Exception:
            return None

def load_setores() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_by_stems(DATA_DIR, ["SetoresCensitarios2023", "SetoresCensitarios"])
    return read_gdf(p) if p else None

def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_by_stems(DATA_DIR, ["isocronas", "isócronas", "isocronas2023"])
    return read_gdf(p) if p else None

def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {
        "Distritos": ["Distritos"],
        "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
        "Subprefeitura": ["Subprefeitura", "subprefeitura"],
        "Isócronas": ["isocronas", "isócronas"],
    }.get(name, [name])
    p = _first_parquet_by_stems(DATA_DIR, stems)
    return read_gdf(p) if p else None

# ============================================================================
# Folium helpers
# ============================================================================
def make_satellite_map(center=(-23.55, -46.63), zoom=11, tiles_opacity=0.6):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap", name="OSM", overlay=False, control=False,
    ).add_to(m)
    try:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Esri Satellite",
            overlay=False, control=False, opacity=tiles_opacity,
        ).add_to(m)
    except Exception:
        pass
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    return m

def inject_label_scaler(m, min_px=14, max_px=26, min_zoom=9, max_zoom=18):
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

def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.2):
    line = gdf[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        line["geometry"] = line.geometry.boundary
    folium.GeoJson(
        data=line.to_json(),
        name=f"{layer_name} (contorno)",
        pane="vectors",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(m)

def add_centroid_labels(m, gdf):
    cols = {c.lower(): c for c in gdf.columns}
    name_col = cols.get("ds_nome") or cols.get("sp_nome") or cols.get("nome")
    if not name_col:
        return
    try:
        reps = gdf.to_crs(4326).representative_point()
    except Exception:
        return
    for name, pt in zip(gdf[name_col].astype(str), reps):
        if pt is None or pt.is_empty:
            continue
        html = (
            "<div class='pb-static-label' "
            "style=\"font: 600 12px/1 Roboto, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;"
            "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
            f"{name}</div>"
        )
        folium.Marker(
            location=[pt.y, pt.x], icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
            z_index_offset=1000,
        ).add_to(m)

def add_gradient_legend(m, title: str, vmin: float, vmax: float, colors: list[str]):
    gradient_css = f"background: linear-gradient(to top, {', '.join(colors)});"
    html = f"""
    <div style="position: absolute; bottom: 24px; left: 16px; z-index: 9999;
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
# Pintura
# ============================================================================
def ramp_color(v: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "#c8c8c8"
    t = 0.0 if vmax == vmin else (float(v) - vmin) / (vmax - vmin)
    t = min(1.0, max(0.0, t))
    n = len(colors) - 1
    i = min(int(t * n), n - 1)
    frac = (t * n) - i
    def h2r(h): h=h.lstrip("#"); return tuple(int(h[k:k+2], 16) for k in (0,2,4))
    def r2h(r): return "#{:02x}{:02x}{:02x}".format(*r)
    c1, c2 = h2r(colors[i]), h2r(colors[i+1])
    mix = tuple(int(c1[k] + frac*(c2[k]-c1[k])) for k in range(3))
    return r2h(mix)

def paint_setores_numeric(m, setores: "gpd.GeoDataFrame", value_col: str, label: str, colors: list[str] = ORANGE_RED_GRAD):
    s = to_float_series(setores[value_col])
    vmin, vmax = float(s.min()), float(s.max())
    df = setores[["geometry"]].copy()
    df["__v__"] = s
    def style_fn(feat):
        v = feat["properties"].get("__v__")
        return {"fillOpacity": 0.8, "weight": 0.0, "color": "#00000000",
                "fillColor": ramp_color(v, vmin, vmax, colors)}
    folium.GeoJson(
        data=df[["geometry", "__v__"]].to_json(),
        name=f"{label}", pane="vectors", style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["__v__"], aliases=[label + ": "], sticky=True, labels=False, class_name="pb-big-tooltip"
        ),
    ).add_to(m)
    add_gradient_legend(m, label, vmin, vmax, colors)

def paint_setores_cluster(m, setores: "gpd.GeoDataFrame", cluster_col: str):
    color_map = {0:"#bf7db2", 1:"#f7bd6a", 2:"#cf651f", 3:"#ede4e6", 4:"#793393"}
    label_map = {
        0:"1 - Periférico com predominância residencial de alta densidade construtiva",
        1:"2 - Uso misto de média densidade construtiva",
        2:"3 - Periférico com predominância residencial de média densidade construtiva",
        3:"4 - Verticalizado de uso-misto",
        4:"5 - Predominância de uso comercial e serviços",
    }
    default_color = "#c8c8c8"
    df = setores[["geometry"]].copy()
    c = setores[cluster_col]
    if c.dtype == "object":
        c = c.apply(lambda x: int(x) if isinstance(x, (Decimal, float, int, str)) and str(x).strip().isdigit() else None)
    else:
        c = pd.to_numeric(c, errors="coerce").astype("Int64")
    df["__c__"] = c
    def style_fn(feat):
        v = feat["properties"].get("__c__")
        col = color_map.get(int(v) if v is not None and not pd.isna(v) else -1, default_color)
        return {"fillOpacity": 0.75, "weight": 0.0, "color": "#00000000", "fillColor": col}
    folium.GeoJson(
        data=df[["geometry", "__c__"]].to_json(),
        name="Cluster (perfil urbano)", pane="vectors", style_function=style_fn,
    ).add_to(m)
    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
    items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items)

def paint_isocronas_area(m, iso: "gpd.GeoDataFrame"):
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
    k = pd.to_numeric(iso[cls], errors="coerce").astype("Int64")
    df["__k__"] = k
    def style_fn(feat):
        v = feat["properties"].get("__k__")
        color = lut.get(int(v) if v is not None and not pd.isna(v) else -1, ("Outros", "#c8c8c8"))[1]
        return {"fillOpacity": 0.65, "weight": 0.0, "color": "#00000000", "fillColor": color}
    folium.GeoJson(
        data=df[["geometry", "__k__"]].to_json(),
        name="Área de influência de bairro", pane="vectors", style_function=style_fn,
    ).add_to(m)
    items = [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()]
    add_categorical_legend(m, "Área de influência de bairro (nova_class)", items)

# ============================================================================
# UI – filtros
# ============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [
            PLACEHOLDER_VAR,
            "População (Pessoa/ha)",
            "Densidade demográfica (hab/ha)",
            "Variação de elevação média",
            "Elevação média",
            "Cluster (perfil urbano)",
            "Área de influência de bairro",
        ],
        index=0, key="pb_var", placeholder="Escolha…",
        help="Variáveis numéricas são pintadas por setor. 'Área de influência' usa isócronas.",
    )
    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        [PLACEHOLDER_LIM, "Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0, key="pb_limite", placeholder="Escolha…",
        help="Exibe o contorno do limite selecionado.",
    )
    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on",
        help="Desenha rótulos centróides com fonte proporcional ao zoom.",
    )

    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo. Selecione novamente a camada/variável.")

    # limpa cache quando seleção muda
    sel_now = {"var": var, "lim": limite}
    sel_prev = st.session_state.get("_pb_prev_sel")
    if sel_prev and (sel_prev["var"] != sel_now["var"] or sel_prev["lim"] != sel_now["lim"]):
        st.cache_data.clear()
    st.session_state["_pb_prev_sel"] = sel_now

    return {"variavel": var, "limite": limite, "labels_on": labels_on}

# ============================================================================
# App
# ============================================================================
def main() -> None:
    inject_css()

    # ---------- Cabeçalho (logo em assets) ----------
    logo_uri = get_logo_data_uri()
    st.markdown(
        f"""
        <div class="pb-header">
            <img class="pb-logo" src="{logo_uri}" alt="PlanBairros logo"/>
            <div>
                <div class="pb-title">PlanBairros</div>
                <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout
    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        center = (-23.55, -46.63)
        fmap = make_satellite_map(center=center, zoom=11, tiles_opacity=0.6)

        limite_gdf = None
        setores = None
        iso = None

        if ui["limite"] != PLACEHOLDER_LIM:
            limite_gdf = load_admin_layer(ui["limite"])
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)

        if ui["variavel"] != PLACEHOLDER_VAR:
            if ui["variavel"] == "Área de influência de bairro":
                iso = load_isocronas()
                if iso is not None and len(iso) > 0 and ui["limite"] == PLACEHOLDER_LIM:
                    center = center_from_bounds(iso)
            else:
                setores = load_setores()
                if setores is not None and len(setores) > 0 and ui["limite"] == PLACEHOLDER_LIM:
                    center = center_from_bounds(setores)

        if center != (-23.55, -46.63):
            fmap.location = center

        inject_label_scaler(fmap, min_px=14, max_px=26, min_zoom=9, max_zoom=18)

        if limite_gdf is not None and len(limite_gdf) > 0:
            add_admin_outline(fmap, limite_gdf, ui["limite"])
            if ui["labels_on"]:
                add_centroid_labels(fmap, limite_gdf)
        elif ui["limite"] != PLACEHOLDER_LIM:
            st.info("Camada de limite não encontrada no diretório 'limites_administrativos/'.")

        var = ui["variavel"]
        if var == PLACEHOLDER_VAR:
            st.info("Selecione uma variável e/ou um limite administrativo para iniciar a visualização.")
        elif var == "Área de influência de bairro":
            if iso is None or len(iso) == 0:
                st.info("Isócronas não encontradas em 'limites_administrativos/'.")
            else:
                paint_isocronas_area(fmap, iso)
        elif var == "Cluster (perfil urbano)":
            if setores is None or len(setores) == 0:
                st.info("Setores não encontrados em 'limites_administrativos/'.")
            else:
                cl = find_col(setores.columns, "cluster", "cluster_label", "label", "classe")
                if cl:
                    paint_setores_cluster(fmap, setores, cl)
                else:
                    st.info("Coluna de cluster não encontrada nos setores.")
        else:
            if setores is None or len(setores) == 0:
                st.info("Setores não encontrados em 'limites_administrativos/'.")
            else:
                mapping = {
                    "População (Pessoa/ha)": "populacao",
                    "Densidade demográfica (hab/ha)": "densidade_demografica",
                    "Variação de elevação média": "diferenca_elevacao",
                    "Elevação média": "elevacao",
                }
                col = find_col(setores.columns, mapping.get(var))
                if col:
                    paint_setores_numeric(fmap, setores, col, var, ORANGE_RED_GRAD)
                else:
                    st.info(f"A coluna para '{var}' não foi encontrada nos setores.")

        st_folium(fmap, height=780, use_container_width=True, key="map_view")


if __name__ == "__main__":
    main()
