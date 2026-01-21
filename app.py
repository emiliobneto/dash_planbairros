# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
from unicodedata import normalize as _ud_norm
import re

import pandas as pd
import streamlit as st

# Geo libs (opcionais – o app funciona sem, só não mostra o mapa)
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:  # noqa: BLE001
    gpd = None  # type: ignore
    folium = None  # type: ignore
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

# Gradiente para variáveis CONTÍNUAS (sem azul/verde) → amarelo→laranja→telha
VAR_GRADIENT = ["#F4DD63", "#D58243", "#C65534"]  # amarelo, laranja, telha

# Paleta CLUSTERS (0..4) + outros
CLUSTER_COLORS = {
    0: "#bf7db2",
    1: "#f7bd6a",
    2: "#cf651f",
    3: "#ede4e6",
    4: "#793393",
    "outros": "#c8c8c8",
}
CLUSTER_LABELS = {
    0: "1 - Periférico com predominância residencial de alta densidade construtiva",
    1: "2 - Uso misto de média densidade construtiva",
    2: "3 - Periférico com predominância residencial de média densidade construtiva",
    3: "4 - Verticalizado de uso-misto",
    4: "5 - Predominancia de uso comercial e serviços",
}

# Paleta ÁREA DE INFLUÊNCIA (nova_class 0..9)
AIB_COLORS = {
    0: ("#542788", "Predominância uso misto"),
    1: ("#f7f7f7", "Zona de transição local"),
    2: ("#d8daeb", "Periférico residencial de média densidade"),
    3: ("#b35806", "Transição central verticalizada"),
    4: ("#b2abd2", "Periférico adensado em transição"),
    5: ("#8073ac", "Centralidade comercial e de serviços"),
    6: ("#fdb863", "Predominância residencial média densidade"),
    7: ("#7f3b08", "Áreas íngremes e de encosta"),
    8: ("#e08214", "Alta densidade residencial"),
    9: ("#fee0b6", "Central verticalizado"),
}


# ============================================================================
# CSS / UI
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
            html, body, .stApp {{
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background: var(--pb-navy); color: #fff; border-radius: 18px;
                padding: 20px 24px; min-height: 110px; display: flex; align-items: center; gap: 16px;
            }}
            .pb-title {{ font-size: 2.6rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.2rem; }}

            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04);
                border-radius:16px; padding:18px;
            }}
            .pb-card h4 {{ font-size: 2.0rem !important; margin: 0 0 .6rem 0; }}
            .pb-card label, .pb-card .stMarkdown p {{ font-size: 1.9rem !important; font-weight: 800 !important; }}
            .pb-card div[role="combobox"] {{ font-size: 1.7rem !important; min-height: 58px !important; }}
            .pb-card [data-baseweb="select"] * {{ font-size: 1.7rem !important; }}
            .pb-card .stSelectbox svg {{ transform: scale(1.7); }}

            .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
                background:transparent; border-bottom:3px solid transparent; font-weight:700; font-size: 1.1rem;
            }}
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
                border-bottom:3px solid var(--pb-teal) !important; color:var(--pb-navy) !important;
            }}
            .main .block-container {{ padding-top: .6rem; padding-bottom: .6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_leaflet_css(m, tooltip_px: int = 28):
    """CSS dentro do mapa para tooltips dos GeoJson (hover)."""
    if folium is None:
        return
    css = f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip,
      .leaflet-tooltip.pb-big-tooltip * {{
        font-size: {tooltip_px}px !important;
        font-weight: 800 !important;
        color: #111 !important;
        line-height: 1.05 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background: #fff !important;
        border: 2px solid #222 !important; border-radius: 8px !important;
        padding: 10px 14px !important; white-space: nowrap !important;
        pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.25) !important;
        z-index: 200000 !important;
      }}
    </style>
    """
    from folium import Element  # type: ignore
    m.get_root().html.add_child(Element(css))


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
# Caminhos / leitura de dados
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_dir(subdir: str) -> Path:
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / subdir
        if p.exists():
            return p
    return REPO_ROOT / subdir


ADM_DIR = _resolve_dir("limites_administrativos")

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def _first_parquet_matching(folder: Path, names: List[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in names}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.stat().st_size < 1024:  # ignora LFS/placeholder
            return None
    except Exception:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_gdf_from_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        gdf = gpd.read_parquet(path)
        try:
            gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
        except Exception:
            pass
        return gdf
    except Exception:
        pdf = _safe_read_parquet(path)
        if pdf is None:
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


# Names fixos conhecidos
ADMIN_NAME_MAP = {
    # (1) REMOVIDO SetoresCensitarios dos LIMITES
    "Distritos": ["Distritos"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
}
SETORES_NAMES = ["SetoresCensitarios2023", "SetoresCensitarios"]
ISO_NAMES = ["isocronas", "isócronas", "isocronas2023", "isocronas_parquet"]


@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str) -> Optional["gpd.GeoDataFrame"]:
    if layer_name not in ADMIN_NAME_MAP:
        return None
    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP[layer_name])
    return load_gdf_from_parquet(path) if path else None


@st.cache_data(show_spinner=False)
def load_setores() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_matching(ADM_DIR, SETORES_NAMES)
    if not p:
        return None
    return load_gdf_from_parquet(p)


@st.cache_data(show_spinner=False)
def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_matching(ADM_DIR, ISO_NAMES)
    if not p:
        return None
    return load_gdf_from_parquet(p)


# ============================================================================
# Helpers folium: mapa e legendas
# ============================================================================
SIMPLIFY_TOL = 0.0005  # ~55m


def make_satellite_map(center=(-23.55, -46.63), zoom=10, tiles_opacity=0.5):
    if folium is None:
        st.error("Instale `folium` e `streamlit-folium` para exibir o mapa.")
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass

    # Esri Imagery (sem rótulos) para não conflitar com tooltips/rótulos
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=False,
        opacity=tiles_opacity,
    ).add_to(m)
    return m


def inject_leaflet_css(m, tooltip_px: int = 28):
    """CSS do tooltip (hover)."""
    if folium is None:
        return
    css = f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip,
      .leaflet-tooltip.pb-big-tooltip * {{
        font-size: {tooltip_px}px !important;
        font-weight: 800 !important;
        color: #111 !important;
        line-height: 1.05 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background: #fff !important;
        border: 2px solid #222 !important; border-radius: 8px !important;
        padding: 10px 14px !important; white-space: nowrap !important;
        pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.25) !important;
        z-index: 200000 !important;
      }}
    </style>
    """
    from folium import Element  # type: ignore
    m.get_root().html.add_child(Element(css))


def inject_label_scaler(m, min_px=16, max_px=26, min_zoom=9, max_zoom=18):
    """
    Redimensiona '.pb-static-label' conforme zoom do Leaflet.
    - min_px/max_px: tamanho da fonte (px) em min_zoom/max_zoom
    - min_zoom/max_zoom: faixa de zoom onde ocorre a transição
    """
    if folium is None:
        return
    from folium import Element  # type: ignore
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
          var px = Math.round(minPx + t * (maxPx - minPx));
          return px;
        }}
        function updatePBLabels() {{
          var z = {map_var}.getZoom();
          var px = scaleFont(z);
          var els = document.querySelectorAll('.pb-static-label');
          els.forEach(function(el) {{
            el.style.fontSize = px + 'px';
          }});
        }}
        {map_var}.on('zoomend', updatePBLabels);
        {map_var}.whenReady(updatePBLabels);
      }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))


def add_outline_layer(m, gdf, layer_name: str, color="#000000", weight=1.2):
    """Desenha contorno (linha) de camadas administrativas selecionadas."""
    if gdf is None or folium is None:
        return
    gf_line = gdf[["geometry"]].copy()
    try:
        gf_line["geometry"] = gf_line.geometry.boundary
        gf_line["geometry"] = gf_line.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass
    folium.GeoJson(
        data=gf_line.to_json(),
        name=f"{layer_name} (contorno)",
        pane="vectors",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(m)


def add_vertical_gradient_legend(m, title: str, colors: List[str], vmin: float, vmax: float, unit: str = ""):
    """Legenda vertical (régua) de gradiente."""
    if folium is None:
        return
    grad_css = f"linear-gradient(to top, {', '.join(colors)})"
    html = f"""
    <div style="
        position: absolute; bottom: 40px; right: 16px; z-index: 9999;
        background: rgba(255,255,255,0.95); padding: 10px 12px; border: 1px solid #aaa;
        border-radius: 8px; font-family: Roboto, Arial; font-size: 12px; color: #111;">
      <div style="font-weight: 700; margin-bottom: 6px;">{title}</div>
      <div style="display:flex; align-items:stretch; gap:10px;">
        <div style="width: 16px; height: 140px; background: {grad_css}; border: 1px solid #888;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height: 140px;">
          <div>{(vmax):,.2f}{(' ' + unit) if unit else ''}</div>
          <div>{(vmin):,.2f}{(' ' + unit) if unit else ''}</div>
        </div>
      </div>
    </div>
    """
    from folium import Element  # type: ignore
    m.get_root().html.add_child(Element(html))


def add_categorical_legend(m, title: str, items: List[tuple[str, str]]):
    """Legenda categórica (lista de cores e labels)."""
    if folium is None:
        return
    rows = "".join(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
          <div style="width:14px;height:14px;background:{color};border:1px solid #555;"></div>
          <div style="font-size:12px;">{label}</div>
        </div>
        """
        for color, label in items
    )
    html = f"""
    <div style="
        position: absolute; bottom: 40px; right: 16px; z-index: 9999;
        background: rgba(255,255,255,0.95); padding: 10px 12px; border: 1px solid #aaa;
        border-radius: 8px; font-family: Roboto, Arial; color: #111; max-width: 360px;">
      <div style="font-weight:700; margin-bottom:6px; font-size:12px;">{title}</div>
      <div>{rows}</div>
    </div>
    """
    from folium import Element  # type: ignore
    m.get_root().html.add_child(Element(html))


def add_centroid_labels(m, gdf, label_col: str):
    """Rótulos permanentes (dinâmicos por zoom) no representative_point()."""
    if folium is None:
        return
    try:
        centers = gdf.copy().to_crs(4326).representative_point()
    except Exception:
        return
    for name, pt in zip(gdf[label_col].astype(str), centers):
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


# ============================================================================
# Visualizações conforme requisitos (1–6)
# ============================================================================
def draw_variables_on_setores(
    m,
    setores: "gpd.GeoDataFrame",
    var_col: str,
    var_title: str,
    unit: str = "",
):
    """
    (2) Desenha variável contínua por Setor:
        - Apenas preenchimento (SEM linhas)
        - Gradiente VAR_GRADIENT (sem azul/verde)
        - Legenda vertical (régua)
    """
    if folium is None or setores is None or setores.empty:
        return

    df = setores.copy()
    if var_col not in df.columns:
        st.info(f"Coluna '{var_col}' não encontrada nos Setores Censitários.")
        return

    # Numérico e limpeza
    s = pd.to_numeric(df[var_col], errors="coerce")
    df[var_col] = s
    df = df.dropna(subset=[var_col])
    if df.empty:
        st.info(f"Sem dados válidos para '{var_col}'.")
        return

    # Simplify e gradiente
    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    vmin, vmax = float(s.min()), float(s.max())

    def ramp_color(val: float) -> str:
        if vmax == vmin:
            return VAR_GRADIENT[-1]
        t = (val - vmin) / (vmax - vmin)
        if t <= 0.5:
            a, b = VAR_GRADIENT[0], VAR_GRADIENT[1]
            tt = t / 0.5
        else:
            a, b = VAR_GRADIENT[1], VAR_GRADIENT[2]
            tt = (t - 0.5) / 0.5

        def _hex_to_rgb(h: str):
            h = h.lstrip("#")
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

        def _rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        ra, ga, ba = _hex_to_rgb(a)
        rb, gb, bb = _hex_to_rgb(b)
        rc = int(ra + tt * (rb - ra))
        gc = int(ga + tt * (gb - ga))
        bc = int(ba + tt * (bb - ba))
        return _rgb_to_hex((rc, gc, bc))

    def style_fn(feat):
        v = feat["properties"].get(var_col)
        if v is None:
            color = "#dddddd"
        else:
            color = ramp_color(float(v))
        return {"fillOpacity": 0.80, "weight": 0.0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        data=df.to_json(),
        name=var_title,
        pane="vectors",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[var_col],
            aliases=[var_title + ": "],
            sticky=True,
            labels=True,
            class_name="pb-big-tooltip",
            style=("background:#fff;color:#111;font-size:24px;font-weight:700;"
                   "border:1px solid #222;border-radius:8px;padding:8px 12px;"),
            max_width=1200,
        ),
    ).add_to(m)

    add_vertical_gradient_legend(m, title=var_title, colors=VAR_GRADIENT, vmin=vmin, vmax=vmax, unit=unit)


def draw_isocronas_limits(m, isoc: "gpd.GeoDataFrame"):
    """
    (5) Nos LIMITES: incluir isócronas como 'área de transição'
        (nova_class ∈ {1,3,4,6}) em #836e60 + legenda.
    """
    if folium is None or isoc is None or isoc.empty:
        return

    df = isoc.copy()
    if "nova_class" not in df.columns:
        st.info("`nova_class` não encontrada em isócronas.")
        return

    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    mask_trans = df["nova_class"].isin([1, 3, 4, 6])
    if not mask_trans.any():
        return

    df_trans = df.loc[mask_trans, ["nova_class", "geometry"]].copy()

    def style_fn(feat):
        return {"fillOpacity": 0.25, "weight": 0.0, "color": "#00000000", "fillColor": "#836e60"}

    folium.GeoJson(
        data=df_trans.to_json(),
        name="Área de transição (isócronas)",
        pane="vectors",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["nova_class"],
            aliases=["Nova Class: "],
            sticky=True,
            labels=True,
            class_name="pb-big-tooltip",
            style=("background:#fff;color:#111;font-size:22px;font-weight:700;"
                   "border:1px solid #222;border-radius:8px;padding:8px 12px;"),
            max_width=1200,
        ),
    ).add_to(m)

    add_categorical_legend(m, "Área de influência de bairro (limites)", [("#836e60", "Área de transição (1,3,4,6)")])
    # Demais classes não preenchidas (foco: transição)


def draw_isocronas_variable(m, isoc: "gpd.GeoDataFrame"):
    """
    (6) Em VARIÁVEIS: usar isócronas com `nova_class` renomeada para
        'Área de influência de bairro' + paleta categórica AIB_COLORS + legenda.
    """
    if folium is None or isoc is None or isoc.empty:
        return

    df = isoc.copy()
    if "nova_class" not in df.columns:
        st.info("`nova_class` não encontrada em isócronas.")
        return

    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    def style_fn(feat):
        v = feat["properties"].get("nova_class")
        try:
            v = int(v)
        except Exception:
            v = None
        color = AIB_COLORS.get(v, ("#c8c8c8", "Outros"))[0]
        return {"fillOpacity": 0.80, "weight": 0.0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        data=df.to_json(),
        name="Área de influência de bairro",
        pane="vectors",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["nova_class"],
            aliases=["Área de influência de bairro: "],
            sticky=True,
            labels=True,
            class_name="pb-big-tooltip",
            style=("background:#fff;color:#111;font-size:24px;font-weight:700;"
                   "border:1px solid #222;border-radius:8px;padding:8px 12px;"),
            max_width=1200,
        ),
    ).add_to(m)

    items = [(hex_, label) for code, (hex_, label) in AIB_COLORS.items()]
    add_categorical_legend(m, "Área de influência de bairro", items)


def draw_clusters(m, setores: "gpd.GeoDataFrame", cluster_col: str = "cluster"):
    """
    (3) e (4) Aba Cluster:
        - Cores fixas por classe (0..4), outros cinza
        - Legenda com rótulos 1..5 (label encoding)
    """
    if folium is None or setores is None or setores.empty:
        return
    df = setores.copy()
    if cluster_col not in df.columns:
        # tenta nome aproximado
        c2 = next((c for c in df.columns if re.match(r"^cluster(_id)?$", c, re.I)), None)
        if c2 is None:
            st.info("Coluna de cluster não encontrada nos Setores.")
            return
        cluster_col = c2

    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    def style_fn(feat):
        v = feat["properties"].get(cluster_col)
        try:
            v = int(v)
        except Exception:
            v = "outros"
        color = CLUSTER_COLORS.get(v, CLUSTER_COLORS["outros"])
        return {"fillOpacity": 0.85, "weight": 0.0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        data=df.to_json(),
        name="Clusters",
        pane="vectors",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[cluster_col],
            aliases=["Cluster: "],
            sticky=True,
            labels=True,
            class_name="pb-big-tooltip",
            style=("background:#fff;color:#111;font-size:24px;font-weight:700;"
                   "border:1px solid #222;border-radius:8px;padding:8px 12px;"),
            max_width=1200,
        ),
    ).add_to(m)

    items = []
    for k in [0, 1, 2, 3, 4]:
        items.append((CLUSTER_COLORS[k], CLUSTER_LABELS[k]))
    items.append((CLUSTER_COLORS["outros"], "Outros"))
    add_categorical_legend(m, "Clusters", items)


# ============================================================================
# UI (cabeçalho, abas e controles)
# ============================================================================
def build_header():
    with st.container():
        c1, c2 = st.columns([1, 7])
        with c1:
            # logo (best-effort)
            for p in [
                "assets/logo_todos.jpg", "assets/logo_paleta.jpg",
                "logo_todos.jpg", "logo_paleta.jpg",
                "/mnt/data/logo_todos.jpg", "/mnt/data/logo_paleta.jpg",
            ]:
                if Path(p).exists():
                    st.image(p, width=140)
                    break
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


def build_controls(key_prefix: str) -> tuple[str, bool]:
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Configurações</h4>", unsafe_allow_html=True)

    # (1) REMOVIDO Setores Censitários dos LIMITES
    limite = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "ZonasOD2023", "Subprefeitura"],
        index=0,
        key=f"{key_prefix}pb_limite",
        help="Escolha a desagregação para contorno.",
    )
    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)",
        value=False,
        key=f"{key_prefix}pb_labels",
        help="Mostra nomes fixos nos centróides e ajusta o tamanho conforme zoom.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return limite, labels_on


# ============================================================================
# App
# ============================================================================
def main() -> None:
    inject_css()
    build_header()
    st.write("")

    # Abas principais: Variáveis & Cluster (mais 2 placeholders)
    tab_vars, tab_cluster, tab3, tab4 = st.tabs(["Variáveis", "Cluster", "Aba 3", "Aba 4"])

    # Carregamentos base (setores/isócronas)
    gdf_setores = load_setores()
    gdf_iso = load_isocronas()

    # ====== Aba VARIÁVEIS ======
    with tab_vars:
        st.markdown("### Variáveis (Setores Censitários e Isócronas)")

        # (2.1) Labels de variáveis corrigidos
        VAR_OPTIONS = {
            "populacao": "População (Pessoa/hec)",
            "densidade_demografica": "Densidade demográfica (hab/he)",
            "diferenca_elevacao": "Variação de elevação média",
            "elevacao": "Elevação média",
            "aib": "Área de influência de bairro",  # isócronas.nova_class
        }
        var_key = st.selectbox(
            "Selecione a variável",
            list(VAR_OPTIONS.keys()),
            format_func=lambda k: VAR_OPTIONS[k],
            index=0,
            key="vars_var_sel",
        )

        # Controles (com prefixo de chave exclusivo para evitar DuplicateWidgetID)
        limite, labels_on = build_controls("vars_")

        # Mapa
        if folium is None or st_folium is None:
            st.info("Para o mapa, instale `folium` e `streamlit-folium`.")
        else:
            center = (-23.55, -46.63)
            gdf_lim = load_admin_layer(limite)
            if gdf_lim is not None and len(gdf_lim) > 0:
                minx, miny, maxx, maxy = gdf_lim.total_bounds
                center = ((miny + maxy) / 2, (minx + maxx) / 2)

            fmap = make_satellite_map(center=center, zoom=10, tiles_opacity=0.5)
            inject_leaflet_css(fmap, tooltip_px=28)
            inject_label_scaler(fmap, min_px=16, max_px=26, min_zoom=9, max_zoom=18)

            # contorno dos limites selecionados
            if gdf_lim is not None:
                add_outline_layer(fmap, gdf_lim, layer_name=limite, color="#000", weight=1.2)

                # rótulos (Distritos/Subprefeitura)
                if labels_on:
                    cols_lower = {c.lower(): c for c in gdf_lim.columns}
                    label_col = cols_lower.get("ds_nome") or cols_lower.get("sp_nome")
                    if label_col:
                        add_centroid_labels(fmap, gdf_lim, label_col)

            # (5) sobrepor isócronas como “área de transição” nos LIMITES
            if gdf_iso is not None:
                draw_isocronas_limits(fmap, gdf_iso)

            # (2) variáveis contínuas por Setor (preenchimento) OU (6) AIB por isócronas
            if var_key != "aib":
                if gdf_setores is None:
                    st.info("Setores Censitários não encontrados.")
                else:
                    VAR_MAP = {
                        "populacao": ("populacao", "Pessoa/hec"),
                        "densidade_demografica": ("densidade_demografica", "hab/he"),
                        "diferenca_elevacao": ("diferenca_elevacao", ""),
                        "elevacao": ("elevacao", ""),
                    }
                    col, unit = VAR_MAP[var_key]
                    draw_variables_on_setores(
                        fmap, gdf_setores, var_col=col, var_title=VAR_OPTIONS[var_key], unit=unit
                    )
            else:
                if gdf_iso is None:
                    st.info("Camada de isócronas não encontrada para 'Área de influência de bairro'.")
                else:
                    draw_isocronas_variable(fmap, gdf_iso)

            st_folium(fmap, use_container_width=True, height=820)

    # ====== Aba CLUSTER ======
    with tab_cluster:
        st.markdown("### Clusters (Setores Censitários)")

        limite, labels_on = build_controls("clus_")

        if folium is None or st_folium is None:
            st.info("Para o mapa, instale `folium` e `streamlit-folium`.")
        else:
            center = (-23.55, -46.63)
            gdf_lim = load_admin_layer(limite)
            if gdf_lim is not None and len(gdf_lim) > 0:
                minx, miny, maxx, maxy = gdf_lim.total_bounds
                center = ((miny + maxy) / 2, (minx + maxx) / 2)

            fmap = make_satellite_map(center=center, zoom=10, tiles_opacity=0.5)
            inject_leaflet_css(fmap, tooltip_px=28)
            inject_label_scaler(fmap, min_px=16, max_px=26, min_zoom=9, max_zoom=18)

            if gdf_lim is not None:
                add_outline_layer(fmap, gdf_lim, layer_name=limite, color="#000", weight=1.2)
                if labels_on:
                    cols_lower = {c.lower(): c for c in gdf_lim.columns}
                    label_col = cols_lower.get("ds_nome") or cols_lower.get("sp_nome")
                    if label_col:
                        add_centroid_labels(fmap, gdf_lim, label_col)

            if gdf_setores is None:
                st.info("Setores Censitários não encontrados.")
            else:
                draw_clusters(fmap, gdf_setores, cluster_col="cluster")

            # (5) opcional: manter a sobreposição de transição das isócronas aqui também
            if gdf_iso is not None:
                draw_isocronas_limits(fmap, gdf_iso)

            st_folium(fmap, use_container_width=True, height=820)

    # Abas 3/4 (placeholders)
    with tab3:
        st.write("Conteúdo a definir.")
    with tab4:
        st.write("Conteúdo a definir.")


if __name__ == "__main__":
    main()
