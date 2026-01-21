# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from unicodedata import normalize as _ud_norm
import math
import re

import pandas as pd
import streamlit as st

# Geo libs (opcionais – o app funciona sem, só não mostra o mapa)
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
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

# Gradiente contínuo de alto contraste (sem azul/verde)
# amarelo claro -> laranja -> marrom
HIGH_CONTRAST = ["#fff7bc", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404"]


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
                padding: 20px 24px; min-height: 94px; display: flex; align-items: center; gap: 16px;
            }}
            .pb-title {{ font-size: 2.0rem; line-height: 1.15; font-weight: 700; letter-spacing: .5px; }}
            .pb-subtitle {{ opacity: .95; margin-top: 4px; font-size: 1.00rem; }}

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
# Caminhos e loaders ROBUSTOS (on-demand)
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _list_parquets(d: Path) -> list[str]:
    try:
        return sorted([p.name for p in d.glob("*.parquet")])
    except Exception:
        return []

def _find_parquet_anywhere(names: list[str]) -> Optional[Path]:
    """Procura por nomes-alvo (sem extensão) em locais comuns do deploy."""
    targets = {_slug(n) for n in names}
    search_dirs = [
        REPO_ROOT,
        REPO_ROOT / "limites_administrativos",
        REPO_ROOT / "dash_planbairros",
        REPO_ROOT / "dash_planbairros" / "limites_administrativos",
        Path.cwd(),
        Path.cwd() / "limites_administrativos",
    ]
    seen = set()
    for base in search_dirs:
        if not base.exists() or base in seen:
            continue
        seen.add(base)
        for p in base.glob("**/*.parquet"):
            if _slug(p.stem) in targets:
                return p
    return None

def _read_gdf_parquet(path: Path) -> "gpd.GeoDataFrame | None":
    if gpd is None:
        st.error("Geopandas não está disponível no ambiente.")
        return None
    try:
        gdf = gpd.read_parquet(path)
    except Exception as exc:
        st.warning(f"Falha ao ler {path.name}: {exc}")
        return None
    try:
        gdf = gdf if gdf.crs is not None else gdf.set_crs(4326)
        # Se não for 4326, converte
        if str(gdf.crs).lower() not in ("epsg:4326", "epsg: 4326", "wgs84", "wgs 84"):
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def _dir_signature() -> tuple:
    """Assinatura simples para invalidar cache quando os arquivos mudam."""
    dirs = [
        REPO_ROOT / "limites_administrativos",
        REPO_ROOT / "dash_planbairros" / "limites_administrativos",
        Path.cwd() / "limites_administrativos",
    ]
    listing = []
    for d in dirs:
        if d.exists():
            listing.extend([d.as_posix()] + _list_parquets(d))
    return tuple(listing)

@st.cache_data(show_spinner=False)
def load_setores(sig: tuple) -> "gpd.GeoDataFrame | None":
    path = _find_parquet_anywhere(["SetoresCensitarios2023", "SetoresCensitários2023"])
    if path is None:
        st.warning("Arquivo 'SetoresCensitarios2023.parquet' não encontrado.")
        _ = _dir_signature()  # força exibir nas próximas chamadas
        return None
    return _read_gdf_parquet(path)

@st.cache_data(show_spinner=False)
def load_isocronas(sig: tuple) -> "gpd.GeoDataFrame | None":
    path = _find_parquet_anywhere(["isocronas", "isócronas"])
    if path is None:
        st.info("Arquivo de isócronas não encontrado.")
        return None
    return _read_gdf_parquet(path)

@st.cache_data(show_spinner=False)
def load_admin_layer(name: str, sig: tuple) -> "gpd.GeoDataFrame | None":
    # Sem setores aqui — apenas limites
    alias = {
        "Distritos": ["Distritos"],
        "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
        "Subprefeitura": ["subprefeitura", "Subprefeituras"],
        "Isócronas": ["isocronas", "isócronas"],
    }.get(name, [name])
    path = _find_parquet_anywhere(alias)
    return _read_gdf_parquet(path) if path else None


# ============================================================================
# Utilidades
# ============================================================================
def find_col(df_cols, *cands) -> Optional[str]:
    """Localiza coluna por lista de candidatos (case-insensitive + normalização)."""
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


# ============================================================================
# UI (esquerda) + mapa (direita)
# ============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
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
        index=1,
        key="pb_var",
        help="Variáveis numéricas pintam os Setores; 'Área de influência' usa Isócronas.",
    )
    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0,
        key="pb_limite",
        help="Exibe contorno do limite selecionado (Setores foram movidos para 'Variáveis').",
    )
    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)",
        value=False,
        key="pb_labels_on",
        help="Rotula centróides com fonte proporcional ao zoom.",
    )
    return {"variavel": var, "limite": limite, "labels_on": labels_on}


# ============================================================================
# Folium – mapa, camadas e legendas
# ============================================================================
SIMPLIFY_TOL = 0.00045  # equilíbrio velocidade/forma

def make_satellite_map(center=(-23.55, -46.63), zoom=11, tiles_opacity=0.5):
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

def inject_tooltip_css(m, font_px: int = 48):
    if Element is None:
        return
    css = f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip,
      .leaflet-tooltip.pb-big-tooltip * {{
        font-size: {font_px}px !important;
        font-weight: 800 !important;
        color: #111 !important;
        line-height: 1 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background: #fff !important;
        border: 2px solid #222 !important; border-radius: 10px !important;
        padding: 10px 14px !important; white-space: nowrap !important;
        pointer-events: none !important; box-shadow: 0 2px 6px rgba(0,0,0,.2) !important;
        z-index: 200000 !important;
      }}
    </style>
    """
    m.get_root().html.add_child(Element(css))

def inject_label_scaler(m, min_px=16, max_px=26, min_zoom=9, max_zoom=18):
    """Redimensiona '.pb-static-label' conforme zoom."""
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

def add_admin_outline(m, gdf, layer_name: str, color="#000000", weight=1.15):
    if gdf is None or folium is None:
        return
    # contorno
    gf_line = gdf[["geometry"]].copy()
    try:
        gf_line["geometry"] = gf_line.geometry.boundary.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        gf_line["geometry"] = gf_line.geometry.boundary
    folium.GeoJson(
        data=gf_line.to_json(),
        name=f"{layer_name} (contorno)",
        pane="vectors",
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight},
    ).add_to(m)

    # Regra especial 5) – isócronas como "área de transição" nos limites
    if layer_name.lower().startswith("isócron") or layer_name.lower().startswith("isocron"):
        cls = find_col(gdf.columns, "nova_class")
        if cls:
            trans = gdf[gdf[cls].isin([1, 3, 4, 6])][["geometry"]].copy()
            if not trans.empty:
                try:
                    trans["geometry"] = trans.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
                except Exception:
                    pass
                folium.GeoJson(
                    data=trans.to_json(),
                    name="Área de transição",
                    pane="vectors",
                    style_function=lambda f: {"fillOpacity": 0.2, "color": "#836e60", "weight": 0.8},
                ).add_to(m)
                add_categorical_legend(m, "Limites — Isócronas", [("Área de transição (1,3,4,6)", "#836e60")], topright=True)

def add_centroid_labels(m, gdf):
    """Rótulos permanentes (tamanho dinâmico com inject_label_scaler)."""
    if folium is None:
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

# ---------- Legendas ----------
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


# ---------- Pintura por Setor (variáveis numéricas e cluster) ----------
def _to_float_series(s: pd.Series) -> pd.Series:
    """Converte valores (inclui Decimal/str) para float."""
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.to_numeric(s.astype(str), errors="coerce")

def ramp_color(value: float, vmin: float, vmax: float, colors: list[str]) -> str:
    """Interpolação linear simples sobre uma lista de cores hex."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#cccccc"
    t = 0.0 if vmax == vmin else (float(value) - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    n = len(colors) - 1
    i = min(int(t * n), n - 1)
    frac = (t * n) - i

    def hex_to_rgb(h):  # "#RRGGBB" -> tuple
        h = h.lstrip("#")
        return tuple(int(h[k:k+2], 16) for k in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    c1 = hex_to_rgb(colors[i])
    c2 = hex_to_rgb(colors[i + 1])
    rgb = tuple(int(c1[k] + frac * (c2[k] - c1[k])) for k in range(3))
    return rgb_to_hex(rgb)

def paint_setores_numeric(
    m, setores: "gpd.GeoDataFrame", value_col: str, label: str, colors: list[str] = HIGH_CONTRAST
):
    if folium is None or setores is None or value_col not in setores.columns:
        return
    df = setores[[value_col, "geometry"]].copy()
    # força float (corrige Decimal e afins)
    df["__v__"] = _to_float_series(df[value_col])
    df = df.dropna(subset=["__v__"])
    if df.empty:
        return
    vmin, vmax = float(df["__v__"].min()), float(df["__v__"].max())
    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    def style_fn(feat):
        v = feat["properties"].get("__v__")
        return {"fillOpacity": 0.78, "weight": 0.0, "color": "#00000000", "fillColor": ramp_color(v, vmin, vmax, colors)}

    folium.GeoJson(
        df.to_json(),
        name=f"{label}",
        pane="vectors",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["__v__"],
            aliases=[label + ": "],
            sticky=True,
            labels=False,
            class_name="pb-big-tooltip",
        ),
    ).add_to(m)
    add_gradient_legend(m, label, vmin, vmax, colors)

def paint_setores_cluster(m, setores: "gpd.GeoDataFrame", cluster_col: str):
    """Regra 3 e 4 — cores + legenda nominal (com fallback para 'Outros')."""
    if folium is None or setores is None or cluster_col not in setores.columns:
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
    df = setores[[cluster_col, "geometry"]].copy()
    # força inteiro (corrige Decimal/str)
    df["__c__"] = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    def style_fn(feat):
        v = feat["properties"].get("__c__")
        col = color_map.get(int(v) if v is not None and not pd.isna(v) else -1, default_color)
        return {"fillOpacity": 0.75, "weight": 0.0, "color": "#00000000", "fillColor": col}

    folium.GeoJson(
        df.to_json(),
        name="Cluster (perfil urbano)",
        pane="vectors",
        style_function=style_fn,
    ).add_to(m)

    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
    items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items)

def paint_isocronas_area(m, iso: "gpd.GeoDataFrame"):
    """Regra 6 — variável 'Área de influência de bairro' via nova_class."""
    if folium is None or iso is None:
        return
    cls = find_col(iso.columns, "nova_class")
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
    df = iso[[cls, "geometry"]].copy()
    df["__k__"] = pd.to_numeric(df[cls], errors="coerce").astype("Int64")
    try:
        df["geometry"] = df.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception:
        pass

    def style_fn(feat):
        v = feat["properties"].get("__k__")
        color = lut.get(int(v) if v is not None and not pd.isna(v) else -1, ("Outros", "#c8c8c8"))[1]
        return {"fillOpacity": 0.65, "weight": 0.0, "color": "#00000000", "fillColor": color}

    folium.GeoJson(
        df.to_json(),
        name="Área de influência de bairro",
        pane="vectors",
        style_function=style_fn,
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
                st.image(lp, width=120)
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

    # Layout: filtros à esquerda, mapa à direita
    left, map_col = st.columns([1, 5], gap="small")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        if folium is None or st_folium is None or gpd is None:
            st.error("Instale `geopandas`, `folium` e `streamlit-folium`.")
            return

        # Assinatura (faz o cache observar mudanças na pasta)
        sig = _dir_signature()

        # Cria o mapa imediatamente (sem dados)
        fmap = make_satellite_map(center=(-23.55, -46.63), zoom=11, tiles_opacity=0.5)
        inject_tooltip_css(fmap, font_px=48)
        inject_label_scaler(fmap, 16, 26, 9, 18)

        # Limites administrativos — carregamento on-demand
        lim_gdf = load_admin_layer(ui["limite"], sig)
        if lim_gdf is not None and not lim_gdf.empty:
            add_admin_outline(fmap, lim_gdf, ui["limite"])
            if ui["labels_on"]:
                add_centroid_labels(fmap, lim_gdf)

        # Variável — carregamento on-demand
        var = ui["variavel"]
        if var == "Área de influência de bairro":
            iso = load_isocronas(sig)
            if iso is None or iso.empty:
                st.info("Isócronas não encontradas.")
            else:
                paint_isocronas_area(fmap, iso)
        else:
            setores = load_setores(sig)
            if setores is None or setores.empty:
                st.info("Setores não encontrados.")
            else:
                mapping = {
                    "População (Pessoa/ha)": "populacao",
                    "Densidade demográfica (hab/ha)": "densidade_demografica",
                    "Variação de elevação média": "diferenca_elevacao",
                    "Elevação média": "elevacao",
                    "Cluster (perfil urbano)": None,
                }
                if var == "Cluster (perfil urbano)":
                    cl = find_col(setores.columns, "cluster", "cluster_label", "label", "classe")
                    if cl is None:
                        st.info("Coluna de cluster não encontrada nos setores.")
                    else:
                        paint_setores_cluster(fmap, setores, cl)
                else:
                    col = find_col(setores.columns, mapping[var])
                    if col is None:
                        st.info(f"A coluna para '{var}' não foi encontrada nos setores.")
                    else:
                        paint_setores_numeric(fmap, setores, col, var, HIGH_CONTRAST)

        # Renderiza mapa
        st_folium(fmap, use_container_width=True, height=820)


if __name__ == "__main__":
    main()
