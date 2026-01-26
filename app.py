# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import base64, math, re, json
import numpy as np
import pandas as pd
import streamlit as st

# ====================== imports obrigatórios ======================
def _import_stack():
    try:
        import geopandas as gpd
        import pydeck as pdk
        from shapely import wkb, wkt
        from pyarrow import parquet as pq
        return gpd, pdk, wkb, wkt, pq
    except ImportError as e:
        st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")
        st.error(f"Dependência ausente: **{e}**.")
        st.stop()

gpd, pdk, wkb, wkt, pq = _import_stack()

# ====================== config / tema ======================
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

PB_NAVY = "#14407D"
ORANGE_RED_GRAD = ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#e34a33","#b30000"]
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

LOGO_HEIGHT = 120
MAP_HEIGHT  = 900
SIMPLIFY_M_SETORES   = 25   # ~25 m
SIMPLIFY_M_ISOCRONAS = 80   # ~80 m
SIMPLIFY_M_OVERLAY   = 20   # ~20 m (rios/linhas/verde)

# Overlays permanentes (cores e espessuras)
REF_GREEN    = "#2E7D32"  # áreas verdes
REF_BLUE     = "#1E88E5"  # rios
REF_DARKGRAY = "#333333"  # trilhos (trem/metro)

# >>> espessuras dobradas:
RIVER_WIDTH_PX = 4.4   # antes ~2.2
RAIL_WIDTH_PX  = 6.4   # antes ~3.2

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{ font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }}
        .main .block-container {{ padding-top: .2rem !important; padding-bottom: .8rem !important; }}
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; }}
        .pb-header {{ background:{PB_NAVY}; color:#fff; border-radius:14px; padding:18px 20px; width:100%; }}
        .pb-title {{ font-size:3.8rem; font-weight:900; line-height:1.05; letter-spacing:.2px }}
        .pb-subtitle {{ font-size:1.9rem; opacity:.95; margin-top:6px }}
        .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04); border-radius:14px; padding:12px; }}
        .legend-card {{ margin-top:12px; background:#fff; border:1px solid rgba(20,64,125,.10); border-radius:12px; padding:10px 12px; }}
        .legend-title {{ font-weight:800; margin-bottom:6px; }}
        .legend-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
        .legend-swatch {{ width:18px; height:18px; border-radius:4px; display:inline-block; border:1px solid rgba(0,0,0,.15); }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ====================== caminhos e util ======================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (REPO_ROOT / "limites_administrativos") if (REPO_ROOT / "limites_administrativos").exists() else REPO_ROOT
GEOM_FILE    = DATA_DIR / "IDCenso2023.parquet"
METRICS_FILE = DATA_DIR / "SetoresCensitarios2023.parquet"
LOGO_PATH    = REPO_ROOT / "assets" / "logo_todos.jpg"

def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f"data:image/{LOGO_PATH.suffix.lstrip('.').lower()};base64,{b64}"
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"

def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        if not c: continue
        if c.lower() in low: return low[c.lower()]
        key = re.sub(r"[^a-z0-9]", "", c.lower())
        if key in norm: return norm[key]
    return None

def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)

def _hex_to_rgba(h: str, a: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0,2,4)] + [a]

# ====================== leitores ======================
def _read_gdf_robusto(path: Path, columns: Optional[List[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    if not path.exists(): return None
    try:
        gdf = gpd.read_parquet(path, columns=columns)
        if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
            raise ValueError
        if gdf.crs is None: gdf = gdf.set_crs(4326)
        return gdf.to_crs(4326)
    except Exception:
        try:
            table = pq.read_table(str(path), columns=columns)
            pdf = table.to_pandas()
            geom_col = find_col(pdf.columns, "geometry", "geom", "wkb", "wkt")
            if geom_col is None: return None
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith(("POLY","MULTI","LINE","POINT")).any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            return gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
        except Exception:
            return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_setores_geom() -> Tuple[Optional["gpd.GeoDataFrame"], Optional[str]]:
    if not GEOM_FILE.exists(): return None, None
    gdf = _read_gdf_robusto(GEOM_FILE, ["fid", "geometry"])
    if gdf is None: return None, None
    try:
        gdfm = gdf.to_crs(3857)
        gdfm["geometry"] = gdfm.geometry.simplify(SIMPLIFY_M_SETORES, preserve_topology=True)
        gdf = gdfm.to_crs(4326)
    except Exception:
        pass
    return gdf, "fid"

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_metric_column(var_label: str) -> Optional[pd.DataFrame]:
    if not METRICS_FILE.exists(): return None
    cols = pq.ParquetFile(str(METRICS_FILE)).schema.names
    mapping: Dict[str, List[str] | str] = {
        "População (Pessoa/ha)": ["populacao", "pop", "pessoa_ha"],
        "Densidade demográfica (hab/ha)": ["densidade_demografica", "densidade", "hab_ha"],
        "Elevação média": ["elevacao_media", "elevacao", "elev_med", "altitude_media"],
        "Cluster (perfil urbano)": ["cluster", "classe", "label"],
    }
    cand = mapping.get(var_label, var_label)
    wanted = find_col(cols, *(cand if isinstance(cand, (list, tuple)) else [cand]))
    if not wanted or "fid" not in cols: return None
    table = pq.read_table(str(METRICS_FILE), columns=["fid", wanted])
    df = table.to_pandas().rename(columns={wanted: "__value__"})
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def load_isocronas_raw() -> Optional["gpd.GeoDataFrame"]:
    for name in ("isocronas.parquet", "isócronas.parquet", "Isocronas.parquet"):
        p = DATA_DIR / name
        if p.exists():
            return _read_gdf_robusto(p, None)
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {"Distritos":"Distritos.parquet","ZonasOD2023":"ZonasOD2023.parquet",
             "Subprefeitura":"Subprefeitura.parquet","Isócronas":"isocronas.parquet"}
    p = DATA_DIR / stems.get(name, "")
    if not p.exists(): return None
    return _read_gdf_robusto(p, ["geometry"])

# ---------- Overlays permanentes ----------
def _read_geojson_any(*names: str) -> Optional["gpd.GeoDataFrame"]:
    for nm in names:
        p = DATA_DIR / nm
        if p.exists():
            try:
                g = gpd.read_file(p)
                if g.crs is None: g = g.set_crs(4326)
                g = g.to_crs(4326)
                try:
                    gm = g.to_crs(3857)
                    gm["geometry"] = gm.geometry.buffer(0).simplify(SIMPLIFY_M_OVERLAY, preserve_topology=True)
                    g = gm.to_crs(4326)
                except Exception:
                    pass
                return g
            except Exception:
                continue
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def load_green_areas() -> Optional["gpd.GeoDataFrame"]:
    return _read_geojson_any("area_verde.geojson", "areas_verdes.geojson")

@st.cache_data(show_spinner=False, ttl=3600)
def load_rivers() -> Optional["gpd.GeoDataFrame"]:
    return _read_geojson_any("rios.geojson")

@st.cache_data(show_spinner=False, ttl=3600)
def load_metro_lines() -> Optional["gpd.GeoDataFrame"]:
    return _read_geojson_any("linhas metro.geojson", "linhas_metro.geojson")

@st.cache_data(show_spinner=False, ttl=3600)
def load_train_lines() -> Optional["gpd.GeoDataFrame"]:
    return _read_geojson_any("linhas trem.geojson", "linhas_trem.geojson")

# ====================== helpers de classes numéricas ======================
def _sample_gradient(colors: List[str], n: int) -> List[str]:
    if n <= 1: return [colors[-1]]
    out = []
    for i in range(n):
        t = i/(n-1)
        pos = t*(len(colors)-1)
        j = int(math.floor(pos)); j = min(j, len(colors)-2)
        frac = pos - j
        def h2r(x): x=x.lstrip("#"); return [int(x[k:k+2],16) for k in (0,2,4)]
        def r2h(r): return "#{:02x}{:02x}{:02x}".format(*r)
        c1, c2 = h2r(colors[j]), h2r(colors[j+1])
        mix = [int(c1[k] + frac*(c2[k]-c1[k])) for k in range(3)]
        out.append(r2h(mix))
    return out

def _equal_edges(vmin: float, vmax: float, k: int = 6) -> List[float]:
    step = (vmax - vmin) / k
    edges = [vmin + i*step for i in range(k+1)]
    edges[-1] = vmax
    return edges

def _quantile_edges(vals: np.ndarray, k: int = 6) -> List[float]:
    qs = np.linspace(0, 1, k+1)
    edges = list(np.quantile(vals, qs))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9
    return edges

def classify_auto6(series: pd.Series) -> Tuple[pd.Series, List[Tuple[int,int]], List[str]]:
    s = pd.to_numeric(series, errors="coerce")
    v = s.dropna()
    if v.empty:
        return pd.Series([-1]*len(series), index=series.index), [], []
    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        idx = pd.cut(s, bins=[vmin-1e-9, vmax+1e-9], labels=False, include_lowest=True)
        palette = _sample_gradient(ORANGE_RED_GRAD, 6)
        return idx.fillna(-1).astype("Int64"), [(int(round(vmin)), int(round(vmax)))], palette
    edges_eq = _equal_edges(vmin, vmax, 6)
    idx_eq = pd.cut(s, bins=[edges_eq[0]-1e-9] + edges_eq[1:], labels=False, include_lowest=True)
    counts = idx_eq.value_counts(dropna=True)
    non_empty = (counts > 0).sum()
    max_share = (counts.max() / counts.sum()) if counts.sum() > 0 else 1.0
    use_quantile = (max_share > 0.35) or (non_empty < 4)
    if use_quantile:
        edges = _quantile_edges(v.to_numpy(), 6)
        idx = pd.cut(s, bins=[edges[0]-1e-9] + edges[1:], labels=False, include_lowest=True)
    else:
        edges = edges_eq; idx = idx_eq
    breaks_int: List[Tuple[int,int]] = []
    for i in range(6):
        a = int(round(edges[i])); b = int(round(edges[i+1]))
        if b < a: b = a
        breaks_int.append((a, b))
    palette = _sample_gradient(ORANGE_RED_GRAD, 6)
    return idx.fillna(-1).astype("Int64"), breaks_int, palette

# ====================== UI (filtros) ======================
def left_controls() -> Dict[str, Any]:
    st.markdown("<div style='margin-top:-6px'></div>", unsafe_allow_html=True)
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [
            PLACEHOLDER_VAR,
            "População (Pessoa/ha)",
            "Densidade demográfica (hab/ha)",
            "Elevação média",
            "Cluster (perfil urbano)",
            "Área de influência de bairro",
        ],
        index=0, key="pb_var", placeholder="Escolha…",
    )

    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        [PLACEHOLDER_LIM,"Distritos","ZonasOD2023","Subprefeitura","Isócronas","Setores Censitários (linhas)"],
        index=0, key="pb_limite", placeholder="Escolha…",
    )

    # >>> Toggle do fundo do mapa
    fundo = st.radio(
        "Fundo do mapa",
        ["Claro (CARTO)", "Satélite (ESRI)"],
        index=0, horizontal=True, key="pb_basemap"
    )

    st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")

    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear(); st.success("Cache limpo. Selecione novamente a camada/variável.")

    legend_ph = st.empty(); st.session_state["_legend_ph"] = legend_ph

    sel_now = {"var": var, "lim": limite, "fundo": fundo}
    sel_prev = st.session_state.get("_pb_prev_sel")
    if sel_prev and (sel_prev != sel_now):
        st.cache_data.clear()
    st.session_state["_pb_prev_sel"] = sel_now
    return {"variavel": var, "limite": limite, "fundo": fundo}

# ====================== Legendas (HTML) ======================
def show_numeric_legend(title: str, breaks: List[Tuple[int,int]], palette: List[str]):
    ph = st.session_state.get("_legend_ph")
    if not ph: return
    if not breaks: ph.empty(); return
    rows = []
    for (a,b), col in zip(breaks, palette):
        label = f"{a:,} – {b:,}".replace(",", ".")
        rows.append(f"<div class='legend-row'><span class='legend-swatch' style='background:{col}'></span><span>{label}</span></div>")
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)

def show_categorical_legend(title: str, items: List[Tuple[str,str]]):
    ph = st.session_state.get("_legend_ph")
    if not ph: return
    rows = [f"<div class='legend-row'><span class='legend-swatch' style='background:{c}'></span><span>{l}</span></div>"
            for l,c in items]
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)

def show_reference_legend():
    items = [
        ("Áreas verdes", REF_GREEN),
        ("Rios", REF_BLUE),
        ("Linhas de trem", REF_DARKGRAY),
        ("Linhas de metrô", REF_DARKGRAY),
    ]
    rows = [f"<div class='legend-row'><span class='legend-swatch' style='background:{c}'></span><span>{l}</span></div>"
            for l,c in items]
    html = f"<div class='legend-card'><div class='legend-title'>Camadas de referência</div>{''.join(rows)}</div>"
    st.markdown(html, unsafe_allow_html=True)

def clear_legend():
    ph = st.session_state.get("_legend_ph")
    if ph: ph.empty()

# ====================== Render (pydeck) ======================
def _show_deck(deck: "pdk.Deck"):
    try:
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)

def make_satellite_tilelayer() -> "pdk.Layer":
    # Esri World Imagery sem token; renderizado como Bitmap por tile
    return pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0, max_zoom=19, tile_size=256,
        render_sub_layers="""
            function (props) {
                return new deck.BitmapLayer({
                    id: 'esri-satellite',
                    image: props.data,
                    bounds: props.tile.bbox
                });
            }
        """
    )

def collect_reference_overlays() -> List["pdk.Layer"]:
    """Camadas permanentes (no topo)."""
    layers: List[pdk.Layer] = []

    g = load_green_areas()
    if g is not None and not g.empty:
        gj = json.loads(g[["geometry"]].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=gj,
            filled=True, stroked=False, pickable=False,
            get_fill_color=_hex_to_rgba(REF_GREEN, 255),
            get_line_width=0
        ))

    r = load_rivers()
    if r is not None and not r.empty:
        gj = json.loads(r[["geometry"]].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=gj,
            filled=False, stroked=True, pickable=False,
            get_line_color=_hex_to_rgba(REF_BLUE, 255),
            get_line_width=RIVER_WIDTH_PX, lineWidthUnits="pixels"
        ))

    t = load_train_lines()
    if t is not None and not t.empty:
        gj = json.loads(t[["geometry"]].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=gj,
            filled=False, stroked=True, pickable=False,
            get_line_color=_hex_to_rgba(REF_DARKGRAY, 255),
            get_line_width=RAIL_WIDTH_PX, lineWidthUnits="pixels"
        ))

    m = load_metro_lines()
    if m is not None and not m.empty:
        gj = json.loads(m[["geometry"]].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=gj,
            filled=False, stroked=True, pickable=False,
            get_line_color=_hex_to_rgba(REF_DARKGRAY, 255),
            get_line_width=RAIL_WIDTH_PX, lineWidthUnits="pixels"
        ))

    return layers

def render_pydeck(center: Tuple[float, float],
                  gdf_layer: Optional["gpd.GeoDataFrame"],
                  limite_gdf: Optional["gpd.GeoDataFrame"],
                  *,
                  tooltip_field: Optional[str],
                  categorical_legend: Optional[List[Tuple[str, str]]] = None,
                  numeric_legend: Optional[Tuple[str, List[Tuple[int,int]], List[str]]] = None,
                  draw_setores_outline: bool = False,
                  basemap: str = "Claro (CARTO)"):
    layers: List[pdk.Layer] = []

    # 0) Base (se Satélite, adiciona primeiro para ficar por baixo)
    if basemap == "Satélite (ESRI)":
        layers.append(make_satellite_tilelayer())
        map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    else:
        map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    # 1) camada temática
    if gdf_layer is not None and not gdf_layer.empty and "__rgba__" in gdf_layer.columns:
        geojson = json.loads(gdf_layer[["geometry","__rgba__"] + ([tooltip_field] if tooltip_field else [])].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=geojson,
            filled=True, stroked=False, pickable=bool(tooltip_field), auto_highlight=True,
            get_fill_color="properties.__rgba__", get_line_width=0,
        ))

    # 2) contornos e setores (se desejar) — ficam abaixo dos overlays
    if limite_gdf is not None and not limite_gdf.empty:
        gj_lim = json.loads(limite_gdf[["geometry"]].to_json())
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=gj_lim, filled=False, stroked=True,
            get_line_color=[20,20,20,180], get_line_width=1, lineWidthUnits="pixels"
        ))

    if draw_setores_outline:
        try:
            geoms_only, _ = load_setores_geom()
        except Exception:
            geoms_only = None
        if geoms_only is not None:
            gj = json.loads(geoms_only[["geometry"]].to_json())
            layers.append(pdk.Layer(
                "GeoJsonLayer", data=gj, filled=False, stroked=True,
                get_line_color=[80,80,80,160], get_line_width=0.6, lineWidthUnits="pixels"
            ))

    # 3) OVERLAYS NO TOPO
    layers.extend(collect_reference_overlays())

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11, bearing=0, pitch=0),
        map_style=map_style,
        tooltip={"text": f"{{{tooltip_field}}}"} if tooltip_field else None,
    )
    _show_deck(deck)

    # legenda temática
    if categorical_legend is not None:
        show_categorical_legend("Área de influência de bairro", categorical_legend)
    elif numeric_legend is not None:
        title, breaks, palette = numeric_legend
        show_numeric_legend(title, breaks, palette)
    else:
        clear_legend()

    # legenda fixa das camadas de referência (sempre)
    show_reference_legend()

# ====================== App ======================
def main() -> None:
    inject_css()
    st.markdown(
        f"""
        <div class="pb-row">
            <img class="pb-logo" src="{_logo_data_uri()}" alt="PlanBairros logo"/>
            <div class="pb-header">
                <div class="pb-title">PlanBairros</div>
                <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        center = (-23.55, -46.63)
        var = ui["variavel"]

        # Limite administrativo
        limite_gdf = None
        draw_setores_outline = (ui["limite"] == "Setores Censitários (linhas)")
        if ui["limite"] not in (PLACEHOLDER_LIM, "Setores Censitários (linhas)"):
            limite_gdf = load_admin_layer(ui["limite"])
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)

        # ====== VARIÁVEIS ======
        if var == "Área de influência de bairro":
            lut_color = {0:"#542788",1:"#f7f7f7",2:"#d8daeb",3:"#b35806",4:"#b2abd2",
                         5:"#8073ac",6:"#fdb863",7:"#7f3b08",8:"#e08214",9:"#fee0b6"}
            lut_label = {
                0:"Predominância uso misto", 1:"Zona de transição local",
                2:"Periférico residencial de média densidade",
                3:"Transição central verticalizada",
                4:"Periférico adensado em transição",
                5:"Centralidade comercial e de serviços",
                6:"Predominância residencial média densidade",
                7:"Áreas íngremes e de encosta",
                8:"Alta densidade residencial",
                9:"Central verticalizado",
            }
            iso_raw = load_isocronas_raw()
            gdf_layer = None
            legend_items = None
            if iso_raw is not None and not iso_raw.empty:
                cls = find_col(iso_raw.columns, "Isocrona","isocrona","nova_class","classe","class","cat","category")
                if cls:
                    g = iso_raw[[cls,"geometry"]].copy()
                    k = pd.to_numeric(g[cls], errors="coerce").astype("Int64")
                    g = g[~k.isna()].copy(); g["__k__"] = k
                    try:
                        gm = g.to_crs(3857)
                        gm["geometry"] = gm.buffer(0).geometry.simplify(SIMPLIFY_M_ISOCRONAS, preserve_topology=True)
                        g = gm.to_crs(4326).explode(index_parts=False, ignore_index=True)
                    except Exception:
                        pass
                    g["__rgba__"]  = g["__k__"].map(lambda v: _hex_to_rgba(lut_color.get(int(v), "#c8c8c8"), 200))
                    g["__label__"] = g["__k__"].map(lambda v: f"{int(v)} – {lut_label.get(int(v),'Outros')}")
                    gdf_layer = g[["geometry","__rgba__","__label__"]]
                    center = center_from_bounds(g)
                    present = sorted({int(v) for v in g["__k__"].dropna().unique().tolist()})
                    legend_items = [(f"{k} – {lut_label[k]}", lut_color[k]) for k in present]

            render_pydeck(center, gdf_layer, limite_gdf,
                          tooltip_field="__label__", categorical_legend=legend_items,
                          numeric_legend=None, draw_setores_outline=draw_setores_outline,
                          basemap=ui["fundo"])
            return

        # Demais variáveis (setores)
        gdf_layer = None
        if var != PLACEHOLDER_VAR:
            geoms, id_col = load_setores_geom()
            if geoms is not None and id_col == "fid":
                metric = load_metric_column(var)
                if metric is not None:
                    joined = geoms.merge(metric, on="fid", how="left")
                    center = center_from_bounds(joined)

                    if var == "Cluster (perfil urbano)":
                        cmap = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}
                        labels = {
                            0:"1 - Periférico com predominância residencial de alta densidade construtiva",
                            1:"2 - Uso misto de média densidade construtiva",
                            2:"3 - Periférico com predominância residencial de média densidade construtiva",
                            3:"4 - Verticalizado de uso-misto",
                            4:"5 - Predominância de uso comercial e serviços",
                        }
                        s = pd.to_numeric(joined["__value__"], errors="coerce")
                        joined["__rgba__"] = s.map(lambda v: _hex_to_rgba(cmap.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200))
                        joined["__label__"] = s.map(lambda v: labels.get(int(v), "Outros") if pd.notna(v) else "Sem dado")
                        gdf_layer = joined[["geometry","__rgba__","__label__"]]
                        render_pydeck(center, gdf_layer, limite_gdf,
                                      tooltip_field="__label__", categorical_legend=[(labels[k], cmap[k]) for k in sorted(cmap)],
                                      numeric_legend=None, draw_setores_outline=draw_setores_outline,
                                      basemap=ui["fundo"])
                        return
                    else:
                        classes, breaks_int, palette = classify_auto6(joined["__value__"])
                        color_map = {i: _hex_to_rgba(palette[i], 200) for i in range(6)}
                        joined["__rgba__"] = classes.map(lambda k: color_map.get(int(k) if pd.notna(k) else -1, _hex_to_rgba("#c8c8c8",200)))
                        joined["__label__"] = pd.to_numeric(joined["__value__"], errors="coerce").round(0).astype("Int64").astype(str)
                        gdf_layer = joined[["geometry","__rgba__","__label__"]]
                        render_pydeck(center, gdf_layer, limite_gdf,
                                      tooltip_field="__label__", categorical_legend=None,
                                      numeric_legend=(var, breaks_int, palette), draw_setores_outline=draw_setores_outline,
                                      basemap=ui["fundo"])
                        return

        # sem variável: mapa base + overlays + legenda
        render_pydeck(center, None, limite_gdf, tooltip_field=None,
                      categorical_legend=None, numeric_legend=None,
                      draw_setores_outline=draw_setores_outline, basemap=ui["fundo"])

if __name__ == "__main__":
    main()
