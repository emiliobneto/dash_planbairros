# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import base64, math, re, json
from decimal import Decimal

import pandas as pd
import streamlit as st

# ===========================  Fail-fast imports  ============================
def _import_stack():
    try:
        import geopandas as gpd              # geometrias
        import pydeck as pdk                 # render WebGL
        from shapely import wkb, wkt         # fallback de geometria
        from pyarrow import parquet as pq    # leitura colunar seletiva
        return gpd, pdk, wkb, wkt, pq
    except ImportError as e:
        st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")
        st.error(f"Dependência ausente: **{e}**. Instale os pacotes do requirements.txt e reinicie.")
        st.stop()

gpd, pdk, wkb, wkt, pq = _import_stack()

# ============================  Page / Theme  ================================
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

PB_NAVY = "#14407D"
ORANGE_RED_GRAD = ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#e34a33","#b30000"]
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

# ================================  CSS  =====================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{
            font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}

        /* remove espaço entre o logo e os filtros */
        .main .block-container {{
            padding-top: 0.2rem !important;
            padding-bottom: .8rem !important;
        }}

        /* header: logo à esquerda, barra azul só na direita */
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height: 88px; width:auto; display:block; }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding: 18px 20px; width:100%;
        }}
        .pb-title   {{ font-size: 3.8rem; font-weight: 900; line-height:1.05; letter-spacing:.2px }}
        .pb-subtitle{{ font-size: 1.9rem; opacity:.95; margin-top:6px }}

        .pb-card {{
            background:#fff; border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px; padding:12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==============================  Paths/Utils  ===============================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (REPO_ROOT / "limites_administrativos") if (REPO_ROOT / "limites_administrativos").exists() else REPO_ROOT
LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"   # caminho solicitado

def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f"data:image/{LOGO_PATH.suffix.lstrip('.').lower()};base64,{b64}"
    # fallback
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _first_parquet_by_stems(folder: Path, stems: List[str]) -> Optional[Path]:
    if not folder.exists(): return None
    wanted = {_slug(n) for n in stems}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        if not c: continue
        if c.lower() in low: return low[c.lower()]
        key = re.sub(r"[^a-z0-9]", "", c.lower())
        if key in norm: return norm[key]
    return None

def to_float_series(s: pd.Series) -> pd.Series:
    if isinstance(s, pd.Series) and s.dtype == "object":
        return s.apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype("Float64")
    return pd.to_numeric(s, errors="coerce").astype("Float64") if isinstance(s, pd.Series) else pd.Series(dtype="Float64")

def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)

# ===================  Leitor GeoParquet robusto (evita crash)  ==============
def _read_gdf_robusto(path: Path, columns: Optional[List[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    """Tenta ler como GeoParquet. Se columns=... remover metadado, reconstrói via WKB/WKT."""
    if not path:
        return None
    try:
        gdf = gpd.read_parquet(path, columns=columns)
        if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
            raise ValueError("Sem metadado geoespacial; usando fallback.")
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        return gdf.to_crs(4326)
    except Exception:
        try:
            table = pq.read_table(str(path), columns=columns)
            pdf = table.to_pandas()
            geom_col = find_col(pdf.columns, "geometry", "geom", "wkb", "wkt")
            if geom_col is None:
                return None
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith(("POLY", "MULTI", "LINE", "POINT")).any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
            return gdf
        except Exception:
            return None

# =====================  I/O eficiente: colunas seletivas  ===================
@st.cache_data(show_spinner=False, ttl=3600)
def _parquet_columns(path: Path) -> List[str]:
    return pq.ParquetFile(str(path)).schema.names

@st.cache_data(show_spinner=False, ttl=3600)
def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_by_stems(DATA_DIR, ["isocronas","isócronas","isocronas2023"])
    if not p: return None
    cols = _parquet_columns(p)
    keep = [c for c in ["geometry","nova_class"] if c in cols]
    return _read_gdf_robusto(p, keep)

@st.cache_data(show_spinner=False, ttl=3600)
def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {"Distritos":["Distritos"],"ZonasOD2023":["ZonasOD2023","ZonasOD"],"Subprefeitura":["Subprefeitura","subprefeitura"],"Isócronas":["isocronas","isócronas"]}.get(name,[name])
    p = _first_parquet_by_stems(DATA_DIR, stems)
    if not p: return None
    return _read_gdf_robusto(p, ["geometry"])

@st.cache_data(show_spinner=False, ttl=3600)
def load_setores_geom() -> Tuple[Optional["gpd.GeoDataFrame"], Optional[str], Optional[Path]]:
    p = _first_parquet_by_stems(DATA_DIR, ["SetoresCensitarios2023","SetoresCensitarios","setores"])
    if not p: return None, None, None
    cols = _parquet_columns(p)
    id_col = find_col(cols, "id","cd_setor","geocodigo","codigo","geocod","id_setor")
    keep = [c for c in [id_col, "geometry"] if c]
    gdf = _read_gdf_robusto(p, keep)
    return gdf, id_col, p

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_metric_column(var_label: str, id_col_hint: Optional[str], path_hint: Optional[Path]) -> Optional[pd.DataFrame]:
    p_metrics = _first_parquet_by_stems(DATA_DIR, ["setores_metrics","setores_metricas","metricas_setores"])
    p = p_metrics or path_hint
    if not p: return None
    cols = _parquet_columns(p)
    id_col = id_col_hint or find_col(cols, "id","cd_setor","geocodigo","codigo","geocod","id_setor")
    mapping = {
        "População (Pessoa/ha)": "populacao",
        "Densidade demográfica (hab/ha)": "densidade_demografica",
        "Variação de elevação média": "diferenca_elevacao",
        "Elevação média": "elevacao",
        "Cluster (perfil urbano)": "cluster",
    }
    wanted = find_col(cols, mapping.get(var_label, var_label))
    if not (id_col and wanted): return None
    table = pq.read_table(str(p), columns=[id_col, wanted])
    df = table.to_pandas()
    df.rename(columns={wanted: "__value__"}, inplace=True)
    return df

# ======================  Cores util (gradiente → rgba)  =====================
def _hex_to_rgba(h: str, a: int = 190) -> list[int]:
    h = h.lstrip("#"); return [int(h[i:i+2], 16) for i in (0, 2, 4)] + [a]

def ramp_color(v: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "#c8c8c8"
    t = 0.0 if vmax == vmin else (float(v) - vmin) / (vmax - vmin)
    t = min(1.0, max(0.0, t)); n = len(colors) - 1
    i = min(int(t * n), n - 1); frac = (t * n) - i
    h2r = lambda hx: tuple(int(hx.lstrip("#")[k:k+2],16) for k in (0,2,4))
    r2h = lambda r: "#{:02x}{:02x}{:02x}".format(*r)
    c1, c2 = h2r(colors[i]), h2r(colors[i+1])
    mix = tuple(int(c1[k] + frac*(c2[k]-c1[k])) for k in range(3))
    return r2h(mix)

# ===============================  UI  =======================================
def left_controls() -> Dict[str, Any]:
    st.markdown("<div style='margin-top:-6px'></div>", unsafe_allow_html=True)

    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [PLACEHOLDER_VAR,"População (Pessoa/ha)","Densidade demográfica (hab/ha)","Variação de elevação média","Elevação média","Cluster (perfil urbano)","Área de influência de bairro"],
        index=0, key="pb_var", placeholder="Escolha…",
    )
    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        [PLACEHOLDER_LIM,"Distritos","ZonasOD2023","Subprefeitura","Isócronas"],
        index=0, key="pb_limite", placeholder="Escolha…",
    )
    labels_on = st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")

    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear(); st.success("Cache limpo. Selecione novamente a camada/variável.")

    sel_now = {"var": var, "lim": limite}
    sel_prev = st.session_state.get("_pb_prev_sel")
    if sel_prev and (sel_prev["var"] != sel_now["var"] or sel_prev["lim"] != sel_now["lim"]):
        st.cache_data.clear()
    st.session_state["_pb_prev_sel"] = sel_now

    return {"variavel": var, "limite": limite, "labels_on": labels_on}

# =============================  Render (pydeck)  ============================
# Função JS sem arrow functions (evita "Unexpected =")
_TILE_RENDER_FN = """
function renderSubLayers(props) {
  const bbox = props.tile.bbox;
  const west = bbox.west, south = bbox.south, east = bbox.east, north = bbox.north;
  return new deck.BitmapLayer(props, {
    data: null,
    image: props.data,
    bounds: [west, south, east, north]
  });
}
"""

def render_pydeck(center: Tuple[float, float],
                  setores_joined: Optional["gpd.GeoDataFrame"],
                  limite_gdf: Optional["gpd.GeoDataFrame"],
                  var_label: Optional[str]):
    layers = []

    # Base ESRI (sem token) — usa função JS clássica (sem arrow)
    esri = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0, max_zoom=19, tile_size=256,
        render_sub_layers=_TILE_RENDER_FN,
    )
    layers.append(esri)

    # Setores (variáveis numéricas e cluster)
    if setores_joined is not None and not setores_joined.empty and "__value__" in setores_joined.columns:
        s = pd.to_numeric(setores_joined["__value__"], errors="coerce")
        vmin, vmax = float(s.min()), float(s.max())
        gdf = setores_joined.copy()

        if var_label == "Cluster (perfil urbano)":
            cmap = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}
            gdf["__rgba__"] = pd.to_numeric(gdf["__value__"], errors="coerce").map(
                lambda v: _hex_to_rgba(cmap.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200)
            )
        else:
            gdf["__rgba__"] = s.map(lambda v: _hex_to_rgba(ramp_color(v, vmin, vmax, ORANGE_RED_GRAD), 200))

        geojson = json.loads(gdf[["geometry","__rgba__","__value__"]].to_json(drop_id=True))
        setores_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            filled=True, stroked=False, pickable=True, auto_highlight=True,
            get_fill_color="properties.__rgba__", get_line_width=0,
        )
        layers.append(setores_layer)

    # Contorno administrativo (opcional)
    if limite_gdf is not None and not limite_gdf.empty:
        gj_lim = json.loads(limite_gdf[["geometry"]].to_json(drop_id=True))
        outline = pdk.Layer(
            "GeoJsonLayer", data=gj_lim, filled=False, stroked=True,
            get_line_color=[0,0,0,255], get_line_width=1.2
        )
        layers.append(outline)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11, bearing=0, pitch=0),
        map_style=None,
        tooltip={"text": f"{var_label}: {{__value__}}"} if var_label and var_label != "Cluster (perfil urbano)" else None,
    )
    st.pydeck_chart(deck, use_container_width=True)  # sem height (compat total)

# ================================  App  =====================================
def main() -> None:
    inject_css()

    # Header: logo (assets/logo_todos.jpg) + barra azul à direita
    st.markdown(
        f"""
        <div class="pb-row">
            <img class="pb-logo" src="{_logo_data_uri()}" alt="PlanBairros logo"/>
            <div class="pb-header">
                <div class="pb-title">PlanBairros</div>
                <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        center = (-23.55, -46.63)

        # Limites (opcional)
        limite_gdf = None
        if ui["limite"] != PLACEHOLDER_LIM:
            limite_gdf = load_admin_layer(ui["limite"])
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)

        var = ui["variavel"]

        # Isócronas (categorias)
        if var == "Área de influência de bairro":
            iso = load_isocronas()
            if iso is None or len(iso) == 0:
                st.info("Isócronas não encontradas.")
                render_pydeck(center, None, limite_gdf, None)
            else:
                lut = {0:"#542788",1:"#f7f7f7",2:"#d8daeb",3:"#b35806",4:"#b2abd2",
                       5:"#8073ac",6:"#fdb863",7:"#7f3b08",8:"#e08214",9:"#fee0b6"}
                cls = find_col(iso.columns, "nova_class")
                if cls:
                    g = iso.copy()
                    g["__value__"] = pd.to_numeric(g[cls], errors="coerce")
                    g["__rgba__"] = g["__value__"].map(lambda v: _hex_to_rgba(lut.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200))
                    geojson = json.loads(g[["geometry","__rgba__","__value__"]].to_json(drop_id=True))
                    # camada base + isócronas
                    deck = pdk.Deck(
                        layers=[
                            pdk.Layer("TileLayer",
                                      data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                      min_zoom=0, max_zoom=19, tile_size=256,
                                      render_sub_layers=_TILE_RENDER_FN),
                            pdk.Layer("GeoJsonLayer", data=geojson, filled=True, stroked=False,
                                      get_fill_color="properties.__rgba__", get_line_width=0)
                        ],
                        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11),
                        map_style=None,
                    )
                    st.pydeck_chart(deck, use_container_width=True)
                else:
                    st.info("A coluna 'nova_class' não foi encontrada nas isócronas.")
                    render_pydeck(center, None, limite_gdf, None)
            return  # encerra, já renderizado

        # Setores (variáveis numéricas / cluster)
        setores_joined = None
        if var != PLACEHOLDER_VAR:
            geoms, id_col, setores_path = load_setores_geom()
            if geoms is None or id_col is None:
                st.info("Setores não encontrados.")
            else:
                metric = load_metric_column(var, id_col, setores_path)
                if metric is None:
                    st.info(f"Coluna para '{var}' não encontrada.")
                else:
                    setores_joined = geoms.merge(metric, on=id_col, how="left")
                    center = center_from_bounds(setores_joined)

        render_pydeck(center=center,
                      setores_joined=setores_joined,
                      limite_gdf=limite_gdf,
                      var_label=None if var in (PLACEHOLDER_VAR, "Área de influência de bairro") else var)

if __name__ == "__main__":
    main()
