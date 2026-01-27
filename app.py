# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import base64
import math
import re

import pandas as pd
import streamlit as st

# ==========================
# Geo libs (hard deps do app)
# ==========================
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


# =============================================================================
# Página + Identidade PlanBairros
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

PB_NAVY = PB_COLORS["navy"]
LOGO_HEIGHT = 46

# Jenks (6 classes) – paleta quente (alto contraste)
JENKS_COLORS_6 = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33"]

# Simplificação (em metros) para acelerar GeoJSON no Folium
SIMPLIFY_M_FILL = 18     # polígonos preenchidos (coroplético)
SIMPLIFY_M_LINE = 55     # linhas (limites / contornos)

# =============================================================================
# CSS (identidade + layout)
# =============================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{
            font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
        .main .block-container {{
            padding-top: .15rem !important;
            padding-bottom: .6rem !important;
        }}

        .pb-row {{
            display:flex; align-items:center; gap:12px; margin-bottom:0;
        }}
        .pb-logo {{
            height:{LOGO_HEIGHT}px; width:auto; display:block;
            border-radius: 10px;
        }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding:14px 15px; width:100%;
        }}
        .pb-title {{
            font-size:2.2rem; font-weight:900; line-height:1.05; letter-spacing:.2px;
        }}
        .pb-subtitle {{
            font-size:1.05rem; opacity:.95; margin-top:5px;
        }}

        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:10px 11px;
        }}

        /* Tooltips grandes (se você quiser reativar em algum momento) */
        .leaflet-tooltip.pb-big-tooltip,
        .leaflet-tooltip.pb-big-tooltip * {{
            font-size: 22px !important;
            font-weight: 900 !important;
            color: #111 !important;
            line-height: 1 !important;
        }}
        .leaflet-tooltip.pb-big-tooltip {{
            background:#fff !important;
            border: 2px solid #222 !important;
            border-radius: 10px !important;
            padding: 8px 10px !important;
            white-space: nowrap !important;
            pointer-events: none !important;
            box-shadow: 0 2px 6px rgba(0,0,0,.2) !important;
            z-index: 200000 !important;
        }}

        /* evita “lateral direita vazia” por container quebrado */
        .st-emotion-cache-1jicfl2 {{
            padding-left: .2rem !important;
            padding-right: .2rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Paths (GitHub)
# =============================================================================
try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    REPO_ROOT = Path.cwd()

DATA_DIR = (
    REPO_ROOT / "limites_administrativos"
    if (REPO_ROOT / "limites_administrativos").exists()
    else REPO_ROOT
)

LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"


def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        ext = LOGO_PATH.suffix.lstrip(".").lower() or "jpg"
        return f"data:image/{ext};base64,{b64}"
    # fallback
    return (
        "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/"
        "logomark/streamlit-mark-color.png"
    )


# =============================================================================
# Utils
# =============================================================================
def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


def find_col(cols: List[str], *cands: str) -> Optional[str]:
    """acha coluna por match case-insensitive + normalização"""
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


def center_from_bounds(gdf: "gpd.GeoDataFrame") -> Tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)


def _list_files(folder: Path) -> Dict[str, Path]:
    """map slug(stem) -> path, incluindo parquet/geojson/json"""
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for fp in folder.iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in (".parquet", ".geojson", ".json"):
            continue
        out[_slug(fp.stem)] = fp
    return out


def _pick_layer_file(folder: Path, stems: List[str]) -> Optional[Path]:
    files = _list_files(folder)
    wanted = {_slug(s) for s in stems}
    for k in wanted:
        if k in files:
            return files[k]
    return None


def _ensure_4326(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def _drop_empty_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def _simplify_m(gdf: "gpd.GeoDataFrame", tol_m: float) -> "gpd.GeoDataFrame":
    """simplifica em metros (EPSG:3857) e volta pra 4326"""
    if gdf.empty:
        return gdf
    try:
        g2 = gdf.to_crs(3857).copy()
        g2["geometry"] = g2.geometry.simplify(tol_m, preserve_topology=True)
        g2 = g2.to_crs(4326)
        return _drop_empty_geoms(g2)
    except Exception:
        return _drop_empty_geoms(gdf)


def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# =============================================================================
# Leitura (parquet + geojson) – on-demand + cache
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf(path: str) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() == ".parquet":
            gdf = gpd.read_parquet(p)
        else:
            # .geojson / .json
            gdf = gpd.read_file(p)
        gdf = _ensure_4326(gdf)
        gdf = _drop_empty_geoms(gdf)
        return gdf
    except Exception:
        return None


def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {
        "Distritos": ["Distritos"],
        "SetoresCensitarios2023": ["SetoresCensitarios2023", "IDCenso2023", "IDCenso"],
        "ZonasOD2023": ["ZonasOD2023", "ZonasOD2023", "ZonasOD"],
        "Subprefeitura": ["Subprefeitura", "subprefeitura"],
        "Isócronas": ["isocronas", "isócronas", "Isocronas"],
    }.get(name, [name])
    p = _pick_layer_file(DATA_DIR, stems)
    return read_gdf(str(p)) if p else None


def load_idcenso_geom() -> Optional["gpd.GeoDataFrame"]:
    p = _pick_layer_file(DATA_DIR, ["IDCenso2023", "IDCenso", "SetoresCensitarios2023"])
    return read_gdf(str(p)) if p else None


def load_setores_metrics_df() -> Optional[pd.DataFrame]:
    p = _pick_layer_file(DATA_DIR, ["SetoresCensitarios2023", "SetoresCensitarios"])
    if not p:
        return None
    try:
        df = pd.read_parquet(p)
        return df
    except Exception:
        return None


def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _pick_layer_file(DATA_DIR, ["isocronas", "isócronas", "Isocronas"])
    return read_gdf(str(p)) if p else None


def load_overlay_geo(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {
        "Áreas verdes": ["area_verde", "areas_verdes", "areasverdes", "areaverde"],
        "Rios": ["rios", "rio"],
        "Trem": ["linhas_trem", "linhasdetrem", "trem", "linha_trem"],
        "Metrô": ["linhas_metro", "linhasdemetro", "metro", "linha_metro"],
    }.get(name, [name])
    p = _pick_layer_file(DATA_DIR, stems)
    return read_gdf(str(p)) if p else None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def build_setores_joined() -> Optional["gpd.GeoDataFrame"]:
    """
    Geometria: IDCenso2023 (ou SetoresCensitarios2023 se for igual)
    Métricas: SetoresCensitarios2023
    Join: fid (normalizado)
    """
    if gpd is None:
        return None

    geom = load_idcenso_geom()
    dfm = load_setores_metrics_df()
    if geom is None or dfm is None:
        return None

    fid_g = find_col(list(geom.columns), "fid", "FID")
    if not fid_g:
        return None

    fid_m = find_col(list(dfm.columns), "fid", "FID")
    if not fid_m:
        return None

    g = geom[[fid_g, "geometry"]].copy()
    g["fid"] = _to_int_series(g[fid_g])
    g = g.drop(columns=[fid_g], errors="ignore")

    m = dfm.copy()
    m["fid"] = _to_int_series(m[fid_m])
    m = m.drop(columns=[fid_m], errors="ignore")

    # remove possíveis colunas geometry em dfm para não confundir
    for c in list(m.columns):
        if c.lower() in ("geometry", "geom"):
            m = m.drop(columns=[c], errors="ignore")

    merged = g.merge(m, on="fid", how="left")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=geom.crs)
    merged = _ensure_4326(merged)
    merged = _drop_empty_geoms(merged)
    return merged


# =============================================================================
# Folium – Base Carto + Panes
# =============================================================================
def make_carto_map(center: Tuple[float, float], zoom: int = 11) -> "folium.Map":
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)

    # Basemap fixo – CartoDB Positron (leve)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Carto Positron",
        overlay=False,
        control=False,
    ).add_to(m)

    # Panes (ordem importa)
    try:
        folium.map.CustomPane("admin", z_index=520).add_to(m)
        folium.map.CustomPane("setores_fill", z_index=560).add_to(m)
        folium.map.CustomPane("setores_line", z_index=610).add_to(m)
        folium.map.CustomPane("rios", z_index=640).add_to(m)
        folium.map.CustomPane("rail", z_index=650).add_to(m)
        folium.map.CustomPane("green", z_index=680).add_to(m)  # acima de tudo
    except Exception:
        pass

    return m


def add_outline_lines(m: "folium.Map", gdf: "gpd.GeoDataFrame", name: str, color: str, weight: float, pane: str):
    if gdf is None or gdf.empty:
        return

    gg = gdf.copy()
    # boundary para polígonos; linhas ficam como estão
    try:
        geom_types = set(gg.geometry.geom_type.unique())
    except Exception:
        geom_types = set()

    if any(t in ("Polygon", "MultiPolygon") for t in geom_types):
        gg["geometry"] = gg.geometry.boundary

    gg = _drop_empty_geoms(gg)
    gg = _simplify_m(gg, SIMPLIFY_M_LINE)

    # IMPORTANTÍSSIMO: não mandar feature com geometry None (evita KeyError no get_bounds)
    gj = gg[["geometry"]].to_json()

    folium.GeoJson(
        data=gj,
        name=name,
        pane=pane,
        style_function=lambda f: {"fillOpacity": 0, "color": color, "weight": weight, "opacity": 1.0},
    ).add_to(m)


def add_green_areas(m: "folium.Map", gdf: "gpd.GeoDataFrame"):
    if gdf is None or gdf.empty:
        return
    gg = gdf.copy()
    gg = _drop_empty_geoms(gg)
    gg = _simplify_m(gg, 12)  # pode ser mais agressivo; é só overlay
    gj = gg[["geometry"]].to_json()

    folium.GeoJson(
        data=gj,
        name="Áreas verdes",
        pane="green",
        style_function=lambda f: {"fillOpacity": 1.0, "color": "#1b7f3a", "weight": 0.6, "opacity": 1.0},
    ).add_to(m)


def add_rivers(m: "folium.Map", gdf: "gpd.GeoDataFrame"):
    if gdf is None or gdf.empty:
        return
    gg = gdf.copy()
    gg = _drop_empty_geoms(gg)
    gg = _simplify_m(gg, 18)
    gj = gg[["geometry"]].to_json()

    folium.GeoJson(
        data=gj,
        name="Rios",
        pane="rios",
        style_function=lambda f: {"fillOpacity": 0, "color": "#1f78b4", "weight": 1.3, "opacity": 1.0},
    ).add_to(m)


def add_rail_lines(m: "folium.Map", gdf: "gpd.GeoDataFrame", label: str):
    if gdf is None or gdf.empty:
        return
    gg = gdf.copy()
    gg = _drop_empty_geoms(gg)
    gg = _simplify_m(gg, 18)
    gj = gg[["geometry"]].to_json()

    folium.GeoJson(
        data=gj,
        name=label,
        pane="rail",
        style_function=lambda f: {"fillOpacity": 0, "color": "#111111", "weight": 1.6, "opacity": 1.0},
    ).add_to(m)


# =============================================================================
# Coroplético (Jenks 6 classes) – mais leve + cache
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=24)
def jenks_classes(values: List[float], k: int = 6) -> List[float]:
    """
    Retorna os breaks (bins) do Jenks com k classes.
    Usa mapclassify se disponível.
    """
    v = pd.Series(values).dropna().astype(float)
    if v.empty:
        return []
    try:
        import mapclassify as mc  # type: ignore
        nb = mc.NaturalBreaks(v.values, k=k)
        return list(nb.bins)  # tamanho k
    except Exception:
        # fallback quantis
        qs = [v.quantile(i / k) for i in range(1, k + 1)]
        return [float(x) for x in qs]


def _bin_index(x: float, bins: List[float]) -> int:
    # bins são os limites superiores de cada classe
    for i, b in enumerate(bins):
        if x <= b:
            return i
    return max(0, len(bins) - 1)


def paint_choropleth_setores(
    m: "folium.Map",
    setores: "gpd.GeoDataFrame",
    value_col: str,
    title: str,
):
    if setores is None or setores.empty:
        return

    s = pd.to_numeric(setores[value_col], errors="coerce").astype(float)
    bins = jenks_classes(s.tolist(), k=6)
    if not bins:
        st.info(f"Sem valores válidos para '{title}'.")
        return

    # GeoJSON mínimo: geometry + class + value
    g = setores[["geometry"]].copy()
    g = _drop_empty_geoms(g)

    # simplifica preenchimento (ganho real)
    g = _simplify_m(g, SIMPLIFY_M_FILL)

    # reindex para alinhar com s (garante mesmo comprimento)
    s2 = pd.to_numeric(setores.loc[g.index, value_col], errors="coerce").astype(float)

    g["__v__"] = s2
    g["__cls__"] = s2.apply(lambda x: _bin_index(float(x), bins) if pd.notna(x) else None)

    # remove nulos para reduzir payload
    g2 = g.dropna(subset=["__v__", "__cls__"]).copy()
    if g2.empty:
        st.info(f"Sem valores válidos para '{title}' após limpeza.")
        return

    def style_fn(feat):
        cls = feat["properties"].get("__cls__")
        try:
            i = int(cls)
        except Exception:
            i = 0
        i = max(0, min(5, i))
        return {
            "fillOpacity": 0.80,
            "weight": 0.0,
            "color": "#00000000",
            "fillColor": JENKS_COLORS_6[i],
        }

    folium.GeoJson(
        data=g2[["geometry", "__v__", "__cls__"]].to_json(),
        name=title,
        pane="setores_fill",
        style_function=style_fn,
        # tooltip opcional (pode pesar em 27k features; deixe simples)
        tooltip=folium.features.GeoJsonTooltip(
            fields=["__v__"],
            aliases=[f"{title}: "],
            localize=True,
            sticky=False,
            labels=False,
            class_name="pb-big-tooltip",
        ),
    ).add_to(m)


def paint_cluster_setores(m: "folium.Map", setores: "gpd.GeoDataFrame", cluster_col: str):
    if setores is None or setores.empty:
        return

    color_map = {0: "#bf7db2", 1: "#f7bd6a", 2: "#cf651f", 3: "#ede4e6", 4: "#793393"}
    default_color = "#c8c8c8"

    g = setores[["geometry"]].copy()
    g = _drop_empty_geoms(g)
    g = _simplify_m(g, SIMPLIFY_M_FILL)

    c = pd.to_numeric(setores.loc[g.index, cluster_col], errors="coerce").astype("Int64")
    g["__c__"] = c

    def style_fn(feat):
        v = feat["properties"].get("__c__")
        try:
            k = int(v)
        except Exception:
            k = -1
        col = color_map.get(k, default_color)
        return {"fillOpacity": 0.75, "weight": 0.0, "color": "#00000000", "fillColor": col}

    folium.GeoJson(
        data=g[["geometry", "__c__"]].to_json(),
        name="Cluster",
        pane="setores_fill",
        style_function=style_fn,
    ).add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "",  # começa vazio
            "Populacao",
            "densidade_demografica",
            "Diferenca_elevacao",
            "elevacao",
            "Isocrona",   # <-- pedida por você (coluna existe no seu print)
            "Cluster",
        ],
        index=0,
        key="pb_var",
        help="Começa vazio (só Carto). Variáveis são pintadas em Setores (join por fid).",
    )

    st.markdown("### Limites Administrativos")
    limite = st.selectbox(
        "Selecione o limite",
        ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0,
        key="pb_limite",
        help="Mostra contorno do limite selecionado + linhas dos setores como referência.",
    )

    st.markdown("### Camadas de referência")
    show_green = st.checkbox("Áreas verdes", value=True, key="pb_green")
    show_rivers = st.checkbox("Rios", value=True, key="pb_rivers")
    show_trem = st.checkbox("Linhas de trem", value=True, key="pb_trem")
    show_metro = st.checkbox("Linhas de metrô", value=True, key="pb_metro")

    st.caption("Se algo mudar no GitHub, limpe cache para recarregar.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.success("Cache limpo. Selecione novamente camada/variável.")

    return {
        "variavel": var,
        "limite": limite,
        "green": show_green,
        "rivers": show_rivers,
        "trem": show_trem,
        "metro": show_metro,
    }


# =============================================================================
# Header (logo no topo)
# =============================================================================
def render_header():
    logo_uri = _logo_data_uri()
    st.markdown(
        f"""
        <div class="pb-header">
          <div class="pb-row">
            <img class="pb-logo" src="{logo_uri}" />
            <div style="display:flex; flex-direction:column;">
              <div class="pb-title">PlanBairros</div>
              <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# App
# =============================================================================
def main() -> None:
    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium` instalados.")
        return

    inject_css()
    render_header()

    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        # 1) Centro base – sempre renderiza Carto (mesmo vazio)
        center = (-23.55, -46.63)

        # 2) Carrega limite (contorno)
        limite_gdf = load_admin_layer(ui["limite"])
        if limite_gdf is not None and not limite_gdf.empty:
            center = center_from_bounds(limite_gdf)

        fmap = make_carto_map(center=center, zoom=11)

        # 3) Sempre desenha: contorno do limite selecionado
        if limite_gdf is not None and not limite_gdf.empty:
            add_outline_lines(
                fmap,
                limite_gdf,
                name=f"{ui['limite']} (contorno)",
                color="#000000",
                weight=1.2,
                pane="admin",
            )
        else:
            st.info("Limite selecionado não encontrado. Verifique nomes dos arquivos em limites_administrativos/")

        # 4) Alteração #1 solicitada: linhas dos setores (IDCenso/Setores) como referência nos limites
        setores_lines = load_idcenso_geom()
        if setores_lines is not None and not setores_lines.empty:
            add_outline_lines(
                fmap,
                setores_lines,
                name="Setores (linhas)",
                color="#444444",
                weight=0.6,
                pane="setores_line",
            )

        # 5) Overlays de referência (agora lendo .geojson também)
        if ui["green"]:
            g = load_overlay_geo("Áreas verdes")
            if g is not None and not g.empty:
                add_green_areas(fmap, g)

        if ui["rivers"]:
            r = load_overlay_geo("Rios")
            if r is not None and not r.empty:
                add_rivers(fmap, r)

        if ui["trem"]:
            t = load_overlay_geo("Trem")
            if t is not None and not t.empty:
                add_rail_lines(fmap, t, "Linhas de trem")

        if ui["metro"]:
            mt = load_overlay_geo("Metrô")
            if mt is not None and not mt.empty:
                add_rail_lines(fmap, mt, "Linhas de metrô")

        # 6) Coroplético (só quando variável selecionada)
        var = ui["variavel"]

        if var:
            setores = build_setores_joined()
            if setores is None or setores.empty:
                st.warning(
                    "Não consegui montar Setores (join por fid). "
                    "Cheque se IDCenso2023.parquet e SetoresCensitarios2023.parquet existem e possuem 'fid'."
                )
            else:
                # nomes conforme seu print: Populacao / Diferenca_elevacao / Isocrona / Cluster / elevacao / densidade_demografica
                col = find_col(list(setores.columns), var, var.lower(), var.upper())
                if not col:
                    # tenta mapear var "densidade_demografica" caso esteja truncada/variante
                    if var == "densidade_demografica":
                        col = find_col(list(setores.columns), "densidade_demografica", "Densidade_demografica", "nsidade_demografica")
                if not col:
                    st.info(f"Coluna '{var}' não encontrada após join. Verifique grafia no parquet.")
                else:
                    if var == "Cluster":
                        paint_cluster_setores(fmap, setores, col)
                    else:
                        paint_choropleth_setores(fmap, setores, col, var)

        # 7) Render
        # Observação: o KeyError do get_bounds (st_folium) acontece quando um GeoJSON tem feature sem coordinates.
        # Nós já filtramos geometrias vazias/nulas e simplificamos antes de to_json.
        st_folium(
            fmap,
            height=790,
            use_container_width=True,
            key="pb_map",
            returned_objects=[],
        )


if __name__ == "__main__":
    main()
