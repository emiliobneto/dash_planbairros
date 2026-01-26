# -*- coding: utf-8 -*-
"""
PlanBairros — versão otimizada (consolidada + fix polígonos & performance)

Principais pontos:
- Exige limite administrativo antes de carregar variável (evita cidade inteira).
- Recorte rápido (sindex+bbox) e preciso (clip) com cache do unary_union.
- Sanitização de geometrias (make_valid, remove vazias) antes de desenhar.
- Contorno dos limites desenhado como boundary (linhas) para evitar PathLayer bug.
- Simplificação DINÂMICA após recorte: tolerância escala com nº de features.
- Redução de precisão antes da serialização (GeoJSON menor) + cache em sessão.
- Propriedades GeoJSON padronizadas: value / fill_color.
- Basemap Positron como padrão.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unicodedata import normalize as _ud_norm
import base64, math, re, json

import pandas as pd
import streamlit as st

from shapely.geometry import box
try:
    # Shapely 2.x
    from shapely import set_precision as _set_precision
except Exception:  # compat Shapely <2.0
    from shapely.set_precision import set_precision as _set_precision  # type: ignore

# make_valid com fallback
try:
    from shapely.validation import make_valid as _make_valid  # Shapely 2.x
except Exception:
    try:
        from shapely import make_valid as _make_valid  # alternativa
    except Exception:
        _make_valid = lambda g: g

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
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PB_NAVY = "#14407D"
ORANGE_RED_GRAD = [
    "#fff7ec",
    "#fee8c8",
    "#fdd49e",
    "#fdbb84",
    "#fc8d59",
    "#e34a33",
    "#b30000",
]
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

LOGO_HEIGHT = 120
MAP_HEIGHT = 900
SIMPLIFY_M = 60   # metros (EPSG:3857) — base
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


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
except NameError:  # streamlit cloud
    REPO_ROOT = Path.cwd()

DATA_DIR = (
    REPO_ROOT / "limites_administrativos"
    if (REPO_ROOT / "limites_administrativos").exists()
    else REPO_ROOT
)
GEOM_FILE = DATA_DIR / "IDCenso2023.parquet"                 # fid + geometry
METRICS_FILE = DATA_DIR / "SetoresCensitarios2023.parquet"   # fid + métricas
LOGO_PATH = REPO_ROOT / "assets" / "logo_todos.jpg"


def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f"data:image/{LOGO_PATH.suffix.lstrip('.').lower()};base64,{b64}"
    return (
        "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"
    )


def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())


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


def center_from_bounds(gdf) -> tuple[float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny + maxy) / 2, (minx + maxx) / 2)


# ====================== leitores ======================

def _read_gdf_robusto(
    path: Path, columns: Optional[List[str]] = None
) -> Optional["gpd.GeoDataFrame"]:
    if not path.exists():
        return None
    try:
        gdf = gpd.read_parquet(path, columns=columns)
        if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
            raise ValueError
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
            if (
                vals.dropna().astype(str).str.startswith(("POLY", "MULTI", "LINE", "POINT")).any()
            ):
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
            gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
            return gdf
        except Exception:
            return None


def _sanitize_gdf(gdf: Optional["gpd.GeoDataFrame"], only_types: Optional[List[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    """Remove geometrias nulas/vazias e corrige inválidas via make_valid. Opcionalmente filtra por tipo."""
    if gdf is None:
        return None
    try:
        g = gdf.copy()
        try:
            g["geometry"] = g.geometry.apply(_make_valid)
        except Exception:
            pass
        g = g[g.geometry.notnull()]
        try:
            g = g[~g.geometry.is_empty]
        except Exception:
            pass
        if only_types is not None:
            try:
                g = g[g.geometry.geom_type.isin(only_types)]
            except Exception:
                pass
        return g
    except Exception:
        return gdf


def _reduce_precision(gdf: "gpd.GeoDataFrame", grid: float = 1e-5) -> "gpd.GeoDataFrame":
    try:
        g2 = gdf.copy()
        g2["geometry"] = g2.geometry.apply(lambda geom: _set_precision(geom, grid))
        return g2
    except Exception:
        return gdf


def _dynamic_simplify(gdf: "gpd.GeoDataFrame", base_tol: int = SIMPLIFY_M) -> "gpd.GeoDataFrame":
    """Simplifica em EPSG:3857 com tolerância que escala com nº de features."""
    n = len(gdf)
    mult = 1.0
    if n > 50000:
        mult = 2.0
    elif n > 25000:
        mult = 1.5
    tol = int(base_tol * mult)
    try:
        jm = gdf.to_crs(3857)
        jm["geometry"] = jm.geometry.simplify(tol, preserve_topology=True)
        return jm.to_crs(4326)
    except Exception:
        return gdf


@st.cache_data(show_spinner=False, ttl=3600)
def load_setores_geom() -> Tuple[Optional["gpd.GeoDataFrame"], Optional[str]]:
    if not GEOM_FILE.exists():
        return None, None
    gdf = _read_gdf_robusto(GEOM_FILE, ["fid", "geometry"])
    if gdf is None:
        return None, None
    return gdf, "fid"


@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_metric_column(var_label: str) -> Optional[pd.DataFrame]:
    if not METRICS_FILE.exists():
        return None
    cols = pq.ParquetFile(str(METRICS_FILE)).schema.names
    mapping = {
        "População (Pessoa/ha)": "populacao",
        "Densidade demográfica (hab/ha)": "densidade_demografica",
        "Variação de elevação média": "diferenca_elevacao",
        "Elevação média": "elevacao",
        "Cluster (perfil urbano)": "cluster",
    }
    wanted = find_col(cols, mapping.get(var_label, var_label))
    if not wanted or "fid" not in cols:
        return None
    table = pq.read_table(str(METRICS_FILE), columns=["fid", wanted])
    df = table.to_pandas()
    df.rename(columns={wanted: "value"}, inplace=True)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    for name in ("isocronas.parquet", "isócronas.parquet"):
        p = DATA_DIR / name
        if p.exists():
            return _read_gdf_robusto(p, ["geometry", "nova_class"])
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    """Carrega camada administrativa por nome, com busca robusta por arquivo."""
    target = _slug(name)
    direct = {
        "distritos": DATA_DIR / "Distritos.parquet",
        "zonasod2023": DATA_DIR / "ZonasOD2023.parquet",
        "subprefeitura": DATA_DIR / "Subprefeitura.parquet",
        "isocronas": DATA_DIR / "isocronas.parquet",
        "isocronas2": DATA_DIR / "isócronas.parquet",
    }
    cand = None
    for k, p in direct.items():
        if p.exists() and (target in k or k in target):
            cand = p
            break
    if cand is None:
        for p in DATA_DIR.rglob("*.parquet"):
            if _slug(p.stem) in (target, f"{target}2023", f"{target}s"):
                cand = p
                break
        if cand is None:
            for p in DATA_DIR.rglob("*.geojson"):
                if _slug(p.stem) in (target, f"{target}2023", f"{target}s"):
                    cand = p
                    break
    if cand is None:
        st.warning(f"Camada '{name}' não encontrada em {DATA_DIR}.")
        return None
    gdf = _read_gdf_robusto(cand, ["geometry"])
    if gdf is None or gdf.empty:
        st.warning(f"Não foi possível ler '{cand.name}'. Verifique arquivo/colunas.")
        return None
    return gdf


# ====================== helpers de performance ======================

def _hex_to_rgba(h: str, a: int = 190) -> List[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [a]


def _sample_gradient(colors: List[str], n: int) -> List[str]:
    if n <= 1:
        return [colors[-1]]
    out = []
    for i in range(n):
        t = i / (n - 1)
        pos = t * (len(colors) - 1)
        j = int(math.floor(pos))
        j = min(j, len(colors) - 2)
        frac = pos - j
        def h2r(x):
            x = x.lstrip("#")
            return [int(x[k : k + 2], 16) for k in (0, 2, 4)]
        def r2h(r):
            return "#{:02x}{:02x}{:02x}".format(*r)
        c1, c2 = h2r(colors[j]), h2r(colors[j + 1])
        mix = [int(c1[k] + frac * (c2[k] - c1[k])) for k in range(3)]
        out.append(r2h(mix))
    return out


def classify_soft6(series: pd.Series) -> Tuple[pd.Series, List[Tuple[float, float]]]:
    v = pd.to_numeric(series, errors="coerce")
    if v.dropna().empty:
        return pd.Series([-1] * len(v), index=v.index), []
    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        bins = [vmin - 1e-9, vmax + 1e-9]
        idx = pd.cut(v, bins=bins, labels=False, include_lowest=True)
        return idx.fillna(-1).astype("Int64"), [(vmin, vmax)]
    step = (vmax - vmin) / 6.0
    edges = [vmin + i * step for i in range(7)]
    edges[-1] = vmax + 1e-9
    idx = pd.cut(v, bins=edges, labels=False, include_lowest=True)
    breaks = [(edges[i], edges[i + 1]) for i in range(6)]
    return idx.fillna(-1).astype("Int64"), breaks


# ---- cache leve de GeoJSON na sessão ----

def _gj_cache_get(key: str):
    return st.session_state.setdefault("_gj_cache", {}).get(key)

def _gj_cache_set(key: str, value):
    st.session_state.setdefault("_gj_cache", {})[key] = value

def geojson_from_gdf(gdf: "gpd.GeoDataFrame", props: List[str], cache_key: str):
    gj = _gj_cache_get(cache_key)
    if gj is None:
        gj = json.loads(gdf[props].to_json())
        _gj_cache_set(cache_key, gj)
    return gj


def _prefilter_by_bbox(
    gdf: Optional["gpd.GeoDataFrame"], limit_gdf: Optional["gpd.GeoDataFrame"]
) -> Optional["gpd.GeoDataFrame"]:
    if gdf is None or gdf.empty or limit_gdf is None or limit_gdf.empty:
        return gdf
    try:
        b = box(*limit_gdf.total_bounds)
        idx = gdf.sindex.query(b, predicate="intersects")
        return gdf.iloc[idx]
    except Exception:
        return gdf


def clip_to_limit(
    gdf: Optional["gpd.GeoDataFrame"], limit_gdf: Optional["gpd.GeoDataFrame"], *, fast: bool = False
) -> Optional["gpd.GeoDataFrame"]:
    if gdf is None or gdf.empty or limit_gdf is None or limit_gdf.empty:
        return gdf
    cand = _prefilter_by_bbox(gdf, limit_gdf)
    if cand is None or cand.empty:
        return cand
    if fast:
        return cand
    try:
        key = (
            f"_limit_union|{len(limit_gdf)}|"
            f"{int(limit_gdf.total_bounds[0]*1e6)}-"
            f"{int(limit_gdf.total_bounds[1]*1e6)}-"
            f"{int(limit_gdf.total_bounds[2]*1e6)}-"
            f"{int(limit_gdf.total_bounds[3]*1e6)}"
        )
        union = st.session_state.get(key)
        if union is None:
            union = limit_gdf.unary_union
            st.session_state[key] = union
        return gpd.clip(cand, union)
    except Exception:
        minx, miny, maxx, maxy = limit_gdf.total_bounds
        try:
            return cand.cx[minx:maxx, miny:maxy]
        except Exception:
            return cand


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
            "Variação de elevação média",
            "Elevação média",
            "Cluster (perfil urbano)",
            "Área de influência de bairro",
        ],
        index=0,
        key="pb_var",
        placeholder="Escolha…",
    )
    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        [
            PLACEHOLDER_LIM,
            "Distritos",
            "ZonasOD2023",
            "Subprefeitura",
            "Isócronas",
            "Setores Censitários (linhas)",
        ],
        index=0,
        key="pb_limite",
        placeholder="Escolha…",
    )
    st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")
    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear()
        st.session_state.pop("_gj_cache", None)
        st.success("Cache limpo. Selecione novamente a camada/variável.")

    legend_ph = st.empty()
    st.session_state["_legend_ph"] = legend_ph

    st.session_state["_pb_prev_sel"] = {"var": var, "lim": limite}

    # gating: exige limite antes de variável
    if var != PLACEHOLDER_VAR and limite == PLACEHOLDER_LIM:
        st.info("Selecione um **limite administrativo** antes de carregar a variável.")

    return {"variavel": var, "limite": limite}


# ====================== Legendas (HTML) ======================

def show_numeric_legend(title: str, breaks: List[Tuple[float, float]], palette: List[str]):
    ph = st.session_state.get("_legend_ph")
    if not ph:
        return
    if not breaks:
        ph.empty(); return
    rows = []
    for (a, b), col in zip(breaks, palette):
        label = f"{a:,.0f} – {b:,.0f}"
        rows.append(
            f"<div class='legend-row'><span class='legend-swatch' style='background:{col}'></span><span>{label}</span></div>"
        )
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)


def show_categorical_legend(title: str, items: List[Tuple[str, str]]):
    ph = st.session_state.get("_legend_ph")
    if not ph:
        return
    rows = [
        f"<div class='legend-row'><span class='legend-swatch' style='background:{c}'></span><span>{l}</span></div>"
        for l, c in items
    ]
    html = f"<div class='legend-card'><div class='legend-title'>{title}</div>{''.join(rows)}</div>"
    ph.markdown(html, unsafe_allow_html=True)


def clear_legend():
    ph = st.session_state.get("_legend_ph")
    if ph:
        ph.empty()


# ====================== Render (pydeck) ======================

def _show_deck(deck: "pdk.Deck"):
    try:
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)


def render_pydeck(
    center: Tuple[float, float],
    setores_joined: Optional["gpd.GeoDataFrame"],
    limite_gdf: Optional["gpd.GeoDataFrame"],
    var_label: Optional[str],
    draw_setores_outline: bool = False,
):
    layers: List[Any] = []

    # camada cloroplética
    legend_done = False
    if setores_joined is not None and not setores_joined.empty and "value" in setores_joined.columns:
        gdf = setores_joined.copy()

        if var_label == "Cluster (perfil urbano)":
            cmap = {0: "#bf7db2", 1: "#f7bd6a", 2: "#cf651f", 3: "#ede4e6", 4: "#793393"}
            labels = {
                0: "1 - Periférico com predominância residencial de alta densidade construtiva",
                1: "2 - Uso misto de média densidade construtiva",
                2: "3 - Periférico com predominância residencial de média densidade construtiva",
                3: "4 - Verticalizado de uso-misto",
                4: "5 - Predominância de uso comercial e serviços",
            }
            s = pd.to_numeric(gdf["value"], errors="coerce")
            gdf["fill_color"] = s.map(
                lambda v: _hex_to_rgba(cmap.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200)
            )
            items = [(labels[k], cmap[k]) for k in sorted(cmap)]
            show_categorical_legend("Cluster (perfil urbano)", items); legend_done = True
        else:
            classes, breaks = classify_soft6(gdf["value"])  # quebras por valor
            palette = _sample_gradient(ORANGE_RED_GRAD, 6)
            color_map = {i: _hex_to_rgba(palette[i], 200) for i in range(6)}
            gdf["fill_color"] = classes.map(
                lambda k: color_map.get(int(k) if pd.notna(k) else -1, _hex_to_rgba("#c8c8c8", 200))
            )
            show_numeric_legend(var_label or "", breaks, palette); legend_done = True

        gdf_small = _reduce_precision(gdf)
        cache_key = f"setores|{var_label}|{len(gdf_small)}|{gdf_small.total_bounds.tobytes()}"
        geojson = geojson_from_gdf(
            gdf_small[["geometry", "fill_color", "value"]],
            ["geometry", "fill_color", "value"],
            cache_key,
        )

        setores_layer = pdk.Layer(
            "GeoJsonLayer",
            id="setores-fill",
            data=geojson,
            filled=True,
            stroked=False,
            pickable=False,
            auto_highlight=False,
            get_fill_color="properties.fill_color",
            get_line_width=0,
        )
        layers.append(setores_layer)

    # contorno administrativo como boundary
    if limite_gdf is not None and not limite_gdf.empty:
        lim_clean = _sanitize_gdf(limite_gdf, only_types=["Polygon", "MultiPolygon"])
        if lim_clean is None or lim_clean.empty:
            lim_clean = limite_gdf
        try:
            out = lim_clean.copy(); out["geometry"] = out.geometry.boundary
        except Exception:
            out = lim_clean
        out2 = _sanitize_gdf(out, only_types=["LineString", "MultiLineString"])
        if out2 is None or out2.empty:
            out2 = out
        if out2 is not None and not out2.empty:
            cache_key = f"limite|{len(out2)}|{out2.total_bounds.tobytes()}"
            gj_lim = geojson_from_gdf(out2[["geometry"]], ["geometry"], cache_key)
            outline = pdk.Layer(
                "GeoJsonLayer",
                id="admin-outline",
                data=gj_lim,
                filled=False,
                stroked=True,
                get_line_color=[20, 20, 20, 220],
                get_line_width=2,
                lineWidthUnits="pixels",
            )
            layers.append(outline)

    # setores censitários como linhas (quando pedido)
    if draw_setores_outline:
        g_outline = None
        if setores_joined is not None and not setores_joined.empty:
            g_outline = setores_joined[["geometry"]]
        else:
            geoms_only, _ = load_setores_geom(); g_outline = geoms_only
        g_outline = _sanitize_gdf(g_outline, only_types=["Polygon", "MultiPolygon"]) if g_outline is not None else None
        if g_outline is not None and not g_outline.empty:
            try:
                g_outline = g_outline.copy(); g_outline["geometry"] = g_outline.geometry.boundary
            except Exception:
                pass
            cache_key = f"outline|{len(g_outline)}|{g_outline.total_bounds.tobytes()}"
            gj = geojson_from_gdf(g_outline[["geometry"]], ["geometry"], cache_key)
            sectors_outline = pdk.Layer(
                "GeoJsonLayer",
                id="setores-outline",
                data=gj,
                filled=False,
                stroked=True,
                get_line_color=[80, 80, 80, 160],
                get_line_width=0.6,
                lineWidthUnits="pixels",
            )
            layers.append(sectors_outline)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11, bearing=0, pitch=0),
        map_style=MAP_STYLE,
        tooltip=(
            {"text": f"{var_label}: {{value}}"}
            if var_label and var_label != "Cluster (perfil urbano)"
            else None
        ),
    )
    _show_deck(deck)

    if not legend_done:
        clear_legend()


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
        var = ui["variavel"]

        # limites (inclui opção "Setores Censitários (linhas)")
        limite_gdf = None
        draw_setores_outline = False
        if ui["limite"] != PLACEHOLDER_LIM:
            if ui["limite"] == "Setores Censitários (linhas)":
                draw_setores_outline = True
            else:
                limite_gdf = load_admin_layer(ui["limite"])
                if limite_gdf is not None and len(limite_gdf) > 0:
                    center = center_from_bounds(limite_gdf)
                else:
                    st.warning(f"Limite '{ui['limite']}' não encontrado/vasio em {DATA_DIR}.")

        # variável — só carrega se houver limite escolhido
        setores_joined: Optional["gpd.GeoDataFrame"] = None
        if var == "Área de influência de bairro" and limite_gdf is not None and not limite_gdf.empty:
            iso = load_isocronas()
            if iso is not None and len(iso) > 0:
                lut = {0: "#542788", 1: "#f7f7f7", 2: "#d8daeb", 3: "#b35806", 4: "#b2abd2",
                       5: "#8073ac", 6: "#fdb863", 7: "#7f3b08", 8: "#e08214", 9: "#fee0b6"}
                cls = find_col(iso.columns, "nova_class")
                if cls:
                    g = iso.copy()
                    g["value"] = pd.to_numeric(g[cls], errors="coerce")
                    g = clip_to_limit(g, limite_gdf, fast=False)
                    g = _sanitize_gdf(g)
                    g = _dynamic_simplify(g, base_tol=SIMPLIFY_M)
                    g["fill_color"] = g["value"].map(lambda v: _hex_to_rgba(lut.get(int(v) if pd.notna(v) else -1, "#c8c8c8"), 200))
                    g = _reduce_precision(g)
                    if g is not None and not g.empty:
                        setores_joined = g
                        center = center_from_bounds(g)
                    items = [(f"{k}", lut[k]) for k in sorted(lut)]
                    show_categorical_legend("Área de influência de bairro (nova_class)", items)
        elif var != PLACEHOLDER_VAR and limite_gdf is not None and not limite_gdf.empty:
            geoms, id_col = load_setores_geom()
            if geoms is not None and id_col == "fid":
                metric = load_metric_column(var)
                if metric is not None:
                    joined = geoms.merge(metric, on="fid", how="left")
                    # recorte + sanitização + simplificação dinâmica (sem amostrar)
                    fast_mode = len(joined) > 30000 or (limite_gdf is not None and len(limite_gdf) > 200)
                    joined = clip_to_limit(joined, limite_gdf, fast=fast_mode)
                    joined = _sanitize_gdf(joined)
                    if joined is not None and not joined.empty:
                        joined = _dynamic_simplify(joined, base_tol=SIMPLIFY_M)
                        joined = _reduce_precision(joined)
                        setores_joined = joined
                        center = center_from_bounds(joined)
        elif var != PLACEHOLDER_VAR and (limite_gdf is None or limite_gdf.empty):
            st.info("Para ver a variável, selecione um **limite administrativo**.")

        render_pydeck(
            center=center,
            setores_joined=setores_joined,
            limite_gdf=limite_gdf,
            var_label=None if var in (PLACEHOLDER_VAR, "Área de influência de bairro") else var,
            draw_setores_outline=draw_setores_outline,
        )


if __name__ == "__main__":
    main()

