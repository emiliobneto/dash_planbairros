# -*- coding: utf-8 -*-
"""
PlanBairros — app geoespacial (Streamlit + Folium)
Fluxo: Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setor Censitário)

Principais decisões de performance:
- Download sob demanda por nível (não baixa tudo no boot)
- Leitura filtrada (quando possível) via geopandas.read_parquet(filters=...)
- Multi-seleção (iso/quadra) com destaque visual
- Camada “sombra” do nível acima (marrom/telha, opaca, borda tracejada e fina)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, List, Iterable
import os
import re
import base64

import streamlit as st

# Geo
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# IDENTIDADE VISUAL
# =============================================================================
PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",  # marrom/telha (sombra do nível acima)
    "teal": "#6FA097",
    "navy": "#14407D",   # destaque seleção
}
PB_NAVY = PB_COLORS["navy"]
PB_TELHA = PB_COLORS["telha"]

CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

# acabamento/linhas
SIMPLIFY_TOL_LINES = 0.0006  # só para outlines (linhas)
DASH_SHADOW = "2, 6"         # tracejado “sombra”
DASH_OUTLINE = "1, 0"        # sólido


# =============================================================================
# CAMINHOS LOCAIS
# =============================================================================
REPO_ROOT = Path.cwd()
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46


# =============================================================================
# IDS E ENCADEAMENTO (VOCÊ VAI GERAR)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

# colunas pai
DIST_PARENT = "subpref_id"
ISO_PARENT = "distrito_id"
QUADRA_PARENT = "iso_id"
LOTE_PARENT = "quadra_id"
CENSO_PARENT = "iso_id"

LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]


# =============================================================================
# GOOGLE DRIVE LINKS (DEFAULT) + SECRETS (OPCIONAL)
# - Você pode colocar esses links em .streamlit/secrets.toml para não hardcodear
# - Mas o app também funciona com os DEFAULTs abaixo
# =============================================================================
DEFAULT_GDRIVE_LINKS = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist":   "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso":    "https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/1XKAYLNdt82ZPNAE-rseSuuaCFmjfn8IP/view?usp=drive_link",
    "lote":   "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo":  "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}

SECRET_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID",
    "dist": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID",
    "quadra": "PB_QUADRAS_FILE_ID",
    "lote": "PB_LOTES_FILE_ID",
    "censo": "PB_CENSO_FILE_ID",
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}


def _get_secret_or_default(secret_key: str, default: str) -> str:
    """Busca em st.secrets > env var > default."""
    v = ""
    try:
        v = str(st.secrets.get(secret_key, "")).strip()
    except Exception:
        v = ""
    if v:
        return v
    ev = str(os.getenv(secret_key, "")).strip()
    return ev if ev else default


def _extract_drive_id(url_or_id: str) -> str:
    """
    Aceita:
    - ID puro: 1AbC...
    - link: https://drive.google.com/file/d/<ID>/view?...
    - link: https://drive.google.com/open?id=<ID>
    - link: ...?id=<ID>
    """
    s = (url_or_id or "").strip()
    if not s:
        return ""
    # Se parece um ID
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return s
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m:
        return m.group(1)
    return ""


# resolve links (secrets sobrescrevem defaults)
GDRIVE_LINKS = {
    k: _get_secret_or_default(SECRET_KEYS[k], DEFAULT_GDRIVE_LINKS[k])
    for k in DEFAULT_GDRIVE_LINKS.keys()
}
GDRIVE_IDS = {k: _extract_drive_id(v) for k, v in GDRIVE_LINKS.items()}


# =============================================================================
# CSS / HEADER
# =============================================================================
def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        suf = LOGO_PATH.suffix.lstrip(".").lower()
        mime = "jpeg" if suf in ("jpg", "jpeg") else suf
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"
    return (
        "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/"
        "logomark/streamlit-mark-color.png"
    )


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
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; border-radius:8px; }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding:14px 15px; width:100%;
        }}
        .pb-title {{ font-size:2.25rem; font-weight:900; line-height:1.05; letter-spacing:.2px; }}
        .pb-subtitle {{ font-size:1.05rem; opacity:.95; margin-top:5px; }}

        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:10px;
        }}
        .pb-badge {{
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:999px;
            border:1px solid rgba(20,64,125,.12);
            background:#fff; color:#111; font-size:12px; font-weight:800;
        }}
        .pb-divider {{ height:1px; background:rgba(20,64,125,.10); margin:10px 0; }}
        .pb-note {{ font-size:12px; color:rgba(17,17,17,.75); line-height:1.35; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="pb-header">
          <div class="pb-row">
            <img src="{_logo_data_uri()}" class="pb-logo" />
            <div style="display:flex;flex-direction:column">
              <div class="pb-title">PlanBairros</div>
              <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STATE
# =============================================================================
def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi

    # final: quais camadas mostrar
    st.session_state.setdefault("show_final_lotes", True)
    st.session_state.setdefault("show_final_censo", True)

    # view
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)


def reset_to(level: str) -> None:
    st.session_state["level"] = level
    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["show_final_lotes"] = True
        st.session_state["show_final_censo"] = True
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()


def _back_one_level() -> None:
    cur = st.session_state["level"]
    idx = LEVELS.index(cur)
    if idx <= 0:
        return
    reset_to(LEVELS[idx - 1])


def _toggle_in_set(key: str, value: Any) -> None:
    s: Set[Any] = st.session_state.get(key, set())
    if value in s:
        s.remove(value)
    else:
        s.add(value)
    st.session_state[key] = s


# =============================================================================
# GOOGLE DRIVE DOWNLOAD (STREAMING)
# =============================================================================
def _ensure_drive_ids(keys: Iterable[str]) -> Optional[str]:
    missing = [k for k in keys if not GDRIVE_IDS.get(k)]
    if missing:
        return "Faltam FILE_IDs (ou links válidos) para: " + ", ".join(missing) + "."
    return None


@st.cache_resource(show_spinner=False)
def _download_drive_file(file_id: str, dst: Path) -> Path:
    """
    Download via requests (streaming), com token de confirmação do Google Drive.
    """
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    def _token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    r = session.get(URL, params={"id": file_id}, stream=True)
    tok = _token(r)
    if tok:
        r = session.get(URL, params={"id": file_id, "confirm": tok}, stream=True)

    if r.status_code != 200:
        raise RuntimeError(f"Download falhou (status={r.status_code}).")

    total = int(r.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024  # 1MB

    prog = st.progress(0, text=f"Baixando {dst.name}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in r.iter_content(chunk_size=chunk):
            if not part:
                continue
            f.write(part)
            downloaded += len(part)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                prog.progress(pct, text=f"Baixando {dst.name}… {pct}%")
    prog.empty()
    return dst


def ensure_local_layer(layer_key: str) -> Path:
    file_id = GDRIVE_IDS.get(layer_key, "")
    if not file_id:
        raise RuntimeError(f"FILE_ID inválido para '{layer_key}'.")
    dst = DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]
    return _download_drive_file(file_id, dst)


# =============================================================================
# LEITURA / LIMPEZA
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def read_gdf_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None or not path.exists():
        return None
    gdf_ = gpd.read_parquet(path)
    try:
        if gdf_.crs is None:
            gdf_ = gdf_.set_crs(4326, allow_override=True)
        else:
            gdf_ = gdf_.to_crs(4326)
    except Exception:
        pass
    return gdf_


@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def read_gdf_parquet_filtered(
    path: Path,
    filters: Optional[list] = None,
    columns: Optional[list] = None,
) -> Optional["gpd.GeoDataFrame"]:
    """
    Tenta leitura filtrada (pushdown) — importante para lotes/censo grandes.
    Se a versão do geopandas/pyarrow não suportar, cai no fallback.
    """
    if gpd is None or not path.exists():
        return None
    try:
        # geopandas.read_parquet suporta filters/columns em versões recentes (pyarrow)
        gdf_ = gpd.read_parquet(path, filters=filters, columns=columns)  # type: ignore
    except Exception:
        # fallback: lê tudo e filtra em memória (último recurso)
        gdf_ = gpd.read_parquet(path)
        if columns:
            keep = [c for c in columns if c in gdf_.columns]
            if "geometry" not in keep:
                keep.append("geometry")
            gdf_ = gdf_[keep]
        if filters:
            # filters no formato [(col, op, val)] ou [[...],[...]]
            # aqui tratamos apenas casos simples usados no app
            for flt in filters:
                if isinstance(flt, (list, tuple)) and len(flt) == 3:
                    col, op, val = flt
                    if col in gdf_.columns:
                        if op == "==":
                            gdf_ = gdf_[gdf_[col] == val]
                        elif op == "in":
                            gdf_ = gdf_[gdf_[col].isin(list(val))]
    try:
        if gdf_.crs is None:
            gdf_ = gdf_.set_crs(4326, allow_override=True)
        else:
            gdf_ = gdf_.to_crs(4326)
    except Exception:
        pass
    return gdf_


def _drop_bad_geoms(gdf_: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf_ is None or gdf_.empty:
        return gdf_
    gdf_ = gdf_.copy()
    gdf_ = gdf_[gdf_.geometry.notna()]
    try:
        gdf_ = gdf_[~gdf_.geometry.is_empty]
    except Exception:
        pass
    return gdf_


def _simplify_lines(gdf_: "gpd.GeoDataFrame", tol: float) -> "gpd.GeoDataFrame":
    if gdf_ is None or gdf_.empty:
        return gdf_
    gdf_ = gdf_.copy()
    try:
        gdf_["geometry"] = gdf_.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    return _drop_bad_geoms(gdf_)


def _validate_cols(gdf_: "gpd.GeoDataFrame", cols: List[str], layer_name: str) -> None:
    missing = [c for c in cols if c not in gdf_.columns]
    if missing:
        st.error(f"[{layer_name}] faltam colunas obrigatórias: {missing}")
        st.stop()


def bounds_center_zoom(gdf_: "gpd.GeoDataFrame") -> Tuple[Tuple[float, float], int]:
    minx, miny, maxx, maxy = gdf_.total_bounds
    center = ((miny + maxy) / 2, (minx + maxx) / 2)
    dx = maxx - minx
    if dx < 0.02:
        z = 16
    elif dx < 0.05:
        z = 15
    elif dx < 0.10:
        z = 14
    elif dx < 0.18:
        z = 13
    elif dx < 0.35:
        z = 12
    else:
        z = 11
    return center, z


def subset_by_parent(gdf_: "gpd.GeoDataFrame", parent_col: str, parent_val: Any) -> "gpd.GeoDataFrame":
    if gdf_ is None or gdf_.empty:
        return gdf_
    gdf_ = _drop_bad_geoms(gdf_)
    if parent_col not in gdf_.columns or parent_val is None:
        return gdf_.iloc[0:0].copy()
    return gdf_[gdf_[parent_col] == parent_val]


def subset_by_parent_multi(gdf_: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if gdf_ is None or gdf_.empty:
        return gdf_
    gdf_ = _drop_bad_geoms(gdf_)
    if parent_col not in gdf_.columns or not parent_vals:
        return gdf_.iloc[0:0].copy()
    return gdf_[gdf_[parent_col].isin(list(parent_vals))]


# =============================================================================
# HITTEST (CLICK)
# =============================================================================
def pick_feature_id(gdf_: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[Any]:
    if gdf_ is None or gdf_.empty or not click_latlon:
        return None
    if id_col not in gdf_.columns:
        return None

    lat = click_latlon.get("lat")
    lng = click_latlon.get("lng")
    if lat is None or lng is None:
        return None

    try:
        from shapely.geometry import Point  # type: ignore

        pt = Point(lng, lat)
        cand = gdf_

        # sindex acelera MUITO quando há muitas feições
        try:
            if hasattr(gdf_, "sindex") and gdf_.sindex is not None:
                idx = list(gdf_.sindex.intersection(pt.bounds))
                if idx:
                    cand = gdf_.iloc[idx]
        except Exception:
            pass

        hit = cand[cand.geometry.contains(pt)]
        if hit.empty:
            hit = cand[cand.geometry.intersects(pt)]
        if hit.empty:
            return None
        return hit.iloc[0][id_col]
    except Exception:
        return None


# =============================================================================
# MAPA (Folium)
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11):
    if folium is None:
        return None

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=CARTO_LIGHT_URL,
        attr=CARTO_ATTR,
        name="Carto Positron",
        overlay=False,
        control=False,
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)

    # panes: sombra abaixo, shapes acima
    try:
        folium.map.CustomPane("shadow", z_index=610).add_to(m)
        folium.map.CustomPane("detail", z_index=640).add_to(m)
        folium.map.CustomPane("outline", z_index=660).add_to(m)
    except Exception:
        pass

    return m


def _line_style(color: str, weight: float, dash: str) -> dict:
    return {
        "color": color,
        "weight": weight,
        "opacity": 1.0,
        "dashArray": dash,
        "lineCap": "round",
        "lineJoin": "round",
        "fillOpacity": 0,
    }


def add_outline_from_polygons(
    m,
    gdf_: "gpd.GeoDataFrame",
    name: str,
    color: str = "#111111",
    weight: float = 1.1,
    show: bool = True,
) -> None:
    """Desenha contorno (boundary) com acabamento melhor."""
    if folium is None or gdf_ is None or gdf_.empty:
        return

    line = gdf_[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary
    except Exception:
        return
    line = _simplify_lines(_drop_bad_geoms(line), SIMPLIFY_TOL_LINES)

    fg = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        data=line.to_json(),
        pane="outline",
        style_function=lambda _f: _line_style(color, weight, DASH_OUTLINE),
    ).add_to(fg)
    fg.add_to(m)


def add_shadow_polygons(
    m,
    gdf_: "gpd.GeoDataFrame",
    name: str,
    id_col: Optional[str] = None,
    keep_ids: Optional[Set[Any]] = None,
    pane: str = "shadow",
    fill_color: str = PB_TELHA,
    fill_opacity: float = 0.22,
    border_color: str = PB_TELHA,
    border_weight: float = 0.55,
    dash: str = DASH_SHADOW,
) -> None:
    """Camada ‘sombra’ do nível acima: marrom mais opaco, borda tracejada fininha."""
    if folium is None or gdf_ is None or gdf_.empty:
        return
    df = gdf_.copy()
    if id_col and keep_ids:
        if id_col in df.columns:
            df = df[df[id_col].isin(list(keep_ids))]
    df = _drop_bad_geoms(df)
    if df.empty:
        return

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=df[["geometry"] + ([id_col] if id_col and id_col in df.columns else [])].to_json(),
        pane=pane,
        style_function=lambda _f: {
            "color": border_color,
            "weight": border_weight,
            "dashArray": dash,
            "opacity": 1.0,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
            "lineCap": "round",
            "lineJoin": "round",
        },
    ).add_to(fg)
    fg.add_to(m)


def add_polygons_selectable(
    m,
    gdf_: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    selected_ids: Optional[Set[Any]] = None,
    pane: str = "detail",
    base_color: str = "#222222",
    base_weight: float = 0.65,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.08,
    selected_color: str = PB_NAVY,
    selected_weight: float = 2.2,
    selected_fill_opacity: float = 0.18,
    tooltip_fields: Optional[List[str]] = None,
    tooltip_aliases: Optional[List[str]] = None,
) -> None:
    if folium is None or gdf_ is None or gdf_.empty:
        return
    if id_col not in gdf_.columns:
        return

    selected_ids = selected_ids or set()

    cols = ["geometry", id_col]
    if tooltip_fields:
        for c in tooltip_fields:
            if c in gdf_.columns and c not in cols:
                cols.append(c)

    mini = gdf_[cols].copy()

    def style_fn(feat):
        props = feat.get("properties", {})
        fid = props.get(id_col)
        is_sel = fid in selected_ids if fid is not None else False
        return {
            "color": (selected_color if is_sel else base_color),
            "weight": (selected_weight if is_sel else base_weight),
            "opacity": 1.0,
            "fillColor": fill_color,
            "fillOpacity": (selected_fill_opacity if is_sel else fill_opacity),
            "lineCap": "round",
            "lineJoin": "round",
        }

    tooltip = None
    if tooltip_fields:
        tooltip = folium.features.GeoJsonTooltip(
            fields=[c for c in tooltip_fields if c in mini.columns],
            aliases=tooltip_aliases,
            sticky=True,
            labels=True,
        )

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=mini.to_json(),
        pane=pane,
        style_function=style_fn,
        highlight_function=lambda _f: {"weight": base_weight + 1.1, "fillOpacity": min(fill_opacity + 0.10, 0.35)},
        tooltip=tooltip,
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI
# =============================================================================
def left_panel() -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setor Censitário)")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("Voltar 1 nível", use_container_width=True, on_click=_back_one_level)
    with c2:
        st.button("Reset", type="secondary", use_container_width=True, on_click=reset_to, args=("subpref",))

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>📌 Seleção</span>", unsafe_allow_html=True)
    st.write(
        {
            "level": st.session_state["level"],
            "subpref_id": st.session_state["selected_subpref_id"],
            "distrito_id": st.session_state["selected_distrito_id"],
            "iso_ids_n": len(st.session_state["selected_iso_ids"]),
            "quadra_ids_n": len(st.session_state["selected_quadra_ids"]),
        }
    )

    if st.session_state["level"] == "final":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Nível final</span>", unsafe_allow_html=True)
        st.session_state["show_final_lotes"] = st.checkbox("Mostrar Lotes", value=st.session_state["show_final_lotes"])
        st.session_state["show_final_censo"] = st.checkbox("Mostrar Setor Censitário", value=st.session_state["show_final_censo"])

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>⚙️ Performance</span>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pb-note'>"
        "O app baixa e lê arquivos sob demanda por nível. "
        "Lotes e Setor Censitário (potencialmente grandes) só são baixados no nível final. "
        "A leitura tenta ser filtrada por ID (quando possível) para reduzir RAM."
        "</div>",
        unsafe_allow_html=True,
    )


def kpis_row() -> None:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>📍 Nível</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['level']}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🏛️ Subpref</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_subpref_id'] or '—'}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🗺️ Distrito</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_distrito_id'] or '—'}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Multi</span>", unsafe_allow_html=True)
        st.markdown(f"**Iso: {len(st.session_state['selected_iso_ids'])} | Quad: {len(st.session_state['selected_quadra_ids'])}**")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# APP
# =============================================================================
def main() -> None:
    init_state()
    inject_css()
    render_header()

    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium`.")
        return

    msg = _ensure_drive_ids(["subpref", "dist", "iso", "quadra", "lote", "censo"])
    if msg:
        st.error(msg)
        st.stop()

    left, right = st.columns([1, 4], gap="large")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        level = st.session_state["level"]

        # mapa base
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        # =====================================================================
        # NÍVEL 1 — SUBPREF
        # =====================================================================
        if level == "subpref":
            st.markdown("### Subprefeituras")

            sub_path = ensure_local_layer("subpref")
            g_sub = read_gdf_parquet(sub_path)
            if g_sub is None or g_sub.empty:
                st.error("Subprefeitura vazia/erro ao ler.")
                st.stop()

            g_sub = _drop_bad_geoms(g_sub)
            _validate_cols(g_sub, [SUBPREF_ID], "Subprefeitura")

            # linhas + polígonos clicáveis
            add_outline_from_polygons(m, g_sub, "Subprefeituras (linha)", color="#111111", weight=1.15, show=True)
            add_polygons_selectable(
                m,
                g_sub,
                "Subprefeituras (clique)",
                id_col=SUBPREF_ID,
                selected_ids=set(),
                base_weight=0.55,
                fill_opacity=0.03,
                selected_color=PB_NAVY,
                selected_weight=2.0,
                selected_fill_opacity=0.10,
                tooltip_fields=[SUBPREF_ID],
                tooltip_aliases=["subpref_id: "],
            )

        # =====================================================================
        # NÍVEL 2 — DISTRITOS (filtrados por subpref)
        # =====================================================================
        elif level == "distrito":
            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref")
                st.rerun()

            dist_path = ensure_local_layer("dist")
            g_dist = read_gdf_parquet(dist_path)
            if g_dist is None or g_dist.empty:
                st.error("Distrito vazio/erro ao ler.")
                st.stop()

            g_dist = _drop_bad_geoms(g_dist)
            _validate_cols(g_dist, [DIST_ID, DIST_PARENT], "Distrito")

            # sombra: subpref selecionada
            sub_path = ensure_local_layer("subpref")
            g_sub = _drop_bad_geoms(read_gdf_parquet(sub_path))
            if g_sub is not None and not g_sub.empty:
                add_shadow_polygons(
                    m,
                    g_sub[g_sub[SUBPREF_ID] == sp],
                    name="Sombra Subpref",
                    fill_opacity=0.24,
                    border_weight=0.55,
                )

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
            st.markdown(f"### Distritos (Subpref {sp})")

            add_outline_from_polygons(m, g_show, "Distritos (linha)", color="#111111", weight=0.95, show=True)
            add_polygons_selectable(
                m,
                g_show,
                "Distritos (clique)",
                id_col=DIST_ID,
                selected_ids=set(),
                base_weight=0.50,
                fill_opacity=0.05,
                tooltip_fields=[DIST_ID],
                tooltip_aliases=["distrito_id: "],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # =====================================================================
        # NÍVEL 3 — ISOCRONAS (multi) filtradas por distrito
        # =====================================================================
        elif level == "isocrona":
            d = st.session_state["selected_distrito_id"]
            if d is None:
                reset_to("distrito")
                st.rerun()

            iso_path = ensure_local_layer("iso")
            g_iso = read_gdf_parquet(iso_path)
            if g_iso is None or g_iso.empty:
                st.error("Isócronas vazias/erro ao ler.")
                st.stop()

            g_iso = _drop_bad_geoms(g_iso)
            _validate_cols(g_iso, [ISO_ID, ISO_PARENT], "Isócronas")

            # sombra: distrito selecionado
            dist_path = ensure_local_layer("dist")
            g_dist = _drop_bad_geoms(read_gdf_parquet(dist_path))
            if g_dist is not None and not g_dist.empty:
                add_shadow_polygons(
                    m,
                    g_dist[g_dist[DIST_ID] == d],
                    name="Sombra Distrito",
                    fill_opacity=0.24,
                    border_weight=0.55,
                )

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)
            st.markdown(f"### Isócronas (Distrito {d})")

            add_polygons_selectable(
                m,
                g_show,
                "Isócronas (multi)",
                id_col=ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_weight=0.55,
                fill_opacity=0.10,
                selected_color=PB_NAVY,
                selected_weight=2.1,
                selected_fill_opacity=0.18,
                tooltip_fields=[ISO_ID],
                tooltip_aliases=["iso_id: "],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # =====================================================================
        # NÍVEL 4 — QUADRAS (multi) filtradas por isócronas selecionadas
        # =====================================================================
        elif level == "quadra":
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            quadra_path = ensure_local_layer("quadra")
            g_q = read_gdf_parquet(quadra_path)
            if g_q is None or g_q.empty:
                st.error("Quadras vazias/erro ao ler.")
                st.stop()

            g_q = _drop_bad_geoms(g_q)
            _validate_cols(g_q, [QUADRA_ID, QUADRA_PARENT], "Quadras")

            # sombra: isócronas selecionadas (nível acima)
            iso_path = ensure_local_layer("iso")
            g_iso = _drop_bad_geoms(read_gdf_parquet(iso_path))
            if g_iso is not None and not g_iso.empty:
                add_shadow_polygons(
                    m,
                    g_iso,
                    name="Sombra Isócronas",
                    id_col=ISO_ID,
                    keep_ids=iso_ids,
                    fill_opacity=0.22,
                    border_weight=0.50,
                )

            g_show = subset_by_parent_multi(g_q, QUADRA_PARENT, iso_ids)
            st.markdown("### Quadras (Isócronas selecionadas)")

            add_polygons_selectable(
                m,
                g_show,
                "Quadras (multi)",
                id_col=QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_weight=0.45,
                fill_opacity=0.09,
                selected_color=PB_NAVY,
                selected_weight=2.0,
                selected_fill_opacity=0.18,
                tooltip_fields=[QUADRA_ID],
                tooltip_aliases=["quadra_id: "],
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

            # CTA para evitar “pular” automaticamente (reduz fricção na multi-seleção)
            st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
            st.caption("Selecione 1+ quadras. Depois avance para carregar Lotes/Setor.")
            st.button(
                "Avançar para nível final",
                type="primary",
                disabled=(len(st.session_state["selected_quadra_ids"]) == 0),
                on_click=lambda: st.session_state.__setitem__("level", "final"),
                use_container_width=True,
            )

        # =====================================================================
        # NÍVEL FINAL — LOTES (por quadra) e SETOR CENSITÁRIO (por iso)
        # =====================================================================
        else:
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            st.markdown("### Nível final — Lotes e Setor Censitário")

            # sombras “de referência”
            # - sombra das isócronas selecionadas (para ambos)
            iso_path = ensure_local_layer("iso")
            g_iso = _drop_bad_geoms(read_gdf_parquet(iso_path))
            if g_iso is not None and not g_iso.empty:
                add_shadow_polygons(
                    m,
                    g_iso,
                    name="Sombra Isócronas (final)",
                    id_col=ISO_ID,
                    keep_ids=iso_ids,
                    fill_opacity=0.22,
                    border_weight=0.50,
                )

            # - sombra das quadras selecionadas (para lotes)
            if quad_ids:
                quadra_path = ensure_local_layer("quadra")
                g_q = _drop_bad_geoms(read_gdf_parquet(quadra_path))
                if g_q is not None and not g_q.empty:
                    add_shadow_polygons(
                        m,
                        g_q,
                        name="Sombra Quadras (final)",
                        id_col=QUADRA_ID,
                        keep_ids=quad_ids,
                        fill_opacity=0.24,
                        border_weight=0.45,
                    )

            # Botões explícitos: evita crash por reruns automáticos
            st.warning("Atenção: Lotes e Setor Censitário podem ser pesados. Carregue sob demanda.")
            cA, cB = st.columns(2)
            with cA:
                load_lotes = st.button("⬇️ Carregar Lotes (por quadras selecionadas)", type="primary", use_container_width=True)
            with cB:
                load_censo = st.button("⬇️ Carregar Setor Censitário (por isócronas selecionadas)", type="primary", use_container_width=True)

            # LOTES
            if st.session_state["show_final_lotes"] and load_lotes:
                if not quad_ids:
                    st.info("Selecione quadras no nível anterior para carregar lotes.")
                else:
                    lote_path = ensure_local_layer("lote")

                    # leitura filtrada (reduz RAM): lote_parent IN quad_ids
                    filters = [(LOTE_PARENT, "in", sorted(list(quad_ids)))]
                    cols = [LOTE_ID, LOTE_PARENT, "geometry"]
                    g_l = read_gdf_parquet_filtered(lote_path, filters=filters, columns=cols)
                    if g_l is None or g_l.empty:
                        st.warning("Nenhum lote retornou no filtro. Verifique LOTE_PARENT/IDs.")
                    else:
                        g_l = _drop_bad_geoms(g_l)
                        _validate_cols(g_l, [LOTE_ID, LOTE_PARENT], "Lotes")

                        add_polygons_selectable(
                            m,
                            g_l,
                            "Lotes",
                            id_col=LOTE_ID,
                            selected_ids=set(),
                            base_weight=0.25,
                            fill_opacity=0.10,
                            selected_color=PB_NAVY,
                            selected_weight=1.6,
                            selected_fill_opacity=0.14,
                            tooltip_fields=[LOTE_ID],
                            tooltip_aliases=["lote_id: "],
                        )

                        center, zoom = bounds_center_zoom(g_l)
                        st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                        m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

            # CENSO
            if st.session_state["show_final_censo"] and load_censo:
                censo_path = ensure_local_layer("censo")

                # leitura filtrada: censo_parent IN iso_ids
                filters = [(CENSO_PARENT, "in", sorted(list(iso_ids)))]
                cols = [CENSO_ID, CENSO_PARENT, "geometry"]
                g_c = read_gdf_parquet_filtered(censo_path, filters=filters, columns=cols)
                if g_c is None or g_c.empty:
                    st.warning("Nenhum setor retornou no filtro. Verifique CENSO_PARENT/IDs.")
                else:
                    g_c = _drop_bad_geoms(g_c)
                    _validate_cols(g_c, [CENSO_ID, CENSO_PARENT], "Setor Censitário")

                    add_polygons_selectable(
                        m,
                        g_c,
                        "Setor Censitário",
                        id_col=CENSO_ID,
                        selected_ids=set(),
                        base_weight=0.35,
                        fill_opacity=0.07,
                        selected_color=PB_NAVY,
                        selected_weight=1.8,
                        selected_fill_opacity=0.12,
                        tooltip_fields=[CENSO_ID],
                        tooltip_aliases=["censo_id: "],
                    )

                    center, zoom = bounds_center_zoom(g_c)
                    st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                    m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # controles de camada
        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        # IMPORTANTÍSSIMO: não usar returned_objects=[]
        # Para performance, pedir só last_clicked.
        out = st_folium(
            m,
            height=780,
            use_container_width=True,
            key="map_view",
            returned_objects=["last_clicked"],
        )

        # =====================================================================
        # CLICK ROUTER (AVANÇO POR CLIQUE)
        # =====================================================================
        click = (out or {}).get("last_clicked")
        if click:
            # SUBPREF → DISTRITO
            if level == "subpref":
                sub_path = ensure_local_layer("subpref")
                g_sub = read_gdf_parquet(sub_path)
                if g_sub is not None and not g_sub.empty:
                    g_sub = _drop_bad_geoms(g_sub)
                    picked = pick_feature_id(g_sub, click, SUBPREF_ID)
                    if picked is not None:
                        st.session_state["selected_subpref_id"] = picked
                        st.session_state["level"] = "distrito"
                        st.rerun()

            # DISTRITO → ISO
            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                dist_path = ensure_local_layer("dist")
                g_dist = read_gdf_parquet(dist_path)
                if g_dist is not None and not g_dist.empty:
                    g_dist = _drop_bad_geoms(g_dist)
                    g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
                    picked = pick_feature_id(g_show, click, DIST_ID)
                    if picked is not None:
                        st.session_state["selected_distrito_id"] = picked
                        st.session_state["selected_iso_ids"] = set()
                        st.session_state["selected_quadra_ids"] = set()
                        st.session_state["level"] = "isocrona"
                        st.rerun()

            # ISO (multi toggle) → continua em ISO (mais rápido do que “pular” nível a cada clique)
            elif level == "isocrona":
                d = st.session_state["selected_distrito_id"]
                iso_path = ensure_local_layer("iso")
                g_iso = read_gdf_parquet(iso_path)
                if g_iso is not None and not g_iso.empty:
                    g_iso = _drop_bad_geoms(g_iso)
                    g_show = subset_by_parent(g_iso, ISO_PARENT, d)
                    picked = pick_feature_id(g_show, click, ISO_ID)
                    if picked is not None:
                        _toggle_in_set("selected_iso_ids", picked)
                        # quando mudar ISO, reset quadras
                        st.session_state["selected_quadra_ids"] = set()
                        # se ao menos 1 iso selecionada, pode ir pra quadra
                        st.session_state["level"] = "quadra"
                        st.rerun()

            # QUADRA (multi toggle) → permanece em QUADRA (não vai pro final automaticamente)
            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                quadra_path = ensure_local_layer("quadra")
                g_q = read_gdf_parquet(quadra_path)
                if g_q is not None and not g_q.empty:
                    g_q = _drop_bad_geoms(g_q)
                    g_show = subset_by_parent_multi(g_q, QUADRA_PARENT, iso_ids)
                    picked = pick_feature_id(g_show, click, QUADRA_ID)
                    if picked is not None:
                        _toggle_in_set("selected_quadra_ids", picked)
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
