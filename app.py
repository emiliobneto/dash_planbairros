# -*- coding: utf-8 -*-
"""
PlanBairros — Streamlit + Folium + GeoParquet (navegação hierárquica)

Fluxo (com seleção múltipla + botão "Prosseguir"):

1) Subprefeitura.parquet  (subpref_id)  -> seleção múltipla
2) Distrito.parquet       (distrito_id; subpref_id como FK) -> seleção múltipla + Prosseguir
3) Isocronas.parquet      (iso_id; distrito_id como FK) -> seleção múltipla + Prosseguir
4) Em Isócronas: usuário escolhe "Quadras" OU "Setor Censitário"
   4a) Quadras.parquet     (quadra_id; iso_id como FK) -> seleção múltipla + Prosseguir
       -> Lotes.parquet    (lote_id; quadra_id como FK)  (resultado)
   4b) Setorcensitario.parquet (censo_id; ligação preferencial via censo_id) -> seleção múltipla + Prosseguir
       -> (resultado: setores selecionados)

Fix do bug do zoom (mapa não recarrega ao dar zoom):
- `st_folium` com `returned_objects` mínimo (sem "center"/"zoom"/"bounds").
  Assim zoom/scroll não disparam rerun completo, evitando recarregar o mapa e
  evitando multiplicar caches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple
import base64
import re

import streamlit as st
import pandas as pd

# Geo (necessário para o app)
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium.features import GeoJsonTooltip  # type: ignore
    from shapely.geometry import Point  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    GeoJsonTooltip = None  # type: ignore
    Point = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# CONFIG / UI
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
PB_BROWN = PB_COLORS["telha"]
PB_BTN = "#1C6880"

# Tiles (Carto)
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

# Visual
SMOOTH_FACTOR = 1.0
LINE_CAP = "round"
LINE_JOIN = "round"

# "Sombra" do nível acima
PARENT_FILL_OPACITY = 0.16
PARENT_STROKE_OPACITY = 0.35
PARENT_STROKE_WEIGHT = 0.7
PARENT_STROKE_DASH = "2,6"

# Simplificação apenas visual
SIMPLIFY_TOL_BY_LAYER = {
    "subpref": 0.0000,
    "distrito": 0.0000,
    "iso": 0.0000,
    "quadra": 0.00000,
    "lote": 0.00000,
    "censo": 0.00000,
}

# =============================================================================
# IDs (colunas esperadas)
# =============================================================================
SUBPREF_ID = "subpref_id"

DIST_ID = "distrito_id"
DIST_SUBPREF_FK = "subpref_id"   # Distrito -> Subprefeitura (FK)

ISO_ID = "iso_id"
ISO_DIST_FK = "distrito_id"      # Isócronas -> Distrito (FK)

QUADRA_ID = "quadra_id"
QUADRA_ISO_FK = "iso_id"         # Quadras -> Isócronas (FK)
QUADRA_UID = "quadra_uid"        # chave composta (iso_id__quadra_id) p/ evitar colisões

LOTE_ID = "lote_id"
LOTE_QUADRA_FK = "quadra_id"     # Lotes -> Quadra (FK)

CENSO_ID = "censo_id"
ISO_CENSO_FK = "censo_id"        # Isócronas -> SetorCensitário (FK), se existir

# =============================================================================
# PATHS / ASSETS / DATA
# =============================================================================
REPO_ROOT = Path.cwd()
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

# Arquivos locais esperados (com aliases para tolerar variações de nome)
LOCAL_FILE_CANDIDATES = {
    "subpref": ["subprefeitura.parquet", "Subprefeitura.parquet"],
    "distrito": ["distrito.parquet", "Distrito.parquet"],
    "iso": ["Isocronas.parquet", "isocronas.parquet"],
    "quadra": ["Quadra.parquet", "Quadras.parquet", "quadras.parquet"],
    "lote": ["Lotes.parquet", "lotes.parquet"],
    "censo": ["Setorcensitario.parquet", "setorcensitario.parquet"],
}

# Nome "primário" usado quando o app baixar do Drive p/ data_cache
LOCAL_FILENAMES_PRIMARY = {
    "subpref": "Subprefeitura.parquet",
    "distrito": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    # Mantém compatibilidade com o seu app original, que usava "Quadras.parquet".
    # Se você preferir "Quadra.parquet", troque aqui (e no Drive, se quiser).
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}

# =============================================================================
# Drive — baixa e cacheia em data_cache/
# =============================================================================
SECRETS_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID",
    "distrito": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID",
    "quadra": "PB_QUADRAS_FILE_ID",
    "lote": "PB_LOTES_FILE_ID",
    "censo": "PB_CENSO_FILE_ID",
}

FALLBACK_URLS = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "distrito": "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso": "https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/17VaA-MlITota7shvbN8mNgf2MjWvUbVW/view?usp=drive_link",
    "lote": "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}

_DRIVE_ID_RE_1 = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_RE_2 = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def _get_secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


def extract_drive_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw
    m = _DRIVE_ID_RE_1.search(raw)
    if m:
        return m.group(1)
    m = _DRIVE_ID_RE_2.search(raw)
    if m:
        return m.group(1)
    m = re.search(r"([a-zA-Z0-9_-]{20,})", raw)
    return m.group(1) if m else ""


def get_drive_raw(layer_key: str) -> str:
    """
    Fonte do arquivo:
    1) input no painel (drive_<layer>_raw)
    2) secrets.toml (SECRETS_KEYS)
    3) fallback URL
    """
    raw_ui = str(st.session_state.get(f"drive_{layer_key}_raw", "")).strip()
    if raw_ui:
        return raw_ui
    raw_secret = _get_secret(SECRETS_KEYS.get(layer_key, ""))
    if raw_secret:
        return raw_secret
    return str(FALLBACK_URLS.get(layer_key, "")).strip()


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
    """Download resiliente do Google Drive (com token de confirmação)."""
    import requests

    file_id = extract_drive_id(file_id_or_url)
    if not file_id:
        raise RuntimeError("FILE_ID inválido (não foi possível extrair ID do link).")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    resp = session.get(url, params={"id": file_id}, stream=True)
    token = get_confirm_token(resp)
    if token:
        resp = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

    if resp.status_code != 200:
        raise RuntimeError(f"Download falhou (status={resp.status_code}).")

    total = int(resp.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024

    ui_label = label or dst.name
    prog = st.progress(0, text=f"Baixando {ui_label}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in resp.iter_content(chunk_size=chunk):
            if not part:
                continue
            f.write(part)
            downloaded += len(part)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                prog.progress(pct, text=f"Baixando {ui_label}… {pct}%")

    prog.empty()
    return dst


def _find_cached_or_repo_file(layer_key: str) -> Optional[Path]:
    """Procura 1º em data_cache, depois no repo root (com aliases)."""
    for name in LOCAL_FILE_CANDIDATES.get(layer_key, []):
        p = DATA_CACHE_DIR / name
        if p.exists() and p.stat().st_size > 0:
            return p
        p2 = REPO_ROOT / name
        if p2.exists() and p2.stat().st_size > 0:
            return p2
    return None


def ensure_local_layer(layer_key: str) -> Path:
    """
    Garante que exista um arquivo local para a camada:
    - se já existir localmente (qualquer alias), usa o existente.
    - senão, baixa do Drive (input/secrets/fallback) para o nome primário em data_cache.
    """
    p = _find_cached_or_repo_file(layer_key)
    if p:
        return p

    raw = get_drive_raw(layer_key)
    if not raw:
        raise RuntimeError(
            f"Layer '{layer_key}' não encontrada localmente e não há link/ID configurado (secrets ou input)."
        )

    dst = DATA_CACHE_DIR / LOCAL_FILENAMES_PRIMARY[layer_key]
    return download_drive_file(raw, dst, label=dst.name)


# =============================================================================
# CSS / HEADER
# =============================================================================
def _logo_data_uri() -> str:
    """Converte logo local em data URI (para não depender de rede)."""
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
            padding:12px;
        }}

        button[data-testid="stBaseButton-primary"],
        div[data-testid="stBaseButton-primary"] > button {{
            background:{PB_BTN} !important;
            color:#fff !important;
            border:1px solid {PB_BTN} !important;
        }}
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
# STATE (controle de fluxo e seleções)
# =============================================================================
LEVELS = ["subpref", "distrito", "iso", "quadra", "censo", "final"]


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("branch_mode", "quadra")  # "quadra" | "censo"

    st.session_state.setdefault("sel_subpref", set())
    st.session_state.setdefault("sel_distrito", set())
    st.session_state.setdefault("sel_iso", set())
    st.session_state.setdefault("sel_quadra", set())  # usa quadra_uid
    st.session_state.setdefault("sel_censo", set())

    st.session_state.setdefault("_last_click_sig", "")
    st.session_state.setdefault("_geojson_cache", {})  # type: ignore

    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)
    st.session_state.setdefault("_entered_level", None)


def clear_downstream(from_level: str) -> None:
    """Mantém consistência: mexeu num nível, zera seleções dependentes abaixo."""
    order = {
        "subpref": ["sel_distrito", "sel_iso", "sel_quadra", "sel_censo"],
        "distrito": ["sel_iso", "sel_quadra", "sel_censo"],
        "iso": ["sel_quadra", "sel_censo"],
        "quadra": [],
        "censo": [],
        "final": [],
    }
    for k in order.get(from_level, []):
        st.session_state[k] = set()


def goto(level: str) -> None:
    if level not in LEVELS:
        return
    st.session_state["level"] = level
    st.session_state["_last_click_sig"] = ""


def reset_all() -> None:
    st.session_state["level"] = "subpref"
    st.session_state["branch_mode"] = "quadra"
    st.session_state["sel_subpref"] = set()
    st.session_state["sel_distrito"] = set()
    st.session_state["sel_iso"] = set()
    st.session_state["sel_quadra"] = set()
    st.session_state["sel_censo"] = set()
    st.session_state["_last_click_sig"] = ""
    st.session_state["_geojson_cache"] = {}  # type: ignore
    st.session_state["view_center"] = (-23.55, -46.63)
    st.session_state["view_zoom"] = 11
    st.session_state["_entered_level"] = None


def _toggle_set(key: str, value: Any, *, level: str) -> None:
    s: Set[str] = set(st.session_state.get(key, set()) or set())
    v = _id_to_str(value)
    if v is None:
        return
    if v in s:
        s.remove(v)
    else:
        s.add(v)
    st.session_state[key] = s
    clear_downstream(level)


# =============================================================================
# ID HELPERS / NORMALIZAÇÃO
# =============================================================================
def _id_to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v).strip()
    if isinstance(v, int):
        return str(v)
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s or None


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


def normalize_quadra_id(v: Any, width: int = 6) -> Optional[str]:
    s = _id_to_str(v)
    if s is None:
        return None
    return s.zfill(width) if s.isdigit() else s


def make_quadra_uid(iso_id: Any, quadra_id: Any) -> Optional[str]:
    iso = _id_to_str(iso_id)
    qid = _id_to_str(quadra_id)
    if not iso or not qid:
        return None
    return f"{iso}__{qid}"


def split_quadra_uid(uid: str) -> Tuple[Optional[str], Optional[str]]:
    if not uid or "__" not in uid:
        return (None, None)
    a, b = uid.split("__", 1)
    return (_id_to_str(a), _id_to_str(b))


# =============================================================================
# READ (cache_data)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_gdf_parquet(path: str) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    gdf = gpd.read_parquet(p)
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        else:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf


def read_layer(layer_key: str) -> Optional["gpd.GeoDataFrame"]:
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        st.error(str(e))
        return None

    gdf = read_gdf_parquet(str(p))
    if gdf is None or gdf.empty:
        st.error(f"Layer '{layer_key}' vazia/erro ao ler: {p.name}")
        return None

    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass

    if layer_key == "subpref":
        gdf = normalize_id_cols(gdf, [SUBPREF_ID])
    elif layer_key == "distrito":
        gdf = normalize_id_cols(gdf, [DIST_ID, DIST_SUBPREF_FK])
    elif layer_key == "iso":
        gdf = normalize_id_cols(gdf, [ISO_ID, ISO_DIST_FK, ISO_CENSO_FK])
    elif layer_key == "quadra":
        gdf = normalize_id_cols(gdf, [QUADRA_ID, QUADRA_ISO_FK, ISO_ID])
        if QUADRA_ID in gdf.columns:
            gdf[QUADRA_ID] = gdf[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))
        if ISO_ID in gdf.columns and QUADRA_ID in gdf.columns:
            gdf[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(gdf[ISO_ID], gdf[QUADRA_ID])]
    elif layer_key == "lote":
        gdf = normalize_id_cols(gdf, [LOTE_ID, LOTE_QUADRA_FK, QUADRA_ID, ISO_ID])
    elif layer_key == "censo":
        gdf = normalize_id_cols(gdf, [CENSO_ID, ISO_ID])

    return gdf


# =============================================================================
# FILTERS (ligações hierárquicas)
# =============================================================================
def subset_by_ids(gdf: "gpd.GeoDataFrame", id_col: str, ids: Set[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty or id_col not in gdf.columns or not ids:
        return gdf.iloc[0:0].copy() if gdf is not None else gdf
    return gdf[gdf[id_col].isin(list(ids))]


def subset_by_parent_ids(gdf: "gpd.GeoDataFrame", parent_col: str, parent_ids: Set[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty or parent_col not in gdf.columns or not parent_ids:
        return gdf.iloc[0:0].copy() if gdf is not None else gdf
    return gdf[gdf[parent_col].isin(list(parent_ids))]


def filter_distritos(g_dist: "gpd.GeoDataFrame", subpref_ids: Set[str]) -> "gpd.GeoDataFrame":
    # Distrito: FK subpref_id -> Subprefeitura.subpref_id
    return subset_by_parent_ids(g_dist, DIST_SUBPREF_FK, subpref_ids)


def filter_isocronas(g_iso: "gpd.GeoDataFrame", distrito_ids: Set[str]) -> "gpd.GeoDataFrame":
    # Isócronas: FK distrito_id -> Distrito.distrito_id
    return subset_by_parent_ids(g_iso, ISO_DIST_FK, distrito_ids)


def filter_quadras(g_quad: "gpd.GeoDataFrame", iso_ids: Set[str]) -> "gpd.GeoDataFrame":
    # Quadras: FK iso_id -> Isocronas.iso_id
    return subset_by_parent_ids(g_quad, QUADRA_ISO_FK, iso_ids)


def filter_lotes(g_lote: "gpd.GeoDataFrame", quadra_uids: Set[str]) -> "gpd.GeoDataFrame":
    """
    Lotes: FK quadra_id -> Quadras.quadra_id

    Se Lotes tiver iso_id+quadra_id, filtramos de forma composta (mais preciso).
    Caso contrário, fallback por quadra_id.
    """
    if g_lote is None or g_lote.empty:
        return g_lote

    pairs = [(split_quadra_uid(uid)[0], split_quadra_uid(uid)[1]) for uid in quadra_uids]
    quadra_ids = {q for _, q in pairs if q}
    iso_ids = {i for i, _ in pairs if i}

    if ISO_ID in g_lote.columns and QUADRA_ID in g_lote.columns and iso_ids and quadra_ids:
        mask = False
        for iso_id, quadra_id in pairs:
            if iso_id and quadra_id:
                mask = mask | ((g_lote[ISO_ID] == iso_id) & (g_lote[QUADRA_ID] == quadra_id))
        return g_lote[mask] if hasattr(mask, "__len__") else g_lote.iloc[0:0].copy()

    return subset_by_parent_ids(g_lote, LOTE_QUADRA_FK, quadra_ids)


def filter_setores(g_censo: "gpd.GeoDataFrame", g_iso_selected: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    SetorCensitário: ligação preferencial via censo_id (solicitado).
    - Se Isócronas tiver censo_id, filtramos setores por censo_id.
    - Fallback: se SetorCensitário tiver iso_id, filtramos por iso_id.
    """
    if g_censo is None or g_censo.empty or g_iso_selected is None or g_iso_selected.empty:
        return g_censo.iloc[0:0].copy() if g_censo is not None else g_censo

    if ISO_CENSO_FK in g_iso_selected.columns and CENSO_ID in g_censo.columns:
        censo_ids = {c for c in g_iso_selected[ISO_CENSO_FK].dropna().astype(str).tolist() if c}
        if censo_ids:
            return subset_by_ids(g_censo, CENSO_ID, censo_ids)

    if ISO_ID in g_iso_selected.columns and ISO_ID in g_censo.columns:
        iso_ids = {i for i in g_iso_selected[ISO_ID].dropna().astype(str).tolist() if i}
        if iso_ids:
            return subset_by_parent_ids(g_censo, ISO_ID, iso_ids)

    return g_censo.iloc[0:0].copy()


# =============================================================================
# MAP HELPERS (GeoJSON cache + estilos)
# =============================================================================
def make_carto_map(center: Tuple[float, float], zoom: int):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles=CARTO_LIGHT_URL,
        attr=CARTO_ATTR,
        name="Carto Positron",
        overlay=False,
        control=False,
        subdomains="abcd",
        max_zoom=20,
    ).add_to(m)
    try:
        folium.map.CustomPane("parent_fill", z_index=610).add_to(m)
        folium.map.CustomPane("detail_shapes", z_index=640).add_to(m)
    except Exception:
        pass
    return m


def _mk_tooltip(field: str, prefix: str) -> Optional[Any]:
    if GeoJsonTooltip is None:
        return None
    return GeoJsonTooltip(
        fields=[field],
        aliases=[prefix],
        sticky=True,
        labels=True,
        localize=True,
        max_width=320,
    )


def _geojson_cache_get(key: str) -> Optional[str]:
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})  # type: ignore
    return cache.get(key)


def _geojson_cache_set(key: str, value: str, *, max_items: int = 80) -> None:
    """
    Cache em sessão + poda simples para evitar crescimento indefinido.
    """
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})  # type: ignore
    cache[key] = value
    if len(cache) > max_items:
        extra = len(cache) - max_items
        for k in list(cache.keys())[:extra]:
            cache.pop(k, None)
    st.session_state["_geojson_cache"] = cache  # type: ignore


def _simplify_to_geojson(gdf: "gpd.GeoDataFrame", simplify_tol: float, keep_cols: list[str]) -> str:
    if gdf is None or gdf.empty:
        return ""
    cols = list(dict.fromkeys(keep_cols + ["geometry"]))
    mini = gdf[cols].copy()
    try:
        mini["geometry"] = mini.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        pass
    try:
        return mini.to_json()
    except Exception:
        return ""


def add_parent_fill(
    m,
    gdf: "gpd.GeoDataFrame",
    cache_key: str,
    *,
    fill_color: str = PB_BROWN,
    fill_opacity: float = PARENT_FILL_OPACITY,
    stroke_color: str = PB_BROWN,
    stroke_weight: float = PARENT_STROKE_WEIGHT,
    stroke_opacity: float = PARENT_STROKE_OPACITY,
    dash_array: str = PARENT_STROKE_DASH,
    simplify_tol: float = 0.0006,
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return
    geojson = _geojson_cache_get(cache_key)
    if not geojson:
        geojson = _simplify_to_geojson(gdf, simplify_tol=simplify_tol, keep_cols=[])
        _geojson_cache_set(cache_key, geojson)
    if not geojson:
        return
    try:
        folium.GeoJson(
            data=geojson,
            pane="parent_fill",
            smooth_factor=SMOOTH_FACTOR,
            style_function=lambda _f: {
                "color": stroke_color,
                "weight": stroke_weight,
                "opacity": stroke_opacity,
                "dashArray": dash_array,
                "lineCap": LINE_CAP,
                "lineJoin": LINE_JOIN,
                "fillColor": fill_color,
                "fillOpacity": fill_opacity,
            },
        ).add_to(m)
    except Exception:
        return


def add_polygons_selectable(
    m,
    gdf: "gpd.GeoDataFrame",
    *,
    id_col: str,
    tooltip_col: Optional[str],
    selected_ids: Set[str],
    cache_key: str,
    tooltip_prefix: str,
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.10,
    selected_color: str = PB_NAVY,
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.26,
    simplify_tol: float = 0.0006,
) -> None:
    """
    Base + overlay de selecionados.
    Tooltip somente na BASE (evita comportamento ruim/“mapa branco”).
    """
    if folium is None or gdf is None or gdf.empty or id_col not in gdf.columns:
        return
    tooltip_col = tooltip_col or id_col
    if tooltip_col not in gdf.columns:
        tooltip_col = id_col

    geojson_base = _geojson_cache_get(cache_key)
    if not geojson_base:
        keep = [id_col] if tooltip_col == id_col else [id_col, tooltip_col]
        mini = gdf[keep + ["geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        mini[tooltip_col] = mini[tooltip_col].map(_id_to_str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol=simplify_tol, keep_cols=keep)
        _geojson_cache_set(cache_key, geojson_base)

    if not geojson_base:
        return

    tooltip = _mk_tooltip(tooltip_col, tooltip_prefix)

    try:
        folium.GeoJson(
            data=geojson_base,
            pane="detail_shapes",
            smooth_factor=SMOOTH_FACTOR,
            style_function=lambda _f: {
                "color": base_color,
                "weight": base_weight,
                "opacity": 0.92,
                "lineCap": LINE_CAP,
                "lineJoin": LINE_JOIN,
                "fillColor": fill_color,
                "fillOpacity": fill_opacity,
            },
            highlight_function=lambda _f: {"weight": base_weight + 1.4, "fillOpacity": min(fill_opacity + 0.12, 0.40)},
            tooltip=tooltip,
        ).add_to(m)
    except Exception:
        return

    sel = {s for s in (_id_to_str(x) for x in selected_ids) if s}
    if not sel:
        return

    sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
    if sel_gdf.empty:
        return
    sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
    geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=simplify_tol, keep_cols=[id_col])
    if not geojson_sel:
        return

    try:
        folium.GeoJson(
            data=geojson_sel,
            pane="detail_shapes",
            smooth_factor=SMOOTH_FACTOR,
            style_function=lambda _f: {
                "color": selected_color,
                "weight": selected_weight,
                "opacity": 0.98,
                "lineCap": LINE_CAP,
                "lineJoin": LINE_JOIN,
                "fillColor": fill_color,
                "fillOpacity": selected_fill_opacity,
            },
        ).add_to(m)
    except Exception:
        return


# =============================================================================
# CLICK / HITTEST
# =============================================================================
def parse_tooltip_id(tooltip: Any) -> Optional[str]:
    if not tooltip:
        return None
    if isinstance(tooltip, dict):
        tooltip = tooltip.get("text") or tooltip.get("tooltip") or str(tooltip)
    s = str(tooltip)
    s = re.sub(r"<[^>]+>", " ", s).strip()
    m = re.search(r":\s*([^\s<]+)", s)
    if m:
        return _id_to_str(m.group(1))
    m2 = re.search(r"([A-Za-z0-9_-]+)\s*$", s)
    return _id_to_str(m2.group(1)) if m2 else None


def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[str]:
    if gdf is None or gdf.empty or not click_latlon or Point is None or id_col not in gdf.columns:
        return None
    lat = click_latlon.get("lat")
    lng = click_latlon.get("lng")
    if lat is None or lng is None:
        return None
    try:
        pt = Point(lng, lat)
        cand = gdf
        try:
            if hasattr(gdf, "sindex") and gdf.sindex is not None:
                idx = list(gdf.sindex.intersection(pt.bounds))
                if idx:
                    cand = gdf.iloc[idx]
        except Exception:
            pass

        hit = cand[cand.geometry.contains(pt)]
        if hit.empty:
            hit = cand[cand.geometry.intersects(pt)]
        if hit.empty:
            return None
        return _id_to_str(hit.iloc[0][id_col])
    except Exception:
        return None


def _click_signature(tooltip_id: Optional[str], click: Optional[Dict[str, Any]]) -> str:
    tip = tooltip_id or ""
    lat = None
    lng = None
    if isinstance(click, dict):
        lat = click.get("lat")
        lng = click.get("lng")
    try:
        if lat is not None and lng is not None:
            return f"{tip}|{float(lat):.7f}|{float(lng):.7f}"
    except Exception:
        pass
    return f"{tip}|"


def consume_map_click(level: str, map_state: Dict[str, Any], g_show: Optional["gpd.GeoDataFrame"]) -> None:
    click = (map_state or {}).get("last_clicked") or None
    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip") or None
    picked_tooltip = parse_tooltip_id(tooltip_raw)

    if not click and not picked_tooltip:
        return

    sig = _click_signature(picked_tooltip, click)
    if sig and sig == st.session_state.get("_last_click_sig", ""):
        return
    st.session_state["_last_click_sig"] = sig

    if level == "subpref":
        id_col = SUBPREF_ID
        key = "sel_subpref"
    elif level == "distrito":
        id_col = DIST_ID
        key = "sel_distrito"
    elif level == "iso":
        id_col = ISO_ID
        key = "sel_iso"
    elif level == "quadra":
        id_col = QUADRA_UID if (g_show is not None and QUADRA_UID in g_show.columns) else QUADRA_ID
        key = "sel_quadra"
    elif level == "censo":
        id_col = CENSO_ID
        key = "sel_censo"
    else:
        return

    picked = picked_tooltip
    if not picked and isinstance(click, dict) and g_show is not None:
        picked = pick_feature_id(g_show, click, id_col)

    if picked:
        _toggle_set(key, picked, level=level)
        # sem depender de zoom/center do mapa (pois não retornamos esses campos)
        if isinstance(click, dict) and click.get("lat") is not None and click.get("lng") is not None:
            st.session_state["view_center"] = (float(click["lat"]), float(click["lng"]))


# =============================================================================
# VIEW (fit)
# =============================================================================
def bounds_center_zoom(gdf: "gpd.GeoDataFrame") -> Tuple[Tuple[float, float], int]:
    minx, miny, maxx, maxy = gdf.total_bounds
    center = ((miny + maxy) / 2, (minx + maxx) / 2)
    dx = maxx - minx
    if dx < 0.03:
        z = 15
    elif dx < 0.08:
        z = 14
    elif dx < 0.15:
        z = 13
    elif dx < 0.30:
        z = 12
    else:
        z = 11
    return center, z


def set_view_to_gdf(gdf: "gpd.GeoDataFrame", bump: int = 0, zmax: int = 19) -> None:
    if gdf is None or gdf.empty:
        return
    try:
        center, zoom = bounds_center_zoom(gdf)
        st.session_state["view_center"] = center
        st.session_state["view_zoom"] = min(zoom + bump, zmax)
    except Exception:
        return


def fit_current_level() -> None:
    level = st.session_state["level"]
    if level == "subpref":
        g = read_layer("subpref")
        if g is not None and st.session_state["sel_subpref"]:
            set_view_to_gdf(subset_by_ids(g, SUBPREF_ID, st.session_state["sel_subpref"]), bump=0)
    elif level == "distrito":
        g = read_layer("distrito")
        if g is not None and st.session_state["sel_distrito"]:
            set_view_to_gdf(subset_by_ids(g, DIST_ID, st.session_state["sel_distrito"]), bump=0)
    elif level == "iso":
        g = read_layer("iso")
        if g is not None and st.session_state["sel_iso"]:
            set_view_to_gdf(subset_by_ids(g, ISO_ID, st.session_state["sel_iso"]), bump=0)
    elif level == "quadra":
        g = read_layer("quadra")
        if g is not None and st.session_state["sel_quadra"]:
            id_col = QUADRA_UID if QUADRA_UID in g.columns else QUADRA_ID
            set_view_to_gdf(subset_by_ids(g, id_col, st.session_state["sel_quadra"]), bump=1)
    elif level == "censo":
        g = read_layer("censo")
        if g is not None and st.session_state["sel_censo"]:
            set_view_to_gdf(subset_by_ids(g, CENSO_ID, st.session_state["sel_censo"]), bump=0)
    elif level == "final":
        branch = st.session_state.get("branch_mode", "quadra")
        if branch == "quadra":
            g = read_layer("lote")
            if g is not None:
                set_view_to_gdf(filter_lotes(g, st.session_state["sel_quadra"]), bump=0)
        else:
            g = read_layer("censo")
            if g is not None:
                set_view_to_gdf(subset_by_ids(g, CENSO_ID, st.session_state["sel_censo"]), bump=0)


# =============================================================================
# UI — navegação hierárquica
# =============================================================================
def level_label(level: str) -> str:
    return {
        "subpref": "Subprefeituras",
        "distrito": "Distritos",
        "iso": "Isócronas",
        "quadra": "Quadras",
        "censo": "Setor Censitário",
        "final": "Resultado",
    }.get(level, level)


def back_level(level: str) -> str:
    if level == "distrito":
        return "subpref"
    if level == "iso":
        return "distrito"
    if level == "quadra":
        return "iso"
    if level == "censo":
        return "iso"
    if level == "final":
        return "quadra" if st.session_state.get("branch_mode") == "quadra" else "censo"
    return "subpref"


def left_panel() -> None:
    level = st.session_state["level"]

    c1, c2 = st.columns(2)
    with c1:
        if level == "subpref":
            st.button("Início", disabled=True, use_container_width=True)
        else:
            st.button(
                f"⬅️ Voltar: {level_label(back_level(level))}",
                type="primary",
                use_container_width=True,
                on_click=lambda: goto(back_level(level)),
            )
    with c2:
        st.button("Reset", type="primary", use_container_width=True, on_click=reset_all)

    st.divider()

    st.markdown("**Caminho atual**")
    st.write(
        f"Subpref ({len(st.session_state['sel_subpref'])}) → "
        f"Distrito ({len(st.session_state['sel_distrito'])}) → "
        f"Isócrona ({len(st.session_state['sel_iso'])}) → "
        f"Quadra ({len(st.session_state['sel_quadra'])}) / "
        f"Censo ({len(st.session_state['sel_censo'])})"
    )

    st.divider()

    if level == "subpref":
        ok = len(st.session_state["sel_subpref"]) > 0
        st.button(
            "Prosseguir → Distritos",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: goto("distrito"),
        )

    elif level == "distrito":
        ok = len(st.session_state["sel_distrito"]) > 0
        st.button(
            "Prosseguir → Isócronas",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: goto("iso"),
        )

    elif level == "iso":
        st.markdown("**Tipo de detalhamento**")
        st.radio(
            "Escolha a camada final",
            options=["Quadras", "Setor Censitário"],
            horizontal=False,
            index=0 if st.session_state.get("branch_mode") == "quadra" else 1,
            key="_branch_ui",
        )
        new_branch = "quadra" if st.session_state["_branch_ui"] == "Quadras" else "censo"
        if new_branch != st.session_state.get("branch_mode"):
            st.session_state["branch_mode"] = new_branch
            clear_downstream("iso")  # mudou ramo => limpa seleções do ramo

        ok = len(st.session_state["sel_iso"]) > 0
        nxt = "quadra" if st.session_state["branch_mode"] == "quadra" else "censo"
        st.button(
            f"Prosseguir → {level_label(nxt)}",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: goto(nxt),
        )

    elif level == "quadra":
        ok = len(st.session_state["sel_quadra"]) > 0
        st.button(
            "Prosseguir → Resultado (Lotes)",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: goto("final"),
        )

    elif level == "censo":
        ok = len(st.session_state["sel_censo"]) > 0
        st.button(
            "Prosseguir → Resultado (Setores)",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: goto("final"),
        )

    st.divider()
    st.markdown("**Ajustes de visualização**")
    st.button("Ajustar ao selecionado (nível atual)", use_container_width=True, on_click=fit_current_level)

    with st.expander("Dados / Drive (opcional)", expanded=False):
        st.caption("Cole links/IDs do Drive aqui apenas se não estiver usando secrets.toml.")
        for k in ["subpref", "distrito", "iso", "quadra", "lote", "censo"]:
            placeholder = _get_secret(SECRETS_KEYS.get(k, "")) or FALLBACK_URLS.get(k, "") or ""
            st.text_input(
                f"{k} ({LOCAL_FILENAMES_PRIMARY[k]})",
                key=f"drive_{k}_raw",
                value=str(st.session_state.get(f"drive_{k}_raw", "")).strip(),
                placeholder=placeholder,
            )


# =============================================================================
# MAP RENDER por nível (encadeamento hierárquico)
# =============================================================================
def render_map_panel() -> None:
    level = st.session_state["level"]

    g_show = None
    parent_fill = None
    title = level_label(level)

    if level == "subpref":
        g_sub = read_layer("subpref")
        if g_sub is None or SUBPREF_ID not in g_sub.columns:
            st.stop()
        g_show = g_sub
        title = "Subprefeituras"

    elif level == "distrito":
        g_sub = read_layer("subpref")
        g_dist = read_layer("distrito")
        if g_sub is None or g_dist is None:
            st.stop()

        g_show = filter_distritos(g_dist, st.session_state["sel_subpref"])
        parent_fill = subset_by_ids(g_sub, SUBPREF_ID, st.session_state["sel_subpref"])
        title = f"Distritos — Subpref(s): {len(st.session_state['sel_subpref'])}"

    elif level == "iso":
        g_dist = read_layer("distrito")
        g_iso = read_layer("iso")
        if g_dist is None or g_iso is None:
            st.stop()

        g_show = filter_isocronas(g_iso, st.session_state["sel_distrito"])
        parent_fill = subset_by_ids(g_dist, DIST_ID, st.session_state["sel_distrito"])
        title = f"Isócronas — Distrito(s): {len(st.session_state['sel_distrito'])}"

    elif level == "quadra":
        g_iso = read_layer("iso")
        g_quad = read_layer("quadra")
        if g_iso is None or g_quad is None:
            st.stop()

        g_show = filter_quadras(g_quad, st.session_state["sel_iso"])
        parent_fill = subset_by_ids(g_iso, ISO_ID, st.session_state["sel_iso"])
        title = f"Quadras — Isócrona(s): {len(st.session_state['sel_iso'])}"

    elif level == "censo":
        g_iso = read_layer("iso")
        g_censo = read_layer("censo")
        if g_iso is None or g_censo is None:
            st.stop()

        g_iso_sel = subset_by_ids(g_iso, ISO_ID, st.session_state["sel_iso"])
        g_show = filter_setores(g_censo, g_iso_sel)
        parent_fill = g_iso_sel
        title = f"Setor Censitário — Isócrona(s): {len(st.session_state['sel_iso'])}"

    elif level == "final":
        branch = st.session_state.get("branch_mode", "quadra")
        if branch == "quadra":
            g_lote = read_layer("lote")
            g_quad = read_layer("quadra")
            if g_lote is None or g_quad is None:
                st.stop()

            g_show = filter_lotes(g_lote, st.session_state["sel_quadra"])
            id_col = QUADRA_UID if QUADRA_UID in g_quad.columns else QUADRA_ID
            parent_fill = subset_by_ids(g_quad, id_col, st.session_state["sel_quadra"])
            title = f"Resultado — Lotes (quadras selecionadas: {len(st.session_state['sel_quadra'])})"
        else:
            g_censo = read_layer("censo")
            if g_censo is None:
                st.stop()
            g_show = subset_by_ids(g_censo, CENSO_ID, st.session_state["sel_censo"])
            title = f"Resultado — Setores (selecionados: {len(st.session_state['sel_censo'])})"

    if g_show is not None and not g_show.empty and st.session_state.get("_entered_level") != level:
        set_view_to_gdf(g_show, bump=0)
        st.session_state["_entered_level"] = level

    st.markdown(f"### {title}")

    m = make_carto_map(center=st.session_state["view_center"], zoom=int(st.session_state["view_zoom"]))
    if m is None:
        st.error("Falha ao inicializar o mapa (folium).")
        return

    if parent_fill is not None and not parent_fill.empty:
        add_parent_fill(
            m,
            parent_fill,
            cache_key=f"parent:{level}:{len(parent_fill)}:{SIMPLIFY_TOL_BY_LAYER.get(level, 0.0006)}",
            simplify_tol=SIMPLIFY_TOL_BY_LAYER.get(level, 0.0006),
        )

    if g_show is not None and not g_show.empty and level != "final":
        if level == "subpref":
            add_polygons_selectable(
                m,
                g_show,
                id_col=SUBPREF_ID,
                tooltip_col=SUBPREF_ID,
                selected_ids=st.session_state["sel_subpref"],
                cache_key=f"subpref:all:{SIMPLIFY_TOL_BY_LAYER['subpref']}",
                tooltip_prefix="Subpref: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["subpref"],
                base_weight=0.9,
                fill_opacity=0.04,
            )
        elif level == "distrito":
            sp_sig = "|".join(sorted(st.session_state["sel_subpref"]))
            add_polygons_selectable(
                m,
                g_show,
                id_col=DIST_ID,
                tooltip_col=DIST_ID,
                selected_ids=st.session_state["sel_distrito"],
                cache_key=f"dist:{sp_sig}:{SIMPLIFY_TOL_BY_LAYER['distrito']}",
                tooltip_prefix="Distrito: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["distrito"],
                base_weight=0.75,
                fill_opacity=0.06,
            )
        elif level == "iso":
            d_sig = "|".join(sorted(st.session_state["sel_distrito"]))
            add_polygons_selectable(
                m,
                g_show,
                id_col=ISO_ID,
                tooltip_col=ISO_ID,
                selected_ids=st.session_state["sel_iso"],
                cache_key=f"iso:{d_sig}:{SIMPLIFY_TOL_BY_LAYER['iso']}",
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["iso"],
                base_weight=0.95,
                fill_opacity=0.14,
                selected_weight=3.0,
                selected_fill_opacity=0.0,
            )
        elif level == "quadra":
            i_sig = "|".join(sorted(st.session_state["sel_iso"]))
            id_col = QUADRA_UID if (QUADRA_UID in g_show.columns) else QUADRA_ID
            add_polygons_selectable(
                m,
                g_show,
                id_col=id_col,
                tooltip_col=QUADRA_ID,
                selected_ids=st.session_state["sel_quadra"],
                cache_key=f"quad:{i_sig}:{SIMPLIFY_TOL_BY_LAYER['quadra']}",
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["quadra"],
                base_weight=0.80,
                fill_opacity=0.12,
                selected_fill_opacity=0.0,
            )
        elif level == "censo":
            i_sig = "|".join(sorted(st.session_state["sel_iso"]))
            add_polygons_selectable(
                m,
                g_show,
                id_col=CENSO_ID,
                tooltip_col=CENSO_ID,
                selected_ids=st.session_state["sel_censo"],
                cache_key=f"censo:{i_sig}:{SIMPLIFY_TOL_BY_LAYER['censo']}",
                tooltip_prefix="Censo: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["censo"],
                base_weight=0.80,
                fill_opacity=0.10,
                selected_fill_opacity=0.0,
            )

    elif g_show is not None and not g_show.empty and level == "final":
        if st.session_state.get("branch_mode") == "quadra":
            add_polygons_selectable(
                m,
                g_show,
                id_col=LOTE_ID if (LOTE_ID in g_show.columns) else LOTE_QUADRA_FK,
                tooltip_col=LOTE_ID if (LOTE_ID in g_show.columns) else LOTE_QUADRA_FK,
                selected_ids=set(),
                cache_key=f"lote:final:{len(g_show)}:{SIMPLIFY_TOL_BY_LAYER['lote']}",
                tooltip_prefix="Lote: " if LOTE_ID in g_show.columns else "Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["lote"],
                base_weight=0.55,
                fill_opacity=0.08,
            )
        else:
            add_polygons_selectable(
                m,
                g_show,
                id_col=CENSO_ID,
                tooltip_col=CENSO_ID,
                selected_ids=set(),
                cache_key=f"censo:final:{len(g_show)}:{SIMPLIFY_TOL_BY_LAYER['censo']}",
                tooltip_prefix="Censo: ",
                simplify_tol=SIMPLIFY_TOL_BY_LAYER["censo"],
                base_weight=0.55,
                fill_opacity=0.08,
            )

    # FIX do zoom: returned_objects mínimo (zoom/scroll não dispara rerun)
    map_state = st_folium(
        m,
        height=780,
        use_container_width=True,
        key="map_view",
        returned_objects=[
            "last_clicked",
            "last_object_clicked",
            "last_object_clicked_tooltip",
        ],
    )

    consume_map_click(level, map_state or {}, g_show)

    if g_show is None or g_show.empty:
        st.info("Nada para exibir neste nível (verifique seleções no nível anterior).")
    else:
        st.caption(f"Elementos exibidos: {len(g_show):,}")


# =============================================================================
# APP
# =============================================================================
def main() -> None:
    init_state()
    inject_css()
    render_header()

    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium` instalados.")
        return

    left, right = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        render_map_panel()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
