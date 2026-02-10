# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, Iterable
import re
import base64

import streamlit as st

# geo
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium.features import GeoJsonTooltip  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    from shapely.geometry import Point  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    GeoJsonTooltip = None  # type: ignore
    st_folium = None  # type: ignore
    Point = None  # type: ignore

import pandas as pd  # type: ignore

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
    "marrom": "#C65534",
}
PB_NAVY = PB_COLORS["navy"]
PB_BROWN = PB_COLORS["telha"]
PB_BTN = "#1C6880"  # Reset/Voltar

# Carto tiles explícito
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

# Suavização / acabamento
SMOOTH_FACTOR = 0.9
LINE_CAP = "round"
LINE_JOIN = "round"

# “Sombra” do nível acima (sutil)
PARENT_FILL_OPACITY = 0.16
PARENT_STROKE_OPACITY = 0.35
PARENT_STROKE_WEIGHT = 0.7
PARENT_STROKE_DASH = "2,6"

# Simplificação por nível (VISUAL)
SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0010,
    "distrito": 0.0008,
    "isocrona": 0.0006,
    "quadra": 0.00025,
    "lote": 0.00015,
    "censo": 0.00025,
}

# =============================================================================
# PATHS / ASSETS
# =============================================================================
REPO_ROOT = Path.cwd()
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

# =============================================================================
# IDS DO ENCADEAMENTO (FK-only) - Nomes CANÔNICOS usados no app
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = SUBPREF_ID     # distritos -> subpref
ISO_PARENT = DIST_ID         # isócronas -> distrito
QUADRA_PARENT = ISO_ID       # quadras -> isócrona
LOTE_PARENT = QUADRA_ID      # lotes -> quadra
CENSO_PARENT = ISO_ID        # setor censitário -> isócrona

LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]

# colunas a normalizar por layer (ID + FKs)
LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT],
    "quadra": [QUADRA_ID, QUADRA_PARENT],
    "lote": [LOTE_ID, LOTE_PARENT],
    "censo": [CENSO_ID, CENSO_PARENT],
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}

# =============================================================================
# DRIVE CONFIG
# =============================================================================
SECRETS_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID",
    "dist": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID",
    "quadra": "PB_QUADRAS_FILE_ID",
    "lote": "PB_LOTES_FILE_ID",
    "censo": "PB_CENSO_FILE_ID",
}

FALLBACK_URLS = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist": "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso": "https://drive.google.com/file/d/18ukyzMiYQ6vMqrU6-ctaPFbXMPX9XS9i/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/1XKAYLNdt82ZPNAE-rseSuuaCFmjfn8IP/view?usp=drive_link",
    "lote": "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
}

# =============================================================================
# COLUMN ALIASES (auto-rename) - para casar seu parquet com os nomes canônicos
# =============================================================================
COL_ALIASES_BY_LAYER: Dict[str, Dict[str, list[str]]] = {
    "subpref": {
        SUBPREF_ID: ["SUBPREF_ID", "SUBPREFEITURA_ID", "subprefeitura_id", "ID_SUBPREF", "id_subpref"],
    },
    "dist": {
        DIST_ID: ["DIST_ID", "DISTRITO_ID", "ID_DISTRITO", "id_distrito"],
        DIST_PARENT: ["DIST_PARENT", "SUBPREF_ID", "SUBPREFEITURA_ID", "subpref_id", "subprefeitura_id"],
    },
    "iso": {
        ISO_ID: ["ISO_ID", "ISOCRONA_ID", "ISOCRO_ID", "id_iso", "ID_ISO"],
        ISO_PARENT: ["ISO_PARENT", "DIST_ID", "DISTRITO_ID", "distrito_id", "id_distrito", "ID_DISTRITO"],
    },
    "quadra": {
        QUADRA_ID: ["QUADRA_ID", "ID_QUADRA", "id_quadra", "cod_quadra", "COD_QUADRA", "quadra"],
        QUADRA_PARENT: ["QUADRA_PARENT", "ISO_ID", "iso_id", "ISOCRONA_ID", "isocrona_id", "id_iso", "ID_ISO", "iso_parent"],
    },
    "lote": {
        LOTE_ID: ["LOTE_ID", "ID_LOTE", "id_lote", "cod_lote", "COD_LOTE"],
        LOTE_PARENT: ["LOTE_PARENT", "QUADRA_ID", "quadra_id", "id_quadra", "ID_QUADRA"],
    },
    "censo": {
        CENSO_ID: ["CENSO_ID", "SETOR_ID", "setor_id", "ID_SETOR", "id_setor"],
        CENSO_PARENT: ["CENSO_PARENT", "ISO_ID", "iso_id", "ISOCRONA_ID", "isocrona_id", "ID_ISO"],
    },
}

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

        /* Header */
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; border-radius:8px; }}
        .pb-header {{
            background:{PB_NAVY}; color:#fff; border-radius:14px;
            padding:14px 15px; width:100%;
        }}
        .pb-title {{ font-size:2.25rem; font-weight:900; line-height:1.05; letter-spacing:.2px; }}
        .pb-subtitle {{ font-size:1.05rem; opacity:.95; margin-top:5px; }}

        /* Minimal card wrapper */
        .pb-card {{
            background:#fff;
            border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:14px;
            padding:12px;
        }}

        /* Primary buttons (Reset/Voltar) */
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stBaseButton-primary"] > button {{
            background:{PB_BTN} !important;
            color:#fff !important;
            border:1px solid {PB_BTN} !important;
        }}
        button[data-testid="stBaseButton-primary"]:hover {{
            filter: brightness(.96);
        }}
        button[data-testid="stBaseButton-primary"]:disabled {{
            opacity:.55;
        }}

        .pb-left .stButton, .pb-left .stSelectbox {{
            margin-bottom: .2rem;
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
# STATE
# =============================================================================
def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("last_level", None)

    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi

    st.session_state.setdefault("final_mode", "lote")          # "lote" | "censo"

    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)

    # anti-repetição de clique
    st.session_state.setdefault("last_click_sig", "")

    # cache geojson (por sessão) para acelerar simplify+to_json
    st.session_state.setdefault("_geojson_cache", {})  # type: ignore

    # debug
    st.session_state.setdefault("debug_fk", False)

    # Métricas (dropdowns)
    st.session_state.setdefault("metric_theme", None)
    st.session_state.setdefault("metric_factor", None)
    st.session_state.setdefault("metric_indicator", None)

    # assinatura de ações de UI (para não consumir clique “stale” do mapa)
    st.session_state.setdefault("_ui_action_sig", 0)
    st.session_state.setdefault("_ui_action_sig_seen", 0)

    # se alguma ação setou view explicitamente, não sobrescreve com center/zoom antigo do mapa
    st.session_state.setdefault("_view_set_by_ui", False)


def mark_ui_action(view_override: bool = False) -> None:
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1
    if view_override:
        st.session_state["_view_set_by_ui"] = True


def reset_to(level: str) -> None:
    st.session_state["level"] = level
    st.session_state["last_click_sig"] = ""
    st.session_state["_geojson_cache"] = {}  # type: ignore

    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
        st.session_state["last_level"] = None
        st.session_state["_view_set_by_ui"] = True

    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"

    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"

    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"


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
# HELPERS: ids / normalize
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
    if s.endswith(".0"):
        core = s[:-2]
        if core.isdigit():
            return core
    return s


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


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


def _harmonize_columns(layer_key: str, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Renomeia colunas para o schema canônico (quadra_id, iso_id etc.), tolerando casing e aliases."""
    if gdf is None or gdf.empty:
        return gdf

    aliases = COL_ALIASES_BY_LAYER.get(layer_key, {})
    if not aliases:
        return gdf

    cols = list(gdf.columns)
    cf_map = {c.casefold(): c for c in cols}

    rename: Dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in gdf.columns:
            continue

        c0 = cf_map.get(canonical.casefold())
        if c0:
            rename[c0] = canonical
            continue

        found = None
        for cand in candidates:
            if cand in gdf.columns:
                found = cand
                break
            c1 = cf_map.get(cand.casefold())
            if c1:
                found = c1
                break
        if found:
            rename[found] = canonical

    if rename:
        gdf = gdf.rename(columns=rename)

    return gdf

# =============================================================================
# DRIVE / LOCAL IO
# =============================================================================
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
    ui_key = f"drive_{layer_key}_raw"
    raw_ui = str(st.session_state.get(ui_key, "")).strip()
    if raw_ui:
        return raw_ui

    secret_key = SECRETS_KEYS.get(layer_key, "")
    raw_secret = _get_secret(secret_key) if secret_key else ""
    if raw_secret:
        return raw_secret

    return str(FALLBACK_URLS.get(layer_key, "")).strip()


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
    import requests

    file_id = extract_drive_id(file_id_or_url)
    if not file_id:
        raise RuntimeError("FILE_ID inválido (não foi possível extrair ID do link).")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    def get_confirm_token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Download falhou (status={response.status_code}).")

    total = int(response.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024

    ui_label = label or dst.name
    prog = st.progress(0, text=f"Baixando {ui_label}…")
    downloaded = 0

    with open(dst, "wb") as f:
        for part in response.iter_content(chunk_size=chunk):
            if not part:
                continue
            f.write(part)
            downloaded += len(part)
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                prog.progress(pct, text=f"Baixando {ui_label}… {pct}%")

    prog.empty()
    return dst


def local_layer_path(layer_key: str) -> Path:
    return DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]


def layer_available_locally(layer_key: str) -> bool:
    p = local_layer_path(layer_key)
    return p.exists() and p.stat().st_size > 0


def ensure_local_layer(layer_key: str) -> Path:
    dst = local_layer_path(layer_key)
    if layer_available_locally(layer_key):
        return dst

    raw = get_drive_raw(layer_key)
    if not raw:
        raise RuntimeError(
            f"Layer '{layer_key}' não encontrada localmente em {dst.name} e não há FILE_ID/link configurado."
        )

    return download_drive_file(raw, dst, label=dst.name)

# =============================================================================
# READ / FILTER (cache_data)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
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


def _drop_bad_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    return gdf


def _missing_required_cols(layer_key: str, gdf: "gpd.GeoDataFrame") -> list[str]:
    required = LAYER_ID_COLS.get(layer_key, [])
    return [c for c in required if c not in gdf.columns]


def read_layer(layer_key: str) -> Optional["gpd.GeoDataFrame"]:
    """Leitura padronizada + harmonização de colunas + normalização de ids (FK-only)."""
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        st.error(str(e))
        return None

    g = read_gdf_parquet(str(p))
    if g is None or g.empty:
        st.error(f"Layer '{layer_key}' vazia/erro ao ler ({p.name}).")
        return None

    g = _drop_bad_geoms(g)

    # harmoniza colunas (QUADRA_PARENT -> iso_id, etc.)
    g = _harmonize_columns(layer_key, g)

    # normaliza ids
    g = normalize_id_cols(g, LAYER_ID_COLS.get(layer_key, []))

    # check mínimo: colunas do contrato existem
    miss = _missing_required_cols(layer_key, g)
    if miss:
        st.error(
            f"Layer '{layer_key}' não atende o contrato de colunas. Faltando: {miss}. "
            f"Colunas disponíveis: {list(g.columns)}"
        )
        return None

    return g


def subset_by_parent(child: "gpd.GeoDataFrame", parent_col: str, parent_val: Any) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    if parent_col not in child.columns or parent_val is None:
        return child.iloc[0:0].copy()
    pv = _id_to_str(parent_val)
    if pv is None:
        return child.iloc[0:0].copy()
    return child[child[parent_col] == pv]


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()
    pset = {v for v in (_id_to_str(x) for x in parent_vals) if v is not None}
    if not pset:
        return child.iloc[0:0].copy()
    return child[child[parent_col].isin(list(pset))]


def subset_by_id(gdf: "gpd.GeoDataFrame", id_col: str, id_val: Any) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or id_val is None:
        return gdf.iloc[0:0].copy()
    iv = _id_to_str(id_val)
    if iv is None:
        return gdf.iloc[0:0].copy()
    return gdf[gdf[id_col] == iv]


def subset_by_id_multi(gdf: "gpd.GeoDataFrame", id_col: str, ids: Set[Any]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    if id_col not in gdf.columns or not ids:
        return gdf.iloc[0:0].copy()
    iset = {v for v in (_id_to_str(x) for x in ids) if v is not None}
    if not iset:
        return gdf.iloc[0:0].copy()
    return gdf[gdf[id_col].isin(list(iset))]

# =============================================================================
# VALIDATION (unicidade + não-nulos + FK)
# =============================================================================
LAYER_SPECS = {
    "subpref": {"id": SUBPREF_ID, "parent": None,         "parent_layer": None,    "parent_id": None},
    "dist":    {"id": DIST_ID,    "parent": DIST_PARENT,  "parent_layer": "subpref","parent_id": SUBPREF_ID},
    "iso":     {"id": ISO_ID,     "parent": ISO_PARENT,   "parent_layer": "dist",   "parent_id": DIST_ID},
    "quadra":  {"id": QUADRA_ID,  "parent": QUADRA_PARENT,"parent_layer": "iso",    "parent_id": ISO_ID},
    "lote":    {"id": LOTE_ID,    "parent": LOTE_PARENT,  "parent_layer": "quadra", "parent_id": QUADRA_ID},
    "censo":   {"id": CENSO_ID,   "parent": CENSO_PARENT, "parent_layer": "iso",    "parent_id": ISO_ID},
}
MAX_FK_UNIQUES = 200_000


def validate_layer_contract(
    layer_key: str,
    gdf: "gpd.GeoDataFrame",
    parent_gdf: Optional["gpd.GeoDataFrame"] = None,
    strict_fk_integrity: bool = False,
) -> bool:
    spec = LAYER_SPECS.get(layer_key)
    if spec is None:
        return True

    id_col = spec["id"]
    parent_col = spec["parent"]
    parent_layer = spec["parent_layer"]
    parent_id_col = spec["parent_id"]

    if id_col not in gdf.columns:
        st.error(f"[{layer_key}] Coluna obrigatória ausente: '{id_col}'")
        return False

    s_id = gdf[id_col]
    n_null_id = int(s_id.isna().sum())
    if n_null_id > 0:
        st.error(f"[{layer_key}] '{id_col}' tem {n_null_id:,} nulos. ID deve ser não-nulo.")
        return False

    dup_mask = s_id.duplicated(keep=False)
    if int(dup_mask.sum()) > 0:
        n_dup_ids = int(s_id[dup_mask].nunique())
        examples = s_id[dup_mask].astype(str).head(5).tolist()
        st.error(f"[{layer_key}] '{id_col}' NÃO é único: {n_dup_ids:,} IDs repetidos. Ex: {examples}")
        return False

    if parent_col:
        if parent_col not in gdf.columns:
            st.error(f"[{layer_key}] Coluna FK obrigatória ausente: '{parent_col}'")
            return False

        s_fk = gdf[parent_col]
        n_null_fk = int(s_fk.isna().sum())
        if n_null_fk > 0:
            st.error(f"[{layer_key}] FK '{parent_col}' tem {n_null_fk:,} nulos. Camada filha precisa de pai.")
            return False

        if strict_fk_integrity and parent_gdf is not None and parent_layer and parent_id_col:
            if parent_id_col not in parent_gdf.columns:
                st.error(f"[{parent_layer}] Coluna obrigatória ausente: '{parent_id_col}'")
                return False

            child_uni = pd.unique(s_fk.dropna())
            if len(child_uni) <= MAX_FK_UNIQUES:
                parent_set = set(pd.unique(parent_gdf[parent_id_col].dropna()).tolist())
                missing = [v for v in child_uni.tolist() if v not in parent_set]
                if missing:
                    st.error(
                        f"[{layer_key}] FK inválida: '{parent_col}' contém valores que não existem em "
                        f"'{parent_layer}.{parent_id_col}'. Ex: {missing[:10]}"
                    )
                    return False
            else:
                st.warning(
                    f"[{layer_key}] FK integrity skip: muitos valores únicos em '{parent_col}' "
                    f"({len(child_uni):,} > {MAX_FK_UNIQUES:,})."
                )

    return True

# =============================================================================
# CLICK HITTEST (fallback)
# =============================================================================
def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[str]:
    if gdf is None or gdf.empty or not click_latlon:
        return None
    if id_col not in gdf.columns:
        return None
    if Point is None:
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

# =============================================================================
# MAPA (Folium)
# =============================================================================
def make_carto_map(center=(-23.55, -46.63), zoom=11):
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


def _mk_tooltip(id_col: str, prefix: str) -> Optional[Any]:
    if GeoJsonTooltip is None:
        return None
    return GeoJsonTooltip(
        fields=[id_col],
        aliases=[prefix],
        sticky=True,
        labels=True,
        localize=True,
        max_width=320,
    )


def _session_geojson_get(key: str) -> Optional[str]:
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})  # type: ignore
    return cache.get(key)


def _session_geojson_set(key: str, value: str) -> None:
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})  # type: ignore
    cache[key] = value
    st.session_state["_geojson_cache"] = cache  # type: ignore


def _simplify_to_geojson(gdf: "gpd.GeoDataFrame", simplify_tol: float, keep_cols: Optional[list[str]] = None) -> str:
    if gdf is None or gdf.empty:
        return ""
    cols = (keep_cols or []) + ["geometry"]
    g = gdf[cols].copy()
    try:
        g["geometry"] = g.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        pass
    g = _drop_bad_geoms(g)
    try:
        return g.to_json()
    except Exception:
        return ""


def add_parent_fill(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    pane: str = "parent_fill",
    fill_color: str = PB_BROWN,
    fill_opacity: float = PARENT_FILL_OPACITY,
    stroke_color: str = PB_BROWN,
    stroke_weight: float = PARENT_STROKE_WEIGHT,
    stroke_opacity: float = PARENT_STROKE_OPACITY,
    dash_array: str = PARENT_STROKE_DASH,
    simplify_tol: float = 0.0006,
    cache_key: Optional[str] = None,
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return

    key = cache_key or f"parent:{name}:{simplify_tol}:{len(gdf)}"
    geojson = _session_geojson_get(key)
    if not geojson:
        geojson = _simplify_to_geojson(gdf, simplify_tol=simplify_tol, keep_cols=[])
        _session_geojson_set(key, geojson)
    if not geojson:
        return

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=geojson,
        pane=pane,
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
    ).add_to(fg)
    fg.add_to(m)


def add_polygons_selectable(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    selected_ids: Optional[Set[Any]] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.10,
    selected_color: str = PB_NAVY,
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.26,
    tooltip_prefix: str = "ID: ",
    simplify_tol: Optional[float] = None,
    cache_key: Optional[str] = None,
) -> None:
    """
    FIX do “tela branca”:
    - tooltip só na camada BASE.
    - NÃO reutiliza tooltip na camada de selecionados.
    """
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        return

    selected_ids = selected_ids or set()
    sel = {v for v in (_id_to_str(x) for x in selected_ids) if v is not None}
    tol = simplify_tol if simplify_tol is not None else 0.0006

    # cache do base geojson (pesado)
    key = cache_key or f"base:{name}:{id_col}:{tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[[id_col, "geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol=tol, keep_cols=[id_col])
        _session_geojson_set(key, geojson_base)

    if not geojson_base:
        return

    tooltip_base = _mk_tooltip(id_col, tooltip_prefix)

    fg_base = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=geojson_base,
        pane=pane,
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
        highlight_function=lambda _f: {
            "weight": base_weight + 1.4,
            "fillOpacity": min(fill_opacity + 0.12, 0.40),
        },
        tooltip=tooltip_base,
    ).add_to(fg_base)
    fg_base.add_to(m)

    # selecionados (sem tooltip)
    if sel:
        sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
        if not sel_gdf.empty:
            sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
            geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=tol, keep_cols=[id_col])
            if geojson_sel:
                fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
                folium.GeoJson(
                    data=geojson_sel,
                    pane=pane,
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
                ).add_to(fg_sel)
                fg_sel.add_to(m)

# =============================================================================
# MAP VIEW HELPERS
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


def set_view_to_gdf(gdf: "gpd.GeoDataFrame", bump: int = 0, zmax: int = 18) -> None:
    if gdf is None or gdf.empty:
        return
    try:
        center, zoom = bounds_center_zoom(gdf)
        st.session_state["view_center"] = center
        st.session_state["view_zoom"] = min(zoom + bump, zmax)
        st.session_state["_view_set_by_ui"] = True
    except Exception:
        pass


def _parse_center_zoom(map_state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
    c = map_state.get("center")
    z = map_state.get("zoom")

    center: Optional[Tuple[float, float]] = None
    zoom: Optional[int] = None

    try:
        if isinstance(c, dict):
            lat = c.get("lat")
            lng = c.get("lng")
            if lat is not None and lng is not None:
                center = (float(lat), float(lng))
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            center = (float(c[0]), float(c[1]))
    except Exception:
        center = None

    try:
        if z is not None:
            zoom = int(z)
    except Exception:
        zoom = None

    return center, zoom


def update_view_from_map_state(map_state: Dict[str, Any]) -> None:
    if st.session_state.get("_view_set_by_ui", False):
        st.session_state["_view_set_by_ui"] = False
        return

    center, zoom = _parse_center_zoom(map_state)
    if center is not None:
        st.session_state["view_center"] = center
    if zoom is not None:
        st.session_state["view_zoom"] = zoom

# =============================================================================
# CONSUME CLICK (ANTES DO MAPA)
# =============================================================================
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


def consume_map_event(
    level: str,
    map_state: Dict[str, Any],
    get_layer: callable,
    allow_click: bool = True,
) -> None:
    """Processa clique antes do mapa. Sem st.rerun()."""
    if not allow_click:
        return

    click = (map_state or {}).get("last_clicked") or None
    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip") or None
    picked_tooltip = parse_tooltip_id(tooltip_raw)

    if not click and not picked_tooltip:
        return

    sig = _click_signature(picked_tooltip, click)
    if sig and sig == st.session_state.get("last_click_sig", ""):
        return
    st.session_state["last_click_sig"] = sig

    # SUBPREF -> DISTRITO
    if level == "subpref":
        picked = picked_tooltip
        if not picked and isinstance(click, dict):
            g_sub = get_layer("subpref")
            if g_sub is not None:
                picked = pick_feature_id(g_sub, click, SUBPREF_ID)
        if picked:
            st.session_state["selected_subpref_id"] = picked
            st.session_state["selected_distrito_id"] = None
            st.session_state["selected_iso_ids"] = set()
            st.session_state["selected_quadra_ids"] = set()
            st.session_state["final_mode"] = "lote"
            st.session_state["level"] = "distrito"
        return

    # DISTRITO -> ISOCRONA
    if level == "distrito":
        sp = _id_to_str(st.session_state.get("selected_subpref_id"))
        if sp is None:
            return
        picked = picked_tooltip
        if not picked and isinstance(click, dict):
            g_dist = get_layer("dist")
            if g_dist is not None:
                g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
                picked = pick_feature_id(g_show, click, DIST_ID)
        if picked:
            st.session_state["selected_distrito_id"] = picked
            st.session_state["selected_iso_ids"] = set()
            st.session_state["selected_quadra_ids"] = set()
            st.session_state["final_mode"] = "lote"
            st.session_state["level"] = "isocrona"
        return

    # ISOCRONA (multi toggle)
    if level == "isocrona":
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if d is None:
            return
        picked = picked_tooltip
        if not picked and isinstance(click, dict):
            g_iso = get_layer("iso")
            if g_iso is not None:
                g_show = subset_by_parent(g_iso, ISO_PARENT, d)
                picked = pick_feature_id(g_show, click, ISO_ID)
        if picked:
            _toggle_in_set("selected_iso_ids", picked)
        return

    # QUADRA (multi toggle)
    if level == "quadra":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
        if not iso_ids:
            return
        picked = picked_tooltip
        if not picked and isinstance(click, dict):
            g_quad = get_layer("quadra")
            if g_quad is not None:
                g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)
                picked = pick_feature_id(g_show, click, QUADRA_ID)
        if picked:
            _toggle_in_set("selected_quadra_ids", picked)
        return


def sanitize_level_state() -> None:
    lvl = st.session_state.get("level", "subpref")

    if lvl == "distrito" and _id_to_str(st.session_state.get("selected_subpref_id")) is None:
        reset_to("subpref")
        return

    if lvl == "isocrona" and _id_to_str(st.session_state.get("selected_distrito_id")) is None:
        reset_to("distrito")
        return

    if lvl == "quadra":
        iso_ids = st.session_state.get("selected_iso_ids", set())
        if not iso_ids:
            reset_to("isocrona")
            return

# =============================================================================
# (Opcional) diagnóstico leve
# =============================================================================
def diag_layer(gdf: "gpd.GeoDataFrame", title: str, cols: list[str]) -> None:
    if not st.session_state.get("debug_fk", False):
        return
    with st.expander(f"🧪 Diagnóstico: {title}", expanded=False):
        if gdf is None or gdf.empty:
            st.warning("GDF vazio.")
            return
        st.write(f"Linhas: **{len(gdf):,}**")
        for c in cols:
            if c not in gdf.columns:
                st.error(f"Coluna ausente: {c}")
                continue
            s = gdf[c]
            n_null = int(s.isna().sum())
            n_unique = int(s.dropna().nunique())
            ex = s.dropna().astype(str).head(5).tolist()
            st.write(f"- **{c}** | dtype={s.dtype} | nulos={n_null} | únicos={n_unique} | ex={ex}")

# =============================================================================
# UI: Métricas + Dados/Config
# =============================================================================
METRICS_TREE: Dict[str, Dict[str, list[str]]] = {
    "Demografia": {
        "População": ["Pop total", "Pop 0-14", "Pop 60+"],
        "Densidade": ["Hab/km²", "Domicílios/km²"],
    },
    "Mobilidade": {
        "Transporte": ["Acesso ônibus", "Acesso metrô", "Tempo ao emprego"],
        "Caminhabilidade": ["Índice walk", "Sinuosidade viária"],
    },
    "Infraestrutura": {
        "Saneamento": ["Água", "Esgoto", "Coleta lixo"],
        "Equipamentos": ["Escolas", "UBS", "Parques"],
    },
}


def ensure_metric_state() -> None:
    themes = list(METRICS_TREE.keys())
    if not themes:
        return

    if st.session_state.get("metric_theme") not in themes:
        st.session_state["metric_theme"] = themes[0]

    theme = st.session_state["metric_theme"]
    factors = list(METRICS_TREE.get(theme, {}).keys())
    if not factors:
        st.session_state["metric_factor"] = None
        st.session_state["metric_indicator"] = None
        return

    if st.session_state.get("metric_factor") not in factors:
        st.session_state["metric_factor"] = factors[0]

    factor = st.session_state["metric_factor"]
    indicators = METRICS_TREE.get(theme, {}).get(factor, [])
    if not indicators:
        st.session_state["metric_indicator"] = None
        return

    if st.session_state.get("metric_indicator") not in indicators:
        st.session_state["metric_indicator"] = indicators[0]


def metrics_panel() -> None:
    ensure_metric_state()

    theme = st.selectbox(
        "Temática",
        options=list(METRICS_TREE.keys()),
        key="metric_theme",
        on_change=lambda: mark_ui_action(False),
    )

    factors = list(METRICS_TREE.get(theme, {}).keys())
    st.selectbox(
        "Fator",
        options=factors,
        key="metric_factor",
        on_change=lambda: mark_ui_action(False),
    )

    factor = st.session_state.get("metric_factor")
    indicators = METRICS_TREE.get(theme, {}).get(factor, []) if factor else []
    st.selectbox(
        "Indicador",
        options=indicators,
        key="metric_indicator",
        on_change=lambda: mark_ui_action(False),
    )


def data_sources_panel() -> None:
    st.caption("Local (data_cache) ou Drive (secrets / input).")

    for k in ["subpref", "dist", "iso", "quadra", "lote", "censo"]:
        p = local_layer_path(k)
        ok = layer_available_locally(k)
        st.write(f"- `{p.name}`: {'✅ local' if ok else '—'}")

    st.divider()

    with st.expander("Configurar links/IDs do Drive (opcional)", expanded=False):
        st.caption("Se tiver secrets.toml, não precisa colar aqui. Se colar aqui, tem prioridade.")
        for k in ["subpref", "dist", "iso", "quadra", "lote", "censo"]:
            placeholder = _get_secret(SECRETS_KEYS.get(k, "")) or FALLBACK_URLS.get(k, "") or "Cole aqui o link/ID"
            st.text_input(
                f"{k} ({LOCAL_FILENAMES[k]})",
                key=f"drive_{k}_raw",
                value=str(st.session_state.get(f"drive_{k}_raw", "")).strip(),
                placeholder=placeholder,
                on_change=lambda: mark_ui_action(False),
            )

    st.divider()
    st.checkbox("Modo diagnóstico (FK/colunas)", key="debug_fk", on_change=lambda: mark_ui_action(False))


def _level_label(level: str) -> str:
    return {
        "subpref": "Subprefeituras",
        "distrito": "Distritos",
        "isocrona": "Isócronas",
        "quadra": "Quadras",
        "final": "Nível final",
    }.get(level, level)


def _prev_level(level: str) -> Optional[str]:
    if level not in LEVELS:
        return None
    idx = LEVELS.index(level)
    if idx <= 0:
        return None
    return LEVELS[idx - 1]


def on_back_click() -> None:
    mark_ui_action(False)
    _back_one_level()


def on_reset_click() -> None:
    mark_ui_action(True)
    reset_to("subpref")


def on_go_quadras() -> None:
    mark_ui_action(False)
    st.session_state["level"] = "quadra"


def on_fit_isos() -> None:
    mark_ui_action(True)
    iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
    if not iso_ids:
        return
    g_iso = read_layer("iso")
    if g_iso is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_iso, ISO_ID, iso_ids), bump=0, zmax=18)


def on_fit_quadras() -> None:
    mark_ui_action(True)
    quad_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_quadra_ids", set())) if v is not None}
    if not quad_ids:
        return
    g_quad = read_layer("quadra")
    if g_quad is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_quad, QUADRA_ID, quad_ids), bump=1, zmax=19)


def left_panel() -> None:
    lvl = st.session_state["level"]
    prev = _prev_level(lvl)

    st.markdown("<div class='pb-left'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if prev is None:
            st.button("Voltar", disabled=True, use_container_width=True)
        else:
            st.button(_level_label(prev), type="primary", use_container_width=True, on_click=on_back_click)

    with c2:
        st.button("Reset", type="primary", use_container_width=True, on_click=on_reset_click)

    if lvl == "isocrona":
        st.divider()
        ok = len(st.session_state.get("selected_iso_ids", set())) > 0
        st.button("➡️ Ir para Quadras", use_container_width=True, disabled=not ok, on_click=on_go_quadras)
        st.button("Ajustar ao selecionado", use_container_width=True, disabled=not ok, on_click=on_fit_isos)

    if lvl == "quadra":
        st.divider()
        okq = len(st.session_state.get("selected_quadra_ids", set())) > 0
        st.button("Ajustar ao selecionado", use_container_width=True, disabled=not okq, on_click=on_fit_quadras)

    st.divider()
    st.subheader("Métricas", anchor=False)
    metrics_panel()

    st.divider()
    with st.expander("Dados / Config", expanded=False):
        data_sources_panel()

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MAP RENDER (sem rerun manual; view só ao entrar no nível)
# =============================================================================
def render_map_panel(get_layer: callable) -> None:
    level = st.session_state["level"]
    title = ""

    # =========================
    # SUBPREF
    # =========================
    if level == "subpref":
        title = "Subprefeituras"
        g_sub = get_layer("subpref")
        if g_sub is None:
            st.stop()

        if not validate_layer_contract("subpref", g_sub):
            st.stop()

        if st.session_state.get("last_level") != "subpref":
            set_view_to_gdf(g_sub, bump=0)
            st.session_state["last_level"] = "subpref"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        add_polygons_selectable(
            m, g_sub, "Subprefeituras", SUBPREF_ID,
            selected_ids=set(),
            base_weight=0.9, fill_opacity=0.04,
            tooltip_prefix="Subpref: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"subpref:all:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )

    # =========================
    # DISTRITO
    # =========================
    elif level == "distrito":
        sp = _id_to_str(st.session_state.get("selected_subpref_id"))
        if sp is None:
            reset_to("subpref")
            return

        title = f"Distritos (Subpref {sp})"

        g_dist = get_layer("dist")
        g_sub = get_layer("subpref")
        if g_dist is None or g_sub is None:
            st.stop()

        if not validate_layer_contract("subpref", g_sub):
            st.stop()
        if not validate_layer_contract("dist", g_dist, parent_gdf=g_sub, strict_fk_integrity=True):
            st.stop()

        g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
        g_show = subset_by_parent(g_dist, DIST_PARENT, sp)

        if st.session_state.get("last_level") != "distrito":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "distrito"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        add_parent_fill(
            m, g_parent, "Subpref selecionada (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"parent:subpref:{sp}:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )

        add_polygons_selectable(
            m, g_show, "Distritos", DIST_ID,
            selected_ids=set(),
            base_weight=0.75, fill_opacity=0.06,
            tooltip_prefix="Distrito: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"dist:sp:{sp}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

    # =========================
    # ISOCRONAS (multi)
    # =========================
    elif level == "isocrona":
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if d is None:
            reset_to("distrito")
            return

        sel_n = len(st.session_state.get("selected_iso_ids", set()))
        title = f"Isócronas (Distrito {d}) — selecionadas: {sel_n}"

        g_iso = get_layer("iso")
        g_dist = get_layer("dist")
        if g_iso is None or g_dist is None:
            st.stop()

        if not validate_layer_contract("dist", g_dist):
            st.stop()
        if not validate_layer_contract("iso", g_iso, parent_gdf=g_dist, strict_fk_integrity=True):
            st.stop()

        g_parent = subset_by_id(g_dist, DIST_ID, d)
        g_show = subset_by_parent(g_iso, ISO_PARENT, d)

        if st.session_state.get("last_level") != "isocrona":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "isocrona"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        add_parent_fill(
            m, g_parent, "Distrito selecionado (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"parent:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

        add_polygons_selectable(
            m, g_show, "Isócronas", ISO_ID,
            selected_ids=st.session_state["selected_iso_ids"],
            base_weight=0.95,
            fill_opacity=0.14,
            selected_weight=3.0,
            selected_fill_opacity=0.30,
            tooltip_prefix="Isócrona: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"iso:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )

    # =========================
    # QUADRAS (multi)
    # =========================
    elif level == "quadra":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
        if not iso_ids:
            reset_to("isocrona")
            return

        sel_nq = len(st.session_state.get("selected_quadra_ids", set()))
        title = f"Quadras — selecionadas: {sel_nq}"

        g_quad = get_layer("quadra")
        g_iso = get_layer("iso")
        if g_quad is None or g_iso is None:
            st.stop()

        # aqui é exatamente a regra: quadra_id único e iso_id (FK) sem nulos e existindo no pai
        if not validate_layer_contract("iso", g_iso):
            st.stop()
        if not validate_layer_contract("quadra", g_quad, parent_gdf=g_iso, strict_fk_integrity=True):
            st.stop()

        g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)
        g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)

        if st.session_state.get("last_level") != "quadra":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=1)
            st.session_state["last_level"] = "quadra"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return

        iso_key = "|".join(sorted(list(iso_ids)))
        add_parent_fill(
            m, g_parent, "Isócronas selecionadas (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"parent:iso:{iso_key}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )

        add_polygons_selectable(
            m, g_show, "Quadras", QUADRA_ID,
            selected_ids=st.session_state["selected_quadra_ids"],
            base_weight=0.80,
            fill_opacity=0.12,
            selected_weight=2.6,
            selected_fill_opacity=0.28,
            tooltip_prefix="Quadra: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
            cache_key=f"quadra:iso:{iso_key}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
        )

    else:
        title = "Nível final (placeholder)"
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            return
        st.info("Nível final não é foco aqui. Mantido como placeholder.")

    # título simples do painel do mapa
    st.markdown(f"### {title}")

    # render do mapa
    st_folium(
        m,
        height=780,
        use_container_width=True,
        key="map_view",
        returned_objects=["last_clicked", "last_object_clicked_tooltip", "center", "zoom"],
    )

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

    # 1) detecta se rerun veio de UI (botões/selects) e não do mapa
    ui_sig = int(st.session_state.get("_ui_action_sig", 0))
    ui_seen = int(st.session_state.get("_ui_action_sig_seen", 0))
    ui_action = ui_sig != ui_seen
    st.session_state["_ui_action_sig_seen"] = ui_sig

    # se veio de UI, libera clique antigo e não consome clique do mapa nesse rerun
    if ui_action:
        st.session_state["last_click_sig"] = ""

    # 2) lê estado do mapa do session_state e atualiza view (pan/zoom)
    map_state = st.session_state.get("map_view", {}) or {}
    update_view_from_map_state(map_state)

    # 3) per-rerun layer cache (evita read_layer múltiplas vezes no mesmo ciclo)
    _run_layers: Dict[str, Optional["gpd.GeoDataFrame"]] = {}

    def get_layer_cached(layer_key: str) -> Optional["gpd.GeoDataFrame"]:
        if layer_key in _run_layers:
            return _run_layers[layer_key]
        g = read_layer(layer_key)
        _run_layers[layer_key] = g
        return g

    # 4) consome clique ANTES do mapa (sem st.rerun)
    current_level = st.session_state.get("level", "subpref")
    consume_map_event(current_level, map_state, get_layer_cached, allow_click=(not ui_action))

    # 5) sanity do fluxo
    sanitize_level_state()

    # 6) layout
    left, right = st.columns([1, 4], gap="large")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        render_map_panel(get_layer_cached)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
