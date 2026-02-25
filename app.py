# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, Iterable, List
import re
import base64
import time

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

PB_BTN = "#1C6880"

CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

SMOOTH_FACTOR = 1.0
LINE_CAP = "round"
LINE_JOIN = "round"

PARENT_FILL_OPACITY = 0.16
PARENT_STROKE_OPACITY = 0.35
PARENT_STROKE_WEIGHT = 0.7
PARENT_STROKE_DASH = "2,6"

SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0000,
    "distrito": 0.0000,
    "isocrona": 0.0000,
    "censo": 0.00000,
    "quadra": 0.00000,
    "lote": 0.00000,
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
# VISUALIZAÇÕES (Clusters / Isócronas)
# =============================================================================
QUADRAS_CSV_FILENAME = "quadras.csv"
QUADRAS_CSV_SECRET_KEY = "PB_QUADRAS_CSV_FILE_ID"
QUADRAS_CSV_FALLBACK_URL = "https://drive.google.com/file/d/1_WKryQlu_jZL1xsgAmQDrI81aSdzKYsc/view?usp=drive_link"
CLUSTER_COL = "Cluster"
ISO_CLASS_COL = "nova_class"

CLUSTER_COLOR_MAP = {
    0: "#bf7db2",
    1: "#f7bd6a",
    2: "#cf651f",
    3: "#cf651f",
    4: "#793393",
}
CLUSTER_LABEL_MAP = {
    0: "Alta Densidade Periférica",
    1: "Uso Misto Intermediário",
    2: "Média Densidade Periférica",
    3: "Uso Misto Verticalizado Central",
    4: "Predominância Comercial e de Serviços",
}
CLUSTER_NULL_COLOR = "#c8c8c8"

ISO_TRANSITION_SET = {1, 3, 6}
ISO_TRANSITION_LABEL = "Área de transição"
ISO_TRANSITION_COLOR = "#7f6a5c"

ISO_VALUE_TO_CLASSNUM = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7}
ISO_CLASSNUM_TO_COLOR = {
    1: "#f7f7f7",
    2: "#d8daeb",
    3: "#8073ac",
    4: "#b2abd2",
    5: "#b35806",
    6: "#e08214",
    7: "#542788",
}
ISO_DEFAULT_COLOR = "#ffffff"

# =============================================================================
# IDS (FK-only)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
QUADRA_UID = "quadra_uid"      # iso_id__quadra_id
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = SUBPREF_ID       # distritos -> subpref
ISO_PARENT = DIST_ID           # isócronas -> distrito

QUADRA_PARENT = ISO_ID         # quadras -> isócrona
LOTE_PARENT = QUADRA_ID        # lotes -> quadra

# Para o ajuste (2) e (3):
# - SetorCensitario será filtrado por iso_id do(s) iso(s) selecionado(s)
CENSO_PARENT = ISO_ID          # setor -> iso (esperado no parquet)
# - E do setor -> quadra, filtramos pelas quadra_id (idealmente setor tem quadra_id)
CENSO_QUADRA_COL = QUADRA_ID   # coluna esperada em Setorcensitario para derivar quadras

# Níveis (inclui "censo" para o ramo do setor censitário)
LEVELS = ["subpref", "distrito", "isocrona", "censo", "quadra", "final"]

LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT],
    "quadra": [QUADRA_ID, QUADRA_PARENT, QUADRA_UID, ISO_ID, CENSO_ID],
    "lote": [LOTE_ID, LOTE_PARENT, QUADRA_ID, ISO_ID],
    "censo": [CENSO_ID, CENSO_PARENT, QUADRA_ID, ISO_ID],
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "quadra": "Quadras.parquet",
    "lote": "Lotes.parquet",
    "censo": "Setorcensitario.parquet",
}

# Aliases comuns (inclui o typo "cendo_id" que você mesmo citou)
CENSO_ID_ALIASES = {
    "censo_id",
    "CENSO_ID",
    "Censo_id",
    "censoId",
    "censoid",
    "censo_id ",
    "cendo_id",
    "CENDO_ID",
}

def standardize_columns(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Normaliza nomes de colunas para evitar erros por:
    - maiúsculas/minúsculas
    - espaços
    - typo "cendo_id"
    """
    if gdf is None or gdf.empty:
        return gdf
    ren = {}
    for c in gdf.columns:
        cl = str(c).strip()
        low = cl.lower()

        # normaliza censo_id
        if cl in CENSO_ID_ALIASES or low in {a.lower().strip() for a in CENSO_ID_ALIASES}:
            ren[c] = CENSO_ID

    return gdf.rename(columns=ren)

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
    "quadra": "https://drive.google.com/file/d/17VaA-MlITota7shvbN8mNgf2MjWvUbVW/view?usp=drive_link",
    "lote": "https://drive.google.com/file/d/1oTFAZff1mVAWD6KQTJSz45I6B6pi6ceP/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
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
# STATE
# =============================================================================
def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("last_level", None)

    # Mantive subpref/distrito como seleção única (como no seu código base)
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)

    # Isócronas multi
    st.session_state.setdefault("selected_iso_ids", set())     # multi

    # NOVO: setor censitário multi
    st.session_state.setdefault("selected_censo_ids", set())   # multi

    # Quadras multi (sempre por UID p/ evitar colisão)
    st.session_state.setdefault("selected_quadra_ids", set())  # multi (quadra_uid)

    # NOVO: escolha feita no nível isócronas (ajuste #2)
    # "quadra" ou "censo"
    st.session_state.setdefault("iso_next_mode", "quadra")
    st.session_state.setdefault("quadra_filter_col", QUADRA_UID)

    # NOVO: quando vem do setor censitário, quadras exibidas são filtradas
    # pelo conjunto de quadra_uid derivado de (iso_id + quadra_id) do setor selecionado
    st.session_state.setdefault("quadra_filter_uids", set())

    # view usada apenas quando o app decide (fit / troca de nível / etc.)
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)

    # anti-repetição de clique
    st.session_state.setdefault("last_click_sig", "")

    # geojson cache em sessão (agora com limite)
    st.session_state.setdefault("_geojson_cache", {})  # type: ignore
    st.session_state.setdefault("_geojson_cache_order", [])  # type: ignore

    # debug
    st.session_state.setdefault("debug_fk", False)

    # UI action sig (evita consumir clique do mapa em reruns por widgets)
    st.session_state.setdefault("_ui_action_sig", 0)
    st.session_state.setdefault("_ui_action_sig_seen", 0)

    # FINAL: proteção opcional para não carregar lotes automaticamente
    st.session_state.setdefault("final_load_sig", "")
    st.session_state.setdefault("final_loaded", False)

    st.session_state.setdefault("_layer_cache", {}) 
    st.session_state.setdefault("_layer_cache_meta", {})


def mark_ui_action() -> None:
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1


def reset_to(level: str) -> None:
    st.session_state["level"] = level

    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
        st.session_state["last_level"] = None
        st.session_state["last_click_sig"] = ""
        _geojson_cache_reset()
        _final_reset()

    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["last_click_sig"] = ""
        _geojson_cache_reset()
        _final_reset()

    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["last_click_sig"] = ""
        _geojson_cache_reset()
        _final_reset()

    elif level == "censo":
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["last_click_sig"] = ""
        _geojson_cache_reset()
        _final_reset()

    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["last_click_sig"] = ""
        _geojson_cache_reset()
        _final_reset()

    elif level == "final":
        _final_reset()


def _final_reset() -> None:
    st.session_state["final_load_sig"] = ""
    st.session_state["final_loaded"] = False


def _geojson_cache_reset() -> None:
    st.session_state["_geojson_cache"] = {}  # type: ignore
    st.session_state["_geojson_cache_order"] = []  # type: ignore


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

def _pick_id_from_last_object(map_state: Dict[str, Any], id_col: str) -> Optional[str]:
    """
    ✅ Performance: tenta pegar o ID diretamente de last_object_clicked.properties,
    evitando parse de tooltip e hit-test geométrico (muito mais rápido).
    """
    obj = (map_state or {}).get("last_object_clicked")
    if not isinstance(obj, dict):
        return None

    props = obj.get("properties") if isinstance(obj.get("properties"), dict) else obj
    if not isinstance(props, dict):
        return None

    return _id_to_str(props.get(id_col))

def normalize_quadra_id(v: Any, width: int = 6) -> Optional[str]:
    s = _id_to_str(v)
    if s is None:
        return None
    s = s.strip()
    return s.zfill(width) if s.isdigit() else s


def make_quadra_uid(iso_id: Any, quadra_id: Any) -> Optional[str]:
    iso = _id_to_str(iso_id)
    qid = _id_to_str(quadra_id)
    if iso is None or qid is None or iso == "" or qid == "":
        return None
    return f"{iso}__{qid}"


def split_quadra_uid(uid: str) -> Tuple[Optional[str], Optional[str]]:
    if not uid or "__" not in uid:
        return (None, None)
    a, b = uid.split("__", 1)
    return (_id_to_str(a), _id_to_str(b))


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


def get_quadras_csv_raw() -> str:
    raw_ui = str(st.session_state.get("drive_quadras_csv_raw", "")).strip()
    if raw_ui:
        return raw_ui
    raw_secret = _get_secret(QUADRAS_CSV_SECRET_KEY) if QUADRAS_CSV_SECRET_KEY else ""
    if raw_secret:
        return raw_secret
    return str(QUADRAS_CSV_FALLBACK_URL or "").strip()


def quadras_csv_local_path() -> Path:
    p1 = REPO_ROOT / QUADRAS_CSV_FILENAME
    if p1.exists() and p1.stat().st_size > 0:
        return p1
    return DATA_CACHE_DIR / QUADRAS_CSV_FILENAME


def ensure_local_quadras_csv() -> Path:
    p = quadras_csv_local_path()
    if p.exists() and p.stat().st_size > 0:
        return p
    raw = get_quadras_csv_raw()
    dst = DATA_CACHE_DIR / QUADRAS_CSV_FILENAME
    if not raw:
        return dst
    try:
        return download_drive_file(raw, dst, label=dst.name)
    except Exception as e:
        st.warning(f"Não foi possível baixar {QUADRAS_CSV_FILENAME} do Drive: {e}")
        return dst


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
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

    response = session.get(url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

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
# READ / FILTER
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def read_gdf_parquet(path: str) -> Optional["gpd.GeoDataFrame"]:
    """
    Leitura base (sem filtros de linhas). Cache global (Streamlit) por path.
    """
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


def read_gdf_parquet_filtered(
    path: str,
    *,
    columns: Optional[List[str]] = None,
    filters: Optional[Any] = None,
) -> Optional["gpd.GeoDataFrame"]:
    """
    Leitura filtrada (pushdown quando possível).

    - Para Lotes.parquet (ajuste #4), tentamos ler somente as quadras selecionadas.
    - NÃO usamos st.cache_data aqui para não explodir o cache com muitas combinações.
      (o zoom não dispara rerun, então a necessidade de cache cai muito).
    """
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None

    try:
        # GeoPandas aceita **kwargs e passa para o backend de parquet (pyarrow).
        # Se filters não for suportado no ambiente, cai no except e faz fallback.
        gdf = gpd.read_parquet(p, columns=columns, filters=filters)  # type: ignore[arg-type]
    except Exception:
        try:
            gdf = gpd.read_parquet(p, columns=columns)
        except Exception:
            return None

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


def read_layer(layer_key: str) -> Optional["gpd.GeoDataFrame"]:
    """
    ✅ Performance: cache em sessão do layer já processado (normalizado + uid criado).
    Isso reduz bastante o custo por clique, porque evita repetir map/normalize em toda rerun.
    """
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        st.error(str(e))
        return None

    # meta simples para invalidar cache se o arquivo mudar
    try:
        meta = (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
    except Exception:
        meta = (str(p), 0.0, 0)

    cache: Dict[str, Any] = st.session_state.get("_layer_cache", {})  # type: ignore
    cache_meta: Dict[str, Any] = st.session_state.get("_layer_cache_meta", {})  # type: ignore

    if layer_key in cache and cache_meta.get(layer_key) == meta:
        g_cached = cache.get(layer_key)
        if g_cached is not None:
            return g_cached

    g = read_gdf_parquet(str(p))
    if g is None or g.empty:
        st.error(f"Layer '{layer_key}' vazia/erro ao ler ({p.name}).")
        return None
        
    g = standardize_columns(g)
    g = _drop_bad_geoms(g)
    g = normalize_id_cols(g, LAYER_ID_COLS.get(layer_key, []))

    if layer_key == "quadra" and QUADRA_ID in g.columns:
        g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))

    # chave composta para quadras (evita colisão entre isócronas)
    if layer_key == "quadra":
        if ISO_ID in g.columns and QUADRA_ID in g.columns:
            iso_s = g[ISO_ID].map(_id_to_str)
            qid_s = g[QUADRA_ID].map(_id_to_str)
            g[QUADRA_UID] = [
                f"{i}__{q}" if (i is not None and q is not None and i != "" and q != "") else None
                for i, q in zip(iso_s, qid_s)
            ]
        else:
            st.warning(f"Quadras.parquet sem colunas '{ISO_ID}' e/ou '{QUADRA_ID}'.")

    # salva em cache de sessão
    cache[layer_key] = g
    cache_meta[layer_key] = meta
    st.session_state["_layer_cache"] = cache  # type: ignore
    st.session_state["_layer_cache_meta"] = cache_meta  # type: ignore

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
# CSV clusters (quadras.csv)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_df_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return None
    try:
        return pd.read_csv(
            p,
            dtype={QUADRA_ID: "string", ISO_ID: "string", CLUSTER_COL: "string"},
        )
    except Exception:
        return None


def get_quadras_csv_df() -> Optional[pd.DataFrame]:
    p = ensure_local_quadras_csv()
    df = read_df_csv(str(p))
    if df is None or df.empty:
        return None

    df = df.copy()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    if QUADRA_ID not in df.columns and "quadra_id" in cols_lower:
        df = df.rename(columns={cols_lower["quadra_id"]: QUADRA_ID})
    if ISO_ID not in df.columns and "iso_id" in cols_lower:
        df = df.rename(columns={cols_lower["iso_id"]: ISO_ID})
    if CLUSTER_COL not in df.columns and "cluster" in cols_lower:
        df = df.rename(columns={cols_lower["cluster"]: CLUSTER_COL})

    if QUADRA_ID in df.columns:
        df[QUADRA_ID] = df[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))

    if ISO_ID in df.columns:
        df[ISO_ID] = df[ISO_ID].map(_id_to_str)
        df[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(df.get(ISO_ID, []), df.get(QUADRA_ID, []))]

    return df


def attach_quadras_csv(g_quad: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if g_quad is None or g_quad.empty:
        return g_quad
    df = get_quadras_csv_df()
    if df is None:
        return g_quad

    if QUADRA_UID in g_quad.columns and QUADRA_UID in df.columns:
        return g_quad.merge(df, on=QUADRA_UID, how="left", suffixes=("", "_csv"))

    if QUADRA_ID not in df.columns:
        return g_quad

    return g_quad.merge(df, on=QUADRA_ID, how="left", suffixes=("", "_csv"))


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        fv = float(v)
        return int(fv)
    except Exception:
        return None


def cluster_label(code: Optional[int]) -> str:
    if code is None:
        return "Sem classe"
    return CLUSTER_LABEL_MAP.get(code, f"Cluster {code}")


def cluster_color(code: Optional[int]) -> str:
    if code is None:
        return CLUSTER_NULL_COLOR
    return CLUSTER_COLOR_MAP.get(code, CLUSTER_NULL_COLOR)


def iso_label_color(nova_class: Any) -> Tuple[str, str]:
    nc = _coerce_int(nova_class)
    if nc is None:
        return ("Sem classe", ISO_DEFAULT_COLOR)
    if nc in ISO_TRANSITION_SET:
        return (ISO_TRANSITION_LABEL, ISO_TRANSITION_COLOR)
    if nc in ISO_VALUE_TO_CLASSNUM:
        k = ISO_VALUE_TO_CLASSNUM[nc]
        return (f"Classe {k}", ISO_CLASSNUM_TO_COLOR.get(k, ISO_DEFAULT_COLOR))
    return ("Outros", ISO_DEFAULT_COLOR)


# =============================================================================
# Legendas
# =============================================================================
def render_legend(title: str, items: list[Tuple[str, str]]) -> None:
    html = f"<div style='margin-top:10px'><b>{title}</b><div style='margin-top:6px'>"
    for lab, col in items:
        html += (
            "<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;background:{col};border:1px solid rgba(0,0,0,.15)'></span>"
            f"<span style='font-size:0.92rem'>{lab}</span>"
            "</div>"
        )
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

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
        folium.map.CustomPane("top_lines", z_index=670).add_to(m)
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


def _session_geojson_set(key: str, value: str, max_items: int = 120) -> None:
    """
    Ajuste #1 (complementar): cache único, com limite (evita crescer indefinidamente).
    """
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})  # type: ignore
    order: list[str] = st.session_state.get("_geojson_cache_order", [])  # type: ignore

    if key in cache:
        cache[key] = value
        # move para o fim (mais recente)
        try:
            order.remove(key)
        except Exception:
            pass
        order.append(key)
    else:
        cache[key] = value
        order.append(key)

    # poda simples (LRU)
    while len(order) > max_items:
        old = order.pop(0)
        cache.pop(old, None)

    st.session_state["_geojson_cache"] = cache  # type: ignore
    st.session_state["_geojson_cache_order"] = order  # type: ignore


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
    try:
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
    except Exception:
        return


def add_polygons_selectable(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    tooltip_col: Optional[str] = None,
    selected_ids: Optional[Set[Any]] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.10,
    selected_color: str = "#14407D",
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.26,
    tooltip_prefix: str = "ID: ",
    simplify_tol: Optional[float] = None,
    cache_key: Optional[str] = None,
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        return
    tooltip_col = tooltip_col or id_col
    if tooltip_col not in gdf.columns:
        return

    selected_ids = selected_ids or set()
    sel = {v for v in (_id_to_str(x) for x in selected_ids) if v is not None}

    tol = simplify_tol if simplify_tol is not None else 0.0006

    key = cache_key or f"base:{name}:{id_col}:{tooltip_col}:{tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        keep = [id_col] if tooltip_col == id_col else [id_col, tooltip_col]
        mini = gdf[keep + ["geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        if tooltip_col != id_col:
            mini[tooltip_col] = mini[tooltip_col].map(_id_to_str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol=tol, keep_cols=keep)
        _session_geojson_set(key, geojson_base)

    if not geojson_base:
        return

    tooltip_base = _mk_tooltip(tooltip_col, tooltip_prefix)

    fg_base = folium.FeatureGroup(name=name, show=True)
    try:
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
    except Exception:
        return

    if sel:
        sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
        if not sel_gdf.empty:
            sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
            geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=tol, keep_cols=[id_col])
            if geojson_sel:
                fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
                try:
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
                except Exception:
                    return


def add_polygons_selectable_colored(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    fill_color_col: str,
    selected_ids: Optional[Set[Any]] = None,
    tooltip_col: Optional[str] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_opacity: float = 0.14,
    selected_color: str = "#14407D",
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.28,
    tooltip_prefix: str = "ID: ",
    simplify_tol: Optional[float] = None,
    cache_key: Optional[str] = None,
    default_fill: str = "#ffffff",
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns or fill_color_col not in gdf.columns:
        return

    tooltip_col = tooltip_col or id_col
    if tooltip_col not in gdf.columns:
        tooltip_col = id_col

    selected_ids = selected_ids or set()
    sel = {v for v in (_id_to_str(x) for x in selected_ids) if v is not None}
    tol = simplify_tol if simplify_tol is not None else 0.0006

    keep = [id_col, fill_color_col]
    if tooltip_col not in keep:
        keep.append(tooltip_col)

    key = cache_key or f"baseC:{name}:{id_col}:{tooltip_col}:{fill_color_col}:{tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[keep + ["geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        mini[tooltip_col] = mini[tooltip_col].map(_id_to_str)
        mini[fill_color_col] = mini[fill_color_col].astype(str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol=tol, keep_cols=keep)
        _session_geojson_set(key, geojson_base)
    if not geojson_base:
        return

    tooltip_base = _mk_tooltip(tooltip_col, tooltip_prefix)

    fg_base = folium.FeatureGroup(name=name, show=True)
    try:
        def _style(f):
            props = (f or {}).get("properties", {}) or {}
            fc = props.get(fill_color_col, default_fill)
            if not fc or str(fc).lower() in ("nan", "none"):
                fc = default_fill
            return {
                "color": base_color,
                "weight": base_weight,
                "opacity": 0.92,
                "lineCap": LINE_CAP,
                "lineJoin": LINE_JOIN,
                "fillColor": fc,
                "fillOpacity": fill_opacity,
            }

        folium.GeoJson(
            data=geojson_base,
            pane=pane,
            smooth_factor=SMOOTH_FACTOR,
            style_function=_style,
            highlight_function=lambda _f: {"weight": base_weight + 1.4, "fillOpacity": min(fill_opacity + 0.12, 1.0)},
            tooltip=tooltip_base,
        ).add_to(fg_base)
        fg_base.add_to(m)
    except Exception:
        return

    if sel:
        sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
        if not sel_gdf.empty:
            sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
            geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=tol, keep_cols=[id_col])
            if geojson_sel:
                fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
                try:
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
                            "fillColor": "#ffffff",
                            "fillOpacity": selected_fill_opacity,
                        },
                    ).add_to(fg_sel)
                    fg_sel.add_to(m)
                except Exception:
                    return

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


def _quadra_ids_from_selected_uids(uids: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for uid in uids:
        _, q = split_quadra_uid(uid)
        if q:
            out.add(q)
    return out


def _quadras_filter_from_censo_selection() -> Tuple[str, Set[str]]:
    """
    Novo caminho (pedido):
    - SetorCensitário -> Quadras via censo_id (em ambos os arquivos).
    - NÃO constrói UID nesta etapa.
    Retorna:
      (id_col_para_filtrar_quadras, conjunto_de_ids)
    onde id_col é QUADRA_UID se existir, senão QUADRA_ID.
    """
    sel_censo: Set[str] = st.session_state.get("selected_censo_ids", set()) or set()
    if not sel_censo:
        return (QUADRA_ID, set())

    g_quad = read_layer("quadra")
    if g_quad is None or g_quad.empty:
        return (QUADRA_ID, set())

    # garante colunas normalizadas (caso não tenha passado no read_layer por algum motivo)
    g_quad = standardize_columns(g_quad)

    if CENSO_ID in g_quad.columns:
        g_q = g_quad[g_quad[CENSO_ID].isin(list(sel_censo))].copy()
        if g_q.empty:
            return (QUADRA_ID, set())

        if QUADRA_UID in g_q.columns and g_q[QUADRA_UID].notna().any():
            ids = {u for u in g_q[QUADRA_UID].dropna().astype(str).tolist() if u}
            return (QUADRA_UID, ids)

        if QUADRA_ID in g_q.columns:
            ids = {u for u in g_q[QUADRA_ID].dropna().astype(str).tolist() if u}
            return (QUADRA_ID, ids)

        return (QUADRA_ID, set())

    # Fallback útil: se Quadras realmente não tiver censo_id,
    # tenta Setor(censo_id) -> Isocronas(censo_id) -> Quadras(iso_id)
    g_iso = read_layer("iso")
    if g_iso is not None and not g_iso.empty:
        g_iso = standardize_columns(g_iso)
        if CENSO_ID in g_iso.columns and ISO_ID in g_iso.columns:
            iso_ids = {i for i in g_iso[g_iso[CENSO_ID].isin(list(sel_censo))][ISO_ID].dropna().astype(str).tolist() if i}
            if iso_ids and QUADRA_PARENT in g_quad.columns:
                g_q = g_quad[g_quad[QUADRA_PARENT].isin(list(iso_ids))].copy()
                if not g_q.empty and QUADRA_UID in g_q.columns:
                    ids = {u for u in g_q[QUADRA_UID].dropna().astype(str).tolist() if u}
                    return (QUADRA_UID, ids)

    # Debug útil: mostra colunas reais se continuar falhando
    st.warning(f"Não encontrei '{CENSO_ID}' em Quadras.parquet. Colunas disponíveis: {list(g_quad.columns)[:40]}")
    return (QUADRA_ID, set())


def consume_map_event(level: str, map_state: Dict[str, Any], allow_click: bool = True) -> bool:
    """
    FIX do 'um clique atrasado':
    - O mapa já foi renderizado neste run.
    - Se você atualiza seleção após st_folium(), a mudança só aparece no próximo run.
    - Então retornamos changed=True e chamamos st.rerun() logo após processar o clique.
    """
    if not allow_click:
        return False

    click = (map_state or {}).get("last_clicked") or None
    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip") or None
    picked_tooltip = parse_tooltip_id(tooltip_raw)

    if not click and not picked_tooltip:
        return False

    sig = _click_signature(picked_tooltip, click if isinstance(click, dict) else None)
    if sig and sig == st.session_state.get("last_click_sig", ""):
        return False
    st.session_state["last_click_sig"] = sig

    changed = False

    # ===== SUBPREF =====
    if level == "subpref":
        picked = picked_tooltip
        # Fallback geométrico é caro; só use se realmente precisar:
        # if not picked and isinstance(click, dict): ...
        if picked:
            st.session_state["selected_subpref_id"] = picked
            reset_to("distrito")
            st.session_state["selected_subpref_id"] = picked
            st.session_state["level"] = "distrito"
            return True  # mudou nível, precisa rerun

    # ===== DISTRITO =====
    if level == "distrito":
        sp = _id_to_str(st.session_state.get("selected_subpref_id"))
        if sp is None:
            return False
        picked = picked_tooltip
        if picked:
            st.session_state["selected_distrito_id"] = picked
            reset_to("isocrona")
            st.session_state["selected_subpref_id"] = sp
            st.session_state["selected_distrito_id"] = picked
            st.session_state["level"] = "isocrona"
            return True

    # ===== ISO =====
    if level == "isocrona":
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if d is None:
            return False
        picked = picked_tooltip
        if picked:
            _toggle_in_set("selected_iso_ids", picked)
            _final_reset()
            return True

    # ===== CENSO =====
    if level == "censo":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v}
        if not iso_ids:
            return False
        picked = picked_tooltip
        if picked:
            _toggle_in_set("selected_censo_ids", picked)
            _final_reset()
            return True

    # ===== QUADRA =====
    if level == "quadra":
        picked = picked_tooltip
        if picked:
            # aqui, picked_tooltip normalmente vai ser QUADRA_ID (tooltip_col=QUADRA_ID)
            # mas o id_col do layer é QUADRA_UID; quando o tooltip vier com quadra_id,
            # a seleção por UID pode não bater. Então, neste modo, mantenha tooltip_col=QUADRA_UID
            # OU mantenha picked_tooltip como UID via propriedades (ver ajuste abaixo).
            _toggle_in_set("selected_quadra_ids", picked)
            _final_reset()
            return True

    return changed


def sanitize_level_state() -> None:
    lvl = st.session_state.get("level", "subpref")

    if lvl == "distrito" and _id_to_str(st.session_state.get("selected_subpref_id")) is None:
        reset_to("subpref")
        return

    if lvl in ("isocrona", "censo", "quadra", "final") and _id_to_str(st.session_state.get("selected_distrito_id")) is None:
        reset_to("distrito")
        return

    if lvl in ("censo", "quadra", "final"):
        iso_ids = st.session_state.get("selected_iso_ids", set())
        if not iso_ids:
            reset_to("isocrona")
            return

    if lvl == "quadra":
        # Migração simples: se ainda tiver quadra_id antigo (sem "__"), limpa
        qset = st.session_state.get("selected_quadra_ids", set()) or set()
        if any(isinstance(x, str) and "__" not in x for x in qset):
            st.session_state["selected_quadra_ids"] = set()

# =============================================================================
# UI: Variável
# =============================================================================
def _variables_for_level(level: str) -> list[str]:
    if level == "subpref":
        return ["Subprefeituras"]
    if level == "distrito":
        return ["Distritos"]
    if level == "isocrona":
        return ["Isócronas", "Isócronas (classes)"]
    if level == "censo":
        return ["Setor censitário"]
    if level == "quadra":
        return ["Quadras", "Cluster"]
    if level == "final":
        return ["Lotes"]
    return ["Nível"]


def ensure_variable_for_level(level: str) -> None:
    opts = _variables_for_level(level)
    cur = st.session_state.get("variable")
    if cur not in opts:
        st.session_state["variable"] = opts[0]


def variable_panel() -> None:
    lvl = st.session_state.get("level", "subpref")
    ensure_variable_for_level(lvl)
    st.selectbox(
        "Variável",
        options=_variables_for_level(lvl),
        key="variable",
        on_change=mark_ui_action,
    )

# =============================================================================
# Data/config panel (mantido)
# =============================================================================
def data_sources_panel() -> None:
    st.caption("Local (data_cache) ou Drive (secrets / input).")
    p_csv = quadras_csv_local_path()
    ok_csv = p_csv.exists() and p_csv.stat().st_size > 0
    st.write(f"- `{p_csv.name}` (clusters): {'✅ local' if ok_csv else '—'}")
    for k in ["subpref", "dist", "iso", "censo", "quadra", "lote"]:
        p = local_layer_path(k)
        ok = layer_available_locally(k)
        st.write(f"- `{p.name}`: {'✅ local' if ok else '—'}")

    st.divider()
    with st.expander("Configurar links/IDs do Drive (opcional)", expanded=False):
        st.caption("Se tiver secrets.toml, não precisa colar aqui. Se colar aqui, tem prioridade.")
        st.text_input(
            "quadras.csv (clusters)",
            key="drive_quadras_csv_raw",
            value=str(st.session_state.get("drive_quadras_csv_raw", "")).strip(),
            placeholder=_get_secret(QUADRAS_CSV_SECRET_KEY) or QUADRAS_CSV_FALLBACK_URL or "Cole aqui o link/ID",
            on_change=mark_ui_action,
        )

        for k in ["subpref", "dist", "iso", "censo", "quadra", "lote"]:
            placeholder = _get_secret(SECRETS_KEYS.get(k, "")) or FALLBACK_URLS.get(k, "") or "Cole aqui o link/ID"
            st.text_input(
                f"{k} ({LOCAL_FILENAMES[k]})",
                key=f"drive_{k}_raw",
                value=str(st.session_state.get(f"drive_{k}_raw", "")).strip(),
                placeholder=placeholder,
                on_change=mark_ui_action,
            )

    st.divider()
    st.checkbox("Modo diagnóstico (FK/colunas)", key="debug_fk", on_change=mark_ui_action)

    st.caption(
        "Obs: se não achar localmente, o app baixa do Drive (pode demorar na primeira vez). "
        "Depois fica cacheado em `data_cache`."
    )

# =============================================================================
# Labels / navigation
# =============================================================================
def _level_label(level: str) -> str:
    return {
        "subpref": "Subprefeituras",
        "distrito": "Distritos",
        "isocrona": "Isócronas",
        "censo": "Setor censitário",
        "quadra": "Quadras",
        "final": "Lotes",
    }.get(level, level)


def _prev_level(level: str) -> Optional[str]:
    if level not in LEVELS:
        return None
    idx = LEVELS.index(level)
    if idx <= 0:
        return None
    return LEVELS[idx - 1]

# =============================================================================
# Fit helpers
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
    except Exception:
        pass


def _fit_selected_isos() -> None:
    iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
    if not iso_ids:
        return
    g_iso = read_layer("iso")
    if g_iso is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_iso, ISO_ID, iso_ids), bump=0, zmax=18)


def _fit_selected_censos() -> None:
    cids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_censo_ids", set())) if v is not None}
    if not cids:
        return
    g_censo = read_layer("censo")
    if g_censo is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_censo, CENSO_ID, cids), bump=0, zmax=18)


def _fit_selected_quadras() -> None:
    quad_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_quadra_ids", set())) if v is not None}
    if not quad_ids:
        return
    g_quad = read_layer("quadra")
    if g_quad is None:
        return
    id_col = QUADRA_UID if (QUADRA_UID in g_quad.columns) else QUADRA_ID
    set_view_to_gdf(subset_by_id_multi(g_quad, id_col, quad_ids), bump=1, zmax=19)

# =============================================================================
# AJUSTE #2 e #3 — controle no painel (nível Isócronas)
# =============================================================================
def _go_from_isos_next() -> None:
    """
    - Se iso_next_mode == 'quadra': vai para quadra (normal)
    - Se iso_next_mode == 'censo': vai para censo (setor censitário), filtrado por iso_id
    """
    mode = st.session_state.get("iso_next_mode", "quadra")
    if mode == "censo":
        st.session_state["level"] = "censo"
        # limpa seleções de ramo
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()
    else:
        st.session_state["level"] = "quadra"
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()


def _go_from_censo_to_quadras() -> None:
    """
    Setor -> Quadras via censo_id (sem construir uid nesta etapa).
    """
    id_col, ids = _quadras_filter_from_censo_selection()
    if not ids:
        st.warning("Nenhuma quadra encontrada para o(s) censo_id selecionado(s).")
        return

    # guarda filtro e qual coluna deve ser usada
    st.session_state["quadra_filter_uids"] = ids
    st.session_state["quadra_filter_col"] = id_col  # ✅ novo
    st.session_state["selected_quadra_ids"] = set()
    st.session_state["level"] = "quadra"
    _final_reset()


def _go_to_final() -> None:
    st.session_state["level"] = "final"
    _final_reset()

# =============================================================================
# Painel de controle (o seu “lateral direito”)
# =============================================================================
def control_panel() -> None:
    lvl = st.session_state["level"]
    prev = _prev_level(lvl)

    c1, c2 = st.columns(2)
    with c1:
        if prev is None:
            st.button(_level_label("subpref"), disabled=True, use_container_width=True)
        else:
            st.button(
                _level_label(prev),
                type="primary",
                use_container_width=True,
                on_click=lambda: (mark_ui_action(), _back_one_level()),
            )

    with c2:
        st.button(
            "Reset",
            type="primary",
            use_container_width=True,
            on_click=lambda: (mark_ui_action(), reset_to("subpref")),
        )

    st.divider()
    st.subheader("Variável", anchor=False)
    variable_panel()

    # Legendas
    var = st.session_state.get("variable")

    if lvl == "isocrona" and var == "Isócronas (classes)":
        render_legend(
            "Legenda — Isócronas (nova_class)",
            [
                (f"{ISO_TRANSITION_LABEL} (nova_class ∈ {sorted(list(ISO_TRANSITION_SET))})", ISO_TRANSITION_COLOR),
                ("Classe 1 (nova_class=0)", ISO_CLASSNUM_TO_COLOR[1]),
                ("Classe 2 (nova_class=2)", ISO_CLASSNUM_TO_COLOR[2]),
                ("Classe 3 (nova_class=4)", ISO_CLASSNUM_TO_COLOR[3]),
                ("Classe 4 (nova_class=5)", ISO_CLASSNUM_TO_COLOR[4]),
                ("Classe 5 (nova_class=7)", ISO_CLASSNUM_TO_COLOR[5]),
                ("Classe 6 (nova_class=8)", ISO_CLASSNUM_TO_COLOR[6]),
                ("Classe 7 (nova_class=9)", ISO_CLASSNUM_TO_COLOR[7]),
                ("Outros (branco)", ISO_DEFAULT_COLOR),
            ],
        )

    if lvl == "quadra" and var == "Cluster":
        render_legend(
            "Legenda — Quadras (Cluster)",
            [
                ("0 — Alta Densidade Periférica", CLUSTER_COLOR_MAP[0]),
                ("1 — Uso Misto Intermediário", CLUSTER_COLOR_MAP[1]),
                ("2 — Média Densidade Periférica", CLUSTER_COLOR_MAP[2]),
                ("3 — Uso Misto Verticalizado Central", CLUSTER_COLOR_MAP[3]),
                ("4 — Predominância Comercial e de Serviços", CLUSTER_COLOR_MAP[4]),
                ("Sem classe (null)", CLUSTER_NULL_COLOR),
            ],
        )

    # ---- Ações por nível (ajustes #2 e #3 entram aqui) ----
    st.divider()
    st.subheader("Ações", anchor=False)

    if lvl == "isocrona":
        ok = len(st.session_state.get("selected_iso_ids", set())) > 0

        # Opção 1: Ajustar ao selecionado
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: (mark_ui_action(), _fit_selected_isos()),
        )

        # Opções 2 e 3: destino do próximo nível
        st.radio(
            "Próximo nível",
            options=["Quadras", "Setor censitário"],
            index=0 if st.session_state.get("iso_next_mode") == "quadra" else 1,
            key="_iso_next_ui",
            on_change=mark_ui_action,
        )
        st.session_state["iso_next_mode"] = "quadra" if st.session_state["_iso_next_ui"] == "Quadras" else "censo"

        st.button(
            "Prosseguir",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: (mark_ui_action(), _go_from_isos_next()),
        )

    if lvl == "censo":
        okc = len(st.session_state.get("selected_censo_ids", set())) > 0
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not okc,
            on_click=lambda: (mark_ui_action(), _fit_selected_censos()),
        )
        st.button(
            "Prosseguir → Quadras (filtrar por quadra_id)",
            use_container_width=True,
            disabled=not okc,
            on_click=lambda: (mark_ui_action(), _go_from_censo_to_quadras()),
        )

    if lvl == "quadra":
        okq = len(st.session_state.get("selected_quadra_ids", set())) > 0
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _fit_selected_quadras()),
        )
        st.button(
            "➡️ Ir para Lotes",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _go_to_final()),
        )

    if lvl == "final":
        okq = len(st.session_state.get("selected_quadra_ids", set())) > 0
        st.caption("Lotes são carregados somente para as quadras selecionadas (otimizado).")
        st.button(
            "Recarregar Lotes selecionados",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _final_reset()),
        )

    st.divider()
    with st.expander("Dados / Config", expanded=False):
        data_sources_panel()

# =============================================================================
# FINAL: leitura filtrada de lotes (ajuste #4)
# =============================================================================
def _make_parquet_filters_for_quadras(quadra_ids: Set[str]) -> Any:
    """
    Monta filtros para parquet (pyarrow) em DNF.
    - Preferimos um filtro 'in' (mais compacto) e caímos para OR de '==' se necessário.

    Observação:
    - Operadores aceitos dependem do backend; se falhar, faremos fallback para filtro em memória.
    """
    vals = sorted({v for v in quadra_ids if v})
    if not vals:
        return None

    # tentativa compacta: ('quadra_id', 'in', [...])
    return [(QUADRA_ID, "in", vals)]


def _read_lotes_for_selected_quadras(selected_uids: Set[str]) -> Optional["gpd.GeoDataFrame"]:
    """
    Ajuste #4:
    - Carrega Lotes.parquet limitado às quadras selecionadas (por quadra_id).
    - Tenta pushdown com `filters` no parquet; fallback para leitura completa + filtro.
    """
    try:
        p = ensure_local_layer("lote")
    except Exception as e:
        st.error(str(e))
        return None

    quadra_ids = _quadra_ids_from_selected_uids(selected_uids)
    if not quadra_ids:
        return None

    # Assinatura do carregamento (para manter "1 load por seleção" no nível final)
    sig = "|".join(sorted(list(quadra_ids)))
    if st.session_state.get("final_loaded", False) and st.session_state.get("final_load_sig") == sig:
        # já carregado para esta seleção
        return st.session_state.get("_final_lotes_gdf")  # type: ignore

    # Colunas mínimas para reduzir IO
    cols = [QUADRA_ID, LOTE_ID, "geometry"]
    filters = _make_parquet_filters_for_quadras(quadra_ids)

    g = read_gdf_parquet_filtered(str(p), columns=cols, filters=filters)
    if g is None:
        return None
    g = _drop_bad_geoms(g)
    g = normalize_id_cols(g, [QUADRA_ID, LOTE_ID])

    # Se o filtro pushdown não funcionou (ou coluna não existe), garante filtro em memória
    if QUADRA_ID in g.columns:
        g = g[g[QUADRA_ID].isin(list(quadra_ids))].copy()

    st.session_state["_final_lotes_gdf"] = g  # type: ignore
    st.session_state["final_loaded"] = True
    st.session_state["final_load_sig"] = sig
    return g

# =============================================================================
# MAP RENDER
# =============================================================================
def render_map_panel() -> None:
    level = st.session_state["level"]
    ensure_variable_for_level(level)

    title = ""

    # -----------------------------
    # SUBPREF
    # -----------------------------
    if level == "subpref":
        title = "Subprefeituras"
        g_sub = read_layer("subpref")
        if g_sub is None:
            st.stop()
        if SUBPREF_ID not in g_sub.columns:
            st.error(f"Coluna obrigatória ausente em Subprefeitura: '{SUBPREF_ID}'.")
            st.stop()

        if st.session_state.get("last_level") != "subpref":
            set_view_to_gdf(g_sub, bump=0)
            st.session_state["last_level"] = "subpref"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_polygons_selectable(
            m, g_sub, "Subprefeituras", SUBPREF_ID,
            tooltip_col=SUBPREF_ID,
            selected_ids=set(),
            base_weight=0.9, fill_opacity=0.04,
            selected_color=PB_NAVY,
            tooltip_prefix="Subpref: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"subpref:all:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )

    # -----------------------------
    # DISTRITO
    # -----------------------------
    elif level == "distrito":
        sp = _id_to_str(st.session_state["selected_subpref_id"])
        if sp is None:
            reset_to("subpref")
            return

        title = f"Distritos (Subpref {sp})"
        g_dist = read_layer("dist")
        g_sub = read_layer("subpref")
        if g_dist is None or g_sub is None:
            st.stop()
        if DIST_ID not in g_dist.columns or DIST_PARENT not in g_dist.columns:
            st.error(f"Colunas obrigatórias ausentes em Distritos: '{DIST_ID}' e/ou '{DIST_PARENT}'.")
            st.stop()

        g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
        g_show = subset_by_parent(g_dist, DIST_PARENT, sp)

        if st.session_state.get("last_level") != "distrito":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "distrito"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_parent_fill(
            m, g_parent, "Subpref selecionada (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"parent:subpref:{sp}:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )
        add_polygons_selectable(
            m, g_show, "Distritos", DIST_ID,
            tooltip_col=DIST_ID,
            selected_ids=set(),
            base_weight=0.75, fill_opacity=0.06,
            selected_color=PB_NAVY,
            tooltip_prefix="Distrito: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"dist:sp:{sp}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

    # -----------------------------
    # ISÓCRONAS (ajuste #2 entra no painel, não aqui)
    # -----------------------------
    elif level == "isocrona":
        d = _id_to_str(st.session_state["selected_distrito_id"])
        if d is None:
            reset_to("distrito")
            return

        sel_n = len(st.session_state.get("selected_iso_ids", set()))
        title = f"Isócronas (Distrito {d}) — selecionadas: {sel_n}"

        g_iso = read_layer("iso")
        g_dist = read_layer("dist")
        if g_iso is None or g_dist is None:
            st.stop()
        if ISO_ID not in g_iso.columns or ISO_PARENT not in g_iso.columns:
            st.error(f"Colunas obrigatórias ausentes em Isócronas: '{ISO_ID}' e/ou '{ISO_PARENT}'.")
            st.stop()

        g_parent = subset_by_id(g_dist, DIST_ID, d)
        g_show = subset_by_parent(g_iso, ISO_PARENT, d)

        if st.session_state.get("last_level") != "isocrona":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "isocrona"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        add_parent_fill(
            m, g_parent, "Distrito selecionado (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"parent:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

        g_show_viz = g_show.copy()
        if ISO_CLASS_COL in g_show_viz.columns:
            tmp = g_show_viz[ISO_CLASS_COL].apply(
                lambda v: pd.Series(iso_label_color(v), index=["__iso_label", "__iso_color"])
            )
            g_show_viz["__iso_label"] = tmp["__iso_label"]
            g_show_viz["__iso_color"] = tmp["__iso_color"]
        else:
            g_show_viz["__iso_label"] = "Outros"
            g_show_viz["__iso_color"] = ISO_DEFAULT_COLOR

        if st.session_state.get("variable") == "Isócronas (classes)":
            add_polygons_selectable_colored(
                m,
                g_show_viz,
                "Isócronas",
                ISO_ID,
                fill_color_col="__iso_color",
                selected_ids=st.session_state["selected_iso_ids"],
                tooltip_col=ISO_ID,
                base_weight=0.95,
                fill_opacity=1.0,
                selected_color=PB_NAVY,
                selected_weight=3.0,
                selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"isoVIZ:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
                default_fill=ISO_DEFAULT_COLOR,
            )
        else:
            add_polygons_selectable(
                m,
                g_show,
                "Isócronas",
                ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_weight=0.95,
                fill_opacity=0.14,
                selected_color=PB_NAVY,
                selected_weight=3.0,
                selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"iso:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
            )

    # -----------------------------
    # CENSO (novo nível, ajuste #2/#3)
    # -----------------------------
    elif level == "censo":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
        if not iso_ids:
            reset_to("isocrona")
            return

        sel_nc = len(st.session_state.get("selected_censo_ids", set()))
        title = f"Setor censitário — selecionados: {sel_nc}"

        g_censo = read_layer("censo")
        g_iso = read_layer("iso")
        if g_censo is None or g_iso is None:
            st.stop()

        # Filtra setores pelo iso_id selecionado (pedido: associado ao iso_id)
        if CENSO_PARENT in g_censo.columns:
            g_show = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)
        elif ISO_ID in g_censo.columns:
            g_show = subset_by_parent_multi(g_censo, ISO_ID, iso_ids)
        else:
            st.error(f"Setorcensitario.parquet precisa ter '{CENSO_PARENT}' ou '{ISO_ID}'.")
            st.stop()

        g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)

        if st.session_state.get("last_level") != "censo":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "censo"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        add_parent_fill(
            m, g_parent, "Isócronas selecionadas (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"parent:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )

        add_polygons_selectable(
            m,
            g_show,
            "Setor censitário",
            CENSO_ID,
            tooltip_col=CENSO_ID,
            selected_ids=st.session_state["selected_censo_ids"],
            base_weight=0.75,
            fill_opacity=0.10,
            selected_color=PB_NAVY,
            selected_weight=2.5,
            selected_fill_opacity=0.0,
            tooltip_prefix="Setor: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["censo"],
            cache_key=f"censo:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['censo']}",
        )

    # -----------------------------
    # QUADRAS (normal ou filtrado por setor)
    # -----------------------------
    elif level == "quadra":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
        if not iso_ids:
            reset_to("isocrona")
            return

        mode = st.session_state.get("iso_next_mode", "quadra")

        # ✅ novo: quando veio do setor, o filtro pode ser por QUADRA_ID ou QUADRA_UID
        filter_ids: Set[str] = st.session_state.get("quadra_filter_uids", set()) or set()
        filter_col: str = st.session_state.get("quadra_filter_col", QUADRA_UID)

        sel_nq = len(st.session_state.get("selected_quadra_ids", set()))
        if mode == "censo":
            title = f"Quadras (filtradas por Setor censitário) — selecionadas: {sel_nq}"
        else:
            title = f"Quadras — selecionadas: {sel_nq}"

        g_quad = read_layer("quadra")
        g_iso = read_layer("iso")
        if g_quad is None or g_iso is None:
            st.stop()

        # ✅ não força mais QUADRA_UID existir (pode não existir nessa etapa)
        g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)

        # ✅ define qual coluna o mapa vai usar como ID da feature nesta tela
        # - se estamos no modo "censo", usamos filter_col (pode ser quadra_id)
        # - senão, preferimos QUADRA_UID se existir, senão QUADRA_ID
        if mode == "censo":
            id_col_map = filter_col if filter_col in g_quad.columns else QUADRA_ID
        else:
            id_col_map = QUADRA_UID if QUADRA_UID in g_quad.columns else QUADRA_ID

        # Ajuste #3:
        # - Se veio do setor censitário, filtra quadras usando (filter_col, filter_ids)
        if mode == "censo":
            if not filter_ids:
                st.warning("Nenhum filtro de quadra derivado do setor censitário. Volte e selecione setores.")
                g_show = g_quad.iloc[0:0].copy()
            else:
                if filter_col not in g_quad.columns:
                    st.warning(f"Quadras.parquet não contém a coluna '{filter_col}'. Vou tentar filtrar por '{QUADRA_ID}'.")
                    g_show = subset_by_id_multi(g_quad, QUADRA_ID, filter_ids)
                    id_col_map = QUADRA_ID
                else:
                    g_show = subset_by_id_multi(g_quad, filter_col, filter_ids)
        else:
            g_show = subset_by_parent_multi(g_quad, QUADRA_PARENT, iso_ids)

        if st.session_state.get("last_level") != "quadra":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=1)
            st.session_state["last_level"] = "quadra"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        add_parent_fill(
            m, g_parent, "Isócronas selecionadas (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"parent:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )

        # Visualização por cluster
        g_show_viz = attach_quadras_csv(g_show)
        if CLUSTER_COL in g_show_viz.columns:
            g_show_viz["__cluster_code"] = g_show_viz[CLUSTER_COL].apply(_coerce_int)
            g_show_viz["__cluster_label"] = g_show_viz["__cluster_code"].apply(cluster_label)
            g_show_viz["__cluster_color"] = g_show_viz["__cluster_code"].apply(cluster_color)
        else:
            g_show_viz["__cluster_code"] = None
            g_show_viz["__cluster_label"] = "Sem classe"
            g_show_viz["__cluster_color"] = CLUSTER_NULL_COLOR

        cache_prefix = "quad_from_censo" if mode == "censo" else "quad"
        key_sig = "|".join(sorted(list(filter_ids))) if mode == "censo" else "|".join(sorted(list(iso_ids)))

        if st.session_state.get("variable") == "Cluster":
            add_polygons_selectable_colored(
                m,
                g_show_viz,
                "Quadras",
                id_col_map,  # ✅ aqui!
                fill_color_col="__cluster_color",
                selected_ids=st.session_state["selected_quadra_ids"],
                tooltip_col=QUADRA_ID if QUADRA_ID in g_show_viz.columns else id_col_map,
                base_weight=0.80,
                fill_opacity=1.0,
                selected_color=PB_NAVY,
                selected_weight=2.6,
                selected_fill_opacity=0.0,
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                cache_key=f"{cache_prefix}:viz:{key_sig}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
                default_fill=CLUSTER_NULL_COLOR,
            )
        else:
            add_polygons_selectable(
                m,
                g_show,
                "Quadras",
                id_col_map,  # ✅ aqui!
                tooltip_col=QUADRA_ID if QUADRA_ID in g_show.columns else id_col_map,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_weight=0.80,
                fill_opacity=0.12,
                selected_color=PB_NAVY,
                selected_weight=2.6,
                selected_fill_opacity=0.0,
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                cache_key=f"{cache_prefix}:{key_sig}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
            )

    # -----------------------------
    # FINAL = LOTES (ajuste #4)
    # -----------------------------
    else:
        title = "Lotes (filtrado por quadras selecionadas)"

        sel_uids: Set[str] = st.session_state.get("selected_quadra_ids", set()) or set()
        if not sel_uids:
            st.warning("Selecione ao menos uma quadra para visualizar lotes.")
            m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        else:
            # Só carrega lotes quando necessário; e sempre filtrado
            if not st.session_state.get("final_loaded", False):
                st.info("Clique em “Carregar Lotes selecionados” para carregar apenas os lotes das quadras atuais.")
                if st.button("⬇️ Carregar Lotes selecionados", type="secondary"):
                    mark_ui_action()
                    _final_reset()  # força recarga
                    st.session_state["final_loaded"] = False
            g_lote = _read_lotes_for_selected_quadras(sel_uids)

            g_quad = read_layer("quadra")
            m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

            if g_quad is not None and QUADRA_UID in g_quad.columns:
                g_parent = subset_by_id_multi(g_quad, QUADRA_UID, sel_uids)
                add_parent_fill(
                    m, g_parent, "Quadras selecionadas (sombra)",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                    cache_key=f"parent:quad:{'|'.join(sorted(list(sel_uids)))}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
                )

            if g_lote is not None and not g_lote.empty:
                # Fit no primeiro carregamento do final
                if st.session_state.get("last_level") != "final":
                    set_view_to_gdf(g_lote, bump=0, zmax=19)
                    st.session_state["last_level"] = "final"

                add_polygons_selectable(
                    m,
                    g_lote,
                    "Lotes",
                    LOTE_ID if LOTE_ID in g_lote.columns else QUADRA_ID,
                    tooltip_col=LOTE_ID if LOTE_ID in g_lote.columns else QUADRA_ID,
                    selected_ids=set(),
                    base_weight=0.55,
                    fill_opacity=0.08,
                    selected_color=PB_NAVY,
                    tooltip_prefix="Lote: " if LOTE_ID in g_lote.columns else "Quadra: ",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["lote"],
                    cache_key=f"lote:sel:{st.session_state.get('final_load_sig','')}:{SIMPLIFY_TOL_BY_LEVEL['lote']}",
                )
            else:
                st.warning("Nenhum lote encontrado para as quadras selecionadas (ou falha de leitura).")

    st.markdown(f"### {title}")

    # -------------------------------------------------------------------------
    # AJUSTE #1 (principal): não retornar center/zoom/bounds
    # -> zoom/scroll NÃO causam rerun, evitando reload do mapa e cascata de caches
    # -------------------------------------------------------------------------
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

    # Consome apenas cliques/tooltip (zoom/pan não chega aqui)
    ui_sig = int(st.session_state.get("_ui_action_sig", 0))
    ui_seen = int(st.session_state.get("_ui_action_sig_seen", 0))
    ui_action = ui_sig != ui_seen
    st.session_state["_ui_action_sig_seen"] = ui_sig

    if ui_action:
        st.session_state["last_click_sig"] = ""

    changed = consume_map_event(level, map_state or {}, allow_click=(not ui_action))
    if changed:
        st.rerun()

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

    sanitize_level_state()

    # Layout: mapa grande + painel de controle na lateral direita (como você descreveu)
    left, right = st.columns([4, 1], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        render_map_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        control_panel()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


