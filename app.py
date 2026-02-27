from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import base64
import re

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

# ✅ Simplificação > 0 para reduzir payload e evitar timeouts do componente em deploy
SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0012,
    "distrito": 0.0008,
    "isocrona": 0.0006,
    "censo": 0.00035,
    "quadra": 0.00025,
    "lote": 0.00012,
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
QUADRAS_CSV_FALLBACK_URL = (
    "https://drive.google.com/file/d/1_WKryQlu_jZL1xsgAmQDrI81aSdzKYsc/view?usp=drive_link"
)

CLUSTER_COL = "Cluster"
ISO_CLASS_COL = "nova_class"

CLUSTER_COLOR_MAP = {
    0: "#bf7db2",
    1: "#f7bd6a",
    2: "#cf651f",
    3: "#cf651f",
    4: "#793393",
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
# IDS / CHAVES
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
QUADRA_UID = "quadra_uid"  # iso_id__quadra_id
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = SUBPREF_ID
ISO_PARENT = DIST_ID

# ✅ Estrutura para filtrar quadras por duas vias
QUADRA_PARENTS = {"iso": ISO_ID, "censo": CENSO_ID}

LOTE_PARENT = QUADRA_ID
CENSO_PARENT = ISO_ID

LEVELS = ["subpref", "distrito", "isocrona", "censo", "quadra", "final"]

LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT],
    "quadra": [QUADRA_ID, ISO_ID, CENSO_ID, QUADRA_UID],
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

# =============================================================================
# NORMALIZAÇÃO DE COLUNAS / IDS
# =============================================================================
def _mk_aliases(base: str) -> Set[str]:
    b = base.strip()
    return {
        b,
        b.upper(),
        b.title(),
        b.replace("_", ""),
        b.replace("_", "").upper(),
        f"{b} ",
        f" {b}",
    }


COL_ALIASES: Dict[str, Set[str]] = {
    SUBPREF_ID: _mk_aliases(SUBPREF_ID),
    DIST_ID: _mk_aliases(DIST_ID),
    ISO_ID: _mk_aliases(ISO_ID),
    QUADRA_ID: _mk_aliases(QUADRA_ID),
    LOTE_ID: _mk_aliases(LOTE_ID),
    CENSO_ID: _mk_aliases(CENSO_ID) | {"cendo_id", "CENDO_ID", "setor_id", "id_setor", "codigo_setor", "cd_setor"},
}


def standardize_columns(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Padroniza nomes de colunas:
    - strip
    - colapsa múltiplos espaços
    - resolve aliases para nomes canônicos
    """
    if gdf is None or gdf.empty:
        return gdf

    ren: Dict[str, str] = {}
    for c in gdf.columns:
        raw = str(c)
        cl = re.sub(r"\s+", " ", raw.strip())
        low = cl.lower()

        if raw != cl:
            ren[raw] = cl

        for canon, aliases in COL_ALIASES.items():
            aliases_norm = {a.strip() for a in aliases}
            aliases_low = {a.strip().lower() for a in aliases}
            if cl in aliases_norm or low in aliases_low:
                ren[cl] = canon

    return gdf.rename(columns=ren)


def _id_to_str(v: Any) -> Optional[str]:
    """Coerção estável para string (1.0 -> '1')."""
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


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g

def ensure_set_of_str(value: Any, *, drop_empty: bool = True) -> Set[str]:
    """
    Normaliza qualquer entrada razoável em `set[str]`.

    Aceita:
      - set/list/tuple de valores (str/int/float/None...)
      - valor único (str/int/float) -> vira conjunto unitário
      - None -> set()

    Converte via `_id_to_str` e remove nulos/vazios.
    """
    if value is None:
        return set()

    # Se já for set/list/tuple, ok. Se for outra coisa, trate como item único.
    if isinstance(value, (set, list, tuple)):
        items: Iterable[Any] = value
    else:
        items = (value,)  # item único

    out: Set[str] = set()
    for x in items:
        s = _id_to_str(x)  # usa a sua função existente
        if s is None:
            continue
        if drop_empty and s == "":
            continue
        out.add(s)

    return out
    
def choose_quadra_parent_col(
    g_quad: "gpd.GeoDataFrame",
    *,
    preferred: str = CENSO_ID,
    fallback: str = ISO_ID,
) -> Optional[str]:
    if g_quad is None or getattr(g_quad, "empty", True):
        return None
    if preferred in g_quad.columns:
        return preferred
    if fallback in g_quad.columns:
        return fallback
    return None

def get_quadras_subset_for_mode(
    g_quad: "gpd.GeoDataFrame",
    *,
    mode: str,
    iso_ids: Set[str],
    filter_censo_ids: Set[str],
) -> "gpd.GeoDataFrame":
    """
    Retorna o subset de quadras que deve estar 'clicável' e/ou renderizado
    no nível quadra, respeitando o modo:
      - mode == 'censo': tenta filtrar por censo_id; se não existir, fallback por iso_id
      - mode != 'censo': filtra por iso_id
    """
    if g_quad is None or g_quad.empty:
        return g_quad.iloc[0:0].copy()

    # normaliza inputs
    iso_ids = ensure_set_of_str(iso_ids)
    filter_censo_ids = ensure_set_of_str(filter_censo_ids)

    if mode == "censo":
        parent_col = choose_quadra_parent_col(g_quad, preferred=CENSO_ID, fallback=ISO_ID)

        if parent_col == CENSO_ID:
            if not filter_censo_ids:
                return g_quad.iloc[0:0].copy()
            return subset_by_id_multi(g_quad, CENSO_ID, filter_censo_ids)

        if parent_col == ISO_ID:
            if not iso_ids:
                return g_quad.iloc[0:0].copy()
            return subset_by_parent_multi(g_quad, ISO_ID, iso_ids)

        return g_quad.iloc[0:0].copy()

    # modo padrão (iso -> quadras)
    if not iso_ids:
        return g_quad.iloc[0:0].copy()
    return subset_by_parent_multi(g_quad, ISO_ID, iso_ids)

# =============================================================================
# HEADER / CSS
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
MAP_KEY = "map_view"  # ✅ um único key para o componente (evita recarga de assets)


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("last_level", None)

    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)

    st.session_state.setdefault("selected_iso_ids", set())
    st.session_state.setdefault("selected_censo_ids", set())
    st.session_state.setdefault("selected_quadra_ids", set())

    st.session_state.setdefault("iso_next_mode", "quadra")  # "quadra" | "censo"

    # filtro usado para transição censo -> quadra
    st.session_state.setdefault("quadra_filter_col", CENSO_ID)
    st.session_state.setdefault("quadra_filter_uids", set())

    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)

    # dedupe de clique
    st.session_state.setdefault("last_click_sig", "")

    # caches
    st.session_state.setdefault("_geojson_cache", {})
    st.session_state.setdefault("_geojson_cache_order", [])
    st.session_state.setdefault("_layer_cache", {})
    st.session_state.setdefault("_layer_cache_meta", {})

    # UI action signature
    st.session_state.setdefault("_ui_action_sig", 0)
    st.session_state.setdefault("_ui_action_sig_seen", 0)

    # nível que renderizou o mapa anterior
    st.session_state.setdefault("_map_level_rendered", None)

    # qual ID está sendo usado para seleção de quadra no mapa atual
    st.session_state.setdefault("_quadra_id_col_map", QUADRA_UID)

    # variável atual
    st.session_state.setdefault("variable", None)

    # final (lotes)
    st.session_state.setdefault("final_load_sig", "")
    st.session_state.setdefault("final_loaded", False)
    st.session_state.setdefault("_final_lotes_gdf", None)

    # debug
    st.session_state.setdefault("debug_schema", False)


def mark_ui_action() -> None:
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1


def _geojson_cache_reset() -> None:
    st.session_state["_geojson_cache"] = {}
    st.session_state["_geojson_cache_order"] = []


def _final_reset() -> None:
    st.session_state["final_load_sig"] = ""
    st.session_state["final_loaded"] = False
    st.session_state["_final_lotes_gdf"] = None


def reset_to(level: str, *, clear_click_sig: bool = True) -> None:
    """Reset por nível mantendo otimizações."""
    st.session_state["level"] = level
    if clear_click_sig:
        st.session_state["last_click_sig"] = ""
    _geojson_cache_reset()
    _final_reset()

    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
        st.session_state["last_level"] = None

    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["iso_next_mode"] = "quadra"
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "censo":
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()

    elif level == "quadra":
        st.session_state["selected_quadra_ids"] = set()

    elif level == "final":
        pass


def _prev_level(level: str) -> Optional[str]:
    if level not in LEVELS:
        return None
    i = LEVELS.index(level)
    if i <= 0:
        return None
    return LEVELS[i - 1]


def _back_one_level() -> None:
    cur = st.session_state.get("level", "subpref")
    prev = _prev_level(cur)
    if prev:
        reset_to(prev)


def _toggle_in_set(key: str, value: Any) -> None:
    s: Set[Any] = st.session_state.get(key, set()) or set()
    if value in s:
        s.remove(value)
    else:
        s.add(value)
    st.session_state[key] = s


def debug_layer_schema(layer_key: str, g: Optional["gpd.GeoDataFrame"], *, path: Optional[Path] = None) -> None:
    """Mostra no UI (Streamlit) o schema real carregado. Útil para debug em deploy."""
    if not st.session_state.get("debug_schema", False):
        return
    if g is None:
        st.write(f"[debug] layer={layer_key}: None")
        return
    if path is not None:
        try:
            st.write(f"[debug] layer={layer_key} path={path} mtime={path.stat().st_mtime} size={path.stat().st_size}")
        except Exception:
            st.write(f"[debug] layer={layer_key} path={path}")
    st.write(f"[debug] layer={layer_key} shape={g.shape}")
    st.write(f"[debug] layer={layer_key} cols={list(g.columns)}")
    try:
        st.write(f"[debug] layer={layer_key} crs={getattr(g, 'crs', None)}")
    except Exception:
        pass


# =============================================================================
# SANITIZE (evita estado inválido)
# =============================================================================
def sanitize_level_state() -> None:
    lvl = st.session_state.get("level", "subpref")

    if lvl == "distrito" and _id_to_str(st.session_state.get("selected_subpref_id")) is None:
        reset_to("subpref")
        return

    if lvl in ("isocrona", "censo", "quadra", "final") and _id_to_str(st.session_state.get("selected_distrito_id")) is None:
        reset_to("distrito")
        return

    if lvl in ("censo", "quadra", "final"):
        iso_ids = st.session_state.get("selected_iso_ids", set()) or set()
        if not iso_ids:
            reset_to("isocrona")
            return

    if lvl == "quadra":
        id_col_map = st.session_state.get("_quadra_id_col_map", QUADRA_UID)
        if id_col_map == QUADRA_UID:
            qset = st.session_state.get("selected_quadra_ids", set()) or set()
            if any(isinstance(x, str) and "__" not in x for x in qset):
                # evita misturar IDs simples com UID no modo iso
                st.session_state["selected_quadra_ids"] = set()


# =============================================================================
# DRIVE / LOCAL IO
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
    "iso": "https://drive.google.com/file/d/1UQkswArEB_1MmPhGW_TgvLahR6dPvZXn/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/1Ivy2PyGHqFgIxSMoK3N9oik2wr5v912U/view?usp=drive_link",
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
    """
    Extrai FILE_ID de:
      - ID puro
      - links /file/d/<id>/...
      - links ?id=<id>
      - links do tipo open?id=<id>
      - links uc?export=download&id=<id>

    Retorna "" se não conseguir extrair.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    # ID puro
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw

    # padrões comuns
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)

    # fallback agressivo: pega um token grande
    m = re.search(r"([a-zA-Z0-9_-]{20,})", raw)
    return m.group(1) if m else ""


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
    """
    Download via Google Drive com:
      - diagnóstico (status/URL final/content-type)
      - suporte a token de confirmação para arquivos grandes
      - detecção de "HTML salvo no lugar do parquet" (permissão/link incorreto)

    Levanta RuntimeError com mensagem útil quando falhar.
    """
    import requests  # local import

    file_id = extract_drive_id(file_id_or_url)
    if not file_id:
        raise RuntimeError(f"FILE_ID inválido: não foi possível extrair ID de: {file_id_or_url!r}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # se já existe e tem tamanho, retorna
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    session = requests.Session()

    # Endpoint mais estável para download
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    def get_confirm_token(resp) -> Optional[str]:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    # 1) primeira tentativa
    resp = session.get(url, params=params, stream=True, allow_redirects=True, timeout=120)
    token = get_confirm_token(resp)

    # 2) se houver token de confirmação (arquivos grandes), tenta com confirm
    if token:
        params2 = {"export": "download", "id": file_id, "confirm": token}
        resp = session.get(url, params=params2, stream=True, allow_redirects=True, timeout=120)

    # 3) valida status
    if resp.status_code != 200:
        final_url = getattr(resp, "url", url)
        ct = resp.headers.get("Content-Type", "")
        raise RuntimeError(
            f"Download falhou para '{label or dst.name}' (HTTP {resp.status_code}). "
            f"URL={final_url} Content-Type={ct}. "
            f"Possíveis causas: link/ID incorreto, arquivo removido/movido, ou permissões (não público)."
        )

    total = int(resp.headers.get("Content-Length", 0) or 0)
    chunk = 1024 * 1024  # 1MB

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

    # 4) sanity check: às vezes o Drive devolve HTML (página de erro) e salva como arquivo "parquet"
    try:
        size = dst.stat().st_size
        if size < 300_000:  # 300KB (parquet real costuma ser maior; ajuste se necessário)
            head = dst.read_bytes()[:512].lower()
            if b"<!doctype html" in head or b"<html" in head:
                dst.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download de '{ui_label}' retornou HTML (não é parquet). "
                    f"Verifique se o arquivo está público e se o ID do Drive está correto."
                )
    except Exception:
        # se o check falhar, prefira falhar explicitamente em vez de aceitar arquivo inválido
        raise

    return dst


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


def local_layer_path(layer_key: str) -> Path:
    return DATA_CACHE_DIR / LOCAL_FILENAMES[layer_key]


def layer_available_locally(layer_key: str) -> bool:
    p = local_layer_path(layer_key)
    return p.exists() and p.stat().st_size > 0


def ensure_local_layer(layer_key: str, *, force_redownload: bool = False) -> Path:
    """
    Garante que o layer esteja no cache local.
    Ajuste: permite forçar re-download (útil quando o arquivo local ficou incompleto/corrompido).
    """
    dst = local_layer_path(layer_key)

    if force_redownload:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass

    if layer_available_locally(layer_key):
        return dst

    raw = get_drive_raw(layer_key)
    if not raw:
        raise RuntimeError(
            f"Layer '{layer_key}' não encontrada localmente ({dst.name}) e não há FILE_ID/link configurado."
        )

    return download_drive_file(raw, dst, label=dst.name)


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


# =============================================================================
# READ / FILTER
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


def read_gdf_parquet_filtered(
    path: str,
    *,
    columns: Optional[List[str]] = None,
    filters: Optional[Any] = None,
) -> Optional["gpd.GeoDataFrame"]:
    """Leitura parcial para acelerar nível final (lotes)."""
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
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
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        st.error(str(e))
        return None

    try:
        meta = (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
    except Exception:
        meta = (str(p), 0.0, 0)

    cache: Dict[str, Any] = st.session_state.get("_layer_cache", {})
    cache_meta: Dict[str, Any] = st.session_state.get("_layer_cache_meta", {})

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

    if layer_key == "quadra":
        # quadra_id padronizado
        if QUADRA_ID in g.columns:
            g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))

        # iso_id em string “limpa”
        if ISO_ID in g.columns:
            g[ISO_ID] = g[ISO_ID].map(_id_to_str)

        # gera UID se possível
        if ISO_ID in g.columns and QUADRA_ID in g.columns:
            g[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(g[ISO_ID], g[QUADRA_ID])]

        # ✅ validação (apenas debug) para detectar incompatibilidade de iso_id
        if st.session_state.get("debug_schema", False) and ISO_ID in g.columns:
            sample = g[ISO_ID].dropna().astype(str).head(10).tolist()
            st.write("[debug] quadra iso_id sample:", sample)

    cache[layer_key] = g
    cache_meta[layer_key] = meta
    st.session_state["_layer_cache"] = cache
    st.session_state["_layer_cache_meta"] = cache_meta

    debug_layer_schema(layer_key, g, path=p)
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
        return pd.read_csv(p, dtype={QUADRA_ID: "string", ISO_ID: "string", CLUSTER_COL: "string"})
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
    if QUADRA_ID in g_quad.columns and QUADRA_ID in df.columns:
        return g_quad.merge(df, on=QUADRA_ID, how="left", suffixes=("", "_csv"))
    return g_quad


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
# GEOJSON cache LRU
# =============================================================================
def _session_geojson_get(key: str) -> Optional[str]:
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})
    return cache.get(key)


def _session_geojson_set(key: str, value: str, max_items: int = 120) -> None:
    cache: Dict[str, str] = st.session_state.get("_geojson_cache", {})
    order: List[str] = st.session_state.get("_geojson_cache_order", [])

    if key in cache:
        cache[key] = value
        try:
            order.remove(key)
        except Exception:
            pass
        order.append(key)
    else:
        cache[key] = value
        order.append(key)

    while len(order) > max_items:
        old = order.pop(0)
        cache.pop(old, None)

    st.session_state["_geojson_cache"] = cache
    st.session_state["_geojson_cache_order"] = order


def _simplify_to_geojson(
    gdf: "gpd.GeoDataFrame",
    simplify_tol: float,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Converte GeoDataFrame em GeoJSON (string) com simplificação opcional.
    Versão defensiva: ignora colunas inexistentes para não retornar "" silenciosamente.
    """
    if gdf is None or gdf.empty:
        return ""

    keep_cols = keep_cols or []
    keep_cols = [c for c in keep_cols if c in gdf.columns]  # evita KeyError

    cols = keep_cols + ["geometry"]
    try:
        g = gdf[cols].copy()
    except Exception:
        return ""

    try:
        if simplify_tol and simplify_tol > 0:
            g["geometry"] = g.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        pass

    g = _drop_bad_geoms(g)

    try:
        return g.to_json()
    except Exception:
        return ""


# =============================================================================
# CLICK HITTEST
# =============================================================================
def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[str]:
    """Retorna o id_col do primeiro polígono que contém/intersecta o ponto do clique."""
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
# FOLIUM HELPERS
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


def add_parent_fill(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    *,
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
    *,
    tooltip_col: Optional[str] = None,
    selected_ids: Optional[Set[Any]] = None,
    extra_props: Optional[List[str]] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.10,
    selected_color: str = PB_NAVY,
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.26,
    tooltip_prefix: str = "ID: ",
    simplify_tol: float = 0.0006,
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

    extra_props = extra_props or []
    extra_props = [c for c in extra_props if c in gdf.columns and c not in (id_col, tooltip_col)]

    keep = [id_col] if tooltip_col == id_col else [id_col, tooltip_col]
    keep = keep + extra_props

    key = cache_key or f"base:{name}:{id_col}:{tooltip_col}:{','.join(extra_props)}:{simplify_tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[keep + ["geometry"]].copy()
        for c in keep:
            mini[c] = mini[c].map(_id_to_str)

        geojson_base = _simplify_to_geojson(mini, simplify_tol=simplify_tol, keep_cols=keep)
        _session_geojson_set(key, geojson_base)

    if not geojson_base:
        return

    tooltip_base = _mk_tooltip(tooltip_col, tooltip_prefix)
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
        highlight_function=lambda _f: {"weight": base_weight + 1.4, "fillOpacity": min(fill_opacity + 0.12, 0.40)},
        tooltip=tooltip_base,
    ).add_to(fg_base)
    fg_base.add_to(m)

    # camada selecionados (sem tooltip)
    if sel:
        sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
        if not sel_gdf.empty:
            sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
            geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=simplify_tol, keep_cols=[id_col])
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



def add_polygons_selectable_colored(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    id_col: str,
    fill_color_col: str,
    *,
    selected_ids: Optional[Set[Any]] = None,
    tooltip_col: Optional[str] = None,
    pane: str = "detail_shapes",
    base_color: str = "#111111",
    base_weight: float = 0.8,
    fill_opacity: float = 0.14,
    selected_color: str = PB_NAVY,
    selected_weight: float = 2.6,
    selected_fill_opacity: float = 0.28,
    tooltip_prefix: str = "ID: ",
    simplify_tol: float = 0.0006,
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

    keep = [id_col, fill_color_col]
    if tooltip_col not in keep:
        keep.append(tooltip_col)

    key = cache_key or f"baseC:{name}:{id_col}:{tooltip_col}:{fill_color_col}:{simplify_tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[keep + ["geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        mini[tooltip_col] = mini[tooltip_col].map(_id_to_str)
        mini[fill_color_col] = mini[fill_color_col].astype(str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol=simplify_tol, keep_cols=keep)
        _session_geojson_set(key, geojson_base)
    if not geojson_base:
        return

    tooltip_base = _mk_tooltip(tooltip_col, tooltip_prefix)

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

    fg_base = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=geojson_base,
        pane=pane,
        smooth_factor=SMOOTH_FACTOR,
        style_function=_style,
        highlight_function=lambda _f: {"weight": base_weight + 1.4, "fillOpacity": min(fill_opacity + 0.12, 1.0)},
        tooltip=tooltip_base,
    ).add_to(fg_base)
    fg_base.add_to(m)

    # camada selecionados (sem tooltip)
    selected_ids = selected_ids or set()
    sel = {v for v in (_id_to_str(x) for x in selected_ids) if v is not None}
    if sel:
        sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
        if not sel_gdf.empty:
            sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
            geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol=simplify_tol, keep_cols=[id_col])
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
                        "fillColor": "#ffffff",
                        "fillOpacity": selected_fill_opacity,
                    },
                ).add_to(fg_sel)
                fg_sel.add_to(m)


# =============================================================================
# CLICK consumption (pré-render)
# =============================================================================
def _pick_id_from_last_object(map_state: Dict[str, Any], id_col: str) -> Optional[str]:
    obj = (map_state or {}).get("last_object_clicked")
    if not isinstance(obj, dict):
        return None
    props = obj.get("properties") if isinstance(obj.get("properties"), dict) else obj
    if not isinstance(props, dict):
        return None
    return _id_to_str(props.get(id_col))


def parse_tooltip_id(tooltip: Any) -> Optional[str]:
    if not tooltip:
        return None
    if isinstance(tooltip, dict):
        tooltip = tooltip.get("text") or tooltip.get("tooltip") or str(tooltip)
    s = re.sub(r"<[^>]+>", " ", str(tooltip)).strip()
    m = re.search(r":\s*([^\s<]+)", s)
    if m:
        return _id_to_str(m.group(1))
    m2 = re.search(r"([A-Za-z0-9_-]+)\s*$", s)
    return _id_to_str(m2.group(1)) if m2 else None


def _click_signature(picked_id: str, click: Optional[Dict[str, Any]]) -> str:
    lat = None
    lng = None
    if isinstance(click, dict):
        lat = click.get("lat")
        lng = click.get("lng")
    try:
        if lat is not None and lng is not None:
            return f"{picked_id}|{float(lat):.7f}|{float(lng):.7f}"
    except Exception:
        pass
    return f"{picked_id}|"


def consume_map_event(level: str, map_state: Dict[str, Any], allow_click: bool = True) -> None:
    if not allow_click:
        return

    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip") or None
    click = (map_state or {}).get("last_clicked") if isinstance((map_state or {}).get("last_clicked"), dict) else None


    # ------------------------------------------------------------------
    # SUBPREF / DISTRITO / ISO / CENSO
    # ------------------------------------------------------------------
    if level in ("subpref", "distrito", "isocrona", "censo"):
        if level == "subpref":
            id_col = SUBPREF_ID
        elif level == "distrito":
            id_col = DIST_ID
        elif level == "isocrona":
            id_col = ISO_ID
        else:
            id_col = CENSO_ID

        picked = _pick_id_from_last_object(map_state, id_col) or parse_tooltip_id(tooltip_raw)

        # Fallback geométrico
        if not picked and isinstance(click, dict):
            if level == "subpref":
                g = read_layer("subpref")
                if g is not None:
                    picked = pick_feature_id(g, click, SUBPREF_ID)
            elif level == "distrito":
                sp = _id_to_str(st.session_state.get("selected_subpref_id"))
                if sp is not None:
                    g = read_layer("dist")
                    if g is not None:
                        picked = pick_feature_id(subset_by_parent(g, DIST_PARENT, sp), click, DIST_ID)
            elif level == "isocrona":
                d = _id_to_str(st.session_state.get("selected_distrito_id"))
                if d is not None:
                    g = read_layer("iso")
                    if g is not None:
                        picked = pick_feature_id(subset_by_parent(g, ISO_PARENT, d), click, ISO_ID)
            elif level == "censo":
                iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v}
                g = read_layer("censo")
                if g is not None and iso_ids:
                    if CENSO_PARENT in g.columns:
                        picked = pick_feature_id(subset_by_parent_multi(g, CENSO_PARENT, iso_ids), click, CENSO_ID)
                    elif ISO_ID in g.columns:
                        picked = pick_feature_id(subset_by_parent_multi(g, ISO_ID, iso_ids), click, CENSO_ID)

        if not picked:
            return

        sig = _click_signature(picked, click)
        if sig == st.session_state.get("last_click_sig", ""):
            return
        st.session_state["last_click_sig"] = sig

        if level == "subpref":
            st.session_state["selected_subpref_id"] = picked
            reset_to("distrito", clear_click_sig=False)
            st.session_state["selected_subpref_id"] = picked
            st.session_state["level"] = "distrito"
            return

        if level == "distrito":
            st.session_state["selected_distrito_id"] = picked
            reset_to("isocrona", clear_click_sig=False)
            st.session_state["level"] = "isocrona"
            return

        if level == "isocrona":
            _toggle_in_set("selected_iso_ids", picked)
            _final_reset()
            return

        if level == "censo":
            _toggle_in_set("selected_censo_ids", picked)
            _final_reset()
            return

    # ------------------------------------------------------------------
    # QUADRA
    # ------------------------------------------------------------------
    if level == "quadra":
        iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v is not None}
        if not iso_ids:
            return
    
        mode = st.session_state.get("iso_next_mode", "quadra")
        id_col_map = st.session_state.get("_quadra_id_col_map", QUADRA_UID)
    
        picked: Optional[str] = None
    
        # 1) tenta obter do objeto clicado (mais rápido)
        obj = (map_state or {}).get("last_object_clicked") or None
        if isinstance(obj, dict):
            props = obj.get("properties") if isinstance(obj.get("properties"), dict) else obj
            if isinstance(props, dict):
                if id_col_map == QUADRA_UID:
                    picked = _id_to_str(props.get(QUADRA_UID)) or make_quadra_uid(props.get(ISO_ID), props.get(QUADRA_ID))
                else:
                    picked = _id_to_str(props.get(id_col_map)) or _id_to_str(props.get(QUADRA_ID))
    
        # 2) hittest geométrico no subset correto (✅ faltava isso funcionar de ponta a ponta)
        g_show: Optional["gpd.GeoDataFrame"] = None
        if not picked and isinstance(click, dict):
            g_quad = read_layer("quadra")
            if g_quad is not None:
                filter_censo_ids = ensure_set_of_str(st.session_state.get("quadra_filter_uids", set()))
                g_show = get_quadras_subset_for_mode(
                    g_quad,
                    mode=mode,
                    iso_ids=iso_ids,
                    filter_censo_ids=filter_censo_ids,
                )
    
                if g_show is not None and not g_show.empty and id_col_map in g_show.columns:
                    picked = pick_feature_id(g_show, click, id_col_map)
    
        # 3) tooltip fallback (mantém sua lógica, mas só faz sentido se g_show existir)
        picked_tooltip = parse_tooltip_id(tooltip_raw)
        if not picked and picked_tooltip and g_show is not None and not g_show.empty:
            if id_col_map == QUADRA_UID and QUADRA_UID in g_show.columns and QUADRA_ID in g_show.columns:
                qid = _id_to_str(picked_tooltip)
                if qid is not None:
                    cand = g_show[g_show[QUADRA_ID] == qid]
                    if len(cand) == 1:
                        picked = _id_to_str(cand.iloc[0][QUADRA_UID])
            elif id_col_map == QUADRA_ID and QUADRA_ID in g_show.columns:
                picked = _id_to_str(picked_tooltip)
    
        if not picked:
            return
    
        sig = _click_signature(picked, click)
        if sig == st.session_state.get("last_click_sig", ""):
            return
        st.session_state["last_click_sig"] = sig
    
        _toggle_in_set("selected_quadra_ids", picked)
        _final_reset()
        return


# =============================================================================
# UI: variável e navegação
# =============================================================================
def _variables_for_level(level: str) -> List[str]:
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
    st.selectbox("Variável", options=_variables_for_level(lvl), key="variable", on_change=mark_ui_action)


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
    iso_ids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_iso_ids", set())) if v}
    if not iso_ids:
        return
    g_iso = read_layer("iso")
    if g_iso is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_iso, ISO_ID, iso_ids), bump=0, zmax=18)


def _fit_selected_censos() -> None:
    cids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_censo_ids", set())) if v}
    if not cids:
        return
    g_censo = read_layer("censo")
    if g_censo is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_censo, CENSO_ID, cids), bump=0, zmax=18)


def _fit_selected_quadras() -> None:
    qids = {v for v in (_id_to_str(x) for x in st.session_state.get("selected_quadra_ids", set())) if v}
    if not qids:
        return
    g_quad = read_layer("quadra")
    if g_quad is None:
        return
    id_col = st.session_state.get("_quadra_id_col_map", QUADRA_UID)
    if id_col not in g_quad.columns:
        id_col = QUADRA_ID if QUADRA_ID in g_quad.columns else id_col
    set_view_to_gdf(subset_by_id_multi(g_quad, id_col, qids), bump=1, zmax=19)


# =============================================================================
# Fluxo (botões do painel)
# =============================================================================
def _go_from_isos_next() -> None:
    mode = st.session_state.get("iso_next_mode", "quadra")
    if mode == "censo":
        st.session_state["level"] = "censo"
        st.session_state["selected_censo_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()
    else:
        st.session_state["level"] = "quadra"
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["quadra_filter_col"] = CENSO_ID
        st.session_state["quadra_filter_uids"] = set()
        _final_reset()


def _go_from_censo_to_quadras() -> None:
    sel_censo: Set[str] = ensure_set_of_str(st.session_state.get("selected_censo_ids", set()))

    # Mantém o modo de navegação como "censo" para que o nível quadra saiba que veio desse caminho.
    st.session_state["iso_next_mode"] = "censo"

    # Define o filtro (pode ser vazio agora)
    st.session_state["quadra_filter_col"] = CENSO_ID
    st.session_state["quadra_filter_uids"] = sel_censo

    # Limpa seleção de quadras ao entrar no nível
    st.session_state["selected_quadra_ids"] = set()

    # Vai para quadra sempre
    st.session_state["level"] = "quadra"
    _final_reset()

    # Feedback opcional
    if not sel_censo:
        st.info("Nenhum setor selecionado. Exibindo quadras pelo recorte das isócronas (fallback por iso_id).")

def _go_to_final() -> None:
    st.session_state["level"] = "final"
    _final_reset()


def control_panel() -> None:
    lvl = st.session_state.get("level", "subpref")
    prev = _prev_level(lvl)

    c1, c2 = st.columns(2)
    with c1:
        if prev is None:
            st.button("Subprefeituras", disabled=True, use_container_width=True)
        else:
            st.button(
                prev.capitalize(),
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

    st.divider()
    st.subheader("Debug", anchor=False)
    st.checkbox("Mostrar schema/colunas carregadas (debug)", key="debug_schema")

    st.divider()
    st.subheader("Ações", anchor=False)

    if lvl == "isocrona":
        ok = len(st.session_state.get("selected_iso_ids", set()) or set()) > 0
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not ok,
            on_click=lambda: (mark_ui_action(), _fit_selected_isos()),
        )
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
        okc = len(st.session_state.get("selected_censo_ids", set()) or set()) > 0
    
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not okc,
            on_click=lambda: (mark_ui_action(), _fit_selected_censos()),
        )
    
        # ✅ sempre habilitado, seleção de setores é opcional
        st.button(
            "Visualizar Quadras",
            use_container_width=True,
            disabled=False,
            on_click=lambda: (mark_ui_action(), _go_from_censo_to_quadras()),
        )


    if lvl == "quadra":
        okq = len(st.session_state.get("selected_quadra_ids", set()) or set()) > 0
        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _fit_selected_quadras()),
        )
        st.button(
            "Ir para Lotes",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _go_to_final()),
        )

    if lvl == "final":
        okq = len(st.session_state.get("selected_quadra_ids", set()) or set()) > 0
        st.caption("Lotes são carregados somente para as quadras selecionadas.")
        st.button(
            "Recarregar Lotes selecionados",
            use_container_width=True,
            disabled=not okq,
            on_click=lambda: (mark_ui_action(), _final_reset()),
        )


# =============================================================================
# FINAL: lotes filtrados
# =============================================================================
def _quadra_ids_from_selected(uids_or_ids: Set[str]) -> Set[str]:
    """
    Converte a seleção de quadras (que pode vir como quadra_uid 'iso__quadra')
    para um conjunto de quadra_id canônicos (string), com normalização (zfill=6 quando numérico).
    """
    out: Set[str] = set()
    for x in uids_or_ids or set():
        s = _id_to_str(x)
        if not s:
            continue

        # Seleção pode vir como UID: iso__quadra
        if "__" in s:
            _iso, q = split_quadra_uid(s)
            qid = _id_to_str(q)
        else:
            qid = s

        if not qid:
            continue

        # Normaliza padding quando for numérico (compatível com quadras)
        nq = normalize_quadra_id(qid, 6) if isinstance(qid, str) else _id_to_str(qid)
        if nq:
            out.add(nq)

    return out


def _make_parquet_filters_for_quadras(quadra_ids: Set[str]) -> Any:
    """
    Monta filtros no formato aceito por pyarrow/geopandas.read_parquet(filters=...).
    Nem todo ambiente suporta; por isso temos fallback em _read_lotes_for_selected_quadras.
    """
    vals = sorted({v for v in (quadra_ids or set()) if v})
    if not vals:
        return None
    return [(QUADRA_ID, "in", vals)]


def _detect_geometry_unit(g: "gpd.GeoDataFrame") -> Dict[str, Any]:
    """
    Identifica informações básicas da geometria (para debug):
    - coluna geometry
    - quantidade de geometrias não nulas
    - se parece ser uma unidade única (1 registro)
    - tipo de geometria predominante
    """
    out: Dict[str, Any] = {"geometry_col": None, "n_geoms": 0, "single_geometry": False, "geom_type": None}
    if g is None or g.empty:
        return out

    try:
        out["geometry_col"] = g.geometry.name
    except Exception:
        out["geometry_col"] = "geometry"

    try:
        n = int(g.geometry.notna().sum())
        out["n_geoms"] = n
        out["single_geometry"] = (len(g) == 1) or (n == 1)
    except Exception:
        pass

    try:
        gt = g.geometry.geom_type.dropna().iloc[0]
        out["geom_type"] = str(gt)
    except Exception:
        pass

    return out


def _canonicalize_lote_schema(g: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Normaliza o schema do layer de lotes para:
    - padronização de colunas
    - remoção de geometrias ruins
    - normalização dos IDs relevantes
    - CRS 4326
    """
    if g is None or g.empty:
        return g

    g = standardize_columns(g)
    g = _drop_bad_geoms(g)

    # Normaliza IDs como string
    g = normalize_id_cols(g, [QUADRA_ID, LOTE_ID])

    # Normaliza quadra_id com zfill(6) quando aplicável (evita mismatch)
    if QUADRA_ID in g.columns:
        g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))

    # CRS -> 4326 (mantém compatível com o resto do app)
    try:
        if g.crs is None:
            g = g.set_crs(4326, allow_override=True)
        else:
            g = g.to_crs(4326)
    except Exception:
        pass

    return g


def _read_lotes_for_selected_quadras(selected_quad_ids: Set[str]) -> Optional["gpd.GeoDataFrame"]:
    """
    Carrega lotes apenas para as quadras selecionadas.

    Estratégia:
    - tenta leitura parcial (columns + filters)
    - fallback: columns (sem filters)
    - se falhar: força re-download e tenta novamente
    """
    # 0) garante arquivo local
    try:
        p = ensure_local_layer("lote")
    except Exception as e:
        st.error(str(e))
        return None

    # 1) converte seleção -> quadra_id canônico
    quadra_ids = _quadra_ids_from_selected(selected_quad_ids)  # <- corrigido aqui
    if not quadra_ids:
        return None

    # 2) cache
    sig = "|".join(sorted(list(quadra_ids)))
    if st.session_state.get("final_loaded", False) and st.session_state.get("final_load_sig") == sig:
        return st.session_state.get("_final_lotes_gdf")

    cols = [QUADRA_ID, LOTE_ID, "geometry"]
    filters = _make_parquet_filters_for_quadras(quadra_ids)

    def _try_partial_read(path: Path) -> Optional["gpd.GeoDataFrame"]:
        g = read_gdf_parquet_filtered(str(path), columns=cols, filters=filters)
        if g is not None and not g.empty:
            return g

        g = read_gdf_parquet_filtered(str(path), columns=cols, filters=None)
        if g is not None and not g.empty:
            return g

        return None

    g = _try_partial_read(p)

    # força re-download em caso de falha (mitiga cache com download incompleto)
    if g is None:
        try:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            p = ensure_local_layer("lote")
            g = _try_partial_read(p)
        except Exception as e:
            st.error(f"Falha ao rebaixar Lotes.parquet: {e}")
            g = None

    if g is None or g.empty:
        st.warning("Não foi possível ler Lotes.parquet (parquet inválido, corrompido ou incompatível).")
        st.session_state["_final_lotes_gdf"] = None
        st.session_state["final_loaded"] = False
        st.session_state["final_load_sig"] = sig
        return None

    g = _canonicalize_lote_schema(g)

    if QUADRA_ID not in g.columns:
        st.error(f"Lotes.parquet não contém a coluna obrigatória '{QUADRA_ID}'. Colunas: {list(g.columns)}")
        st.session_state["_final_lotes_gdf"] = None
        st.session_state["final_loaded"] = False
        st.session_state["final_load_sig"] = sig
        return None

    # filtro final em memória (garantia)
    g = g[g[QUADRA_ID].isin(list(quadra_ids))].copy()

    if st.session_state.get("debug_schema", False):
        st.write("[debug] lotes geometry unit:", _detect_geometry_unit(g))
        st.write("[debug] lotes filtrados shape:", getattr(g, "shape", None))
        st.write("[debug] lotes cols:", list(g.columns))
        st.write("[debug] lotes path:", str(p))

    st.session_state["_final_lotes_gdf"] = g
    st.session_state["final_loaded"] = True
    st.session_state["final_load_sig"] = sig
    return g

# =============================================================================
# MAP RENDER (todos os níveis)
# =============================================================================
def render_map_panel() -> None:
    level = st.session_state.get("level", "subpref")
    ensure_variable_for_level(level)

    title = ""
    m = None

    # -----------------------------
    # SUBPREF (FIX UnboundLocalError)
    # -----------------------------
    if level == "subpref":
        title = "Subprefeituras"
        g_sub = read_layer("subpref")
        if g_sub is None or g_sub.empty:
            st.stop()
        if SUBPREF_ID not in g_sub.columns:
            st.error(f"Subprefeitura.parquet sem '{SUBPREF_ID}'.")
            st.stop()

        if st.session_state.get("last_level") != "subpref":
            set_view_to_gdf(g_sub, bump=0)
            st.session_state["last_level"] = "subpref"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_polygons_selectable(
            m,
            g_sub,
            "Subprefeituras",
            SUBPREF_ID,
            tooltip_col=SUBPREF_ID,
            selected_ids=set(),
            fill_opacity=0.06,
            selected_fill_opacity=0.0,
            tooltip_prefix="Subpref: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"subpref:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )

    elif level == "distrito":
        sp = _id_to_str(st.session_state.get("selected_subpref_id"))
        if sp is None:
            reset_to("subpref")
            return

        title = f"Distritos (Subpref {sp})"
        g_dist = read_layer("dist")
        g_sub = read_layer("subpref")
        if g_dist is None or g_sub is None:
            st.stop()

        g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
        g_show = subset_by_parent(g_dist, DIST_PARENT, sp)

        if st.session_state.get("last_level") != "distrito":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "distrito"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_parent_fill(
            m,
            g_parent,
            "Subpref selecionada (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"parent:subpref:{sp}:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )
        add_polygons_selectable(
            m,
            g_show,
            "Distritos",
            DIST_ID,
            tooltip_col=DIST_ID,
            selected_ids=set(),
            fill_opacity=0.06,
            tooltip_prefix="Distrito: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"dist:sp:{sp}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

    elif level == "isocrona":
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if d is None:
            reset_to("distrito")
            return

        sel_n = len(st.session_state.get("selected_iso_ids", set()) or set())
        title = f"Isócronas (Distrito {d}) — selecionadas: {sel_n}"

        g_iso = read_layer("iso")
        g_dist = read_layer("dist")
        if g_iso is None or g_dist is None:
            st.stop()

        g_parent = subset_by_id(g_dist, DIST_ID, d)
        g_show = subset_by_parent(g_iso, ISO_PARENT, d)

        if st.session_state.get("last_level") != "isocrona":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=0)
            st.session_state["last_level"] = "isocrona"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_parent_fill(
            m,
            g_parent,
            "Distrito selecionado (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"parent:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

        g_show_viz = g_show.copy()

        # Gera cor por classe de forma robusta (sem depender de apply->DataFrame)
        if ISO_CLASS_COL in g_show_viz.columns:
            pairs = g_show_viz[ISO_CLASS_COL].map(iso_label_color)  # retorna (label, color)

            # pairs pode ter NaN/None inesperado; protege
            safe_pairs = []
            for p in pairs.tolist():
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    safe_pairs.append((str(p[0]), str(p[1])))
                else:
                    safe_pairs.append(("Sem classe", ISO_DEFAULT_COLOR))

            labels, colors = zip(*safe_pairs) if safe_pairs else ([], [])
            g_show_viz["__iso_label"] = list(labels)
            g_show_viz["__iso_color"] = list(colors)
        else:
            g_show_viz["__iso_label"] = "Sem classe"
            g_show_viz["__iso_color"] = ISO_DEFAULT_COLOR

        if st.session_state.get("variable") == "Isócronas (classes)":
            add_polygons_selectable_colored(
                m,
                g_show_viz,
                "Isócronas",
                ISO_ID,
                fill_color_col="__iso_color",
                selected_ids=st.session_state.get("selected_iso_ids", set()),
                tooltip_col=ISO_ID,
                fill_opacity=1.0,
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
                selected_ids=st.session_state.get("selected_iso_ids", set()),
                tooltip_col=ISO_ID,
                fill_opacity=0.14,
                selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"iso:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
            )

    elif level == "censo":
        raw_iso_ids = st.session_state.get("selected_iso_ids")  # sem default
        iso_ids = ensure_set_of_str(raw_iso_ids)
        if not iso_ids:
            reset_to("isocrona")
            return

        sel_nc = len(st.session_state.get("selected_censo_ids", set()) or set())
        title = f"Setor censitário — selecionados: {sel_nc}"

        g_censo = read_layer("censo")
        g_iso = read_layer("iso")
        if g_censo is None or g_iso is None:
            st.stop()

        # setor->iso pode estar em CENSO_PARENT ou ISO_ID
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
            m,
            g_parent,
            "Isócronas selecionadas (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"parent:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )
        add_polygons_selectable(
            m,
            g_show,
            "Setor censitário",
            CENSO_ID,
            tooltip_col=CENSO_ID,
            selected_ids=st.session_state.get("selected_censo_ids", set()),
            fill_opacity=0.10,
            selected_fill_opacity=0.0,
            tooltip_prefix="Setor: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["censo"],
            cache_key=f"censo:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['censo']}",
        )

    # -----------------------------
    # QUADRA (FIX: filtro por censo_id OU iso_id)
    # -----------------------------
    elif level == "quadra":
        raw_iso_ids = st.session_state.get("selected_iso_ids")  # sem default
        iso_ids = ensure_set_of_str(raw_iso_ids)
        if not iso_ids:
            reset_to("isocrona")
            return

        mode = st.session_state.get("iso_next_mode", "quadra")  # "quadra" | "censo"
        filter_ids: Set[str] = ensure_set_of_str(st.session_state.get("quadra_filter_uids", set()))

        sel_nq = len(st.session_state.get("selected_quadra_ids", set()) or set())
        title = (
            f"Quadras — selecionadas: {sel_nq}"
            if mode != "censo"
            else f"Quadras (filtradas por setor) — selecionadas: {sel_nq}"
        )

        g_quad = read_layer("quadra")
        g_iso = read_layer("iso")
        if g_quad is None or g_iso is None:
            st.stop()

        g_parent = subset_by_id_multi(g_iso, ISO_ID, iso_ids)

        # FIX: coluna de clique/seleção no mapa
        if mode == "censo":
            # Em modo censo, a seleção por quadra pode ser só QUADRA_ID mesmo.
            id_col_map = QUADRA_ID if QUADRA_ID in g_quad.columns else (QUADRA_UID if QUADRA_UID in g_quad.columns else QUADRA_ID)
        else:
            id_col_map = QUADRA_UID if QUADRA_UID in g_quad.columns else QUADRA_ID
        st.session_state["_quadra_id_col_map"] = id_col_map

        # FIX: escolher parent col corretamente
        if mode == "censo":
            # preferir CENSO_ID; se não existir, cair para ISO_ID
            parent_col = choose_quadra_parent_col(g_quad, preferred=CENSO_ID, fallback=ISO_ID)

            # debug opcional
            if st.session_state.get("debug_schema", False):
                st.write("[debug] quadra mode=censo parent_col escolhido:", parent_col)
                st.write("[debug] quadra filter_ids (setores):", len(filter_ids))
                st.write("[debug] quadra iso_ids:", len(iso_ids))
        
            if parent_col is None:
                st.error(f"Quadras.parquet precisa ter '{CENSO_ID}' ou '{ISO_ID}'. Colunas: {list(g_quad.columns)}")
                st.stop()

            if parent_col == CENSO_ID:
                if not filter_ids:
                    # ✅ seleção opcional: sem setores, exibir por iso_id
                    g_show = subset_by_parent_multi(g_quad, ISO_ID, iso_ids)
                    st.info("Nenhum setor selecionado. Exibindo quadras pelo recorte das isócronas (fallback por iso_id).")
                else:
                    g_show = subset_by_id_multi(g_quad, CENSO_ID, filter_ids)

            else:
                # fallback: se não há censo_id nas quadras, filtramos por iso_id (mantém o app operável)
                g_show = subset_by_parent_multi(g_quad, ISO_ID, iso_ids)
        else:
            if ISO_ID not in g_quad.columns:
                st.error(f"Quadras.parquet não tem '{ISO_ID}'. Colunas: {list(g_quad.columns)}")
                st.stop()
            g_show = subset_by_parent_multi(g_quad, ISO_ID, iso_ids)

        if st.session_state.get("last_level") != "quadra":
            set_view_to_gdf(g_show if not g_show.empty else g_parent, bump=1)
            st.session_state["last_level"] = "quadra"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        add_parent_fill(
            m,
            g_parent,
            "Isócronas selecionadas (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
            cache_key=f"parent:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
        )

        # visualização cluster
        g_show_viz = attach_quadras_csv(g_show)
        if CLUSTER_COL in g_show_viz.columns:
            g_show_viz["__cluster_code"] = g_show_viz[CLUSTER_COL].apply(_coerce_int)
            g_show_viz["__cluster_color"] = g_show_viz["__cluster_code"].apply(cluster_color)

        # FIX: incluir CENSO_ID no GeoJSON quando aplicável (para clique/props)
        extra_props = [CENSO_ID] if (mode == "censo" and CENSO_ID in g_show.columns) else []

        if st.session_state.get("variable") == "Cluster" and "__cluster_color" in g_show_viz.columns:
            add_polygons_selectable_colored(
                m,
                g_show_viz,
                "Quadras",
                id_col_map,
                fill_color_col="__cluster_color",
                selected_ids=st.session_state.get("selected_quadra_ids", set()),
                tooltip_col=QUADRA_ID if QUADRA_ID in g_show_viz.columns else id_col_map,
                fill_opacity=1.0,
                selected_fill_opacity=0.0,
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                cache_key=f"quad:{mode}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
                default_fill=CLUSTER_NULL_COLOR,
            )
        else:
            add_polygons_selectable(
                m,
                g_show,
                "Quadras",
                id_col_map,
                tooltip_col=QUADRA_ID if QUADRA_ID in g_show.columns else id_col_map,
                selected_ids=st.session_state.get("selected_quadra_ids", set()),
                extra_props=extra_props,
                fill_opacity=0.12,
                selected_fill_opacity=0.0,
                tooltip_prefix="Quadra: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                cache_key=f"quadB:{mode}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
            )

    else:
        # FINAL
        title = "Lotes (filtrado por quadras selecionadas)"
        sel_ids: Set[str] = st.session_state.get("selected_quadra_ids", set()) or set()

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        if not sel_ids:
            st.warning("Selecione ao menos uma quadra para visualizar lotes.")
        else:
            g_lote = _read_lotes_for_selected_quadras(sel_ids)
            if g_lote is not None and not g_lote.empty:
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
                    fill_opacity=0.08,
                    tooltip_prefix="Lote: " if LOTE_ID in g_lote.columns else "Quadra: ",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["lote"],
                    cache_key=f"lote:{st.session_state.get('final_load_sig','')}:{SIMPLIFY_TOL_BY_LEVEL['lote']}",
                )
            else:
                st.warning("Nenhum lote encontrado para as quadras selecionadas (ou falha de leitura).")

    st.markdown(f"### {title}")

    if st_folium is None:
        st.error("Falha ao importar `streamlit_folium` (dependência ausente ou quebrada).")
        return

    _ = st_folium(
        m,
        height=780,
        use_container_width=True,
        key=MAP_KEY,
        returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_tooltip"],
    )
    st.session_state["_map_level_rendered"] = level

# =============================================================================
# APP (main)
# =============================================================================
def main() -> None:
    init_state()
    inject_css()
    render_header()

    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium`.")
        return

    # ------------------------------------------------------------
    # Pré-consumo do clique do mapa:
    # - Consome o clique do RUN anterior antes do render do mapa.
    # - Só consome se o mapa anterior foi renderizado no mesmo nível atual.
    # ------------------------------------------------------------
    ui_sig = int(st.session_state.get("_ui_action_sig", 0))
    ui_seen = int(st.session_state.get("_ui_action_sig_seen", 0))
    ui_action = ui_sig != ui_seen
    st.session_state["_ui_action_sig_seen"] = ui_sig

    # Se houve ação de UI (botões, selectbox, radio), limpamos dedupe de clique
    if ui_action:
        st.session_state["last_click_sig"] = ""

    cur_level = st.session_state.get("level", "subpref")
    rendered_level = st.session_state.get("_map_level_rendered")
    map_state_prev = st.session_state.get(MAP_KEY, {}) or {}

    allow_click = (not ui_action) and (rendered_level == cur_level)

    if allow_click and isinstance(map_state_prev, dict) and map_state_prev:
        consume_map_event(cur_level, map_state_prev, allow_click=True)

    sanitize_level_state()

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

























