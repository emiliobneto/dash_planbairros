# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import base64
import re
from typing import Optional, Dict, Any, Tuple, Set, List

import streamlit as st

# Map stack
try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    st_folium = None  # type: ignore


# =============================================================================
# CONFIG / IDENTIDADE (UI)
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

SIMPLIFY_TOL = 0.0006  # simplificar só LINHAS/outlines
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

REPO_ROOT = Path.cwd()
ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

# =============================================================================
# GOOGLE DRIVE (PASTA)
# - A pasta precisa estar compartilhada como "Qualquer pessoa com o link" (Viewer).
# - Usamos gdown para baixar a pasta inteira uma vez (cache_resource).
# =============================================================================
GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1YdJuACMnudDgb6__wAl_TYibvnxlJz0M?usp=drive_link"
)

DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DRIVE_SYNC_DIR = DATA_CACHE_DIR / "drive_folder_sync"


def _extract_drive_folder_id(url: str) -> Optional[str]:
    # Aceita "folders/<ID>" ou "?id=<ID>"
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def ensure_drive_folder_synced(folder_url: str, out_dir: Path) -> Path:
    """
    Baixa a pasta do Google Drive para o disco local.
    Usa cache_resource para não rebaixar a cada rerun (enquanto o processo estiver vivo).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # gdown é o caminho mais robusto para Drive (lida com confirmações em arquivos grandes).
    try:
        import gdown  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Dependência ausente: 'gdown'. Adicione em requirements.txt: gdown\n"
            f"Erro: {e}"
        )

    # Se já tem arquivos, não baixa de novo (pode ajustar a regra se quiser “forçar sync”)
    has_any = any(out_dir.rglob("*"))
    if has_any:
        return out_dir

    folder_id = _extract_drive_folder_id(folder_url)
    if not folder_id:
        raise RuntimeError("Não consegui extrair o folder_id do link do Google Drive.")

    # URL normalizada
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    # Download folder
    # use_cookies=False tende a funcionar melhor em ambientes server.
    gdown.download_folder(
        url=url,
        output=str(out_dir),
        quiet=True,
        use_cookies=False,
    )
    return out_dir


def _slug(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9]+", "", str(s).strip().lower())
    return s2


def find_file_in_folder(folder: Path, candidates: List[str], exts: Tuple[str, ...]) -> Optional[Path]:
    """
    Procura por arquivo na pasta sincronizada, aceitando nomes aproximados.
    Ex.: candidates=["Subprefeitura","subprefeituras"] encontra "Subprefeitura.parquet".
    """
    if not folder.exists():
        return None
    wanted = {_slug(c) for c in candidates}
    for fp in folder.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts and _slug(fp.stem) in wanted:
            return fp
    return None


# =============================================================================
# IDs EXPLÍCITOS (seu encadeamento)
# =============================================================================
SUBPREF_ID = "subpref_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
QUADRA_ID = "quadra_id"
LOTE_ID = "lote_id"
CENSO_ID = "censo_id"

DIST_PARENT = "subpref_id"
ISO_PARENT = "distrito_id"
QUADRA_PARENT = "iso_id"
LOTE_PARENT = "quadra_id"
CENSO_PARENT = "iso_id"


# =============================================================================
# CSS (estética)
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
            padding:9px;
        }}
        .pb-badge {{
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:999px;
            border:1px solid rgba(20,64,125,.12);
            background:#fff; color:#111; font-size:12px; font-weight:700;
        }}
        .pb-note {{ font-size:12px; color:rgba(17,17,17,.75); line-height:1.35; }}
        .pb-divider {{ height:1px; background:rgba(20,64,125,.10); margin:10px 0; }}
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
# ESTADO (drill-down)
# =============================================================================
LEVELS = ["subpref", "distrito", "isocrona", "quadra", "final"]  # final = (lote | censo)


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)
    st.session_state.setdefault("selected_iso_ids", set())     # multi
    st.session_state.setdefault("selected_quadra_ids", set())  # multi
    st.session_state.setdefault("final_mode", "lote")          # "lote" | "censo"
    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)


def reset_to(level: str) -> None:
    st.session_state["level"] = level
    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        st.session_state["selected_quadra_ids"] = set()
        st.session_state["final_mode"] = "lote"
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
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
    elif level == "final":
        st.session_state["final_mode"] = "lote"


def _back_one_level() -> None:
    cur = st.session_state["level"]
    idx = LEVELS.index(cur)
    if idx == 0:
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
# IO / SANITIZAÇÃO
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def read_layer_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    if not path.exists():
        return None
    gdf = gpd.read_parquet(path)
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


def _simplify_lines(gdf: "gpd.GeoDataFrame", tol: float) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    except Exception:
        pass
    return _drop_bad_geoms(gdf)


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


def pick_feature_id(gdf: "gpd.GeoDataFrame", click_latlon: Dict[str, float], id_col: str) -> Optional[Any]:
    if gdf is None or gdf.empty or not click_latlon:
        return None
    if id_col not in gdf.columns:
        return None

    lat = click_latlon.get("lat")
    lng = click_latlon.get("lng")
    if lat is None or lng is None:
        return None

    try:
        from shapely.geometry import Point  # type: ignore
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
        return hit.iloc[0][id_col]
    except Exception:
        return None


def subset_by_parent(child: "gpd.GeoDataFrame", parent_col: str, parent_val: Any) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or parent_val is None:
        return child.iloc[0:0].copy()
    try:
        return child[child[parent_col] == parent_val]
    except Exception:
        return child.iloc[0:0].copy()


def subset_by_parent_multi(child: "gpd.GeoDataFrame", parent_col: str, parent_vals: Set[Any]) -> "gpd.GeoDataFrame":
    if child is None or child.empty:
        return child
    child = _drop_bad_geoms(child)
    if parent_col not in child.columns or not parent_vals:
        return child.iloc[0:0].copy()
    try:
        return child[child[parent_col].isin(list(parent_vals))]
    except Exception:
        return child.iloc[0:0].copy()


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
        folium.map.CustomPane("detail_shapes", z_index=640).add_to(m)
        folium.map.CustomPane("top_lines", z_index=660).add_to(m)
    except Exception:
        pass

    return m


def add_outline(m, gdf: "gpd.GeoDataFrame", name: str, color="#111111", weight=1.2, show=True) -> None:
    if folium is None or gdf is None or gdf.empty:
        return

    line = gdf[["geometry"]].copy()
    try:
        line["geometry"] = line.geometry.boundary
    except Exception:
        return
    line = _simplify_lines(line, SIMPLIFY_TOL)

    fg = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        data=line.to_json(),
        pane="top_lines",
        style_function=lambda _f: {"fillOpacity": 0, "color": color, "weight": weight},
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
    fill_opacity: float = 0.08,
    selected_color: str = "#14407D",
    selected_weight: float = 2.4,
    selected_fill_opacity: float = 0.18,
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return
    if id_col not in gdf.columns:
        id_col = ""

    selected_ids = selected_ids or set()
    cols = ["geometry"] + ([id_col] if id_col else [])
    mini = gdf[cols].copy()

    def style_fn(feat):
        props = feat.get("properties", {})
        fid = props.get(id_col) if id_col else None
        is_sel = (fid in selected_ids) if fid is not None else False
        return {
            "color": (selected_color if is_sel else base_color),
            "weight": (selected_weight if is_sel else base_weight),
            "fillColor": fill_color,
            "fillOpacity": (selected_fill_opacity if is_sel else fill_opacity),
        }

    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        data=mini.to_json(),
        pane=pane,
        style_function=style_fn,
        highlight_function=lambda _f: {
            "weight": base_weight + 1.2,
            "fillOpacity": min(fill_opacity + 0.10, 0.35),
        },
    ).add_to(fg)
    fg.add_to(m)


# =============================================================================
# UI LATERAL
# =============================================================================
def left_panel() -> None:
    st.markdown("<span class='pb-badge'>🧭 Fluxo</span>", unsafe_allow_html=True)
    st.caption("Subprefeitura → Distrito → Isócronas → Quadras → (Lotes | Setores)")

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.button("Voltar 1 nível", use_container_width=True, on_click=_back_one_level)
    with c2:
        st.button("Reset", type="secondary", use_container_width=True, on_click=reset_to, args=("subpref",))

    if st.session_state["level"] == "final":
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Nível final</span>", unsafe_allow_html=True)
        mode = st.radio(
            "Visualizar",
            ["lote", "censo"],
            index=0 if st.session_state["final_mode"] == "lote" else 1,
            horizontal=True,
            help="No nível final, alterna entre Lotes (por quadra_id) e Setores (por iso_id).",
        )
        st.session_state["final_mode"] = mode

    st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<span class='pb-badge'>📌 Seleção</span>", unsafe_allow_html=True)
    st.write(
        {
            "level": st.session_state["level"],
            "final_mode": st.session_state["final_mode"],
            "subpref_id": st.session_state["selected_subpref_id"],
            "distrito_id": st.session_state["selected_distrito_id"],
            "iso_ids": sorted(list(st.session_state["selected_iso_ids"]))[:30],
            "quadra_ids": sorted(list(st.session_state["selected_quadra_ids"]))[:30],
        }
    )
    st.markdown(
        "<div class='pb-note'>Clique no mapa para avançar. Isócronas/Quadras: clique alterna seleção (multi).</div>",
        unsafe_allow_html=True,
    )


def kpis_row() -> None:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>📍 Nível</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['level']}**")
        st.caption("Camada em exibição")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🏛️ Subpref</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_subpref_id'] or '—'}**")
        st.caption("Seleção")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🗺️ Distrito</span>", unsafe_allow_html=True)
        st.markdown(f"**{st.session_state['selected_distrito_id'] or '—'}**")
        st.caption("Seleção")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🧩 Multi</span>", unsafe_allow_html=True)
        st.markdown(
            f"**Iso: {len(st.session_state['selected_iso_ids'])} | Quad: {len(st.session_state['selected_quadra_ids'])}**"
        )
        st.caption("Seleções ativas")
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

    # 1) Sincroniza pasta do Drive (uma vez)
    with st.spinner("Sincronizando dados do Google Drive…"):
        try:
            drive_dir = ensure_drive_folder_synced(GOOGLE_DRIVE_FOLDER_URL, DRIVE_SYNC_DIR)
        except Exception as e:
            st.error("Falha ao acessar/baixar a pasta do Google Drive.")
            st.code(str(e))
            st.info("Garanta: pasta pública ('Qualquer pessoa com o link') e dependency 'gdown' no requirements.txt.")
            return

    # 2) Descobre caminhos dos arquivos dentro da pasta baixada
    p_subpref = find_file_in_folder(drive_dir, ["Subprefeitura", "Subprefeituras"], (".parquet",))
    p_dist = find_file_in_folder(drive_dir, ["Distritos", "Distrito"], (".parquet",))
    p_iso = find_file_in_folder(drive_dir, ["Isocronas", "Isócronas", "Isocronas2023"], (".parquet",))
    p_quadra = find_file_in_folder(drive_dir, ["Quadras", "Quadra"], (".parquet",))
    p_lote = find_file_in_folder(drive_dir, ["Lotes", "Lote"], (".parquet",))
    p_censo = find_file_in_folder(drive_dir, ["SetoresCensitarios", "SetoresCensitarios2023", "Censo", "Setores"], (".parquet",))

    left, right = st.columns([1, 4], gap="large")

    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        left_panel()
        st.markdown("<div class='pb-divider'></div>", unsafe_allow_html=True)
        st.markdown("<span class='pb-badge'>🗂️ Dados</span>", unsafe_allow_html=True)
        st.write(
            {
                "drive_dir": str(drive_dir),
                "subpref": str(p_subpref) if p_subpref else None,
                "distritos": str(p_dist) if p_dist else None,
                "isocronas": str(p_iso) if p_iso else None,
                "quadras": str(p_quadra) if p_quadra else None,
                "lotes": str(p_lote) if p_lote else None,
                "setores": str(p_censo) if p_censo else None,
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        kpis_row()
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)

        # 3) Carrega camadas sob demanda (cache_data)
        g_subpref = read_layer_parquet(p_subpref) if p_subpref else None
        g_dist = read_layer_parquet(p_dist) if p_dist else None
        g_iso = read_layer_parquet(p_iso) if p_iso else None
        g_quadra = read_layer_parquet(p_quadra) if p_quadra else None
        g_lote = read_layer_parquet(p_lote) if p_lote else None
        g_censo = read_layer_parquet(p_censo) if p_censo else None

        if g_subpref is None or g_subpref.empty:
            st.warning("Subprefeituras não encontradas/vazias na pasta do Drive.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # saneia geoms
        g_subpref = _drop_bad_geoms(g_subpref)
        if g_dist is not None:
            g_dist = _drop_bad_geoms(g_dist)
        if g_iso is not None:
            g_iso = _drop_bad_geoms(g_iso)
        if g_quadra is not None:
            g_quadra = _drop_bad_geoms(g_quadra)
        if g_lote is not None:
            g_lote = _drop_bad_geoms(g_lote)
        if g_censo is not None:
            g_censo = _drop_bad_geoms(g_censo)

        # mapa
        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])
        if m is None:
            st.error("Falha ao inicializar o mapa.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        level = st.session_state["level"]

        # ---------------------------------------------------------
        # NÍVEL 0: SUBPREF (abre com LINHAS)
        # ---------------------------------------------------------
        if level == "subpref":
            st.markdown("### Subprefeituras")
            st.caption("Clique em uma subprefeitura para abrir os distritos (recorte).")
            add_outline(m, g_subpref, "Subprefeituras (linha)", color="#111111", weight=1.25, show=True)
            st.session_state["view_center"] = (-23.55, -46.63)
            st.session_state["view_zoom"] = 11

        # ---------------------------------------------------------
        # NÍVEL 1: DISTRITOS (da subpref selecionada)
        # ---------------------------------------------------------
        elif level == "distrito":
            if g_dist is None or g_dist.empty:
                st.warning("Distritos não encontrados/vazios na pasta do Drive.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            sp = st.session_state["selected_subpref_id"]
            if sp is None:
                reset_to("subpref")
                st.rerun()

            g_show = subset_by_parent(g_dist, DIST_PARENT, sp)

            st.markdown(f"### Distritos (Subpref {sp})")
            st.caption("Clique em um distrito para abrir isócronas.")

            add_outline(m, g_show, "Distritos (linha)", color="#111111", weight=1.0, show=True)
            add_polygons_selectable(
                m,
                g_show,
                name="Distritos (clique)",
                id_col=DIST_ID,
                selected_ids=set(),
                base_color="#111111",
                base_weight=0.6,
                fill_color="#ffffff",
                fill_opacity=0.06,
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # ---------------------------------------------------------
        # NÍVEL 2: ISÓCRONAS (do distrito selecionado)
        # ---------------------------------------------------------
        elif level == "isocrona":
            if g_iso is None or g_iso.empty:
                st.warning("Isócronas não encontradas/vazias na pasta do Drive.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            d = st.session_state["selected_distrito_id"]
            if d is None:
                reset_to("distrito")
                st.rerun()

            g_show = subset_by_parent(g_iso, ISO_PARENT, d)

            st.markdown(f"### Isócronas (Distrito {d})")
            st.caption("Clique para selecionar múltiplas isócronas. Ao selecionar, abre Quadras.")

            add_polygons_selectable(
                m,
                g_show,
                name="Isócronas (multi)",
                id_col=ISO_ID,
                selected_ids=st.session_state["selected_iso_ids"],
                base_color="#111111",
                base_weight=0.6,
                fill_color="#ffffff",
                fill_opacity=0.10,
                selected_color=PB_NAVY,
                selected_weight=2.4,
                selected_fill_opacity=0.20,
            )
            add_outline(m, g_show, "Isócronas (linha)", color="#111111", weight=0.7, show=False)

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, zoom
                m.location, m.zoom_start = center, zoom

        # ---------------------------------------------------------
        # NÍVEL 3: QUADRAS (das isócronas selecionadas)
        # ---------------------------------------------------------
        elif level == "quadra":
            if g_quadra is None or g_quadra.empty:
                st.warning("Quadras não encontradas/vazias na pasta do Drive.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)

            st.markdown("### Quadras (Isócronas selecionadas)")
            st.caption("Clique para selecionar múltiplas quadras. Ao selecionar, abre o nível final.")

            add_polygons_selectable(
                m,
                g_show,
                name="Quadras (multi)",
                id_col=QUADRA_ID,
                selected_ids=st.session_state["selected_quadra_ids"],
                base_color="#111111",
                base_weight=0.45,
                fill_color="#ffffff",
                fill_opacity=0.10,
                selected_color=PB_NAVY,
                selected_weight=2.2,
                selected_fill_opacity=0.22,
            )

            if not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 17)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        # ---------------------------------------------------------
        # NÍVEL 4: FINAL (Lotes OU Setores)
        # - Lotes recortados por quadra_id selecionadas
        # - Setores recortados por iso_id selecionadas
        # ---------------------------------------------------------
        else:  # "final"
            iso_ids: Set[Any] = st.session_state["selected_iso_ids"]
            quad_ids: Set[Any] = st.session_state["selected_quadra_ids"]
            if not iso_ids:
                reset_to("isocrona")
                st.rerun()

            mode = st.session_state["final_mode"]

            if mode == "lote":
                if g_lote is None or g_lote.empty:
                    st.warning("Lotes não encontrados/vazios na pasta do Drive.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return
                if not quad_ids:
                    reset_to("quadra")
                    st.rerun()

                g_show = subset_by_parent_multi(g_lote, LOTE_PARENT, quad_ids)
                st.markdown("### Lotes (Quadras selecionadas)")
                st.caption("Nível final. Você pode alternar para Setores no painel à esquerda.")

                add_polygons_selectable(
                    m,
                    g_show,
                    name="Lotes",
                    id_col=LOTE_ID if (LOTE_ID in g_show.columns) else "",
                    selected_ids=set(),
                    base_color="#111111",
                    base_weight=0.25,
                    fill_color="#ffffff",
                    fill_opacity=0.10,
                )
            else:
                if g_censo is None or g_censo.empty:
                    st.warning("Setores Censitários não encontrados/vazios na pasta do Drive.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                g_show = subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)
                st.markdown("### Setores Censitários (Isócronas selecionadas)")
                st.caption("Nível final (paralelo a Lotes).")

                add_polygons_selectable(
                    m,
                    g_show,
                    name="Setores Censitários",
                    id_col=CENSO_ID if (CENSO_ID in g_show.columns) else "",
                    selected_ids=set(),
                    base_color="#111111",
                    base_weight=0.35,
                    fill_color="#ffffff",
                    fill_opacity=0.08,
                    selected_color=PB_NAVY,
                    selected_weight=2.0,
                    selected_fill_opacity=0.16,
                )

            if g_show is not None and not g_show.empty:
                center, zoom = bounds_center_zoom(g_show)
                st.session_state["view_center"], st.session_state["view_zoom"] = center, min(zoom + 1, 18)
                m.location, m.zoom_start = st.session_state["view_center"], st.session_state["view_zoom"]

        try:
            folium.LayerControl(position="bottomright", collapsed=False).add_to(m)
        except Exception:
            pass

        out = st_folium(m, height=780, use_container_width=True, key="map_view", returned_objects=[])

        # ---------------------------------------------------------
        # CLIQUES: avança fluxo
        # ---------------------------------------------------------
        click = (out or {}).get("last_clicked")
        if click:
            if level == "subpref":
                picked = pick_feature_id(g_subpref, click, SUBPREF_ID)
                if picked is not None:
                    st.session_state["selected_subpref_id"] = picked
                    st.session_state["level"] = "distrito"
                    st.rerun()

            elif level == "distrito":
                sp = st.session_state["selected_subpref_id"]
                g_show = subset_by_parent(g_dist, DIST_PARENT, sp)  # type: ignore
                picked = pick_feature_id(g_show, click, DIST_ID)
                if picked is not None:
                    st.session_state["selected_distrito_id"] = picked
                    st.session_state["selected_iso_ids"] = set()
                    st.session_state["selected_quadra_ids"] = set()
                    st.session_state["final_mode"] = "lote"
                    st.session_state["level"] = "isocrona"
                    st.rerun()

            elif level == "isocrona":
                d = st.session_state["selected_distrito_id"]
                g_show = subset_by_parent(g_iso, ISO_PARENT, d)  # type: ignore
                picked = pick_feature_id(g_show, click, ISO_ID)
                if picked is not None:
                    _toggle_in_set("selected_iso_ids", picked)
                    if len(st.session_state["selected_iso_ids"]) >= 1:
                        st.session_state["selected_quadra_ids"] = set()
                        st.session_state["final_mode"] = "lote"
                        st.session_state["level"] = "quadra"
                    st.rerun()

            elif level == "quadra":
                iso_ids = st.session_state["selected_iso_ids"]
                g_show = subset_by_parent_multi(g_quadra, QUADRA_PARENT, iso_ids)  # type: ignore
                picked = pick_feature_id(g_show, click, QUADRA_ID)
                if picked is not None:
                    _toggle_in_set("selected_quadra_ids", picked)
                    if len(st.session_state["selected_quadra_ids"]) >= 1:
                        st.session_state["level"] = "final"
                    st.rerun()

            else:
                # nível final: por enquanto sem clique (só visual)
                pass

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
