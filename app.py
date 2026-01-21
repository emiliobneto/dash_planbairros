# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from unicodedata import normalize as _ud_norm
from decimal import Decimal
import gc
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # entrega o HTML do mapa (evita reruns)

# --- Geo libs (opcionais) ---
try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import box  # type: ignore
    import folium  # type: ignore
    try:
        from branca.element import Element  # type: ignore
    except Exception:
        from folium import Element  # type: ignore
except Exception:
    gpd = None; box = None; folium = None; Element = None

# =============================================================================
# Página / identidade visual
# =============================================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde":   "#B1BF7C",
    "laranja": "#D58243",
    "telha":   "#C65534",
    "teal":    "#6FA097",
    "navy":    "#14407D",
}

# Paleta numérica (alto contraste, sem azul/verde)
NUM_PALETTE = ["#fff0d6", "#ffd36a", "#f2a740", "#d46f1e", "#ad3c19", "#7a1e10"]

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
          html, body, .stApp {{ font-family: Roboto, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif !important; }}
          .pb-header {{
            background: {PB_COLORS['navy']}; color:#fff; border-radius:18px;
            padding:20px 24px; min-height:110px; display:flex; align-items:center; gap:16px;
          }}
          .pb-title {{ font-size: 2.2rem; font-weight: 700; }}
          .pb-card {{
            background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04);
            border-radius:16px; padding:16px;
          }}
          .main .block-container {{ padding-top:.6rem; padding-bottom:.6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_logo_path() -> Optional[str]:
    for p in ["assets/logo_todos.jpg","assets/logo_paleta.jpg","logo_todos.jpg","logo_paleta.jpg",
              "/mnt/data/logo_todos.jpg","/mnt/data/logo_paleta.jpg"]:
        if Path(p).exists(): return p
    return None

# =============================================================================
# Caminhos / utilidades
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
def _resolve_dir(subdir: str) -> Path:
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / subdir
        if p.exists(): return p
    return REPO_ROOT / subdir

ADM_DIR = _resolve_dir("limites_administrativos")  # Setores + Isócronas + Distritos + Subprefeituras + ZonasOD

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII","ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+","", s2.strip().lower())

def find_file(folder: Path, *names: str) -> Optional[Path]:
    if not folder.exists(): return None
    wanted = {_slug(n) for n in names}
    for f in folder.glob("*.parquet"):
        if _slug(f.stem) in wanted: return f
    return None

def find_col(df_cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in df_cols}
    for c in cands:
        if c and c.lower() in low: return low[c.lower()]
    norm = {re.sub(r"[^a-z0-9]","",k.lower()): v for k,v in low.items()}
    for c in cands:
        key = re.sub(r"[^a-z0-9]","",(c or "").lower())
        if key in norm: return norm[key]
    return None

def bounds_tuple(gdf) -> Optional[tuple[float,float,float,float]]:
    if gdf is None or gdf.empty: return None
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)

def center_from_bounds(gdf) -> tuple[float,float]:
    minx,miny,maxx,maxy = gdf.total_bounds
    return ((miny+maxy)/2,(minx+maxx)/2)

def filter_bbox(gdf, b):
    if gdf is None or gdf.empty or b is None or box is None: return gdf
    try: return gdf[gdf.intersects(box(*b))]
    except Exception: return gdf

# =============================================================================
# Leitura robusta (GeoParquet) — CRS=4326 + fallback via WKB/WKT
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def read_gdf_from_parquet(path: Path) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None or path is None: return None
    try:
        gdf = gpd.read_parquet(path)
        if gdf.crs is None: gdf = gdf.set_crs(4326)
        else: gdf = gdf.to_crs(4326)
        return gdf
    except Exception:
        pass
    try:
        pdf = pd.read_parquet(path)
    except Exception:
        return None
    geom_col = next((c for c in pdf.columns if c.lower() in ("geometry","geom","wkb","wkt")), None)
    if geom_col is None: return None
    try:
        from shapely import wkb, wkt
        vals = pdf[geom_col]
        if vals.dropna().astype(str).str.startswith(("POLY","MULTI","LINE","POINT")).any():
            geo = vals.dropna().apply(wkt.loads)
        else:
            geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b,str)))
        gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
        return gdf
    except Exception:
        return None

# loaders on-demand
@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def load_limits(name: str) -> Optional["gpd.GeoDataFrame"]:
    fn = {
        "Distritos": ["Distritos"],
        "ZonasOD2023": ["ZonasOD2023","ZonasOD"],
        "Subprefeitura": ["subprefeitura","Subprefeitura"],
        "Isócronas": ["isocronas","isócronas"],
    }.get(name, [name])
    p = find_file(ADM_DIR, *fn)
    return read_gdf_from_parquet(p) if p else None

@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def load_setores() -> Optional["gpd.GeoDataFrame"]:
    p = find_file(ADM_DIR, "SetoresCensitarios2023")
    return read_gdf_from_parquet(p) if p else None

# =============================================================================
# Sanitização universal para evitar `Decimal`/NaN no GeoJSON
# =============================================================================
def sanitize_df_for_geojson(df: pd.DataFrame, int_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Mantém somente as colunas passadas no df (já deve estar filtrado),
    converte Decimals para float/int, força numéricos e troca NaN/NA por None.
    """
    int_cols = int_cols or []
    out = df.copy()

    for c in out.columns:
        if c == "geometry":
            continue

        # Converte Decimal -> float (ou int, se solicitado)
        out[c] = out[c].apply(lambda x: (int(x) if isinstance(x, Decimal) else float(x))
                              if isinstance(x, Decimal) else x)

        if c in int_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
        else:
            # força float para tudo que for numérico (evita objects mistos)
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
            else:
                # se ainda houver Decimal ou algo bizarro
                def _coerce(v):
                    if isinstance(v, Decimal):
                        return float(v)
                    return v
                out[c] = out[c].map(_coerce)

    # substitui NaN/NA por None (null no JSON)
    out = out.where(pd.notnull(out), None)
    return out

def simplify_safe(gdf):
    try:
        return gdf.set_geometry(gdf.geometry.simplify(0.0004, preserve_topology=True))
    except Exception:
        return gdf

# =============================================================================
# Folium (mapa, tooltip/legendas) – apenas colunas necessárias
# =============================================================================
def make_base_map(center=(-23.55,-46.63), zoom=11, tiles_opacity=0.55):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try: folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception: pass
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", overlay=False, control=False, opacity=tiles_opacity
    ).add_to(m)
    return m

def inject_tooltip_css(m, font_px: int = 64):
    if Element is None: return
    m.get_root().html.add_child(Element(f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip, .leaflet-tooltip.pb-big-tooltip * {{
        font-size:{font_px}px !important; font-weight:800 !important; color:#111 !important; line-height:1 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background:#fff !important; border:2px solid #222 !important; border-radius:10px !important;
        padding:10px 14px !important; white-space:nowrap !important; pointer-events:none !important;
        box-shadow:0 2px 6px rgba(0,0,0,.2) !important; z-index:999999 !important;
      }}
    </style>"""))

def inject_label_scaler(m, min_px=16, max_px=26, min_zoom=9, max_zoom=18):
    if Element is None: return
    mv = m.get_name()
    m.get_root().html.add_child(Element(f"""
    <script>
      (function(){{
        var minZ={min_zoom}, maxZ={max_zoom}, minPx={min_px}, maxPx={max_px};
        function scale(z){{ z=Math.max(minZ,Math.min(maxZ,z)); return Math.round(minPx+(z-minZ)*(maxPx-minPx)/(maxZ-minZ)); }}
        function upd(){{ var px=scale({mv}.getZoom()); document.querySelectorAll('.pb-static-label').forEach(e=>e.style.fontSize=px+'px'); }}
        {mv}.on('zoomend',upd); {mv}.whenReady(upd);
      }})();
    </script>"""))

def add_outline(m, gdf, layer_name: str, weight=1.1):
    if gdf is None or gdf.empty or folium is None: return
    try:
        line = gdf[["geometry"]].copy()
        line["geometry"] = line.geometry.boundary
        line = simplify_safe(line)
        # só geometry (sem propriedades -> sem Decimals)
        folium.GeoJson(
            data=line.to_json(), name=f"{layer_name} (contorno)", pane="vectors",
            style_function=lambda f: {"fillOpacity":0, "color":"#000000", "weight":weight}, smooth_factor=0.8,
        ).add_to(m)
    except Exception as exc:
        st.warning(f"Falha ao desenhar contorno de {layer_name}: {exc}")

def add_centroids(m, gdf):
    if folium is None or gdf is None or gdf.empty: return
    cols = {c.lower(): c for c in gdf.columns}
    nm = cols.get("ds_nome") or cols.get("sp_nome") or cols.get("nome")
    if nm is None: return
    try:
        pts = gdf.to_crs(4326).representative_point()
    except Exception:
        return
    for name, pt in zip(gdf[nm].astype(str), pts):
        html = ("<div class='pb-static-label' style=\"font:600 12px/1 Roboto, sans-serif;"
                "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
                f"{name}</div>")
        try:
            folium.Marker(
                location=[pt.y, pt.x],
                icon=folium.DivIcon(html=html, icon_size=(0,0), icon_anchor=(0,0)),
                z_index_offset=1000,
            ).add_to(m)
        except Exception:
            continue

def add_gradient_legend(m, title: str, ticks: List[float], colors: List[str]):
    if Element is None: return
    gradient_css = f"background: linear-gradient(to top, {', '.join(colors)});"
    html = f"""
    <div style="position:absolute; left:16px; bottom:24px; z-index:999999; background:rgba(255,255,255,.97);
                padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25);
                font:500 12px Roboto, sans-serif; border:1px solid #333;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      <div style="display:flex; gap:10px; align-items:stretch;">
        <div style="width:18px; height:120px; {gradient_css}; border-radius:4px; border:1px solid #333;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:120px;">
          <div>{ticks[-1]:,.0f}</div>
          <div>{ticks[len(ticks)//2]:,.0f}</div>
          <div>{ticks[0]:,.0f}</div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(html))

def add_categorical_legend(m, title: str, items: List[tuple[str, str]], topright: bool = False):
    if Element is None: return
    pos = "right: 16px; top: 24px;" if topright else "left: 16px; bottom: 24px;"
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
        f'<span style="width:16px;height:16px;border-radius:3px;background:{color};display:inline-block;border:1px solid #333"></span>'
        f'<span>{label}</span></div>'
        for label, color in items
    )
    html = f"""
    <div style="position:absolute; {pos} z-index:999999; background:rgba(255,255,255,.97);
                padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25);
                font:500 12px Roboto, sans-serif; border: 1px solid #333;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(Element(html))

# =============================================================================
# Variáveis – apenas colunas necessárias + sanitização
# =============================================================================
def quantile_bins(s: pd.Series, k: int = 6) -> List[float]:
    s = pd.to_numeric(s, errors="coerce").replace([float("inf"), -float("inf")], pd.NA).dropna()
    if s.empty: return [0,1,2,3,4,5]
    qs = s.quantile([i/(k-1) for i in range(k)]).tolist()
    eps = 1e-9
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]: qs[i] = qs[i-1] + eps
    return qs

def draw_numeric(m, setores, value_col: str, label: str, colors=NUM_PALETTE):
    try:
        df = setores[[value_col, "geometry"]].copy()
        # sanitiza (converte Decimal -> float; NaN->None)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = sanitize_df_for_geojson(df)
        df = simplify_safe(df)

        ticks = quantile_bins(df[value_col], k=len(colors))

        def color_for(v):
            if v is None or pd.isna(v): return "#cccccc"
            for i in range(1, len(ticks)):
                if float(v) <= ticks[i]: return colors[i-1]
            return colors[-1]

        def style_fn(feat):
            v = feat["properties"].get(value_col)
            try: v = float(v)
            except Exception: v = float("nan")
            return {"fillOpacity":0.78, "weight":0.0, "color":"#00000000", "fillColor": color_for(v)}

        folium.GeoJson(
            data=df.to_json(),  # só value_col + geometry
            name=label, pane="vectors", style_function=style_fn, smooth_factor=0.8,
            tooltip=folium.features.GeoJsonTooltip(
                fields=[value_col], aliases=[label + ": "], sticky=True, labels=False, class_name="pb-big-tooltip"
            ),
        ).add_to(m)
        add_gradient_legend(m, label, ticks, colors)
    except Exception as exc:
        st.warning(f"Falha ao desenhar '{label}': {exc}")

def draw_cluster(m, setores, cluster_col: str):
    try:
        df = setores[[cluster_col, "geometry"]].copy()
        # força inteiro + sanitiza (evita Decimal)
        df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce").round(0).astype("Int64")
        df = sanitize_df_for_geojson(df, int_cols=[cluster_col])
        df = simplify_safe(df)

        vals = df[cluster_col].dropna().astype(int)
        offset = 1 if (not vals.empty and vals.between(1, 5).all()) else 0

        color_map = {0:"#bf7db2", 1:"#f7bd6a", 2:"#cf651f", 3:"#ede4e6", 4:"#793393"}
        label_map = {
            0:"1 - Periférico com predominância residencial de alta densidade construtiva",
            1:"2 - Uso misto de média densidade construtiva",
            2:"3 - Periférico com predominância residencial de média densidade construtiva",
            3:"4 - Verticalizado de uso-misto",
            4:"5 - Predominância de uso comercial e serviços",
        }
        default_color = "#c8c8c8"

        def style_fn(feat):
            v = feat["properties"].get(cluster_col)
            try: vi = int(v) - offset
            except Exception: vi = -1
            return {"fillOpacity":0.75, "weight":0.0, "color":"#00000000",
                    "fillColor": color_map.get(vi, default_color)}

        folium.GeoJson(
            data=df.to_json(),  # só cluster + geometry
            name="Cluster (perfil urbano)", pane="vectors", style_function=style_fn, smooth_factor=0.8,
        ).add_to(m)

        items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
        items.append(("Outros", default_color))
        add_categorical_legend(m, "Cluster (perfil urbano)", items)
    except Exception as exc:
        st.warning(f"Falha ao desenhar cluster: {exc}")

def draw_isoc_area(m, iso):
    try:
        col = find_col(iso.columns, "nova_class") or "nova_class"
        df = iso[[col, "geometry"]].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")
        df = sanitize_df_for_geojson(df, int_cols=[col])
        df = simplify_safe(df)

        lut = {
            0: ("Predominância uso misto", "#542788"),
            1: ("Zona de transição local", "#f7f7f7"),
            2: ("Periférico residencial de média densidade", "#d8daeb"),
            3: ("Transição central verticalizada", "#b35806"),
            4: ("Periférico adensado em transição", "#b2abd2"),
            5: ("Centralidade comercial e de serviços", "#8073ac"),
            6: ("Predominância residencial média densidade", "#fdb863"),
            7: ("Áreas íngremes e de encosta", "#7f3b08"),
            8: ("Alta densidade residencial", "#e08214"),
            9: ("Central verticalizado", "#fee0b6"),
        }

        def style_fn(feat):
            v = feat["properties"].get(col)
            try: vi = int(v)
            except Exception: vi = -1
            return {"fillOpacity":0.65, "weight":0.0, "color":"#00000000",
                    "fillColor": lut.get(vi, ("Outros", "#c8c8c8"))[1]}

        folium.GeoJson(
            data=df.to_json(), name="Área de influência de bairro", pane="vectors",
            style_function=style_fn, smooth_factor=0.8,
        ).add_to(m)

        legend_items = [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()]
        add_categorical_legend(m, "Área de influência de bairro (nova_class)", legend_items)
    except Exception as exc:
        st.warning(f"Falha ao desenhar isócronas: {exc}")

# =============================================================================
# UI (filtros verticais à esquerda)
# =============================================================================
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "— Nenhuma —",
            "População (Pessoa/ha)",
            "Densidade demográfica (hab/ha)",
            "Variação de elevação média",
            "Elevação média",
            "Cluster (perfil urbano)",
            "Área de influência de bairro",
        ],
        index=0, key="pb_var",
    )
    st.markdown("### Configurações")
    lim = st.selectbox(
        "Limites Administrativos",
        ["— Nenhum —", "Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0, key="pb_limite",
    )
    labels_on = st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")
    return {"variavel": var, "limite": lim, "labels_on": labels_on}

# =============================================================================
# App
# =============================================================================
def main() -> None:
    inject_css()

    # Header
    c1, c2 = st.columns([1, 7])
    with c1:
        lp = get_logo_path()
        if lp: st.image(lp, width=140)
    with c2:
        st.markdown(
            """
            <div class="pb-header">
                <div>
                    <div class="pb-title">PlanBairros</div>
                    <div>Plataforma de visualização e planejamento em escala de bairro</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a1, a2, a3, a4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    with a1:
        left, map_col = st.columns([1, 5], gap="small")

        with left:
            st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
            ui = left_controls()
            st.markdown("</div>", unsafe_allow_html=True)

        ui_sig = f"{ui['variavel']}|{ui['limite']}|{int(ui['labels_on'])}"
        need_rebuild = st.session_state.get("_ui_sig") != ui_sig

        if need_rebuild:
            # 1) Carrega apenas o necessário (on-demand)
            lim_gdf = None
            if ui["limite"] != "— Nenhum —":
                with st.spinner(f"Carregando limite: {ui['limite']}"):
                    lim_gdf = load_limits(ui["limite"])

            var_gdf = None
            if ui["variavel"] != "— Nenhuma —":
                if ui["variavel"] == "Área de influência de bairro":
                    with st.spinner("Carregando isócronas…"):
                        var_gdf = load_limits("Isócronas")
                else:
                    with st.spinner("Carregando setores…"):
                        var_gdf = load_setores()

            # 2) Centro
            center = (-23.55, -46.63)
            if lim_gdf is not None and not lim_gdf.empty:
                center = center_from_bounds(lim_gdf)

            # 3) Folium base
            fmap = make_base_map(center=center, zoom=11, tiles_opacity=0.55)
            inject_tooltip_css(fmap, font_px=64)
            inject_label_scaler(fmap, 16, 26, 9, 18)

            # 4) Limites (contorno) + regra especial de transição (isócronas 1/3/4/6)
            if lim_gdf is not None and not lim_gdf.empty:
                add_outline(fmap, lim_gdf, ui["limite"], weight=1.1)
                if ui["labels_on"]:
                    add_centroids(fmap, lim_gdf)

                if ui["limite"].startswith("Isócron"):
                    try:
                        cls = find_col(lim_gdf.columns, "nova_class")
                        if cls:
                            trans = lim_gdf[lim_gdf[cls].isin([1,3,4,6])][["geometry"]]
                            if not trans.empty:
                                trans = simplify_safe(trans)
                                folium.GeoJson(
                                    data=trans.to_json(), name="Área de transição", pane="vectors",
                                    style_function=lambda f: {"fillOpacity": 0.2, "color": "#836e60", "weight": 0.8},
                                    smooth_factor=0.8,
                                ).add_to(fmap)
                                add_categorical_legend(
                                    fmap, "Limites — Isócronas", [("Área de transição (1,3,4,6)", "#836e60")], topright=True
                                )
                    except Exception as exc:
                        st.warning(f"Falha ao desenhar 'Área de transição': {exc}")

            # 5) Variável
            if var_gdf is not None and not var_gdf.empty:
                try:
                    # recorte bbox para reduzir GeoJSON
                    if lim_gdf is not None and not lim_gdf.empty:
                        var_gdf = filter_bbox(var_gdf, bounds_tuple(lim_gdf))

                    var = ui["variavel"]
                    if var == "Área de influência de bairro":
                        draw_isoc_area(fmap, var_gdf)
                    elif var == "Cluster (perfil urbano)":
                        ccol = find_col(var_gdf.columns, "cluster", "cluster_label", "classe", "label") or "cluster"
                        draw_cluster(fmap, var_gdf, ccol)
                    else:
                        col_map = {
                            "População (Pessoa/ha)": "populacao",
                            "Densidade demográfica (hab/ha)": "densidade_demografica",
                            "Variação de elevação média": "diferenca_elevacao",
                            "Elevação média": "elevacao",
                        }
                        col = find_col(var_gdf.columns, col_map[var]) or col_map[var]
                        draw_numeric(fmap, var_gdf, col, var, NUM_PALETTE)
                except Exception as exc:
                    st.warning(f"Falha ao desenhar variável '{ui['variavel']}': {exc}")

            # 6) HTML estático no estado (interações não rerodam o script)
            html = fmap.get_root().render()
            st.session_state["_ui_sig"] = ui_sig
            st.session_state["_map_html"] = html

            del lim_gdf, var_gdf, fmap
            gc.collect()

        components.html(st.session_state.get("_map_html", ""), height=850, scrolling=False)

    with a2: st.info("Conteúdo a definir.")
    with a3: st.info("Conteúdo a definir.")
    with a4: st.info("Conteúdo a definir.")

if __name__ == "__main__":
    main()
