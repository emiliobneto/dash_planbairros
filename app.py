# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from unicodedata import normalize as _ud_norm
import gc
import math
import re

import pandas as pd
import streamlit as st

# Geo libs
try:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import box  # type: ignore
    import folium  # type: ignore
    try:
        # folium 0.16 usa branca.element.Element
        from branca.element import Element  # type: ignore
    except Exception:
        from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None; box = None; folium = None; Element = None; st_folium = None

# =============================================================================
# Página / identidade visual
# =============================================================================
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde":   "#B1BF7C",
    "laranja": "#D58243",
    "telha":   "#C65534",
    "teal":    "#6FA097",
    "navy":    "#14407D",
}

# Paleta de ALTO CONTRASTE (sem azul/verde) para choropleth discreto (6 classes)
NUM_PALETTE = ["#fff0d6", "#ffd36a", "#f2a93b", "#d46f1e", "#ad3c19", "#7a1e10"]

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

# =============================================================================
# Leitura on-demand (assumindo EPSG:4326)
# =============================================================================
def read_gdf(path: Path, columns: Optional[list[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None or path is None: return None
    try:
        gdf = gpd.read_parquet(path, columns=columns) if columns else gpd.read_parquet(path)
    except Exception:
        return None
    # Todos já em 4326 segundo o usuário; apenas garante a definição
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        elif int(str(gdf.crs).split(":")[-1]) != 4326:
            gdf = gdf.to_crs(4326)
    except Exception:
        try: gdf = gdf.set_crs(4326)
        except Exception: pass
    return gdf

@st.cache_data(max_entries=2, ttl=300, show_spinner=False)
def load_limits(name: str) -> Optional["gpd.GeoDataFrame"]:
    fn = {
        "Distritos": ["Distritos"],
        "ZonasOD2023": ["ZonasOD2023","ZonasOD"],
        "Subprefeitura": ["subprefeitura","Subprefeitura"],
        "Isócronas": ["isocronas","isócronas"],
    }.get(name, [name])
    p = find_file(ADM_DIR, *fn)
    cols = ["geometry","ds_nome","sp_nome","nova_class"]
    return read_gdf(p, columns=cols) if p else None

@st.cache_data(max_entries=2, ttl=300, show_spinner=False)
def load_setores_for(kind: str) -> Optional["gpd.GeoDataFrame"]:
    p = find_file(ADM_DIR, "SetoresCensitarios2023")
    if not p: return None
    if kind == "cluster":
        cols = ["geometry","cluster","cluster_label","classe","label"]
    elif kind == "pop":
        cols = ["geometry","populacao"]
    elif kind == "dens":
        cols = ["geometry","densidade_demografica"]
    elif kind == "elev_diff":
        cols = ["geometry","diferenca_elevacao"]
    else:  # "elev"
        cols = ["geometry","elevacao"]
    return read_gdf(p, columns=cols)

def filter_bbox(gdf: "gpd.GeoDataFrame", b: Optional[tuple[float,float,float,float]]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty or b is None or box is None: return gdf
    try: return gdf[gdf.intersects(box(*b))]
    except Exception: return gdf

# =============================================================================
# Folium (mapa/legendas)
# =============================================================================
def make_satellite_map(center=(-23.55,-46.63), zoom=11, tiles_opacity=0.55):
    if folium is None: return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try: folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception: pass
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Esri Satellite", overlay=False, control=False, opacity=tiles_opacity
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
    line = gdf[["geometry"]].copy()
    line["geometry"] = line.geometry.boundary
    folium.GeoJson(
        data=line.__geo_interface__, name=f"{layer_name} (contorno)", pane="vectors",
        style_function=lambda f: {"fillOpacity":0, "color":"#000000", "weight":weight},
        smooth_factor=0.8,
    ).add_to(m)

def add_centroids(m, gdf):
    if folium is None or gdf is None or gdf.empty: return
    cols = {c.lower(): c for c in gdf.columns}
    nm = cols.get("ds_nome") or cols.get("sp_nome") or cols.get("nome")
    if nm is None: return
    try: pts = gdf.to_crs(4326).representative_point()
    except Exception: return
    for name, pt in zip(gdf[nm].astype(str), pts):
        html = ("<div class='pb-static-label' style=\"font:600 12px/1 Roboto, sans-serif;"
                "color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"
                f"{name}</div>")
        folium.Marker(
            location=[pt.y, pt.x],
            icon=folium.DivIcon(html=html, icon_size=(0,0), icon_anchor=(0,0)),
            z_index_offset=1000,
        ).add_to(m)

def add_gradient_legend(m, title: str, ticks: list[float], colors: list[str]):
    if Element is None: return
    # monta gradiente (de baixo p/ cima) e rótulos de quantis
    gradient_css = f"background: linear-gradient(to top, {', '.join(colors)});"
    labels = "".join(f"<div>{v:,.0f}</div>" for v in [ticks[-1], ticks[len(ticks)//2], ticks[0]])
    html = f"""
    <div style="position:absolute; left:16px; bottom:24px; z-index:999999; background:rgba(255,255,255,.97);
                padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25);
                font:500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      <div style="display:flex; gap:10px; align-items:stretch;">
        <div style="width:18px; height:120px; {gradient_css}; border-radius:4px; border:1px solid #333;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:120px;">
          {labels}
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(html))

def add_categorical_legend(m, title: str, items: list[tuple[str, str]], topright: bool = False):
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
                font:500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(Element(html))

# =============================================================================
# Pintura das camadas
# =============================================================================
def quantile_bins(s: pd.Series, k: int = 6) -> list[float]:
    qs = s.quantile([i/(k-1) for i in range(k)]).tolist()
    # Garantir monotonia estrita (evita bins iguais para dados com pouca variação)
    eps = 1e-9
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = qs[i-1] + eps
    return qs

def draw_numeric(m, setores: "gpd.GeoDataFrame", value_col: str, label: str, colors: list[str] = NUM_PALETTE):
    s = pd.to_numeric(setores[value_col], errors="coerce")
    s = s.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if s.empty: return
    ticks = quantile_bins(s, k=len(colors))

    def color_for(v: float) -> str:
        # encontra classe por quantil
        if v is None or pd.isna(v): return "#cccccc"
        for i in range(1, len(ticks)):
            if v <= ticks[i]: return colors[i-1]
        return colors[-1]

    def style_fn(feat):
        v = feat["properties"].get(value_col)
        try: v = float(v)
        except Exception: v = float("nan")
        return {"fillOpacity": 0.78, "weight": 0.0, "color": "#00000000", "fillColor": color_for(v)}

    folium.GeoJson(
        setores.__geo_interface__,
        name=label,
        pane="vectors",
        style_function=style_fn,
        smooth_factor=0.8,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[value_col], aliases=[label + ": "], sticky=True, labels=False, class_name="pb-big-tooltip"
        ),
    ).add_to(m)
    add_gradient_legend(m, label, ticks, colors)

def draw_cluster(m, setores: "gpd.GeoDataFrame", cluster_col: str):
    # aceita 0..4 ou 1..5
    vals = pd.to_numeric(setores[cluster_col], errors="coerce")
    offset = 1 if vals.dropna().between(1,5).all() else 0

    color_map = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}
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
        return {"fillOpacity": 0.75, "weight": 0.0, "color": "#00000000",
                "fillColor": color_map.get(vi, default_color)}

    folium.GeoJson(
        setores.__geo_interface__,
        name="Cluster (perfil urbano)",
        pane="vectors",
        style_function=style_fn,
        smooth_factor=0.8,
    ).add_to(m)
    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]
    items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items)

def draw_isoc_area(m, iso: "gpd.GeoDataFrame"):
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
    col = find_col(iso.columns, "nova_class")

    def style_fn(feat):
        v = feat["properties"].get(col)
        try: vi = int(v)
        except Exception: vi = -1
        return {"fillOpacity": 0.65, "weight": 0.0, "color": "#00000000",
                "fillColor": lut.get(vi, ("Outros", "#c8c8c8"))[1]}

    folium.GeoJson(
        iso.__geo_interface__,
        name="Área de influência de bairro",
        pane="vectors",
        style_function=style_fn,
        smooth_factor=0.8,
    ).add_to(m)
    legend_items = [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()]
    add_categorical_legend(m, "Área de influência de bairro (nova_class)", legend_items)

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

        with map_col:
            if folium is None or st_folium is None or gpd is None:
                st.error("Instale `geopandas`, `folium` e `streamlit-folium` para exibir o mapa.")
                return

            # 1) Limite (lazy)
            lim_gdf = None
            if ui["limite"] != "— Nenhum —":
                with st.spinner(f"Carregando limite: {ui['limite']}"):
                    lim_gdf = load_limits(ui["limite"])

            # 2) Centro
            center = (-23.55, -46.63)
            if lim_gdf is not None and not lim_gdf.empty:
                center = center_from_bounds(lim_gdf)

            fmap = make_satellite_map(center=center, zoom=11, tiles_opacity=0.55)
            inject_tooltip_css(fmap, font_px=64)   # tooltips legíveis
            inject_label_scaler(fmap, 16, 26, 9, 18)

            # 3) Contorno + regra Isócronas (área de transição)
            if lim_gdf is not None and not lim_gdf.empty:
                add_outline(fmap, lim_gdf, ui["limite"], weight=1.1)
                if ui["labels_on"]:
                    add_centroids(fmap, lim_gdf)
                if ui["limite"].startswith("Isócron"):
                    cls = find_col(lim_gdf.columns, "nova_class")
                    if cls:
                        trans = lim_gdf[lim_gdf[cls].isin([1,3,4,6])][["geometry"]]
                        if not trans.empty:
                            folium.GeoJson(
                                data=trans.__geo_interface__, name="Área de transição", pane="vectors",
                                style_function=lambda f: {"fillOpacity": 0.2, "color": "#836e60", "weight": 0.8},
                                smooth_factor=0.8,
                            ).add_to(fmap)
                            add_categorical_legend(
                                fmap, "Limites — Isócronas", [("Área de transição (1,3,4,6)", "#836e60")], topright=True
                            )

            # 4) Variável (lazy)
            var = ui["variavel"]
            var_gdf = None
            if var != "— Nenhuma —":
                if var == "Área de influência de bairro":
                    with st.spinner("Carregando isócronas…"):
                        var_gdf = load_limits("Isócronas")
                else:
                    kind = {
                        "População (Pessoa/ha)": "pop",
                        "Densidade demográfica (hab/ha)": "dens",
                        "Variação de elevação média": "elev_diff",
                        "Elevação média": "elev",
                        "Cluster (perfil urbano)": "cluster",
                    }[var]
                    with st.spinner(f"Carregando setores ({var})…"):
                        var_gdf = load_setores_for(kind)

                if var_gdf is not None and lim_gdf is not None and not lim_gdf.empty:
                    var_gdf = filter_bbox(var_gdf, bounds_tuple(lim_gdf))

            # 5) Desenho (sem linhas para setores)
            if var_gdf is not None and not var_gdf.empty:
                if var == "Área de influência de bairro":
                    draw_isoc_area(fmap, var_gdf)
                elif var == "Cluster (perfil urbano)":
                    ccol = find_col(var_gdf.columns, "cluster", "cluster_label", "classe", "label") or "cluster"
                    draw_cluster(fmap, var_gdf, ccol)
                else:
                    col = {
                        "População (Pessoa/ha)": "populacao",
                        "Densidade demográfica (hab/ha)": "densidade_demografica",
                        "Variação de elevação média": "diferenca_elevacao",
                        "Elevação média": "elevacao",
                    }[var]
                    col = find_col(var_gdf.columns, col) or col
                    draw_numeric(fmap, var_gdf, col, var, NUM_PALETTE)

            st_folium(fmap, use_container_width=True, height=850)
            del lim_gdf, var_gdf
            gc.collect()

    with a2: st.info("Conteúdo a definir.")
    with a3: st.info("Conteúdo a definir.")
    with a4: st.info("Conteúdo a definir.")

if __name__ == "__main__":
    main()
