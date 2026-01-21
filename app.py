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

try:
    import geopandas as gpd  # type: ignore
    from shapely import set_precision  # type: ignore
    from shapely.geometry import box  # type: ignore
    import folium  # type: ignore
    from folium import Element  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None; set_precision = None; folium = None; Element = None; st_folium = None

# -----------------------------------------------------------------------------
# Página / paleta
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

PB_COLORS = {"amarelo":"#F4DD63","verde":"#B1BF7C","laranja":"#D58243","telha":"#C65534","teal":"#6FA097","navy":"#14407D"}
GRADIENT = ["#F4DD63","#D58243","#C65534"]  # sem azul/verde

def css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
      :root {{ --pb-navy:{PB_COLORS['navy']}; }}
      html, body, .stApp {{ font-family: Roboto, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif !important; }}
      .pb-header {{ background: var(--pb-navy); color:#fff; border-radius:18px; padding:20px 24px; min-height:110px; display:flex; align-items:center; gap:16px; }}
      .pb-title {{ font-size: 2.2rem; font-weight: 700; }}
      .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.1); box-shadow:0 1px 2px rgba(0,0,0,.04); border-radius:16px; padding:16px; }}
      .main .block-container {{ padding-top:.6rem; padding-bottom:.6rem; }}
    </style>""", unsafe_allow_html=True)

def get_logo() -> Optional[str]:
    for p in ["assets/logo_todos.jpg","assets/logo_paleta.jpg","logo_todos.jpg","logo_paleta.jpg","/mnt/data/logo_todos.jpg","/mnt/data/logo_paleta.jpg"]:
        if Path(p).exists(): return p
    return None

# -----------------------------------------------------------------------------
# Caminhos / util
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
def _resolve_dir(subdir: str) -> Path:
    for base in (REPO, REPO / "dash_planbairros"):
        p = base / subdir
        if p.exists(): return p
    return REPO / subdir

ADM_DIR = _resolve_dir("limites_administrativos")  # setores + isócronas + distritos etc.

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

def ramp_color(v: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if v is None or math.isnan(v): return "#cccccc"
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    t = max(0, min(1, t)); n = len(colors)-1
    i = min(int(t*n), n-1); frac = (t*n)-i
    h1 = colors[i].lstrip("#"); h2 = colors[i+1].lstrip("#")
    c1 = tuple(int(h1[k:k+2],16) for k in (0,2,4)); c2 = tuple(int(h2[k:k+2],16) for k in (0,2,4))
    rgb = tuple(int(c1[j] + frac*(c2[j]-c1[j])) for j in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)

# -----------------------------------------------------------------------------
# Leitura on-demand (com cache pequeno) + simplificação ao desenhar
# -----------------------------------------------------------------------------
def read_gdf(path: Path, columns: Optional[list[str]] = None) -> Optional["gpd.GeoDataFrame"]:
    if gpd is None: return None
    try:
        gdf = gpd.read_parquet(path, columns=columns) if columns else gpd.read_parquet(path)
        if gdf.crs is None: gdf = gdf.set_crs(4326)
        else: gdf = gdf.to_crs(4326)
        return gdf
    except Exception:
        return None

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
def load_setores_for(value_kind: str) -> Optional["gpd.GeoDataFrame"]:
    """value_kind: 'pop','dens','elev','elev_med','cluster'"""
    p = find_file(ADM_DIR, "SetoresCensitarios2023")
    if not p: return None
    if value_kind == "cluster":
        cols = ["geometry","cluster","cluster_label","classe","label"]
    elif value_kind == "pop":
        cols = ["geometry","populacao"]
    elif value_kind == "dens":
        cols = ["geometry","densidade_demografica"]
    elif value_kind == "elev_diff":
        cols = ["geometry","diferenca_elevacao"]
    else:
        cols = ["geometry","elevacao"]
    return read_gdf(p, columns=cols)

def simplify_3857(gdf: "gpd.GeoDataFrame", tol_m: int = 60, precision_m: float = 1.0) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty or gpd is None: return gdf
    g = gdf.to_crs(3857).copy()
    try:
        g["geometry"] = g.geometry.simplify(tol_m, preserve_topology=True)
        if set_precision is not None:
            g["geometry"] = set_precision(g.geometry, grid_size=precision_m)
    except Exception:
        pass
    return g.to_crs(4326)

def filter_bbox(gdf: "gpd.GeoDataFrame", b: Optional[tuple[float,float,float,float]]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty or b is None: return gdf
    try: return gdf[gdf.intersects(box(*b))]
    except Exception: return gdf

def pick_tol(limite_gdf, fallback_gdf) -> int:
    g = limite_gdf if (limite_gdf is not None and len(limite_gdf)>0) else fallback_gdf
    if g is None or g.empty: return 60
    minx,miny,maxx,maxy = g.total_bounds
    largura_km = (maxx-minx)*111
    return 160 if largura_km>100 else 120 if largura_km>50 else 60 if largura_km>20 else 30 if largura_km>8 else 15

# -----------------------------------------------------------------------------
# Folium helpers
# -----------------------------------------------------------------------------
def make_map(center=(-23.55,-46.63), zoom=11):
    if folium is None: return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try: folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception: pass
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Esri Satellite", overlay=False, control=False, opacity=.55
    ).add_to(m)
    return m

def inject_tooltip_css(m, px=34):
    if Element is None: return
    m.get_root().html.add_child(Element(f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip, .leaflet-tooltip.pb-big-tooltip * {{
        font-size:{px}px !important; font-weight:700 !important; color:#111 !important; line-height:1 !important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background:#fff !important; border:2px solid #222 !important; border-radius:10px !important;
        padding:6px 10px !important; white-space:nowrap !important; pointer-events:none !important;
        box-shadow:0 2px 6px rgba(0,0,0,.2) !important; z-index:200000 !important;
      }}
    </style>"""))

def inject_label_scaler(m, min_px=14, max_px=24, min_zoom=9, max_zoom=18):
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

def add_outline(m, gdf, name, tol_m):
    if gdf is None or gdf.empty or folium is None: return
    line = simplify_3857(gdf[["geometry"]], tol_m=max(30,tol_m))
    line["geometry"] = line.geometry.boundary
    folium.GeoJson(
        line.to_json(), name=f"{name} (contorno)", pane="vectors",
        style_function=lambda f: {"fillOpacity":0, "color":"#000", "weight":1.1}, smooth_factor=1.2
    ).add_to(m)

def add_centroids(m, gdf):
    if folium is None or gdf is None or gdf.empty: return
    cols = {c.lower():c for c in gdf.columns}
    nm = cols.get("ds_nome") or cols.get("sp_nome") or cols.get("nome")
    if nm is None: return
    try: pts = gdf.to_crs(4326).representative_point()
    except Exception: return
    for name, pt in zip(gdf[nm].astype(str), pts):
        html = "<div class='pb-static-label' style=\"font:600 12px/1 Roboto, sans-serif; color:#111; text-shadow:0 0 2px #fff, 0 0 6px #fff; white-space:nowrap;\">"+name+"</div>"
        folium.Marker(location=[pt.y,pt.x], icon=folium.DivIcon(html=html,icon_size=(0,0),icon_anchor=(0,0)), z_index_offset=1000).add_to(m)

def add_grad_legend(m, title, vmin, vmax, colors):
    if Element is None: return
    grad = f"background: linear-gradient(to top, {', '.join(colors)});"
    Element_html = f"""
    <div style="position:absolute; left:16px; bottom:24px; z-index:9999; background:rgba(255,255,255,.95); padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25); font:500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>
      <div style="display:flex; gap:10px; align-items:stretch;">
        <div style="width:18px; height:120px; {grad}; border-radius:4px;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:120px;">
          <div>{vmax:,.0f}</div><div>{(vmin+vmax)/2:,.0f}</div><div>{vmin:,.0f}</div>
        </div>
      </div>
    </div>"""
    m.get_root().html.add_child(Element(Element_html))

def add_cat_legend(m, title, items, topright=False):
    if Element is None: return
    pos = "right:16px; top:24px;" if topright else "left:16px; bottom:24px;"
    rows = "".join([f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0"><span style="width:16px;height:16px;border-radius:3px;background:{c};display:inline-block"></span><span>{t}</span></div>' for t,c in items])
    m.get_root().html.add_child(Element(f"""
    <div style="position:absolute; {pos} z-index:9999; background:rgba(255,255,255,.95); padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25); font:500 12px Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:6px">{title}</div>{rows}
    </div>"""))

# -----------------------------------------------------------------------------
# Camadas desenhadas (sempre a partir de leitura sob demanda)
# -----------------------------------------------------------------------------
def draw_numeric(m, setores, value_col, label):
    s = pd.to_numeric(setores[value_col], errors="coerce")
    vmin, vmax = float(s.min()), float(s.max())

    def sty(f):
        v = f["properties"].get(value_col)
        try: v = float(v)
        except Exception: v = float("nan")
        return {"fillOpacity":0.75,"weight":0,"color":"#0000","fillColor":ramp_color(v,vmin,vmax,GRADIENT)}
    folium.GeoJson(
        setores.to_json(), name=label, pane="vectors", style_function=sty, smooth_factor=1.2,
        tooltip=folium.features.GeoJsonTooltip(fields=[value_col], aliases=[label+": "], sticky=True, labels=False, class_name="pb-big-tooltip"),
    ).add_to(m)
    add_grad_legend(m, label, vmin, vmax, GRADIENT)

def draw_cluster(m, setores, col):
    cmap = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}; other="#c8c8c8"
    lab = {0:"1 - Periférico com predominância residencial de alta densidade construtiva",
           1:"2 - Uso misto de média densidade construtiva",
           2:"3 - Periférico com predominância residencial de média densidade construtiva",
           3:"4 - Verticalizado de uso-misto",
           4:"5 - Predominância de uso comercial e serviços"}

    def sty(f):
        v = f["properties"].get(col)
        try: vi = int(v)
        except Exception: vi = -1
        return {"fillOpacity":0.75,"weight":0,"color":"#0000","fillColor":cmap.get(vi,other)}
    folium.GeoJson(setores.to_json(), name="Cluster (perfil urbano)", pane="vectors", style_function=sty, smooth_factor=1.2).add_to(m)
    items = [(lab[k], cmap[k]) for k in sorted(cmap)]; items.append(("Outros", other))
    add_cat_legend(m, "Cluster (perfil urbano)", items)

def draw_isoc_area(m, iso):
    lut = {0:("Predominância uso misto","#542788"),1:("Zona de transição local","#f7f7f7"),
           2:("Periférico residencial de média densidade","#d8daeb"),3:("Transição central verticalizada","#b35806"),
           4:("Periférico adensado em transição","#b2abd2"),5:("Centralidade comercial e de serviços","#8073ac"),
           6:("Predominância residencial média densidade","#fdb863"),7:("Áreas íngremes e de encosta","#7f3b08"),
           8:("Alta densidade residencial","#e08214"),9:("Central verticalizado","#fee0b6")}
    col = find_col(iso.columns, "nova_class")
    def sty(f):
        v = f["properties"].get(col)
        try: vi = int(v)
        except Exception: vi = -1
        return {"fillOpacity":0.65,"weight":0,"color":"#0000","fillColor":lut.get(vi,("Outros","#c8c8c8"))[1]}
    folium.GeoJson(iso.to_json(), name="Área de influência de bairro", pane="vectors", style_function=sty, smooth_factor=1.2).add_to(m)
    add_cat_legend(m, "Área de influência de bairro (nova_class)", [(f"{k} - {v[0]}", v[1]) for k,v in lut.items()])

# -----------------------------------------------------------------------------
# UI (filtros à esquerda)
# -----------------------------------------------------------------------------
def controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox("Selecione a variável",
        ["— Nenhuma —","População (Pessoa/ha)","Densidade demográfica (hab/ha)","Variação de elevação média","Elevação média","Cluster (perfil urbano)","Área de influência de bairro"],
        index=0, key="var_sel",
        help="Com '— Nenhuma —', nada é carregado.")
    st.markdown("### Configurações")
    lim = st.selectbox("Limites Administrativos",
        ["— Nenhum —","Distritos","ZonasOD2023","Subprefeitura","Isócronas"],
        index=0, key="lim_sel",
        help="Ao trocar, o limite anterior é descarregado.")
    labels = st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="labels_on")
    return {"var":var, "lim":lim, "labels":labels}

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    css()

    # Header
    c1,c2 = st.columns([1,7])
    with c1:
        lp = get_logo()
        if lp: st.image(lp, width=140)
    with c2:
        st.markdown("""<div class="pb-header"><div><div class="pb-title">PlanBairros</div><div>Plataforma de visualização e planejamento em escala de bairro</div></div></div>""", unsafe_allow_html=True)

    a1,_,_,_ = st.tabs(["Aba 1","Aba 2","Aba 3","Aba 4"])
    with a1:
        left, mapc = st.columns([1,5], gap="small")
        with left:
            st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
            ui = controls()
            st.markdown("</div>", unsafe_allow_html=True)

        with mapc:
            if folium is None or st_folium is None or gpd is None:
                st.error("Instale `geopandas`, `folium` e `streamlit-folium` para o mapa.")
                return

            # Descarga explícita se a seleção mudou
            if st.session_state.get("_last_lim") != ui["lim"]:
                st.session_state.pop("_lim_gdf", None)
                st.session_state["_last_lim"] = ui["lim"]
                gc.collect()
            if st.session_state.get("_last_var") != ui["var"]:
                st.session_state.pop("_var_gdf", None)
                st.session_state["_last_var"] = ui["var"]
                gc.collect()

            # Mapa base
            fmap = make_map()
            inject_tooltip_css(fmap, px=34)
            inject_label_scaler(fmap, 14, 24, 9, 18)

            # Carrega limite somente se escolhido
            lim_gdf = st.session_state.get("_lim_gdf")
            if ui["lim"] != "— Nenhum —" and lim_gdf is None:
                lim_gdf = load_limits(ui["lim"])
                st.session_state["_lim_gdf"] = lim_gdf

            # Centraliza no limite, se houver
            if lim_gdf is not None and not lim_gdf.empty:
                fmap.location = center_from_bounds(lim_gdf)

            # Tolerância e bbox (para filtrar variáveis) calculadas só se limite existir
            tol = pick_tol(lim_gdf, None)
            bbox = bounds_tuple(lim_gdf) if lim_gdf is not None else None

            # Desenha limite (sem carregar outros)
            if lim_gdf is not None and not lim_gdf.empty:
                add_outline(fmap, lim_gdf, ui["lim"], tol_m=tol)
                if ui["labels"]: add_centroids(fmap, lim_gdf)
                # Isócronas como "área de transição" nos limites (regra 5)
                if ui["lim"].startswith("Isócron"):
                    cls = find_col(lim_gdf.columns, "nova_class")
                    if cls:
                        trans = lim_gdf[lim_gdf[cls].isin([1,3,4,6])][["geometry"]]
                        if not trans.empty:
                            trans = simplify_3857(trans, tol_m=max(30,tol))
                            folium.GeoJson(
                                trans.to_json(), name="Área de transição", pane="vectors",
                                style_function=lambda f: {"fillOpacity":.2,"color":"#836e60","weight":.8}, smooth_factor=1.2
                            ).add_to(fmap)
                            add_cat_legend(fmap, "Limites — Isócronas", [("Área de transição (1,3,4,6)", "#836e60")], topright=True)

            # Carrega variável somente se escolhida
            var_gdf = st.session_state.get("_var_gdf")
            var = ui["var"]
            if var != "— Nenhuma —" and var_gdf is None:
                if var == "Área de influência de bairro":
                    g = load_limits("Isócronas")  # só carrega iso quando necessário
                    if g is not None and bbox is not None: g = filter_bbox(g, bbox)
                    st.session_state["_var_gdf"] = g
                else:
                    kind = {"População (Pessoa/ha)":"pop","Densidade demográfica (hab/ha)":"dens","Variação de elevação média":"elev_diff","Elevação média":"elev","Cluster (perfil urbano)":"cluster"}[var]
                    g = load_setores_for(kind)
                    if g is not None and bbox is not None: g = filter_bbox(g, bbox)
                    st.session_state["_var_gdf"] = g
                var_gdf = st.session_state.get("_var_gdf")

            # Desenha variável (se houver e somente a atual)
            if var_gdf is not None and not var_gdf.empty:
                tolv = max(15, tol)  # mesma ordem de grandeza do limite
                gsim = simplify_3857(var_gdf, tol_m=tolv, precision_m=1.0)
                if var == "Área de influência de bairro":
                    draw_isoc_area(fmap, gsim)
                elif var == "Cluster (perfil urbano)":
                    col = find_col(gsim.columns, "cluster","cluster_label","classe","label") or "cluster"
                    draw_cluster(fmap, gsim, col)
                else:
                    col = {"População (Pessoa/ha)":"populacao","Densidade demográfica (hab/ha)":"densidade_demografica","Variação de elevação média":"diferenca_elevacao","Elevação média":"elevacao"}[var]
                    col = find_col(gsim.columns, col) or col
                    draw_numeric(fmap, gsim, col, var)

            st_folium(fmap, use_container_width=True, height=850)

if __name__ == "__main__":
    main()
