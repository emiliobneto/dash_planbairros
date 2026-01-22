# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from unicodedata import normalize as _ud_norm
import base64, math, re
from decimal import Decimal

import pandas as pd
import streamlit as st

# ---- Fail-fast: geostack obrigatório
def _import_geostack():
    try:
        import geopandas as gpd
        import folium
        from branca.element import Element
        from streamlit_folium import st_folium
        from pyarrow import parquet as pq
        return gpd, folium, Element, st_folium, pq
    except ImportError as e:
        st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")
        st.error(f"Dependência ausente: **{e}**. Instale os pacotes de requirements.txt e reinicie.")
        st.stop()

gpd, folium, Element, st_folium, pq = _import_geostack()

# ---- Page config
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

PB_COLORS = {"amarelo":"#F4DD63","verde":"#B1BF7C","laranja":"#D58243","telha":"#C65534","teal":"#6FA097","navy":"#14407D"}
ORANGE_RED_GRAD = ["#fff7ec","#fee8c8","#fdd49e","#fdbb84","#fc8d59","#e34a33","#b30000"]
SIMPLIFY_M = 35  # simplificar geometria ~35m
PLACEHOLDER_VAR = "— selecione uma variável —"
PLACEHOLDER_LIM = "— selecione o limite —"

# ---- CSS
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        :root {{
          --pb-navy: {PB_COLORS['navy']};
        }}
        html, body, .stApp {{ font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }}
        .main .block-container {{ padding-top:.6rem; padding-bottom:.8rem; }}

        /* header: barra azul só no bloco de texto (coluna direita) */
        .pb-header-text {{
          background: var(--pb-navy); color:#fff; border-radius: 14px;
          padding: 16px 20px; width: 100%; box-sizing: border-box;
        }}
        .pb-title   {{ font-size: 3.8rem; font-weight: 800; letter-spacing: .2px; line-height:1.05; }}
        .pb-subtitle{{ font-size: 1.9rem; opacity:.95; margin-top:6px; }}

        .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.10); box-shadow:0 1px 2px rgba(0,0,0,.04);
                   border-radius:14px; padding:12px; }}

        /* tooltips grandes */
        .leaflet-tooltip.pb-big-tooltip, .leaflet-tooltip.pb-big-tooltip * {{
          font-size: 28px !important; font-weight: 800 !important; color:#111 !important; line-height:1 !important;
        }}
        .leaflet-tooltip.pb-big-tooltip {{
          background:#fff !important; border:2px solid #222 !important; border-radius:10px !important;
          padding:10px 14px !important; white-space:nowrap !important; pointer-events:none !important;
          box-shadow:0 2px 6px rgba(0,0,0,.2) !important; z-index:200000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- Paths & utils
try: REPO_ROOT = Path(__file__).resolve().parent
except NameError: REPO_ROOT = Path.cwd()

DATA_DIR = (REPO_ROOT / "limites_administrativos") if (REPO_ROOT / "limites_administrativos").exists() else REPO_ROOT

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII","ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+","", s2.strip().lower())

def _first_parquet_by_stems(folder: Path, stems: List[str]) -> Optional[Path]:
    if not folder.exists(): return None
    wanted = {_slug(n) for n in stems}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

def find_col(cols, *cands) -> Optional[str]:
    low = {c.lower():c for c in cols}
    norm = {re.sub(r"[^a-z0-9]","",k): v for k,v in low.items()}
    for c in cands:
        if not c: continue
        k = c.lower(); if1 = low.get(k); 
        if if1: return if1
        k2 = re.sub(r"[^a-z0-9]","",k)
        if k2 in norm: return norm[k2]
    return None

def to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        return s.apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype("Float64")
    return pd.to_numeric(s, errors="coerce").astype("Float64")

def center_from_bounds(gdf) -> tuple[float, float]:
    minx,miny,maxx,maxy = gdf.total_bounds
    return ((miny+maxy)/2, (minx+maxx)/2)

# ---- Logo
def _logo_data_uri() -> str:
    cand = ["logo_todos.jpg"]
    assets = REPO_ROOT / "assets"
    if assets.exists():
        for n in cand:
            p = assets / n
            if p.exists():
                mime = "image/" + p.suffix.lower().lstrip(".")
                import base64
                return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"

# ============================================================================
# I/O eficiente: leitura seletiva + caches
# ----------------------------------------------------------------------------
# sniff de colunas sem carregar tudo
@st.cache_data(show_spinner=False, ttl=3600)
def _list_columns(parquet_path: Path) -> List[str]:
    return [pq.ParquetFile(str(parquet_path)).schema.names]

# leitura geopandas só com as colunas pedidas (quando possível)
@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def _read_geo_cols(parquet_path: Path, columns: List[str]) -> "gpd.GeoDataFrame|None":
    try:
        gdf = gpd.read_parquet(parquet_path, columns=columns)  # pyarrow engine
        if gdf.crs is None: gdf = gdf.set_crs(4326)
        return gdf.to_crs(4326)
    except Exception:
        # fallback: pyarrow -> pandas -> shapely
        try:
            table = pq.read_table(str(parquet_path), columns=columns)
            pdf = table.to_pandas()
            from shapely import wkb, wkt
            geom_col = find_col(pdf.columns, "geometry","geom","wkb","wkt")
            if geom_col is None: return None
            vals = pdf[geom_col]
            if vals.dropna().astype(str).str.startswith("POLY").any():
                geo = vals.dropna().apply(wkt.loads)
            else:
                geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b,str)))
            return gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)
        except Exception:
            return None

# geometria dos setores (apenas id+geometry) + simplificação métrica
@st.cache_data(show_spinner=False, ttl=3600)
def load_setores_geom(simplify_m: float = SIMPLIFY_M) -> tuple[Optional["gpd.GeoDataFrame"], Optional[str]]:
    p = _first_parquet_by_stems(DATA_DIR, ["SetoresCensitarios2023","SetoresCensitarios","setores_geom"])
    if not p: return None, None
    # descobrir id
    cols = pq.ParquetFile(str(p)).schema.names
    id_col = find_col(cols, "id","cd_setor","codigo","geocodigo","geocod","id_setor")
    needed = [c for c in [id_col,"geometry"] if c]
    gdf = _read_geo_cols(p, needed)  # só id + geometry
    if gdf is None: return None, None
    # simplificação métrica (projeta -> simplifica em metros -> volta)
    try:
        gdfm = gdf.to_crs(3857)
        gdfm["geometry"] = gdfm.geometry.simplify(simplify_m, preserve_topology=True)
        gdf = gdfm.to_crs(4326)
    except Exception:
        pass
    return gdf, id_col

# métrica on-demand (lê só a coluna pedida)
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_metric_column(var_name: str, id_col_hint: Optional[str]) -> Optional[pd.DataFrame]:
    # tenta um arquivo só de métricas; se não houver, usa o mesmo dos setores
    p_metrics = _first_parquet_by_stems(DATA_DIR, ["setores_metrics","setores_metricas","metricas_setores"])
    p_all     = _first_parquet_by_stems(DATA_DIR, ["SetoresCensitarios2023","SetoresCensitarios"])
    p = p_metrics or p_all
    if not p: return None
    cols = pq.ParquetFile(str(p)).schema.names
    id_col = id_col_hint or find_col(cols, "id","cd_setor","codigo","geocodigo","geocod","id_setor")
    # mapeia nomes amigáveis -> nomes no arquivo
    name_map = {
        "População (Pessoa/ha)": "populacao",
        "Densidade demográfica (hab/ha)": "densidade_demografica",
        "Variação de elevação média": "diferenca_elevacao",
        "Elevação média": "elevacao",
        "Cluster (perfil urbano)": "cluster",
    }
    wanted = find_col(cols, name_map.get(var_name, var_name))
    if not (id_col and wanted): return None
    table = pq.read_table(str(p), columns=[id_col, wanted])
    pdf = table.to_pandas()
    pdf.rename(columns={wanted: "__value__"}, inplace=True)
    return pdf

# demais camadas
@st.cache_data(show_spinner=False, ttl=3600)
def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _first_parquet_by_stems(DATA_DIR, ["isocronas","isócronas","isocronas2023"])
    if not p: return None
    return _read_geo_cols(p, ["geometry","nova_class"])

@st.cache_data(show_spinner=False, ttl=3600)
def load_admin_layer(name: str) -> Optional["gpd.GeoDataFrame"]:
    stems = {"Distritos":["Distritos"],"ZonasOD2023":["ZonasOD2023","ZonasOD"],"Subprefeitura":["Subprefeitura","subprefeitura"],"Isócronas":["isocronas","isócronas"]}.get(name,[name])
    p = _first_parquet_by_stems(DATA_DIR, stems)
    if not p: return None
    return _read_geo_cols(p, ["geometry"])

# ============================================================================
# Folium helpers
def make_satellite_map(center=(-23.55,-46.63), zoom=11, tiles_opacity=0.6):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="© OpenStreetMap", name="OSM", overlay=False, control=False).add_to(m)
    try:
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri World Imagery", name="Esri Satellite", overlay=False, control=False, opacity=tiles_opacity).add_to(m)
    except Exception: pass
    try: folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception: pass
    return m

def inject_label_scaler(m, min_px=14, max_px=26, min_zoom=9, max_zoom=18):
    js = f"""
    <script>(function(){{var minZ={min_zoom},maxZ={max_zoom},minPx={min_px},maxPx={max_px};
    function s(z){{z=Math.max(minZ,Math.min(maxZ,z));var t=(z-minZ)/(maxZ-minZ);return Math.round(minPx+t*(maxPx-minPx));}}
    function u(){{var z={m.get_name()}.getZoom(),px=s(z);document.querySelectorAll('.pb-static-label').forEach(el=>el.style.fontSize=px+'px');}}
    {m.get_name()}.on('zoomend',u);{m.get_name()}.whenReady(u);}})();</script>"""
    m.get_root().html.add_child(Element(js))

def add_admin_outline(m, gdf, layer_name: str, color="#000", weight=1.2):
    line = gdf[["geometry"]].copy()
    try: line["geometry"] = line.geometry.boundary.simplify(0.0005, preserve_topology=True)
    except Exception: line["geometry"] = line.geometry.boundary
    folium.GeoJson(data=line.to_json(), name=f"{layer_name} (contorno)", pane="vectors",
                   style_function=lambda f: {"fillOpacity":0,"color":color,"weight":weight}).add_to(m)

def add_categorical_legend(m, title: str, items: list[tuple[str,str]], topright: bool=False):
    pos = "right: 16px; top: 24px;" if topright else "left: 16px; bottom: 24px;"
    rows = "".join(f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0"><span style="width:16px;height:16px;border-radius:3px;background:{c};display:inline-block"></span><span>{l}</span></div>' for l,c in items)
    html = f"""<div style="position:absolute; {pos} z-index:9999; background:rgba(255,255,255,.95); padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25); font:500 12px Roboto,sans-serif;">
    <div style="font-weight:700; margin-bottom:6px">{title}</div>{rows}</div>"""
    m.get_root().html.add_child(Element(html))

def add_gradient_legend(m, title: str, vmin: float, vmax: float, colors: list[str]):
    grad = f"background: linear-gradient(to top, {', '.join(colors)});"
    html = f"""<div style="position:absolute; left:16px; bottom:24px; z-index:9999; background:rgba(255,255,255,.95); padding:10px 12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,.25); font:500 12px Roboto,sans-serif;">
    <div style="font-weight:700; margin-bottom:6px">{title}</div>
    <div style="display:flex; gap:10px;"><div style="width:18px; height:120px; {grad}; border-radius:4px;"></div>
    <div style="display:flex; flex-direction:column; justify-content:space-between;"><div>{vmax:,.0f}</div><div>{(vmin+vmax)/2:,.0f}</div><div>{vmin:,.0f}</div></div></div></div>"""
    m.get_root().html.add_child(Element(html))

def ramp_color(v: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if v is None or (isinstance(v,float) and math.isnan(v)): return "#c8c8c8"
    t = 0.0 if vmax==vmin else (float(v)-vmin)/(vmax-vmin); t = min(1.0, max(0.0, t))
    n = len(colors)-1; i = min(int(t*n), n-1); frac = (t*n)-i
    h2r = lambda h: tuple(int(h.lstrip("#")[k:k+2],16) for k in (0,2,4))
    r2h = lambda r: "#{:02x}{:02x}{:02x}".format(*r)
    c1,c2 = h2r(colors[i]), h2r(colors[i+1])
    mix = tuple(int(c1[k] + frac*(c2[k]-c1[k])) for k in range(3))
    return r2h(mix)

def paint_setores_numeric(m, geoms: "gpd.GeoDataFrame", value: pd.Series, label: str):
    s = to_float_series(value); vmin, vmax = float(s.min()), float(s.max())
    df = geoms[["geometry"]].copy(); df["__v__"] = s.reset_index(drop=True)
    def style_fn(feat):
        v = feat["properties"].get("__v__")
        return {"fillOpacity":0.8,"weight":0.0,"color":"#00000000","fillColor": ramp_color(v, vmin, vmax, ORANGE_RED_GRAD)}
    folium.GeoJson(data=df[["geometry","__v__"]].to_json(), name=label, pane="vectors",
                   style_function=style_fn,
                   tooltip=folium.features.GeoJsonTooltip(fields=["__v__"], aliases=[label+": "],
                                                          sticky=True, labels=False, class_name="pb-big-tooltip")).add_to(m)
    add_gradient_legend(m, label, vmin, vmax, ORANGE_RED_GRAD)

def paint_setores_cluster(m, geoms: "gpd.GeoDataFrame", cluster: pd.Series):
    color_map = {0:"#bf7db2",1:"#f7bd6a",2:"#cf651f",3:"#ede4e6",4:"#793393"}
    label_map = {0:"1 - Periférico com predominância residencial de alta densidade construtiva",
                 1:"2 - Uso misto de média densidade construtiva",
                 2:"3 - Periférico com predominância residencial de média densidade construtiva",
                 3:"4 - Verticalizado de uso-misto", 4:"5 - Predominância de uso comercial e serviços"}
    default_color="#c8c8c8"
    df = geoms[["geometry"]].copy(); c = pd.to_numeric(cluster, errors="coerce").astype("Int64"); df["__c__"]=c.reset_index(drop=True)
    def style_fn(feat):
        v = feat["properties"].get("__c__"); col = color_map.get(int(v) if v is not None and not pd.isna(v) else -1, default_color)
        return {"fillOpacity":0.75,"weight":0.0,"color":"#00000000","fillColor":col}
    folium.GeoJson(data=df[["geometry","__c__"]].to_json(), name="Cluster (perfil urbano)", pane="vectors", style_function=style_fn).add_to(m)
    items = [(label_map[k], color_map[k]) for k in sorted(color_map)]; items.append(("Outros", default_color))
    add_categorical_legend(m, "Cluster (perfil urbano)", items)

def paint_isocronas_area(m, iso: "gpd.GeoDataFrame"):
    cls = find_col(iso.columns,"nova_class")
    if not cls: st.info("A coluna 'nova_class' não foi encontrada em isócronas."); return
    lut = {0:("Predominância uso misto","#542788"),1:("Zona de transição local","#f7f7f7"),
           2:("Periférico residencial média densidade","#d8daeb"),3:("Transição central verticalizada","#b35806"),
           4:("Periférico adensado em transição","#b2abd2"),5:("Centralidade comercial e serviços","#8073ac"),
           6:("Predominância residencial média densidade","#fdb863"),7:("Áreas íngremes e de encosta","#7f3b08"),
           8:("Alta densidade residencial","#e08214"),9:("Central verticalizado","#fee0b6")}
    df = iso[["geometry"]].copy(); k = pd.to_numeric(iso[cls], errors="coerce").astype("Int64"); df["__k__"]=k
    def style_fn(feat):
        v = feat["properties"].get("__k__"); color = lut.get(int(v) if v is not None and not pd.isna(v) else -1, ("Outros","#c8c8c8"))[1]
        return {"fillOpacity":0.65,"weight":0.0,"color":"#00000000","fillColor":color}
    folium.GeoJson(data=df[["geometry","__k__"]].to_json(), name="Área de influência de bairro", pane="vectors", style_function=style_fn).add_to(m)
    items = [(f"{k} - {v[0]}", v[1]) for k,v in lut.items()]; add_categorical_legend(m, "Área de influência de bairro (nova_class)", items)

# ============================================================================
# UI
def left_controls() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox("Selecione a variável",
                       [PLACEHOLDER_VAR,"População (Pessoa/ha)","Densidade demográfica (hab/ha)","Variação de elevação média","Elevação média","Cluster (perfil urbano)","Área de influência de bairro"],
                       index=0, key="pb_var", placeholder="Escolha…")
    st.markdown("### Configurações")
    limite = st.selectbox("Limites Administrativos",
                          [PLACEHOLDER_LIM,"Distritos","ZonasOD2023","Subprefeitura","Isócronas"],
                          index=0, key="pb_limite", placeholder="Escolha…")
    labels_on = st.checkbox("Rótulos permanentes (dinâmicos por zoom)", value=False, key="pb_labels_on")

    st.caption("Use o botão abaixo para limpar os caches de dados em memória.")
    if st.button("🧹 Limpar cache de dados", type="secondary"):
        st.cache_data.clear(); st.success("Cache limpo. Selecione novamente a camada/variável.")

    sel_now = {"var": var, "lim": limite}
    sel_prev = st.session_state.get("_pb_prev_sel")
    if sel_prev and (sel_prev["var"] != sel_now["var"] or sel_prev["lim"] != sel_now["lim"]):
        st.cache_data.clear()
    st.session_state["_pb_prev_sel"] = sel_now
    return {"variavel": var, "limite": limite, "labels_on": labels_on}

# ============================================================================
# App
def main() -> None:
    inject_css()

    # Header: logo (coluna esquerda) e barra azul (direita)
    c1, c2 = st.columns([1, 9], gap="small")
    with c1:
        st.image(_logo_data_uri(), width=110)
    with c2:
        st.markdown(
            f"""
            <div class="pb-header-text">
              <div class="pb-title">PlanBairros</div>
              <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, map_col = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col:
        center = (-23.55,-46.63)
        fmap = make_satellite_map(center=center, zoom=11, tiles_opacity=0.6)

        limite_gdf = None
        if ui["limite"] != PLACEHOLDER_LIM:
            limite_gdf = load_admin_layer(ui["limite"])
            if limite_gdf is not None and len(limite_gdf) > 0:
                center = center_from_bounds(limite_gdf)
                fmap.location = center

        inject_label_scaler(fmap, min_px=14, max_px=26, min_zoom=9, max_zoom=18)

        if limite_gdf is not None and len(limite_gdf) > 0:
            add_admin_outline(fmap, limite_gdf, ui["limite"])
            if ui["labels_on"]:
                # rótulos centróides se quiser
                pass

        var = ui["variavel"]
        if var == PLACEHOLDER_VAR:
            st.info("Selecione uma variável e/ou um limite administrativo para iniciar a visualização.")
        elif var == "Área de influência de bairro":
            iso = load_isocronas()
            if iso is None or len(iso) == 0:
                st.info("Isócronas não encontradas.")
            else:
                if ui["limite"] == PLACEHOLDER_LIM:
                    fmap.location = center_from_bounds(iso)
                paint_isocronas_area(fmap, iso)
        else:
            # setores: geometria cacheada + métrica on-demand
            geoms, id_col = load_setores_geom(SIMPLIFY_M)
            if geoms is None or id_col is None:
                st.info("Setores não encontrados.")
            else:
                metric_df = load_metric_column(var, id_col)
                if metric_df is None:
                    st.info(f"Coluna para '{var}' não encontrada.")
                else:
                    joined = geoms.reset_index(drop=True).join(metric_df.set_index(id_col), on=id_col)
                    if var == "Cluster (perfil urbano)":
                        paint_setores_cluster(fmap, joined, "__value__")
                    else:
                        paint_setores_numeric(fmap, joined, "__value__", var)

        st_folium(fmap, height=780, use_container_width=True, key="map_view")

if __name__ == "__main__":
    main()
