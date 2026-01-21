# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from unicodedata import normalize as _ud_norm
import math
import re

import pandas as pd
import streamlit as st

# ---- Geo (opcional; sem estas libs o app abre, mas sem mapa) ----
try:
    import geopandas as gpd            # type: ignore
    import folium                      # type: ignore
    from folium import Element         # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    gpd = None         # type: ignore
    folium = None      # type: ignore
    Element = None     # type: ignore
    st_folium = None   # type: ignore


# ============================================================================
# Página e estilo
# ============================================================================
st.set_page_config(page_title="PlanBairros", page_icon="🏙️", layout="wide")

PB = {
    "amarelo": "#F4DD63",
    "verde":   "#B1BF7C",
    "laranja": "#D58243",
    "telha":   "#C65534",
    "teal":    "#6FA097",
    "navy":    "#14407D",
}

# gradiente mais contrastado (quente) para coroplético
HEAT_GRADIENT = ["#fff5b1", "#f7d154", "#f49b3e", "#d55e00", "#7f2704"]

def css():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            html, body, .stApp {{
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background:{PB['navy']}; color:#fff; border-radius:14px; padding:16px 20px;
                display:flex; align-items:center; gap:12px;
            }}
            .pb-title {{ font-size:1.8rem; font-weight:700; }}
            .pb-subtitle {{ opacity:.95; font-size:.95rem; }}
            .pb-card {{
                background:#fff; border:1px solid rgba(20,64,125,.12);
                border-radius:14px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04);
            }}
        </style>
        """, unsafe_allow_html=True
    )

css()


# ============================================================================
# Caminhos e carregamento on-demand
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent

def _resolve_dir(name: str) -> Path:
    # aceita repo raiz OU dash_planbairros/ (para quem clonou diferente)
    for base in (REPO_ROOT, REPO_ROOT / "dash_planbairros"):
        p = base / name
        if p.exists():
            return p
    return REPO_ROOT / name

ADM_DIR = _resolve_dir("limites_administrativos")   # Setores + Isócronas aqui

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _find_parquet(folder: Path, base_names: list[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    want = {_slug(n) for n in base_names}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in want:
            return fp
    return None

@st.cache_data(show_spinner=False)
def _read_gdf(path: Path) -> Optional["gpd.GeoDataFrame"]:
    """Leitura robusta em EPSG:4326 (o seu dado já está em 4326, então só garante)."""
    if gpd is None or path is None or not path.exists():
        return None
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        # fallback: pandas -> geopandas
        try:
            pdf = pd.read_parquet(path)
        except Exception:
            return None
        geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
        if geom_col is None:
            return None
        from shapely import wkb, wkt
        vals = pdf[geom_col]
        geo = (vals.dropna().apply(wkt.loads) if vals.dropna().astype(str).str.startswith("POLY").any()
               else vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str))))
        gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo, crs=4326)

    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def _find_col(cols, *cands) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c and c.lower() in low:
            return low[c.lower()]
    # normalização (tira acento/underscore)
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in low.items()}
    for c in cands:
        key = re.sub(r"[^a-z0-9]", "", (c or "").lower())
        if key in norm:
            return norm[key]
    return None


# ---- Carregadores lazy (on-demand) ------------------------------------------
@st.cache_data(show_spinner=False)
def load_setores() -> Optional["gpd.GeoDataFrame"]:
    p = _find_parquet(ADM_DIR, ["SetoresCensitarios2023"])
    return _read_gdf(p) if p else None

@st.cache_data(show_spinner=False)
def load_isocronas() -> Optional["gpd.GeoDataFrame"]:
    p = _find_parquet(ADM_DIR, ["isocronas", "isócronas"])
    return _read_gdf(p) if p else None

@st.cache_data(show_spinner=False)
def load_limite(name: str) -> Optional["gpd.GeoDataFrame"]:
    # Setores NÃO entram aqui (ficam em Variáveis)
    alias = {
        "Distritos": "Distritos",
        "ZonasOD2023": "ZonasOD2023",
        "Subprefeitura": "subprefeitura",
        "Isócronas": "isocronas",
    }.get(name, name)
    p = _find_parquet(ADM_DIR, [alias])
    return _read_gdf(p) if p else None


# ============================================================================
# UI (filtros à esquerda)
# ============================================================================
def left_panel() -> Dict[str, Any]:
    st.markdown("### Variáveis (Setores Censitários e Isócronas)")
    var = st.selectbox(
        "Selecione a variável",
        [
            "População (Pessoa/ha)",
            "Densidade demográfica (hab/ha)",
            "Variação de elevação média",
            "Elevação média",
            "Cluster (perfil urbano)",
            "Área de influência de bairro",
        ],
        index=1,
        key="pb_var",
        help="Variáveis numéricas pintam os setores; 'Área de influência' usa isócronas."
    )

    st.markdown("### Configurações")
    limite = st.selectbox(
        "Limites Administrativos",
        ["Distritos", "ZonasOD2023", "Subprefeitura", "Isócronas"],
        index=0,
        key="pb_limite",
        help="Contorno sobre o satélite."
    )
    labels_on = st.checkbox(
        "Rótulos permanentes (dinâmicos por zoom)",
        value=False, key="pb_labels",
        help="Mostra rótulos fixos; o tamanho acompanha o zoom."
    )
    return {"var": var, "limite": limite, "labels_on": labels_on}


# ============================================================================
# Folium helpers (mapa à direita)
# ============================================================================
SIMPLIFY_TOL = 0.0005  # ~55m

def make_map(center=(-23.55, -46.63), zoom=11, tiles_opacity=0.5):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    try:
        folium.map.CustomPane("vectors", z_index=650).add_to(m)
    except Exception:
        pass
    # Fundo satélite (sem labels) para dar contraste
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False, control=False, opacity=tiles_opacity
    ).add_to(m)
    return m

def inject_big_tooltip_css(m, px=44):
    if Element is None: return
    css = f"""
    <style>
      .leaflet-tooltip.pb-big-tooltip, .leaflet-tooltip.pb-big-tooltip * {{
        font-size:{px}px!important; font-weight:800!important; color:#111!important; line-height:1!important;
      }}
      .leaflet-tooltip.pb-big-tooltip {{
        background:#fff!important; border:2px solid #222!important; border-radius:10px!important;
        padding:10px 14px!important; white-space:nowrap!important; box-shadow:0 2px 6px rgba(0,0,0,.2)!important;
        z-index:200000!important;
      }}
    </style>"""
    m.get_root().html.add_child(Element(css))

def inject_label_scaler(m, min_px=14, max_px=24, min_z=9, max_z=18):
    if Element is None: return
    mv = m.get_name()
    js = f"""
    <script>
      (function(){{
        var minZ={min_z}, maxZ={max_z}, minPx={min_px}, maxPx={max_px};
        function lerp(z) {{ z=Math.max(minZ, Math.min(maxZ,z)); return Math.round(minPx + (z-minZ)*(maxPx-minPx)/(maxZ-minZ)); }}
        function upd(){{
          var px=lerp({mv}.getZoom());
          document.querySelectorAll('.pb-static-label').forEach(e => e.style.fontSize = px + 'px');
        }}
        {mv}.on('zoomend', upd); {mv}.whenReady(upd);
      }})();
    </script>
    """
    m.get_root().html.add_child(Element(js))

def bounds_center(gdf) -> Tuple[float,float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return ((miny+maxy)/2, (minx+maxx)/2)

def add_outline(m, gdf, label_col: Optional[str] = None, color="#000", weight=1.2):
    if gdf is None or folium is None: return
    gl = gdf[["geometry"]].copy()
    gl["geometry"] = gl.geometry.boundary
    try: gl["geometry"] = gl.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    except Exception: pass
    folium.GeoJson(
        gl.to_json(), name="limite", pane="vectors",
        style_function=lambda f: {"fillOpacity":0, "color":color, "weight":weight},
    ).add_to(m)

    if label_col:
        ga = gdf[[label_col, "geometry"]].copy()
        try: ga["geometry"] = ga.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
        except Exception: pass
        folium.GeoJson(
            ga.to_json(), name="hover", pane="vectors",
            style_function=lambda f: {"fillOpacity":0.05, "opacity":0, "weight":0, "color":"#0000"},
            tooltip=folium.features.GeoJsonTooltip(
                fields=[label_col], aliases=[""], sticky=True, labels=False,
                class_name="pb-big-tooltip"
            )
        ).add_to(m)

def add_fixed_labels(m, gdf, label_col: str):
    if folium is None: return
    pts = gdf.copy().to_crs(4326).representative_point()
    for name, pt in zip(gdf[label_col].astype(str), pts):
        html = ("<div class='pb-static-label' style=\"font:600 12px/1 Roboto, Segoe UI, Helvetica, Arial;"
                "color:#111;text-shadow:0 0 2px #fff,0 0 6px #fff;white-space:nowrap;\">"+name+"</div>")
        folium.Marker([pt.y, pt.x], icon=folium.DivIcon(html=html, icon_size=(0,0), icon_anchor=(0,0)),
                      z_index_offset=1000).add_to(m)

def _ramp_hex(v: float, vmin: float, vmax: float, colors: list[str]) -> str:
    if v is None or pd.isna(v): return "#c8c8c8"
    t = 0.0 if vmax==vmin else (float(v) - vmin)/(vmax - vmin)
    t = max(0.0, min(1.0, t))
    n = len(colors)-1
    i = min(int(t*n), n-1)
    f = (t*n) - i
    def h2r(h): h=h.lstrip("#"); return tuple(int(h[k:k+2],16) for k in (0,2,4))
    def r2h(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
    c1, c2 = h2r(colors[i]), h2r(colors[i+1])
    mix = tuple(int(c1[k] + f*(c2[k]-c1[k])) for k in range(3))
    return r2h(mix)

def legend_gradient(m, title: str, vmin: float, vmax: float, colors: list[str]):
    if Element is None: return
    grad_css = f"background: linear-gradient(to top, {', '.join(colors)});"
    html = f"""
    <div style="position:absolute;left:16px;bottom:20px;z-index:9999;background:rgba(255,255,255,.95);
                padding:10px 12px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.25);font:500 12px Roboto;">
      <div style="font-weight:700;margin-bottom:6px">{title}</div>
      <div style="display:flex;align-items:stretch;gap:10px">
        <div style="width:18px;height:120px;{grad_css};border-radius:4px"></div>
        <div style="display:flex;flex-direction:column;justify-content:space-between;height:120px">
          <div>{vmax:,.0f}</div><div>{(vmin+vmax)/2:,.0f}</div><div>{vmin:,.0f}</div>
        </div>
      </div>
    </div>"""
    m.get_root().html.add_child(Element(html))

def legend_cat(m, title: str, items: list[tuple[str,str]], topright=False):
    if Element is None: return
    pos = "right:16px;top:20px" if topright else "left:16px;bottom:20px"
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
        f"<span style='width:16px;height:16px;border-radius:3px;background:{c};display:inline-block'></span>"
        f"<span>{t}</span></div>"
        for t,c in items
    )
    html = f"""
    <div style="position:absolute;{pos};z-index:9999;background:rgba(255,255,255,.95);
                padding:10px 12px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.25);font:500 12px Roboto;">
      <div style="font-weight:700;margin-bottom:6px">{title}</div>{rows}
    </div>"""
    m.get_root().html.add_child(Element(html))

def paint_numeric(m, setores: "gpd.GeoDataFrame", value_col: str, label: str):
    # garante tipo numérico simples (evita Decimal/Int64 na serialização)
    s = pd.to_numeric(setores[value_col], errors="coerce").astype(float)
    vmin, vmax = float(s.min()), float(s.max())
    df = setores[[value_col, "geometry"]].copy()
    df["__v__"] = s

    folium.GeoJson(
        df.to_json(), name=label, pane="vectors",
        style_function=lambda f: {
            "fillOpacity": 0.78,
            "weight": 0.0,
            "color": "#00000000",
            "fillColor": _ramp_hex(f["properties"].get("__v__"), vmin, vmax, HEAT_GRADIENT),
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=[value_col], aliases=[label + ": "], sticky=True,
            class_name="pb-big-tooltip", labels=False,
        ),
    ).add_to(m)
    legend_gradient(m, label, vmin, vmax, HEAT_GRADIENT)

def paint_cluster(m, setores: "gpd.GeoDataFrame", cluster_col: str):
    lut = {
        0: ("1 - Periférico com predominância residencial de alta densidade construtiva", "#bf7db2"),
        1: ("2 - Uso misto de média densidade construtiva", "#f7bd6a"),
        2: ("3 - Periférico com predominância residencial de média densidade construtiva", "#cf651f"),
        3: ("4 - Verticalizado de uso-misto", "#ede4e6"),
        4: ("5 - Predominância de uso comercial e serviços", "#793393"),
    }
    df = setores[[cluster_col, "geometry"]].copy()
    df["__c__"] = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")

    folium.GeoJson(
        df.to_json(), name="Cluster (perfil urbano)", pane="vectors",
        style_function=lambda f: {
            "fillOpacity": 0.78,
            "weight": 0.0,
            "color": "#00000000",
            "fillColor": lut.get(int(f["properties"].get("__c__")) if f["properties"].get("__c__") is not None else -1,
                                 ("Outros", "#c8c8c8"))[1],
        },
    ).add_to(m)
    legend_cat(m, "Cluster (perfil urbano)", [(v[0], v[1]) for _, v in lut.items()] + [("Outros", "#c8c8c8")])

def paint_isocronas(m, iso: "gpd.GeoDataFrame"):
    cls = _find_col(iso.columns, "nova_class")
    if cls is None:
        st.info("Coluna 'nova_class' não encontrada em isócronas.")
        return
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
    df = iso[[cls, "geometry"]].copy()
    df["__k__"] = pd.to_numeric(df[cls], errors="coerce").astype("Int64")

    folium.GeoJson(
        df.to_json(), name="Área de influência de bairro", pane="vectors",
        style_function=lambda f: {
            "fillOpacity": 0.65, "weight": 0.0, "color": "#00000000",
            "fillColor": lut.get(int(f["properties"].get("__k__")) if f["properties"].get("__k__") is not None else -1,
                                 ("Outros", "#c8c8c8"))[1],
        },
    ).add_to(m)
    legend_cat(m, "Área de influência de bairro (nova_class)",
               [(f"{k} - {v[0]}", v[1]) for k, v in lut.items()] + [("Outros", "#c8c8c8")])


# ============================================================================
# App (filtros à esquerda, mapa à direita)
# ============================================================================
def main():
    # Header
    with st.container():
        c1, c2 = st.columns([1, 7])
        with c1: st.image("https://raw.githubusercontent.com/streamlit/brand/master/logo/mark/streamlit-mark-color.png", width=64)
        with c2:
            st.markdown(
                "<div class='pb-header'><div><div class='pb-title'>PlanBairros</div>"
                "<div class='pb-subtitle'>Plataforma de visualização e planejamento em escala de bairro</div></div></div>",
                unsafe_allow_html=True,
            )

    # ====== layout: filtros à ESQUERDA, mapa à DIREITA ======
    left, right = st.columns([1, 4], gap="large")
    with left:
        st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
        ui = left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if folium is None or st_folium is None or gpd is None:
            st.error("Instale `geopandas`, `folium` e `streamlit-folium`.")
            return

        # Carrega SOMENTE o necessário (lazy)
        limite_gdf = load_limite(ui["limite"])
        setores = None
        iso = None

        if ui["var"] == "Área de influência de bairro":
            iso = load_isocronas()
        elif ui["var"] == "Cluster (perfil urbano)" or ui["var"]:
            setores = load_setores()

        # Centro do mapa
        center = (-23.55, -46.63)
        for gdf in (limite_gdf, setores, iso):
            if gdf is not None and len(gdf) > 0:
                center = bounds_center(gdf)
                break

        m = make_map(center=center, zoom=11, tiles_opacity=0.5)
        inject_big_tooltip_css(m, px=44)
        inject_label_scaler(m, min_px=14, max_px=24, min_z=9, max_z=18)

        # Limite (contorno + hover)
        if limite_gdf is not None:
            name_col = (
                _find_col(limite_gdf.columns, "ds_nome") or
                _find_col(limite_gdf.columns, "sp_nome") or
                _find_col(limite_gdf.columns, "nome")
            )
            add_outline(m, limite_gdf, label_col=name_col, color="#000", weight=1.2)
            if ui["labels_on"] and name_col:
                add_fixed_labels(m, limite_gdf, name_col)

            # regra especial: área de transição para isócronas nos LIMITES
            if _find_col(limite_gdf.columns, "nova_class"):
                trans = limite_gdf[limite_gdf[_find_col(limite_gdf.columns, "nova_class")].isin([1,3,4,6])]
                if not trans.empty:
                    folium.GeoJson(
                        trans[["geometry"]].to_json(), name="Área de transição", pane="vectors",
                        style_function=lambda f: {"fillOpacity":0.20, "color":"#836e60", "weight":0.8}
                    ).add_to(m)
                    legend_cat(m, "Limites — Isócronas", [("Área de transição (1,3,4,6)", "#836e60")], topright=True)

        # Variável (coroplético)
        if ui["var"] == "Área de influência de bairro":
            if iso is None: st.info("Isócronas não encontradas."); 
            else: paint_isocronas(m, iso)

        elif ui["var"] == "Cluster (perfil urbano)":
            if setores is None: st.info("Setores não encontrados.")
            else:
                cl = _find_col(setores.columns, "cluster", "cluster_label", "label", "classe")
                if cl is None: st.info("Coluna de cluster não encontrada nos setores.")
                else: paint_cluster(m, setores, cl)

        else:
            # numéricas dos setores
            if setores is None:
                st.info("Setores não encontrados.")
            else:
                mapping = {
                    "População (Pessoa/ha)": "populacao",
                    "Densidade demográfica (hab/ha)": "densidade_demografica",
                    "Variação de elevação média": "diferenca_elevacao",
                    "Elevação média": "elevacao",
                }
                col = _find_col(setores.columns, mapping[ui["var"]])
                if col is None:
                    st.info(f"Coluna para '{ui['var']}' não encontrada em Setores.")
                else:
                    paint_numeric(m, setores, col, ui["var"])

        # Render
        st_folium(m, use_container_width=True, height=820, key="map_main")

    # Abas 2–4 (placeholders)
    a2, a3, a4 = st.tabs(["Aba 2", "Aba 3", "Aba 4"])
    with a2: st.info("Conteúdo a definir.")
    with a3: st.info("Conteúdo a definir.")
    with a4: st.info("Conteúdo a definir.")

if __name__ == "__main__":
    main()
