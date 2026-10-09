# -*- coding: utf-8 -*-
"""PlanBairros — Dashboard Streamlit/Folium."""
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="PlanBairros", layout="wide")

try:
    import geopandas as gpd
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    import folium
    from folium.plugins import Draw
    from branca.element import Element, JavascriptLink, MacroElement, Template
    from streamlit_folium import st_folium
except ImportError as e:  # pragma: no cover
    st.error(f"Este app requer `geopandas`, `folium` e `streamlit-folium`: {e}")
    st.stop()

# =============================================================================
# CAMINHOS (relativos ao repositório — funciona local e no Streamlit Cloud)
# =============================================================================
APP_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "limites_administrativos").is_dir():
            return p
    return start


REPO_ROOT = _find_repo_root(APP_DIR)
LIMITES_DIR = REPO_ROOT / "limites_administrativos"
CACHE_DIR = REPO_ROOT / "data_cache"
SEARCH_DIRS = [
    LIMITES_DIR,
    LIMITES_DIR / "data",
    LIMITES_DIR / "tematicos",
    REPO_ROOT / "data",
    REPO_ROOT,
    APP_DIR,
    CACHE_DIR,
]

CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

ADMIN_FILES = {
    "subpref": "subprefeitura.parquet",
    "distrito": "Distritos.parquet",
    "iso": "isocronas.parquet",
    "censo": "SetoresCensitarios2023.parquet",
    "od": "ZonasOD2023.parquet",
    "quadras": "Quadras.parquet",
    "lotes": "Lotes.parquet",
}
ADMIN_STYLE = {  # cor, espessura, título
    "subpref": ("#1a237e", 2.5, "Subprefeituras"),
    "distrito": ("#4a148c", 2.0, "Distritos"),
    "iso": ("#d84315", 2.0, "Isócronas"),
    "censo": ("#455a64", 0.8, "Setores censitários"),
    "od": ("#00838f", 1.2, "Zonas OD"),
    "quadras": ("#6d4c41", 0.6, "Quadras"),
    "lotes": ("#8d6e63", 0.4, "Lotes"),
}


# =============================================================================
# LEITURA DE ARQUIVOS
# =============================================================================
@st.cache_data(show_spinner=False)
def _file_index() -> dict:
    idx = {}
    for d in SEARCH_DIRS:
        if d.is_dir():
            for f in d.glob("*.parquet"):
                idx.setdefault(f.name.lower(), str(f))
    if LIMITES_DIR.is_dir():
        for f in LIMITES_DIR.rglob("*.parquet"):
            idx.setdefault(f.name.lower(), str(f))
    return idx


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def _download_drive(key: str, fname: str):
    fid = _secret(f"PB_{key.upper()}_FILE_ID")
    if not fid:
        return None
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / fname
    if out.exists():
        return out
    url = f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def find_file(key: str, fname: str):
    p = _file_index().get(fname.lower())
    if p:
        return Path(p)
    try:
        return _download_drive(key, fname)
    except Exception:
        return None


def _fix_crs(gdf):
    if gdf.crs is None:
        minx = gdf.total_bounds[0]
        gdf = gdf.set_crs(31983 if abs(minx) > 180 else 4326, allow_override=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


@st.cache_resource(show_spinner="Carregando camada...")
def read_layer(key: str, fname: str):
    path = find_file(key, fname)
    if path is None:
        return None
    try:
        gdf = gpd.read_parquet(path)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        return _fix_crs(gdf)
    except Exception:
        return pd.read_parquet(path)  # tabela sem geometria


def require(key: str):
    gdf = read_layer(key, ADMIN_FILES[key])
    if gdf is None:
        dirs = "\n".join(f"- `{d}`" for d in SEARCH_DIRS)
        st.error(
            f"Camada **{key}** (`{ADMIN_FILES[key]}`) não encontrada.\n\n"
            f"Raiz do repositório detectada: `{REPO_ROOT}`\n\nPastas verificadas:\n{dirs}"
        )
    return gdf


def name_col(gdf, prefs=()):
    for c in prefs:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if c != "geometry" and re.search(r"(nome|nm_|name|sigla)", c, re.I):
            return c
    for c in gdf.columns:
        if c != "geometry" and gdf[c].dtype == object:
            return c
    return gdf.columns[0]


# =============================================================================
# CAMADAS TEMÁTICAS
# =============================================================================
def _num(v):
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None


def cls_corredor(v):
    s = str(v).lower()
    if "poliniz" in s:
        return "Corredor polinizador", "#f9a825"
    return "Corredor verde", "#2e7d32"


FAVELA = {
    "sem informação": "#9e9e9e", "pública": "#1e88e5",
    "particular": "#e53935", "pública/particular": "#8e24aa",
}


def cls_favela(v):
    s = str(v).strip().lower().replace(" / ", "/")
    s = s.replace("publica", "pública")
    if s in FAVELA:
        return s, FAVELA[s]
    return "sem informação", FAVELA["sem informação"]


DECLIV = {
    "1": ("1 - 0 a 5%", "#c8e6c9"), "2": ("2 - 5 a 25%", "#fff59d"),
    "3": ("3 - 25 a 60%", "#ffb74d"), "4": ("4 - acima de 60%", "#d84315"),
}


def cls_decliv(v):
    m = re.match(r"\s*([1-4])", str(v))
    return DECLIV[m.group(1)] if m else ("Sem classe", "#bdbdbd")


RISCO = {
    "R1": "#fff176", "R2": "#ffb74d", "R3": "#ff7043", "R4": "#c62828",
    "Área em monitoramento": "#90a4ae", "Área encerrada": "#cfd8dc",
}


def cls_risco(v):
    s = str(v).upper()
    if "MONITORAMENTO" in s:
        k = "Área em monitoramento"
    elif "ENCERRAD" in s:
        k = "Área encerrada"
    else:
        m = re.search(r"R\s*([1-4])", s)
        k = f"R{m.group(1)}" if m else None
    return (k, RISCO[k]) if k else ("Sem classe", "#bdbdbd")


VIAS = {
    "Local": "#1e88e5", "Coletora": "#fdd835", "Arterial": "#e53935",
    "Via de trânsito rápido": "#757575", "Via de pedestres": "#fb8c00",
}


def cls_vias(v):
    s = str(v).upper()
    if "PEDESTRE" in s:
        k = "Via de pedestres"
    elif "VTR" in s or "RÁPIDO" in s or "RAPIDO" in s:
        k = "Via de trânsito rápido"
    elif "ARTERIAL" in s:
        k = "Arterial"
    elif "COLETORA" in s:
        k = "Coletora"
    elif "LOCAL" in s:
        k = "Local"
    else:
        return "Outra", "#bdbdbd"
    return k, VIAS[k]


DENS = [
    (2000, "0 – 2.000 hab/hec", "#fff5eb"), (4000, "2.001 – 4.000 hab/hec", "#fdd0a2"),
    (6000, "4.001 – 6.000 hab/hec", "#fdae6b"), (8000, "6.001 – 8.000 hab/hec", "#f16913"),
    (10000, "8.001 – 10.000 hab/hec", "#d94801"), (float("inf"), "> 10.000 hab/hec", "#7f2704"),
]


def cls_dens(v):
    n = _num(v)
    if n is None:
        return "Sem dado", "#bdbdbd"
    for lim, lab, col in DENS:
        if n <= lim:
            return lab, col


LCZ = {
    1: ("1 - Alto-compacto", "#8c0000"), 2: ("2 - Médio-compacto", "#d10000"),
    3: ("3 - Baixo-compacto", "#ff0000"), 4: ("4 - Alto-aberto", "#bf4d00"),
    5: ("5 - Médio-aberto", "#ff6600"), 6: ("6 - Baixo-aberto", "#ff9955"),
    7: ("7 - Baixo-precário", "#faee05"), 8: ("8 - Baixo-grande", "#bcbcbc"),
    9: ("9 - Ocupação esparsa", "#ffccaa"), 10: ("10 - Indústria pesada", "#555555"),
    101: ("101 - Arborização densa", "#006a00"), 102: ("102 - Arborização esparsa", "#00aa00"),
    103: ("103 - Vegetação arbustiva", "#648525"), 104: ("104 - Vegetação herbácea", "#b9db79"),
    105: ("105 - Rocha ou pavimento", "#000000"), 106: ("106 - Solo exposto", "#fbf7ae"),
    107: ("107 - Água", "#6a6aff"),
}


def cls_lcz(v):
    n = _num(v)
    return LCZ.get(int(n), ("Sem classe", "#bdbdbd")) if n is not None else ("Sem classe", "#bdbdbd")


THEMATIC = {
    "apa": dict(file="apa.parquet", title="APA", geom="poly", label="NOME_CAPS",
                cls=None, color="#a5d6a7"),
    "bacia": dict(file="bacia.parquet", title="Bacia hidrográfica", geom="poly",
                  label="nm_bacia_hidrografica_principal", cls=None, color="#81d4fa"),
    "praca": dict(file="praca.parquet", title="Praças", geom="poly", label="nome",
                  cls=None, color="#66bb6a"),
    "corredor": dict(file="corredor_verde.parquet", title="Corredores", geom="line",
                     label="tx_proposta_planpavel", cls_col="tx_proposta_corredor_planpavel",
                     cls=cls_corredor,
                     legend=[("Corredor verde", "#2e7d32"), ("Corredor polinizador", "#f9a825")]),
    "favela": dict(file="favela.parquet", title="Favelas", geom="poly", label="nome",
                   cls_col="propriedade_area", cls=cls_favela, legend=list(FAVELA.items())),
    "decliv": dict(file="Declividade.parquet", title="Declividade", geom="poly",
                   label="classe", cls_col="classe", cls=cls_decliv, legend=list(DECLIV.values())),
    "risco": dict(file="risco_geologico.parquet", title="Risco geológico", geom="poly",
                  label="rg_process", cls_col="rg_process", cls=cls_risco,
                  legend=list(RISCO.items())),
    "arvores": dict(file="arvores.parquet", title="Árvores", geom="point", label=None,
                    cls=None, color="#2e7d32"),
    "vias": dict(file="vias.parquet", title="Vias", geom="line", label="cvc_classe",
                 cls_col="cvc_classe", cls=cls_vias, legend=list(VIAS.items())),
    "dens": dict(file="densidade_demografica.parquet", title="Densidade demográfica (hab/hec)",
                 geom="poly", label="hab_hec", cls_col="hab_hec", cls=cls_dens,
                 legend=[(l, c) for _, l, c in DENS]),
    "lcz": dict(file="LCZ.parquet", title="Zona climática local", geom="poly", label="DN",
                cls_col="DN", cls=cls_lcz, legend=list(LCZ.values())),
}


def clip_to(gdf, area, ids=None):
    if gdf is None or len(gdf) == 0:
        return gdf
    if ids is not None and "iso_ID" in gdf.columns:
        return gdf[gdf["iso_ID"].astype(str).isin([str(i) for i in ids])]
    idx = list(gdf.sindex.query(area, predicate="intersects"))
    return gdf.iloc[idx]


def load_thematic(key, area, iso_ids, censo_sel):
    cfg = THEMATIC[key]
    if key == "dens":
        dens = read_layer(key, cfg["file"])
        if dens is None:
            return None
        if isinstance(dens, gpd.GeoDataFrame) and "dd_setor" not in dens.columns:
            gdf = clip_to(dens, area, iso_ids)
        else:
            if censo_sel is None or "censo_ID" not in censo_sel.columns:
                return None
            tab = pd.DataFrame(dens.drop(columns="geometry", errors="ignore"))
            tab["dd_setor"] = tab["dd_setor"].astype(str)
            base = censo_sel[["censo_ID", "geometry"]].copy()
            base["censo_ID"] = base["censo_ID"].astype(str)
            gdf = base.merge(tab, left_on="censo_ID", right_on="dd_setor", how="inner")
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)
    else:
        src = read_layer(key, cfg["file"])
        if src is None or not isinstance(src, gpd.GeoDataFrame):
            return None
        gdf = clip_to(src, area, iso_ids)
    gdf = gdf.copy()
    if cfg.get("cls"):
        col = cfg["cls_col"]
        res = gdf[col].map(cfg["cls"]) if col in gdf.columns else pd.Series(
            [("Sem classe", "#bdbdbd")] * len(gdf), index=gdf.index)
        gdf["_classe"] = res.map(lambda t: t[0])
        gdf["_color"] = res.map(lambda t: t[1])
    else:
        gdf["_classe"] = cfg["title"]
        gdf["_color"] = cfg["color"]
    lab = cfg.get("label")
    gdf["_label"] = gdf[lab].astype(str) if lab and lab in gdf.columns else gdf["_classe"]
    if key == "dens":
        gdf["_label"] = gdf["_label"] + " hab/hec"
    return gdf


# =============================================================================
# MAPA
# =============================================================================
def _gj(gdf, cols):
    out = gdf[[c for c in cols if c in gdf.columns] + ["geometry"]].copy()
    for c in out.columns:
        if c != "geometry":
            out[c] = out[c].astype(str)
    return out


def add_thematic(m, key, gdf):
    cfg = THEMATIC[key]
    if gdf is None or len(gdf) == 0:
        return
    fg = folium.FeatureGroup(name=cfg["title"])
    data = _gj(gdf, ["_label", "_classe", "_color"])
    tip = folium.GeoJsonTooltip(fields=["_label"], aliases=[cfg["title"]])
    if cfg["geom"] == "point":
        data = data.head(8000)
        folium.GeoJson(
            data, marker=folium.CircleMarker(radius=2.5, fill=True, fill_opacity=0.9, weight=0),
            style_function=lambda f: {"color": f["properties"]["_color"],
                                      "fillColor": f["properties"]["_color"]},
        ).add_to(fg)
    elif cfg["geom"] == "line":
        w = 4 if key == "vias" else 3
        folium.GeoJson(
            data, tooltip=tip,
            style_function=lambda f, w=w: {"color": f["properties"]["_color"], "weight": w,
                                           "opacity": 0.9},
        ).add_to(fg)
    else:
        folium.GeoJson(
            data, tooltip=tip,
            style_function=lambda f: {"color": f["properties"]["_color"], "weight": 0.5,
                                      "fillColor": f["properties"]["_color"],
                                      "fillOpacity": 0.6},
        ).add_to(fg)
    fg.add_to(m)


def add_boundary(m, key, gdf, label_col, highlight=None):
    color, weight, title = ADMIN_STYLE[key]
    if gdf is None or len(gdf) == 0:
        return
    data = _gj(gdf, [label_col])
    hl = set(map(str, highlight or []))
    folium.GeoJson(
        data, name=title,
        tooltip=folium.GeoJsonTooltip(fields=[label_col], aliases=[title]),
        style_function=lambda f: {
            "color": color, "weight": weight * (1.8 if f["properties"][label_col] in hl else 1),
            "fillColor": "#ff7043" if f["properties"][label_col] in hl else color,
            "fillOpacity": 0.25 if f["properties"][label_col] in hl else 0.03,
        },
    ).add_to(m)


def add_legend(m, keys):
    if not keys:
        return
    blocks = []
    for k in keys:
        cfg = THEMATIC[k]
        items = cfg.get("legend") or [(cfg["title"], cfg["color"])]
        rows = "".join(
            f'<div><span style="display:inline-block;width:12px;height:12px;'
            f'background:{c};margin-right:6px;border:1px solid #555"></span>{l}</div>'
            for l, c in items)
        blocks.append(f"<b>{cfg['title']}</b>{rows}")
    html = ('<div style="position:absolute;bottom:20px;right:10px;z-index:9999;background:#fff;'
            'padding:8px 10px;border-radius:6px;font-size:11px;max-height:320px;overflow:auto;'
            'box-shadow:0 1px 4px rgba(0,0,0,.3)">' + "<hr style='margin:4px 0'>".join(blocks)
            + "</div>")
    m.get_root().html.add_child(Element(html))


def add_print_button(m):
    m.get_root().header.add_child(JavascriptLink(
        "https://cdn.jsdelivr.net/npm/leaflet-easyprint@2.1.9/dist/bundle.min.js"))
    el = MacroElement()
    el._template = Template("""
    {% macro script(this, kwargs) %}
    (function addPrint(n){
      if (window.L && L.easyPrint) {
        L.easyPrint({title:'Baixar imagem (PNG)', position:'topleft', sizeModes:['Current'],
                     exportOnly:true, filename:'planbairros_mapa', hideControlContainer:false
        }).addTo({{this._parent.get_name()}});
      } else if (n < 40) { setTimeout(function(){ addPrint(n+1); }, 250); }
    })(0);
    {% endmacro %}""")
    m.add_child(el)


def feature_at(gdf, col, lat, lng):
    pt = Point(lng, lat)
    idx = list(gdf.sindex.query(pt, predicate="intersects"))
    return str(gdf.iloc[idx[0]][col]) if idx else None


# =============================================================================
# ESTADO
# =============================================================================
for k, v in {"subpref": None, "distrito": None, "isos": [], "level": "Subprefeituras",
             "last_click": None, "last_draw": None}.items():
    st.session_state.setdefault(k, v)


def reset_below(level):
    if level == "subpref":
        st.session_state.distrito = None
        st.session_state.isos = []
    elif level == "distrito":
        st.session_state.isos = []


# =============================================================================
# DADOS BASE
# =============================================================================
subpref = require("subpref")
if subpref is None:
    st.stop()
SP_COL = name_col(subpref, ("sp_nome", "nm_subprefeitura", "nome"))
distritos = read_layer("distrito", ADMIN_FILES["distrito"])
DS_COL = name_col(distritos, ("ds_nome", "nm_distrito", "nome")) if distritos is not None else None
isos = read_layer("iso", ADMIN_FILES["iso"])
ISO_COL = "iso_ID" if isos is not None and "iso_ID" in isos.columns else (
    name_col(isos) if isos is not None else None)

sp_names = sorted(subpref[SP_COL].astype(str).unique())
sel_sp_geom = sel_ds_geom = None
ds_in = iso_in = None

if st.session_state.subpref and distritos is not None:
    sel_sp_geom = unary_union(subpref[subpref[SP_COL].astype(str) == st.session_state.subpref].geometry)
    rp = distritos.geometry.representative_point()
    ds_in = distritos[rp.within(sel_sp_geom)]
if st.session_state.distrito and ds_in is not None and isos is not None:
    sel_ds_geom = unary_union(ds_in[ds_in[DS_COL].astype(str) == st.session_state.distrito].geometry)
    iso_in = clip_to(isos, sel_ds_geom)

levels = ["Subprefeituras"]
if ds_in is not None and len(ds_in):
    levels.append("Distritos")
if iso_in is not None and len(iso_in):
    levels.append("Isócronas")
if st.session_state.isos:
    levels.append("Visualização detalhada")
if st.session_state.level not in levels:
    st.session_state.level = levels[-1]

# =============================================================================
# LAYOUT
# =============================================================================
st.title("PlanBairros")
col_map, col_panel = st.columns([3, 1])

with col_panel:
    st.subheader("Navegação")
    st.selectbox("Nível", levels, key="level")
    st.selectbox("Subprefeitura", [None] + sp_names, key="subpref",
                 format_func=lambda x: "— selecione —" if x is None else x,
                 on_change=reset_below, args=("subpref",))
    ds_opts = sorted(ds_in[DS_COL].astype(str).unique()) if ds_in is not None else []
    st.selectbox("Distrito", [None] + ds_opts, key="distrito", disabled=not ds_opts,
                 format_func=lambda x: "— selecione —" if x is None else x,
                 on_change=reset_below, args=("distrito",))
    iso_opts = sorted(iso_in[ISO_COL].astype(str).unique()) if iso_in is not None else []
    st.session_state.isos = [i for i in st.session_state.isos if i in iso_opts]
    st.multiselect("Isócronas", iso_opts, key="isos", disabled=not iso_opts)

    detailed = st.session_state.level == "Visualização detalhada"
    limites_sel = st.multiselect(
        "Limites", ["censo", "quadras", "lotes", "od"], default=["censo"], disabled=not detailed,
        format_func=lambda k: ADMIN_STYLE[k][2])
    tem_sel = st.multiselect(
        "Camadas temáticas", list(THEMATIC), disabled=not detailed,
        format_func=lambda k: THEMATIC[k]["title"])
    st.caption("Dica: clique no mapa ou desenhe um polígono para selecionar feições.")

# =============================================================================
# CONSTRUÇÃO DO MAPA
# =============================================================================
level = st.session_state.level
m = folium.Map(tiles=None, control_scale=True, prefer_canvas=False)
folium.TileLayer(CARTO_LIGHT_URL, attr=CARTO_ATTR, name="CARTO Light").add_to(m)
csv_parts = []

if level == "Subprefeituras":
    base, base_col = subpref, SP_COL
    add_boundary(m, "subpref", subpref, SP_COL, [st.session_state.subpref])
elif level == "Distritos":
    base, base_col = ds_in, DS_COL
    add_boundary(m, "distrito", ds_in, DS_COL, [st.session_state.distrito])
elif level == "Isócronas":
    base, base_col = iso_in, ISO_COL
    add_boundary(m, "iso", iso_in, ISO_COL, st.session_state.isos)
else:
    sel_iso = iso_in[iso_in[ISO_COL].astype(str).isin(st.session_state.isos)]
    base, base_col = sel_iso, ISO_COL
    area = unary_union(sel_iso.geometry)
    ids = st.session_state.isos if ISO_COL == "iso_ID" else None

    lim_data = {}
    for k in set(limites_sel) | ({"censo"} if "dens" in tem_sel else set()):
        g = read_layer(k, ADMIN_FILES[k])
        if isinstance(g, gpd.GeoDataFrame):
            lim_data[k] = clip_to(g, area, ids)

    # 1) temáticas (abaixo)
    for k in tem_sel:
        g = load_thematic(k, area, ids, lim_data.get("censo"))
        if g is None:
            st.warning(f"Camada temática '{THEMATIC[k]['title']}' ({THEMATIC[k]['file']}) indisponível.")
            continue
        add_thematic(m, k, g)
        part = pd.DataFrame(g.drop(columns="geometry"))
        part.insert(0, "camada", THEMATIC[k]["title"])
        csv_parts.append(part)
    add_legend(m, tem_sel)

    # 2) limites (acima das temáticas)
    for k in limites_sel:
        g = lim_data.get(k)
        if g is not None:
            add_boundary(m, k, g, name_col(g, ("censo_ID", "NumeroZona", "nome")))
    add_boundary(m, "iso", sel_iso, ISO_COL, st.session_state.isos)
    iso_tab = pd.DataFrame(sel_iso.drop(columns="geometry"))
    iso_tab.insert(0, "camada", "Isócronas")
    csv_parts.insert(0, iso_tab)

if base is not None and len(base):
    b = base.total_bounds
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

Draw(export=False, draw_options={"polyline": False, "circle": False, "marker": False,
                                 "circlemarker": False}).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)
add_print_button(m)

with col_map:
    out = st_folium(m, height=700, use_container_width=True, key=f"map_{level}",
                    returned_objects=["last_object_clicked", "all_drawings"])

# =============================================================================
# INTERAÇÃO (clique / desenho)
# =============================================================================
changed = False
click = (out or {}).get("last_object_clicked")
if click and base is not None and len(base):
    ck = (round(click["lat"], 6), round(click["lng"], 6), level)
    if ck != st.session_state.last_click:
        st.session_state.last_click = ck
        fid = feature_at(base, base_col, click["lat"], click["lng"])
        if fid:
            if level == "Subprefeituras" and fid != st.session_state.subpref:
                st.session_state.subpref = fid
                reset_below("subpref")
                changed = True
            elif level == "Distritos" and fid != st.session_state.distrito:
                st.session_state.distrito = fid
                reset_below("distrito")
                changed = True
            elif level == "Isócronas":
                s = list(st.session_state.isos)
                s.remove(fid) if fid in s else s.append(fid)
                st.session_state.isos = s
                changed = True

draws = (out or {}).get("all_drawings") or []
if draws and level == "Isócronas" and iso_in is not None:
    key = str(draws)
    if key != st.session_state.last_draw:
        st.session_state.last_draw = key
        poly = unary_union([shape(d["geometry"]) for d in draws])
        new = sorted(clip_to(iso_in, poly)[ISO_COL].astype(str).unique())
        if new and new != st.session_state.isos:
            st.session_state.isos = new
            changed = True

if changed:
    st.rerun()

# =============================================================================
# EXPORTAÇÃO
# =============================================================================
with col_panel:
    st.subheader("Exportar")
    if csv_parts:
        csv = pd.concat(csv_parts, ignore_index=True).to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV da visualização", csv, "planbairros_dados.csv", "text/csv")
    else:
        st.caption("CSV disponível na visualização detalhada (com isócronas selecionadas).")
    st.caption("Imagem: use o botão 🖨️ no canto superior esquerdo do mapa.")
