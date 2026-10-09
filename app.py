# -*- coding: utf-8 -*-
"""PlanBairros — Dashboard Streamlit/Folium."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import base64
import html
import json
import os
import re
import shutil
import unicodedata

import streamlit as st
import pandas as pd  # type: ignore

try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium.features import GeoJsonTooltip  # type: ignore
    from folium.plugins import Draw  # type: ignore
    from branca.element import Template, MacroElement, JavascriptLink  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    from shapely.geometry import Point, shape  # type: ignore
except Exception:
    gpd = folium = GeoJsonTooltip = Draw = st_folium = Point = shape = None  # type: ignore
    Template = MacroElement = JavascriptLink = None  # type: ignore

# =============================================================================
# CONFIG / UI
# =============================================================================
st.set_page_config(page_title="PlanBairros", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="collapsed")

PB_NAVY = "#14407D"
PB_BROWN = "#C65534"
PB_BTN = "#14407D"
PB_BLACK = "#000000"

# Fundo SEM API key (Esri Light Gray) + rótulos opcionais
BASEMAP_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
BASEMAP_LABELS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
BASEMAP_ATTR = "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors"

SMOOTH_FACTOR = 1.0
LINE_CAP = "round"
LINE_JOIN = "round"

PARENT_FILL_OPACITY = 0.20
PARENT_STROKE_OPACITY = 1.0
PARENT_STROKE_WEIGHT = 1.0
PARENT_STROKE_DASH = None

SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0002, "distrito": 0.0001, "isocrona": 0.00005,
    "censo": 0.00003, "od": 0.00005, "quadra": 0.0, "lote": 0.0,
}

ISO_FILL_OPACITY_DEFAULT = 0.05
ISO_FILL_OPACITY_CLASSES = 0.05

FIXED_LAYER_STYLE = {
    "area_verde":   {"file": "area_verde.geojson",  "color": "#B1BF7C", "weight": 0,   "fill": True,  "fill_opacity": 0.55, "name": "Áreas verdes"},
    "rios":         {"file": "rios.geojson",         "color": "#14407D", "weight": 2.0, "fill": True,  "fill_opacity": 0.35, "name": "Rios"},
    "linhas_metro": {"file": "linhas metro.geojson", "color": "#000000", "weight": 2.8, "fill": False, "fill_opacity": 0.0,  "name": "Metrô"},
    "linhas_trem":  {"file": "linhas trem.geojson",  "color": "#000000", "weight": 2.2, "fill": False, "fill_opacity": 0.0,  "name": "Trem", "dash": "6 4"},
}

# =============================================================================
# PATHS (relativos ao repositório — local e Streamlit Cloud)
# =============================================================================
APP_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "limites_administrativos").is_dir():
            return p
    cwd = Path.cwd()
    return cwd if (cwd / "limites_administrativos").is_dir() else start


REPO_ROOT = _find_repo_root(APP_DIR)
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = REPO_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo_todos.jpg"
LOGO_HEIGHT = 46

LIMITES_DIR = REPO_ROOT / "limites_administrativos"
DATA_SEARCH_DIRS = [LIMITES_DIR, LIMITES_DIR / "data", LIMITES_DIR / "tematicos",
                    REPO_ROOT, REPO_ROOT / "data", APP_DIR, DATA_CACHE_DIR]

# =============================================================================
# MAPAS TEMÁTICOS
# =============================================================================
THEMATIC_DIR_SECRET_KEY = "PB_THEMATIC_DIR"
THEMATIC_CACHE_DIR = DATA_CACHE_DIR / "tematicos"
THEMATIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

THEMATIC_DRIVE_LINKS: Dict[str, str] = {
    "apa": "", "bacia": "", "praca": "", "corredor_verde": "", "favela": "",
    "declividade": "", "risco_geologico": "", "arvores": "", "vias": "",
    "densidade": "", "lcz": "",
}

LCZ_LABELS = {
    1: "Alto-compacto", 2: "Médio-compacto", 3: "Baixo-compacto", 4: "Alto-aberto",
    5: "Médio-aberto", 6: "Baixo-aberto", 7: "Baixo-precário", 8: "Baixo-grande",
    9: "Ocupação esparsa", 10: "Indústria pesada", 101: "Arborização densa",
    102: "Arborização esparsa", 103: "Vegetação arbustiva", 104: "Vegetação herbácea",
    105: "Rocha ou pavimento", 106: "Solo exposto", 107: "Água",
}
LCZ_COLORS_BY_CODE = {
    1: "#7f3b08", 2: "#b35807", 3: "#e08215", 4: "#542688", 5: "#8073ac", 6: "#b2abd2",
    7: "#b2182b", 8: "#d8daeb", 9: "#fddbc7", 10: "#c65534", 101: "#6ea097",
    102: "#b2be7b", 103: "#F4DD63", 104: "#fee0b6", 105: "#f7f7f7", 106: "#f4a582", 107: "#14407D",
}
DENS_BINS = [(2000, "0 – 2.000 hab/hec"), (4000, "2.001 – 4.000 hab/hec"),
             (6000, "4.001 – 6.000 hab/hec"), (8000, "6.001 – 8.000 hab/hec"),
             (10000, "8.001 – 10.000 hab/hec"), (float("inf"), "> 10.000 hab/hec")]
# Escala monocromática de vermelhos
DENS_COLORS = ["#fddbc7", "#f4a582", "#d6604d", "#c65534", "#b2182b", "#67001f"]

THEMATIC_LAYERS: Dict[str, Dict[str, Any]] = {
    "apa": {"title": "APA", "file": "apa.parquet", "label": "NOME_CAPS", "kind": "auto"},
    "bacia": {"title": "Bacias hidrográficas", "file": "bacia.parquet",
              "label": "nm_bacia_hidrografica_principal", "kind": "auto"},
    "praca": {"title": "Praças", "file": "praca.parquet", "label": "nome", "kind": "single", "color": "#6FA097"},
    "corredor_verde": {"title": "Corredores verdes", "file": "corredor_verde.parquet",
                       "label": "tx_proposta_planpavel", "kind": "class",
                       "class_col": "tx_proposta_corredor_planpavel", "fn": "corredor", "weight": 3,
                       "colors": {"Corredor verde": "#6FA097", "Corredor polinizador": "#F4DD63"}},
    "favela": {"title": "Favelas", "file": "favela.parquet", "label": "nome", "kind": "class",
               "class_col": "propriedade_area", "fn": "favela",
               "colors": {"Sem informação": "#d8daeb", "Pública": "#8073ac",
                          "Particular": "#e08215", "Pública/particular": "#542688"}},
    "declividade": {"title": "Declividade", "file": "Declividade.parquet", "label": "classe",
                    "kind": "class", "class_col": "classe", "fn": "decliv", "tt_leg": True,
                    "colors": {"1 - 0 a 5%": "#fee0b6", "2 - 5 a 25%": "#fdb863",
                               "3 - 25 a 60%": "#b35807", "4 - acima de 60%": "#7f3b08"}},
    "risco_geologico": {"title": "Risco geológico", "file": "risco_geologico.parquet",
                        "label": "rg_process", "kind": "class", "class_col": "rg_process", "fn": "risco",
                        "colors": {"R1": "#fddbc7", "R2": "#f4a582", "R3": "#c65534", "R4": "#b2182b",
                                   "Área em monitoramento": "#b2abd2", "Área encerrada": "#d8daeb"}},
    "arvores": {"title": "Árvores", "file": "arvores.parquet", "label": None, "kind": "points", "color": "#6FA097"},
    "vias": {"title": "Vias", "file": "vias.parquet", "label": "cvc_classe", "kind": "class",
             "class_col": "cvc_classe", "fn": "vias", "weight": 4, "tt_leg": True,
             "colors": {"Local": "#d8daeb", "Coletora": "#F4DD63", "Arterial": "#C65534",
                        "Via de trânsito rápido": "#542688", "Via de pedestres": "#6FA097"}},
    "densidade": {"title": "Densidade demográfica", "file": "densidade_demografica.parquet",
                  "label": "hab_hec", "kind": "class", "class_col": "hab_hec", "fn": "dens",
                  "colors": dict(zip([b[1] for b in DENS_BINS], DENS_COLORS))},
    "lcz": {"title": "Zona climática local", "file": "LCZ.parquet", "label": "DN", "kind": "class",
            "class_col": "DN", "fn": "lcz", "tt_leg": True,
            "colors": {LCZ_LABELS[k]: LCZ_COLORS_BY_CODE[k] for k in LCZ_LABELS}},
}
PALETTE_AUTO = ["#6FA097", "#D58243", "#8073ac", "#F4DD63", "#14407D", "#B1BF7C",
                "#C65534", "#542688", "#fdb863", "#b2abd2", "#7f3b08", "#f4a582"]
MAX_POINTS = 8000
STATION_RADIUS = 2.5

# =============================================================================
# TABELAS EXTERNAS (OD 2023 / IPTU 2026)
# =============================================================================
OD23_URL = "https://drive.google.com/file/d/1jkgABlxJY4tTpD1CXB8h968hxSQFQqMG/view?usp=drive_link"
IPTU_URL = "https://drive.google.com/file/d/1mTY_Y1uCAQ2GeZ1o8Kf9pZrqTn9v_sKE/view?usp=drive_link"
# src: (nomes locais, secret, url, nome no cache, candidatos à coluna-chave, coluna da camada)
EXT_TABLES = {
    "od": (["od_23.parquet", "od_23.csv", "od_23.xlsx", "od_23"], "PB_OD23_FILE", OD23_URL,
           "od_23.dat", ["Zona_D", "zona_d", "zonad", "zona_destino"], "od_id"),
    "iptu": (["dashiptu_2026.csv"], "PB_IPTU_FILE", IPTU_URL, "dashiptu_2026.csv",
             ["Lote_id", "lote_id", "id_lote", "lote", "sql"], "lote_id"),
}
ID_LIKE_RE = re.compile(r"(^|_)(zona|id|cod|codigo|sql|setor|lote|quadra|cd)(_|$)", re.I)
# Escala monocromática de marrons (intensidade OD / IPTU)
CHORO_COLORS = ["#fee0b6", "#fdb863", "#e08215", "#b35807", "#7f3b08"]

# =============================================================================
# VISUALIZAÇÕES / CSV
# =============================================================================
QUADRAS_CSV_FILENAME = "quadras.csv"
QUADRAS_CSV_SECRET_KEY = "PB_QUADRAS_CSV_FILE_ID"
QUADRAS_CSV_FALLBACK_URL = "https://drive.google.com/file/d/1_WKryQlu_jZL1xsgAmQDrI81aSdzKYsc/view?usp=drive_link"

CLUSTER_COL = "Cluster"
ISO_CLASS_COL = "nova_class"
CLUSTER_COLOR_MAP = {0: "#C65534", 1: "#F4DD63", 2: "#D58243", 3: "#6FA097", 4: "#14407D"}
CLUSTER_LABELS = {0: "Tipo 1 — Periféricas de alta densidade", 1: "Tipo 2 — Mistas de densidade intermediária",
                  2: "Tipo 3 — Periféricas de média densidade", 3: "Tipo 4 — Centrais verticalizadas e mistas",
                  4: "Tipo 5 — Predominância de comércio e serviços"}
CLUSTER_NULL_COLOR = "#c8c8c8"
ISO_TRANSITION_SET = {1, 3, 6}
ISO_TRANSITION_LABEL = "Área de transição"
ISO_TRANSITION_COLOR = "#fdb863"
ISO_VALUE_TO_CLASSNUM = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7}
ISO_CLASSNUM_TO_COLOR = {1: "#f7f7f7", 2: "#d8daeb", 3: "#8073ac", 4: "#b2abd2",
                         5: "#b35806", 6: "#e08214", 7: "#542788"}
ISO_DEFAULT_COLOR = "#ffffff"

# =============================================================================
# IDS
# =============================================================================
SUBPREF_ID = "sp_id"
DIST_ID = "distrito_id"
ISO_ID = "iso_id"
OD_ID = "od_id"
QUADRA_ID = "quadra_id"
QUADRA_UID = "quadra_uid"
CENSO_ID = "censo_id"
LOTE_ID = "lote_id"
DS_CODIGO = "ds_codigo"

DIST_PARENT = SUBPREF_ID
ISO_PARENT = DIST_ID
CENSO_PARENT = ISO_ID

LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID], "dist": [DIST_ID, DIST_PARENT, DS_CODIGO],
    "iso": [ISO_ID, ISO_PARENT, SUBPREF_ID, DS_CODIGO],
    "censo": [CENSO_ID, CENSO_PARENT, QUADRA_ID, ISO_ID],
    "od": [OD_ID, ISO_ID], "quadra": [QUADRA_ID, ISO_ID, CENSO_ID, QUADRA_UID],
    "lote": [LOTE_ID, ISO_ID, DIST_ID],
}

LOCAL_FILENAMES = {
    "subpref": "subprefeitura.parquet",
    "dist": "Distritos.parquet",
    "iso": "isocronas.parquet",
    "censo": "SetoresCensitarios2023.parquet",
    "od": "ZonasOD2023.parquet",
    "quadra": "Quadras.parquet",
}

# =============================================================================
# LOTES / QUADRA — links
# =============================================================================
LOTES_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/17-lA2P_D4oV1joysDf7BOAgp358IcoEG?usp=drive_link"
LOTES_SECRET_KEY = "PB_LOTES_FOLDER_URL"

_LOTES_IDS = {
    "1": "1Kbn6RXKXoxdpcdTbI9txBBSf63Yd14zZ", "2": "1q8oaYurUmEyJIgluOmos-nDytAtLur7Q",
    "3": "15Gg4GVDabqZwzY1ubIwhhPZXXU4hkrs8", "4": "1OAbjgydEp2E5UhWqHwCkKj_6EziDMAYL",
    "5": "1jiruqIpGIGhU7q0Uu4wCSl_xUsLdPnZY", "6": "1IcYBxz1aa-1Aoelsm6_Vf3UM_fQ_y4Iv",
    "7": "1us8VgARE1VLjFEWLMfH4qjOadiJ4Y0vj", "8": "1yGeLMS5zYUKJGi1i5mlvvV4BgJmvEZgW",
    "9": "1ueN0KUmZ5ijd5wK087dVK0Fq9ATob5pS", "10": "1KCwDfaqVBMEoFm4pqAYGPEcYPW9IDIir",
    "11": "1g7wg3aHO4r2UpuEPcNG9qCgqygR4PLvY", "12": "1mG4zs2HqmzTaeFwovJdw-nt11G27L7CC",
    "13": "1BlpnBTDQsBpfPXL_xhom7TtNkt3w9k9I", "14": "1iBkLh3atFZG61jORGXNcwYjM7c_wTPXU",
    "15": "1Kx3ttk0Tmow0aJqPhnI-B0p-mpasMsxp", "16": "1IyFlPR1nGRfbSPn8J3yrySJAWFkI5L2q",
    "17": "1_kmHNaY0k5zl6Xk1dSGVHrUDtGYWaCMj", "18": "15jngv3COgPLIwizOmbQhJuEDlwBSF3M5",
    "19": "1IlvxySwpW3OoRktzooJ2bk4XGkuhcis8", "20": "1KcCz5bIl1cMcE_7eFqLXN0zA4QAZ5Q0U",
    "21": "1wU7EBRnMbJNXTk0J8d0uENE-FGrGfNRc", "22": "1jjfwx59Cni8eJzqiUEixhuP1rcra2m79",
    "23": "1XWhKjd1sm78LgYYf_64-LgsOu4E-_gxl", "24": "1JtBX5QB2rzL5Be4tYTHZD5jW0bnNMca2",
    "25": "1Mnbq4eqLlPZRbcsktuUrJQ0P_1llI8a3", "26": "1gCCKQLKBRLCcWHfuMrMTuW3LR_jIBXCF",
    "27": "1SO2TkRNpFCQ15tH5yPixlsngNpu4Pqxv", "28": "1e-G0lIXe1gn8iK2HMn5ZwJdRMGObjN15",
    "29": "1LXamIFPz_yAdxpWoachxebX5KqTJkJAN", "30": "16oq92B67QzeKIdQ3GTzQex5WGLfKj4Am",
    "31": "1Sp7RXDbxjLVAK1VwkF2q4SBfu1-drm16", "32": "16rw5zxnJi3nxBDFZJyjp8PQZzRB9K7LM",
    "33": "1DYhc60NvClbNDCCr-tHYyyaO_1k_SZQL", "34": "16EN1W9H_EbbT2TbhLknJykODdAgfjB4w",
    "35": "1QrPXvd23iX2duiTgTDmpu_nNFTyDuIS5", "36": "1xToRhnE_ugRwoa_qdWFu3wMloiXGSwoP",
    "37": "1pGeH3Mrfg6xJfFmiKJBgRvHxdAB7N1KU", "38": "1XI2nZIcdOd2hv9MZzVLuYonCEg_s79IC",
    "39": "1WsKRq-sqC5_-zWg0tpVnfjRUyLYoojUB", "40": "1SdoU-K7AmzSoRln3BCuAPxHWEfNi292m",
    "41": "1TT9nkYdA8Dw2mwci599053KpTT-_2sml", "42": "1EqjCjnfW80EaM2F5PaG22qEb3WSspFsf",
    "43": "1s2P3XpxfmGsIjIjaRoC8FXoF5h3TDjGQ", "44": "1fUCV4U34DJaPRF-s1oVdX67DM-Z7Rln_",
    "45": "1UeUQyeAckTabd4BmiHJJqClsgKarfEhL", "46": "1SAMCChCPxWqejoXudY8XFID8jd5o_ngU",
    "47": "1oN_6xqyk7HS1sieoEKk8nttjy8-xS_yu", "48": "1lGEaNPhtHOnSJYJsB7nA1lkaeIPrNNFz",
    "49": "1jB4pIZStVa-vwuvOSDdiWnsCJEpvaa9t", "50": "1dfGxJv0utpr7E-vdjQQprAz_Ipx5ZJwZ",
    "51": "1URtL1odgNGjh2nI2dD3lnjzJiBbBjqh4", "52": "1dPCx22YRyEf1jx0N8Bn8OVedFrETA9VD",
    "53": "1CMRsOPSVKXOTOfDQj202IMJX6aGOfF3V", "54": "17zVEIEUQrefj9BCsHT200hC984qL6L36",
    "55": "1oQUwYO0TgW3ryY5pPG3ixcJ4TdpvT5hg", "56": "1WKXOCAX1SmBa9_N8a9zNxhIO1E93pMId",
    "57": "12RJ7NktS6SoqQ4MsJxsqKt2Zh3qy3iH1", "58": "1wKXvQNTcYmYt5sSEUKdvCsgFNp9m0Q5k",
    "59": "1DY6qMT1MD7EddZd5fj3SBCTJNWyjuqNX", "60": "1zyf9ouOfgoW5mns5N7unQrriQJ6xjdUi",
    "61": "15zON1tja5ou3y7L8d8nu-fdc4XYGlnqb", "62": "1VmzWAtjAFlM1DC-tKG0Dw1-otrVkdpti",
    "63": "1TH5wYLMeBeprot6-bPFZyh_dVhLGLR08", "64": "1ScM1ebCH51wTi0mmMTSIPahoZofm_xuL",
    "65": "1qbWAHdca8ZSx4G7OUnMXQlMsg9neBCW8", "66": "1ASSj1YdIvUe64BfAzWQ9IbDEumof615i",
    "67": "1GKCozATY0I8UVxhnjdpco3-VBPS27Zr7", "68": "1cy7nAHarikbvVXyXAco6EIDrsvZzlV8F",
    "69": "1eiJHdo0GjqECrqYeBeS5xzXCKOkNGJab", "70": "1pyA-Dh0P5AN3-ip9MfSu22ebInsr5R7h",
    "71": "1PA4ovneuLD2qHHwv6Qx6DQxbKIyuLYux", "72": "1aXnU47uzXtW1ZcD9SUg9nHJubVBaz3az",
    "73": "1p7HZEr4s_RgAe57hMDXj5-4dpXlsHMt9", "74": "1dfP9qK9eGnvo0hK--l0lnThxACfIZ9_f",
    "75": "1prPh84jY3BYkRMTBcFtAAPeREOP9EQwe", "76": "1VVcJajqLHrEr8XDAfVF3_QlvGuAqjuSd",
    "77": "1Pw-1y6ZFgFzEDzKDdI8VdOg4O-hJHajy", "78": "12F-MQZFQFjZPAcVKz4Y__T7fcNlyjTpO",
    "79": "1RYbBaPOp2CWYfj8ltEDI_y84FdxOxlEW", "80": "1BHy41a8XZhNitEqRm1sL3n7SontFMVka",
    "81": "1Yo9gPsjPwjvTPAz_nfCGNVF5ruxFcwVU", "82": "1mCGAXLnLPMmo7ZGxjwpPV-qES8S_0XKg",
    "83": "1eggbh76-bq54M1ve_MkxrS-SfH_Uwohk", "84": "1ZJy1KOwMiE2eL14Bgo7iU-G3Hc_BhC6x",
    "85": "1fxfK10uzPHO_Q77myUDsI91gr7_e19Ut", "86": "1OdqJ9IKe2oAN1VE3a94rt1JAwcn-16Ji",
    "87": "1SodUj07iUFiC7oddWNvB6GRzHMZ9eigS", "88": "1ZppypRlP-AbK5gQqRkh_LavTbblq4016",
    "89": "1oQkZUWR5LbffNMHShbiXxSEDR3CFcTNM", "90": "17dAYRSJciVVBIhmTtI_FMbmZBZcXQxBM",
    "91": "10ceUaLciuAGkRMJNo7VHVD7ajVdSOtfj", "92": "1szxz5749c2WcXkXTiV4ZYn_oeuIFi4ke",
    "93": "1dOlYZzidB3Yoo5mjmoDSUC65dpWogoox", "94": "17dfihWuDJsFnhwpE4y_ow493z1IXncCf",
    "95": "1R9m8HCQTOYqSuSlT2FiWSBzCQ2Ry9jE5", "96": "1yjd8bnRuSrsfGTpZY3DwEuiZJs5DXqfs",
}
LOTES_LINKS_BY_DISTRITO = {k: f"https://drive.google.com/file/d/{v}/view?usp=drive_link"
                           for k, v in _LOTES_IDS.items()}

SECRETS_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID", "dist": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID", "censo": "PB_CENSO_FILE_ID",
    "od": "PB_OD_FILE_ID", "quadra": "PB_QUADRAS_FILE_ID",
}
FALLBACK_URLS = {
    "subpref": "", "dist": "", "iso": "", "censo": "", "od": "",
    "quadra": "https://drive.google.com/file/d/1Ivy2PyGHqFgIxSMoK3N9oik2wr5v912U/view?usp=drive_link",
}

# =============================================================================
# NORMALIZAÇÃO
# =============================================================================
def _mk_aliases(base: str) -> Set[str]:
    b = base.strip()
    return {b, b.upper(), b.title(), b.replace("_", ""), b.replace("_", "").upper()}


COL_ALIASES: Dict[str, Set[str]] = {
    SUBPREF_ID: _mk_aliases(SUBPREF_ID),
    DIST_ID: _mk_aliases(DIST_ID) | {"id_distrito", "dist_id"},
    ISO_ID: _mk_aliases(ISO_ID),
    OD_ID: _mk_aliases(OD_ID) | {"OD_ID", "zona_od", "zonaod", "id_od", "od"},
    QUADRA_ID: _mk_aliases(QUADRA_ID),
    CENSO_ID: _mk_aliases(CENSO_ID) | {"cendo_id", "setor_id", "id_setor", "codigo_setor", "cd_setor", "cd_geocodi"},
    LOTE_ID: _mk_aliases(LOTE_ID) | {"id_lote", "codigo_lote", "cd_lote", "lote"},
}
_ALIAS_LOOKUP = {a.strip().lower(): canon for canon, al in COL_ALIASES.items() for a in al}


def standardize_columns(gdf):
    if gdf is None or gdf.empty:
        return gdf
    existing = {str(c) for c in gdf.columns}
    ren: Dict[str, str] = {}
    used: Set[str] = set()
    for c in gdf.columns:
        raw = str(c)
        cl = re.sub(r"\s+", " ", raw.strip())
        canon = _ALIAS_LOOKUP.get(cl.lower())
        if canon and canon != raw:
            # só renomeia se a coluna canônica ainda não existe
            if canon not in existing and canon not in used:
                ren[raw] = canon
                used.add(canon)
        elif raw != cl:
            ren[raw] = cl
    g = gdf.rename(columns=ren)
    return g.loc[:, ~g.columns.duplicated()]


def _id_to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (bool, int)):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v).strip()
    s = str(v).strip()
    if not s:
        return None
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        return s[:-2]
    return s


def normalize_quadra_id(v, width=6):
    s = _id_to_str(v)
    if s is None:
        return None
    return s.zfill(width) if s.isdigit() else s


def make_quadra_uid(iso_id, quadra_id):
    iso, qid = _id_to_str(iso_id), _id_to_str(quadra_id)
    return f"{iso}__{qid}" if iso and qid else None


def normalize_id_cols(gdf, cols):
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


def ensure_set_of_str(value) -> Set[str]:
    if value is None:
        return set()
    items = value if isinstance(value, (set, list, tuple)) else (value,)
    return {s for s in (_id_to_str(x) for x in items) if s}


def first_non_null_value(gdf, col):
    if gdf is None or gdf.empty or col not in gdf.columns:
        return None
    vals = gdf[col].dropna()
    if len(vals) == 0:
        return None
    s = str(vals.iloc[0]).strip()
    return s or None


def label_or_id(gdf, *, label_col, fallback_col, fallback_prefix=""):
    lbl = first_non_null_value(gdf, label_col)
    if lbl:
        return lbl
    fb = first_non_null_value(gdf, fallback_col)
    return f"{fallback_prefix}{fb}" if fb else (fallback_prefix.strip() or "")


def _empty(g):
    return g.iloc[0:0].copy() if g is not None else g


def subset_by_parent(child, parent_col, parent_val):
    if child is None or child.empty or parent_col not in child.columns or parent_val is None:
        return _empty(child)
    pv = _id_to_str(parent_val)
    if pv is None:
        return _empty(child)
    out = child[child[parent_col] == pv]
    if out.empty:  # tolera "01" x "1", "1.0" x "1", espaços etc.
        nk = _norm_key(pv)
        out = child[child[parent_col].map(_norm_key) == nk]
    return out


def subset_by_parent_multi(child, parent_col, parent_vals):
    if child is None or child.empty or parent_col not in child.columns or not parent_vals:
        return _empty(child)
    pset = ensure_set_of_str(parent_vals)
    if not pset:
        return _empty(child)
    out = child[child[parent_col].isin(list(pset))]
    if out.empty:
        nset = {_norm_key(x) for x in pset}
        out = child[child[parent_col].map(_norm_key).isin(nset)]
    return out


subset_by_id = subset_by_parent
subset_by_id_multi = subset_by_parent_multi


def choose_quadra_parent_col(g_quad, *, preferred=CENSO_ID, fallback=ISO_ID):
    if g_quad is None or getattr(g_quad, "empty", True):
        return None
    if preferred in g_quad.columns and g_quad[preferred].notna().any():
        return preferred
    return fallback if fallback in g_quad.columns else None


def get_quadras_subset_for_mode(g_quad, *, iso_ids, filter_censo_ids):
    if g_quad is None or g_quad.empty:
        return _empty(g_quad)
    iso_ids = ensure_set_of_str(iso_ids)
    filter_censo_ids = ensure_set_of_str(filter_censo_ids)
    if choose_quadra_parent_col(g_quad) == CENSO_ID and filter_censo_ids:
        return subset_by_id_multi(g_quad, CENSO_ID, filter_censo_ids)
    if ISO_ID in g_quad.columns and iso_ids:
        return subset_by_parent_multi(g_quad, ISO_ID, iso_ids)
    return _empty(g_quad)


def get_censo_subset_for_isos(g_censo, iso_ids):
    if g_censo is None or g_censo.empty:
        return _empty(g_censo)
    col = CENSO_PARENT if CENSO_PARENT in g_censo.columns else None
    return subset_by_parent_multi(g_censo, col, iso_ids) if col else _empty(g_censo)


def get_lotes_subset_for_isos(g_lote, iso_ids):
    if g_lote is None or g_lote.empty or ISO_ID not in g_lote.columns:
        return _empty(g_lote)
    return subset_by_parent_multi(g_lote, ISO_ID, iso_ids)

# =============================================================================
# DRIVE / IO
# =============================================================================
def _get_secret(key):
    try:
        return str(st.secrets.get(key, "")).strip()
    except Exception:
        return ""


def extract_drive_id(raw):
    raw = (raw or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw
    for pat in (r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)", r"([a-zA-Z0-9_-]{20,})"):
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    return ""


def _drive_candidates(file_id_or_url):
    raw = (file_id_or_url or "").strip()
    fid = extract_drive_id(raw)
    urls = []
    if fid:
        urls.append(f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t")
        urls.append(f"https://drive.google.com/uc?export=download&id={fid}")
    if raw.lower().startswith("http"):
        urls.append(raw)
    return list(dict.fromkeys(urls))


def _looks_like_html(path: Path):
    try:
        with open(path, "rb") as f:
            head = f.read(2048).lower()
        return b"<!doctype html" in head or b"<html" in head or b"<head" in head
    except Exception:
        return False


def _valid_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0 and not _looks_like_html(p)
    except Exception:
        return False


def download_drive_file(file_id_or_url, dst: Path, label=""):
    import requests
    raw = (file_id_or_url or "").strip()
    fid = extract_drive_id(raw)
    if not raw and not fid:
        raise RuntimeError(f"FILE_ID inválido: {file_id_or_url!r}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _valid_file(dst):
        return dst
    session = requests.Session()
    ui = label or dst.name
    last_error = None
    for base_url in _drive_candidates(file_id_or_url):
        try:
            resp = session.get(base_url, stream=True, allow_redirects=True, timeout=120)
            if "drive.google.com/uc" in base_url and fid:
                token = next((v for k, v in resp.cookies.items() if k.startswith("download_warning")), None)
                if token:
                    resp = session.get(f"https://drive.google.com/uc?export=download&id={fid}&confirm={token}",
                                       stream=True, allow_redirects=True, timeout=120)
            if resp.status_code != 200:
                last_error = RuntimeError(f"HTTP {resp.status_code} em '{ui}'"); continue
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            prog = st.progress(0, text=f"Baixando {ui}…")
            with open(dst, "wb") as f:
                for part in resp.iter_content(chunk_size=1024 * 1024):
                    if not part:
                        continue
                    f.write(part); downloaded += len(part)
                    if total > 0:
                        pct = min(int(downloaded * 100 / total), 100)
                        prog.progress(pct, text=f"Baixando {ui}… {pct}%")
            prog.empty()
            if not _valid_file(dst):
                dst.unlink(missing_ok=True); last_error = RuntimeError("arquivo vazio ou HTML"); continue
            return dst
        except Exception as e:
            try:
                if dst.exists() and not _valid_file(dst):
                    dst.unlink(missing_ok=True)
            except Exception:
                pass
            last_error = e
    raise RuntimeError(str(last_error) if last_error else f"Falha ao baixar '{ui}'.")


def get_drive_raw(layer_key):
    sk = SECRETS_KEYS.get(layer_key, "")
    raw_secret = _get_secret(sk) if sk else ""
    return raw_secret or str(FALLBACK_URLS.get(layer_key, "")).strip()


@st.cache_data(show_spinner=False, ttl=600)
def _repo_file_index() -> Dict[str, str]:
    """Índice (nome em minúsculas -> caminho) de todo o repositório relevante."""
    idx: Dict[str, str] = {}
    for d in DATA_SEARCH_DIRS:
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    idx.setdefault(f.name.lower(), str(f))
    for root in (LIMITES_DIR, REPO_ROOT / "data", REPO_ROOT / "tematicos", REPO_ROOT / "dash"):
        if root.is_dir():
            for f in root.rglob("*"):
                if f.is_file():
                    idx.setdefault(f.name.lower(), str(f))
    return idx


def _find_local_file(filename) -> Optional[Path]:
    for d in DATA_SEARCH_DIRS:
        for name in (filename, filename.lower()):
            p = d / name
            if _valid_file(p):
                return p
    hit = _repo_file_index().get(str(filename).lower())
    if hit and _valid_file(Path(hit)):
        return Path(hit)
    return None


def ensure_local_layer(layer_key):
    filename = LOCAL_FILENAMES[layer_key]
    found = _find_local_file(filename)
    if found is not None:
        return found
    raw = get_drive_raw(layer_key)
    if not raw:
        raise RuntimeError(
            f"Layer '{layer_key}' ({filename}) não encontrada. Raiz detectada: {REPO_ROOT} — "
            f"verifique se o arquivo está em limites_administrativos/ com esse nome.")
    return download_drive_file(raw, DATA_CACHE_DIR / filename, label=filename)

# --- TEMÁTICOS ---
def thematic_dirs() -> List[Path]:
    dirs: List[Path] = []
    custom = _get_secret(THEMATIC_DIR_SECRET_KEY) or os.environ.get(THEMATIC_DIR_SECRET_KEY, "")
    if custom:
        dirs.append(Path(custom))
    return dirs + [REPO_ROOT / "dash", REPO_ROOT / "tematicos", THEMATIC_CACHE_DIR] + DATA_SEARCH_DIRS


def ensure_thematic_file(key) -> Optional[Path]:
    fn = THEMATIC_LAYERS[key]["file"]
    for d in thematic_dirs():
        for p in (d / fn, d / fn.lower()):
            if _valid_file(p):
                return p
    found = _find_local_file(fn)
    if found is not None:
        return found
    raw = _get_secret(f"PB_TEM_{key.upper()}") or THEMATIC_DRIVE_LINKS.get(key, "")
    if raw:
        try:
            return download_drive_file(raw, THEMATIC_CACHE_DIR / fn, label=fn)
        except Exception as e:
            st.warning(f"Falha ao baixar {fn}: {e}")
    return None

# --- LOTES ---
def get_lotes_folder_raw():
    return _get_secret(LOTES_SECRET_KEY) or LOTES_DRIVE_FOLDER_URL


def extract_drive_folder_id(raw):
    raw = (raw or "").strip()
    if not raw:
        return ""
    for pat in (r"/folders/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"):
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    return raw if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower() else ""


def lotes_local_dir() -> Path:
    p = DATA_CACHE_DIR / "lotes"; p.mkdir(parents=True, exist_ok=True); return p


def lote_filename_for_distrito(distrito_id):
    return f"Distrito_{_id_to_str(distrito_id) or ''}.parquet"


def try_copy_lote_from_local_sources(distrito_id, dst: Path):
    fn = lote_filename_for_distrito(distrito_id)
    cands = [REPO_ROOT / fn, REPO_ROOT / "data" / fn, REPO_ROOT / "lotes" / fn,
             LIMITES_DIR / fn, LIMITES_DIR / "lotes" / fn, DATA_CACHE_DIR / fn]
    hit = _repo_file_index().get(fn.lower())
    if hit:
        cands.append(Path(hit))
    for p in cands:
        if _valid_file(p):
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
            return dst
    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def list_drive_folder_files(folder_id):
    import requests
    folder_id = (folder_id or "").strip()
    if not folder_id:
        return {}
    found = {}
    for url in (f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing",
                f"https://drive.google.com/drive/u/0/folders/{folder_id}"):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if resp.status_code != 200:
                continue
            for fid, name in re.findall(r'\["([a-zA-Z0-9_-]{20,})","([^"]+\.parquet)"', resp.text):
                found[name] = fid
            for name, fid in re.findall(r'\["([^"]+\.parquet)","([a-zA-Z0-9_-]{20,})"', resp.text):
                found[name] = fid
            if found:
                return found
        except Exception:
            continue
    return found


def ensure_local_lote_file(distrito_id):
    did = _id_to_str(distrito_id)
    if not did:
        raise RuntimeError("distrito_id inválido para lotes.")
    fn = lote_filename_for_distrito(did)
    dst = lotes_local_dir() / fn
    if _valid_file(dst):
        return dst
    if try_copy_lote_from_local_sources(did, dst) is not None:
        return dst
    direct = str(LOTES_LINKS_BY_DISTRITO.get(did, "")).strip()
    if direct:
        return download_drive_file(direct, dst, label=fn)
    folder = extract_drive_folder_id(get_lotes_folder_raw())
    fid = list_drive_folder_files(folder).get(fn, "") if folder else ""
    if fid:
        return download_drive_file(fid, dst, label=fn)
    raise RuntimeError(f"Lotes '{fn}' não localizado (distrito {did}).")

# =============================================================================
# READ
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def read_gdf_parquet(path):
    if gpd is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    gdf = gpd.read_parquet(p)
    if gdf.crs is None:
        minx = gdf.total_bounds[0] if len(gdf) else 0
        gdf = gdf.set_crs(31983 if abs(minx) > 180 else 4326, allow_override=True)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_gdf_geojson(path):
    if gpd is None:
        return None
    try:
        gdf = gpd.read_file(Path(path))
        return gdf.set_crs(4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        return None


def _drop_bad_geoms(gdf):
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    return gdf


def _cache_get(ck, meta):
    cache = st.session_state.get("_layer_cache", {})
    if ck in cache and st.session_state.get("_layer_cache_meta", {}).get(ck) == meta:
        return cache[ck]
    return None


def _cache_set(ck, meta, g):
    st.session_state.setdefault("_layer_cache", {})[ck] = g
    st.session_state.setdefault("_layer_cache_meta", {})[ck] = meta


def _meta(p: Path):
    try:
        return (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
    except Exception:
        return (str(p), 0.0, 0)


def read_layer(layer_key):
    try:
        p = ensure_local_layer(layer_key)
    except Exception as e:
        if layer_key == "quadra":
            st.info("Camada de quadras indisponível — visualização de quadra/cluster desativada.")
        else:
            st.error(str(e))
        return None
    meta = _meta(p)
    cached = _cache_get(layer_key, meta)
    if cached is not None:
        return cached
    try:
        g = read_gdf_parquet(str(p))
    except Exception as e:
        st.error(f"Erro ao ler {p.name}: {e}"); return None
    if g is None or g.empty:
        st.error(f"Layer '{layer_key}' vazia/erro ({p.name})."); return None
    g = standardize_columns(g)
    g = _drop_bad_geoms(g)
    g = normalize_id_cols(g, LAYER_ID_COLS.get(layer_key, []))
    if layer_key == "quadra":
        if QUADRA_ID in g.columns:
            g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))
        if ISO_ID in g.columns and QUADRA_ID in g.columns:
            g[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(g[ISO_ID], g[QUADRA_ID])]
    _cache_set(layer_key, meta, g)
    return g


def read_lotes_by_distrito(distrito_id):
    did = _id_to_str(distrito_id)
    if not did:
        return None
    try:
        p = ensure_local_lote_file(did)
    except Exception as e:
        st.warning(str(e)); return None
    ck, meta = f"lote__{did}", _meta(p)
    cached = _cache_get(ck, meta)
    if cached is not None:
        return cached
    g = read_gdf_parquet(str(p))
    if g is None or g.empty:
        st.warning(f"Lotes do distrito '{did}' vazio/inválido."); return None
    g = normalize_id_cols(_drop_bad_geoms(standardize_columns(g)), LAYER_ID_COLS["lote"])
    _cache_set(ck, meta, g)
    return g

# =============================================================================
# TEMÁTICOS — leitura / filtro / classificação
# =============================================================================
def _col(g, name) -> Optional[str]:
    if g is None or not name:
        return None
    return {str(c).strip().lower(): c for c in g.columns}.get(str(name).strip().lower())


def _norm_txt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()


def read_thematic(key):
    p = ensure_thematic_file(key)
    if p is None:
        st.warning(f"Arquivo '{THEMATIC_LAYERS[key]['file']}' não encontrado no repositório.")
        return None
    ck, meta = f"tem__{key}", _meta(p)
    cached = _cache_get(ck, meta)
    if cached is not None:
        return cached
    g = None
    try:
        g = read_gdf_parquet(str(p))
    except Exception:
        g = None
    if g is None and key == "densidade":
        try:
            df = pd.read_parquet(p)
            dd = _col(df, "dd_setor")
            censo = read_layer("censo")
            if dd and censo is not None:
                df["__k"] = df[dd].map(_id_to_str)
                g = censo[[CENSO_ID, "geometry"]].merge(df, left_on=CENSO_ID, right_on="__k", how="inner")
                g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
        except Exception as e:
            st.warning(f"Densidade: falha ao associar dd_setor = censo_id ({e}).")
    if g is None or g.empty:
        st.warning(f"Camada temática '{key}' vazia ou inválida."); return None
    g = normalize_id_cols(_drop_bad_geoms(standardize_columns(g)), [ISO_ID, CENSO_ID])
    _cache_set(ck, meta, g)
    return g


def filter_thematic(key, g, iso_ids, g_iso_sel):
    if g is None or g.empty:
        return g
    if key == "densidade":
        censo = read_layer("censo")
        ids = set(get_censo_subset_for_isos(censo, iso_ids)[CENSO_ID].dropna()) if censo is not None else set()
        col = _col(g, "dd_setor") or (CENSO_ID if CENSO_ID in g.columns else None)
        if col and ids:
            return g[g[col].map(_id_to_str).isin(ids)]
    if ISO_ID in g.columns:
        return subset_by_parent_multi(g, ISO_ID, iso_ids)
    if g_iso_sel is None or g_iso_sel.empty:
        return g.iloc[0:0]
    try:
        union = g_iso_sel.geometry.union_all()
    except Exception:
        union = g_iso_sel.geometry.unary_union
    minx, miny, maxx, maxy = g_iso_sel.total_bounds
    c = g.cx[minx:maxx, miny:maxy]
    return c[c.intersects(union)]


def _coerce_int(v):
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
            if not v:
                return None
        return int(float(v))
    except Exception:
        return None


def _cls_corredor(v):
    return "Corredor polinizador" if "poliniz" in _norm_txt(v) else "Corredor verde"


def _cls_favela(v):
    s = _norm_txt(v)
    if "public" in s and "particular" in s:
        return "Pública/particular"
    if "public" in s:
        return "Pública"
    if "particular" in s:
        return "Particular"
    return "Sem informação"


def _cls_decliv(v):
    m = re.match(r"\s*([1-4])", str(v))
    n = int(m.group(1)) if m else _coerce_int(v)
    return {1: "1 - 0 a 5%", 2: "2 - 5 a 25%", 3: "3 - 25 a 60%", 4: "4 - acima de 60%"}.get(n, "Sem classe")


def _cls_risco(v):
    s = _norm_txt(v)
    m = re.search(r"r\s*([1-4])", s)
    if "monitor" in s:
        return "Área em monitoramento"
    if "encerr" in s:
        return "Área encerrada"
    return f"R{m.group(1)}" if m else "Outros"


def _cls_vias(v):
    s = _norm_txt(v)
    if "vtr" in s or "transito rapido" in s:
        return "Via de trânsito rápido"
    if "pedest" in s:
        return "Via de pedestres"
    for k, lab in (("arterial", "Arterial"), ("coletora", "Coletora"), ("local", "Local")):
        if k in s:
            return lab
    return "Outros"


def _cls_dens(v):
    try:
        x = float(str(v).replace(",", "."))
    except Exception:
        return "Sem informação"
    if pd.isna(x):
        return "Sem informação"
    for lim, lab in DENS_BINS:
        if x <= lim:
            return lab
    return DENS_BINS[-1][1]


def _cls_lcz(v):
    return LCZ_LABELS.get(_coerce_int(v), "Sem classe")


CLASSIFIERS = {"corredor": _cls_corredor, "favela": _cls_favela, "decliv": _cls_decliv,
               "risco": _cls_risco, "vias": _cls_vias, "dens": _cls_dens, "lcz": _cls_lcz}


def classify_thematic(key, g):
    cfg = THEMATIC_LAYERS[key]
    g = g.copy()
    lab_col = _col(g, cfg.get("label"))
    g["__tt"] = g[lab_col].astype(str) if lab_col else cfg["title"]
    kind = cfg["kind"]
    if kind == "auto":
        vals = sorted(g[lab_col].dropna().astype(str).unique()) if lab_col else []
        cmap = {v: PALETTE_AUTO[i % len(PALETTE_AUTO)] for i, v in enumerate(vals)}
        g["__leg"] = g[lab_col].astype(str) if lab_col else cfg["title"]
        g["__color"] = g["__leg"].map(cmap).fillna("#888888")
        legend = list(cmap.items())
    elif kind in ("single", "points"):
        g["__leg"] = cfg["title"]; g["__color"] = cfg["color"]
        legend = [(cfg["title"], cfg["color"])]
    else:
        src = _col(g, cfg["class_col"])
        g["__leg"] = g[src].map(CLASSIFIERS[cfg["fn"]]) if src else "Sem informação"
        colors = cfg["colors"]
        g["__color"] = g["__leg"].map(colors).fillna("#bdbdbd")
        present = set(g["__leg"].unique())
        legend = [(k, v) for k, v in colors.items() if k in present]
        if cfg.get("tt_leg"):
            g["__tt"] = g["__leg"]
        if key == "densidade":
            g["__tt"] = g["__tt"] + " hab/hec"
    return g, legend

# =============================================================================
# CSV CLUSTER
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def read_df_csv(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return None
    try:
        return pd.read_csv(p, dtype={QUADRA_ID: "string", ISO_ID: "string", CLUSTER_COL: "string"})
    except Exception:
        return None


def ensure_local_quadras_csv():
    found = _find_local_file(QUADRAS_CSV_FILENAME)
    if found is not None:
        return found
    raw = _get_secret(QUADRAS_CSV_SECRET_KEY) or QUADRAS_CSV_FALLBACK_URL
    dst = DATA_CACHE_DIR / QUADRAS_CSV_FILENAME
    if not raw:
        return dst
    try:
        return download_drive_file(raw, dst, label=dst.name)
    except Exception:
        st.warning(f"Não foi possível baixar {QUADRAS_CSV_FILENAME}."); return dst


def get_quadras_csv_df():
    df = read_df_csv(str(ensure_local_quadras_csv()))
    if df is None or df.empty:
        return None
    df = df.copy()
    cl = {str(c).strip().lower(): c for c in df.columns}
    for canon, low in ((QUADRA_ID, "quadra_id"), (ISO_ID, "iso_id"), (CLUSTER_COL, "cluster")):
        if canon not in df.columns and low in cl:
            df = df.rename(columns={cl[low]: canon})
    if QUADRA_ID in df.columns:
        df[QUADRA_ID] = df[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))
    if ISO_ID in df.columns and QUADRA_ID in df.columns:
        df[ISO_ID] = df[ISO_ID].map(_id_to_str)
        df[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(df[ISO_ID], df[QUADRA_ID])]
    return df


def attach_quadras_csv(g_quad):
    if g_quad is None or g_quad.empty:
        return g_quad
    df = get_quadras_csv_df()
    if df is None:
        return g_quad
    for k in (QUADRA_UID, QUADRA_ID):
        if k in g_quad.columns and k in df.columns:
            return g_quad.merge(df, on=k, how="left", suffixes=("", "_csv"))
    return g_quad


# =============================================================================
# TABELAS EXTERNAS — leitura / agregação / vínculo
# =============================================================================
def _norm_key(v):
    s = _id_to_str(v)
    if s is None:
        return None
    t = re.sub(r"[\s.\-/]", "", s)
    if not t:
        return None
    return (t.lstrip("0") or "0") if t.isdigit() else t.upper()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _read_table(path, mtime=0.0):
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(4096)
    if head[:4] == b"PAR1":
        df = pd.read_parquet(p)
        return df.drop(columns=[c for c in df.columns if str(c).lower() == "geometry"])
    if head[:2] == b"PK":
        return pd.read_excel(p, dtype=str)
    first = head.decode("utf-8", "ignore").splitlines()[0] if head else ""
    sep = max([";", ",", "\t", "|"], key=first.count)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, sep=sep, dtype=str, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return None


def _to_numeric_cols(df, skip):
    out = df.copy()
    for c in out.columns:
        if c in skip or ID_LIKE_RE.search(str(c)) or pd.api.types.is_numeric_dtype(out[c]):
            continue
        s = out[c].astype("string").str.strip()
        if s.str.contains(",", regex=False).any():
            s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        n = pd.to_numeric(s, errors="coerce")
        nn = out[c].notna().sum()
        if nn and n.notna().sum() >= 0.9 * nn:
            out[c] = n
    return out


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _load_joined_table(path, mtime, key_cands):
    df = _read_table(path, mtime)
    if df is None or df.empty:
        return None, None, [], []
    cols_all = [str(c) for c in df.columns]
    low = {_norm_txt(c).replace(" ", "_"): c for c in df.columns}
    key = next((low[_norm_txt(k)] for k in key_cands if _norm_txt(k) in low), None)
    if key is None:
        return None, None, [], cols_all
    df = _to_numeric_cols(df, {key})
    df["__k"] = df[key].map(_norm_key)
    df = df[df["__k"].notna()]
    num = [c for c in df.columns if c not in (key, "__k") and pd.api.types.is_numeric_dtype(df[c])
           and not ID_LIKE_RE.search(str(c))]
    grp = df.groupby("__k")
    agg = grp[num].sum(min_count=1) if num else pd.DataFrame(index=grp.size().index)
    agg.insert(0, "n_registros", grp.size())
    txt = [c for c in df.columns if c not in num and c != "__k"]
    if txt:
        agg = agg.join(grp[txt].first())
    return agg.reset_index(), key, [str(c) for c in num], cols_all


def load_ext_table(src):
    names, sk, url, dst, cands, _ = EXT_TABLES[src]
    p = None
    for n in names:
        p = _find_local_file(n)
        if p is not None:
            break
    if p is None:
        raw = _get_secret(sk) or url
        try:
            p = download_drive_file(raw, DATA_CACHE_DIR / dst, label=dst)
        except Exception as e:
            st.session_state[f"_diag_{src}"] = {"erro": f"Falha ao baixar: {e}"}
            return None, None, [], [], None
    agg, key, num, cols = _load_joined_table(str(p), _meta(p)[1], tuple(cands))
    return agg, key, num, cols, p


def ext_variables(src):
    agg, _k, num, _c, _p = load_ext_table(src)
    return (["n_registros"] + num) if agg is not None else []


def attach_table(g, id_col, src):
    if g is None or g.empty or id_col not in g.columns:
        return g
    agg, key, _num, cols, p = load_ext_table(src)
    if agg is None:
        if p is not None:
            st.session_state[f"_diag_{src}"] = {
                "arquivo": p.name, "erro": f"coluna-chave {EXT_TABLES[src][4][0]!r} não encontrada",
                "colunas_no_arquivo": cols[:80]}
        return g
    g2 = g.copy()
    g2["__k"] = g2[id_col].map(_norm_key)
    out = g2.merge(agg, on="__k", how="left", suffixes=("", f"_{src}"))
    miss = out[out["n_registros"].isna()]
    st.session_state[f"_diag_{src}"] = {
        "arquivo": p.name, "coluna_chave_tabela": key, "coluna_chave_camada": id_col,
        "feicoes_na_tela": int(len(out)), "feicoes_vinculadas": int(out["n_registros"].notna().sum()),
        "chaves_na_tabela": int(len(agg)),
        "ex_ids_camada_sem_vinculo": miss[id_col].head(5).tolist(),
        "ex_chaves_tabela_normalizadas": agg["__k"].head(5).tolist(),
        "colunas_no_arquivo": cols[:80]}
    return out.drop(columns="__k")


def _fmt_num(x):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)


def add_choropleth(m, gdf, id_col, val_col, name, prefix, selected_ids, cache_key, opacity=0.75):
    g = gdf.copy()
    v = pd.to_numeric(g[val_col], errors="coerce")
    vv = v.dropna()
    if vv.empty:
        st.info(f"Sem valores para '{val_col}' nas feições exibidas."); return False
    edges = sorted(set(vv.quantile([0, .2, .4, .6, .8, 1]).tolist()))
    if len(edges) < 2:
        edges = [float(vv.min()), float(vv.max()) + 1e-9]
    k = len(edges) - 1
    cols = CHORO_COLORS[-k:]
    idx = pd.cut(v, bins=edges, labels=False, include_lowest=True)
    g["__ch_color"] = [cols[int(i)] if pd.notna(i) else CLUSTER_NULL_COLOR for i in idx]
    g["__ch_tt"] = [f"{_id_to_str(a)} — {val_col}: {_fmt_num(b)}" for a, b in zip(g[id_col], v)]
    add_polygons_selectable_colored(m, g, name, id_col, "__ch_color", selected_ids=selected_ids,
        tooltip_col="__ch_tt", fill_opacity=opacity, selected_fill_opacity=0.0,
        tooltip_prefix=prefix, cache_key=cache_key, default_fill=CLUSTER_NULL_COLOR)
    add_legend(m, val_col, [(f"{_fmt_num(edges[i])} – {_fmt_num(edges[i + 1])}", cols[i])
                            for i in range(k)] + [("Sem dado", CLUSTER_NULL_COLOR)])
    return True


def cluster_color(code):
    return CLUSTER_NULL_COLOR if code is None else CLUSTER_COLOR_MAP.get(code, CLUSTER_NULL_COLOR)


def iso_label_color(nova_class):
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
# EXPORT (CSV / imagem / legenda)
# =============================================================================
def gdf_to_csv_bytes(gdf) -> bytes:
    if gdf is None or len(gdf) == 0:
        return b""
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")).copy()
    if "__leg" in df.columns:
        df = df.rename(columns={"__leg": "legenda"})
    df = df[[c for c in df.columns if not str(c).startswith("__")]]
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def set_export(slot, name, gdf):
    st.session_state[slot] = (name, gdf_to_csv_bytes(gdf)) if gdf is not None and len(gdf) else None


def add_print_button(m):
    """Botão PNG com retry — não quebra o mapa se o plugin demorar a carregar."""
    if MacroElement is None or m is None:
        return
    m.get_root().header.add_child(JavascriptLink(
        "https://cdn.jsdelivr.net/npm/leaflet-easyprint@2.1.9/dist/bundle.min.js"))
    el = MacroElement()
    el._template = Template("""
    {% macro script(this, kwargs) %}
    (function addPrint(n){
      try {
        if (window.L && L.easyPrint) {
          L.easyPrint({title:'Baixar imagem (PNG)', position:'topleft', sizeModes:['Current'],
                       exportOnly:true, filename:'planbairros_mapa', hideControlContainer:false
          }).addTo({{this._parent.get_name()}});
        } else if (n < 40) { setTimeout(function(){ addPrint(n+1); }, 250); }
      } catch(e) { console.warn('easyPrint:', e); }
    })(0);
    {% endmacro %}""")
    m.add_child(el)


def add_legend(m, title, items):
    if MacroElement is None or m is None or not items:
        return
    rows = "".join(
        f"<div style='margin:2px 0'><span style='background:{c};width:14px;height:14px;"
        f"display:inline-block;margin-right:6px;border:1px solid #555;vertical-align:middle'></span>"
        f"{html.escape(str(l))}</div>" for l, c in items[:25])
    box = (f"<div style='background:#fff;padding:8px 10px;border-radius:8px;"
           f"box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px Roboto,Arial;max-height:320px;overflow:auto'>"
           f"<b>{html.escape(title)}</b>{rows}</div>")
    el = MacroElement()
    el._template = Template("""
    {% macro script(this, kwargs) %}
    var lg = L.control({position:'bottomright'});
    lg.onAdd = function(){var d = L.DomUtil.create('div'); d.innerHTML = """ + json.dumps(box) + """; return d;};
    lg.addTo({{this._parent.get_name()}});
    {% endmacro %}""")
    m.add_child(el)

# =============================================================================
# HEADER / CSS
# =============================================================================
def _logo_data_uri():
    if LOGO_PATH.exists():
        suf = LOGO_PATH.suffix.lstrip(".").lower()
        mime = "jpeg" if suf in ("jpg", "jpeg") else suf
        return f"data:image/{mime};base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode('utf-8')}"
    return "https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logomark/streamlit-mark-color.png"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
        html, body, .stApp {{ font-family: 'Roboto', Arial, sans-serif; }}
        .main .block-container {{ padding-top:.15rem !important; padding-bottom:.6rem !important; }}
        .pb-row {{ display:flex; align-items:center; gap:12px; margin-bottom:0; }}
        .pb-logo {{ height:{LOGO_HEIGHT}px; width:auto; display:block; border-radius:8px; }}
        .pb-header {{ background:{PB_NAVY}; color:#fff; border-radius:14px; padding:14px 15px; width:100%; }}
        .pb-title {{ font-size:2.25rem; font-weight:900; line-height:1.05; }}
        .pb-subtitle {{ font-size:1.2rem; opacity:.95; margin-top:5px; }}
        .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{
            font-size:1.12rem !important; line-height:1.65 !important; }}
        .stMarkdown h2 {{ font-size:1.9rem !important; }}
        .stMarkdown h3 {{ font-size:1.45rem !important; }}
        div[data-testid="stCaptionContainer"] p {{ font-size:0.98rem !important; }}
        label p, .stButton button p, .stDownloadButton button p,
        div[data-baseweb="select"] div, button[data-baseweb="tab"] p {{ font-size:1.05rem !important; }}
        .pb-card {{ background:#fff; border:1px solid rgba(20,64,125,.10);
            box-shadow:0 1px 2px rgba(0,0,0,.04); border-radius:14px; padding:12px; }}
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stBaseButton-primary"] > button {{
            background:{PB_BTN} !important; color:#fff !important; border:1px solid {PB_BTN} !important; }}
        </style>
        """, unsafe_allow_html=True)


def render_header():
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
        """, unsafe_allow_html=True)

# =============================================================================
# STATE
# =============================================================================
MAP_KEY = "map_view"
LEVEL_LABELS = {"subpref": "Subprefeituras", "distrito": "Distritos",
                "isocrona": "Isócronas", "quadra": "Visualização detalhada"}


def init_state():
    defaults = {
        "level": "subpref", "last_level": None, "selected_subpref_id": None,
        "selected_distrito_id": None, "selected_iso_ids": set(), "selected_censo_ids": set(),
        "selected_od_ids": set(), "selected_quadra_ids": set(), "selected_lote_ids": set(),
        "view_center": (-23.55, -46.63), "view_zoom": 11, "last_click_sig": "", "last_draw_sig": "",
        "_geojson_cache": {}, "_geojson_cache_order": [], "_layer_cache": {}, "_layer_cache_meta": {},
        "_ui_action_sig": 0, "_ui_action_sig_seen": 0, "_map_level_rendered": None,
        "variable": None, "selection_draw_mode": False, "post_iso_view": "quadra",
        "show_fixed_layers": True, "thematic_layer": "Nenhum", "_export_view": None,
        "_export_thematic": None, "_export_map_html": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def mark_ui_action():
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1


def _geojson_cache_reset():
    st.session_state["_geojson_cache"] = {}
    st.session_state["_geojson_cache_order"] = []


def reset_post_iso_state():
    st.session_state["post_iso_view"] = "quadra"
    for k in ("selected_censo_ids", "selected_od_ids", "selected_quadra_ids", "selected_lote_ids"):
        st.session_state[k] = set()


def reset_to(level, *, clear_click_sig=True):
    ss = st.session_state
    ss["level"] = level
    if clear_click_sig:
        ss["last_click_sig"] = ""; ss["last_draw_sig"] = ""
    _geojson_cache_reset()
    if level == "subpref":
        ss["selected_subpref_id"] = None; ss["selected_distrito_id"] = None
        ss["selected_iso_ids"] = set(); reset_post_iso_state()
        ss["view_center"] = (-23.55, -46.63); ss["view_zoom"] = 11; ss["last_level"] = None
    elif level == "distrito":
        ss["selected_distrito_id"] = None; ss["selected_iso_ids"] = set(); reset_post_iso_state()
    elif level == "isocrona":
        ss["selected_iso_ids"] = set(); reset_post_iso_state()
    elif level == "quadra":
        reset_post_iso_state()


def _available_levels():
    ss = st.session_state
    out = ["subpref"]
    if _id_to_str(ss.get("selected_subpref_id")):
        out.append("distrito")
    if _id_to_str(ss.get("selected_distrito_id")):
        out.append("isocrona")
    if ensure_set_of_str(ss.get("selected_iso_ids")):
        out.append("quadra")
    return out


def _on_nav_change():
    target = st.session_state.get("nav_level", "subpref")
    if target != st.session_state.get("level"):
        st.session_state["level"] = target
        st.session_state["last_level"] = None
        _geojson_cache_reset()
    mark_ui_action()


def _toggle_in_set(key, value):
    s = set(st.session_state.get(key, set()) or set())
    s.discard(value) if value in s else s.add(value)
    st.session_state[key] = s


def sanitize_level_state():
    ss = st.session_state
    lvl = ss.get("level", "subpref")
    if lvl == "distrito" and _id_to_str(ss.get("selected_subpref_id")) is None:
        reset_to("subpref"); return
    if lvl in ("isocrona", "quadra") and _id_to_str(ss.get("selected_distrito_id")) is None:
        reset_to("distrito" if _id_to_str(ss.get("selected_subpref_id")) else "subpref"); return
    if lvl == "quadra" and not ss.get("selected_iso_ids"):
        reset_to("isocrona")

# =============================================================================
# GEOJSON CACHE
# =============================================================================
def _session_geojson_get(key):
    return st.session_state.get("_geojson_cache", {}).get(key)


def _session_geojson_set(key, value, max_items=120):
    cache = st.session_state.get("_geojson_cache", {})
    order = st.session_state.get("_geojson_cache_order", [])
    if key in order:
        order.remove(key)
    cache[key] = value; order.append(key)
    while len(order) > max_items:
        cache.pop(order.pop(0), None)
    st.session_state["_geojson_cache"] = cache
    st.session_state["_geojson_cache_order"] = order


def _simplify_to_geojson(gdf, simplify_tol, keep_cols=None):
    if gdf is None or gdf.empty:
        return ""
    keep_cols = [c for c in (keep_cols or []) if c in gdf.columns]
    g = _drop_bad_geoms(gdf[keep_cols + ["geometry"]].copy())
    if simplify_tol and simplify_tol > 0:
        try:
            g["geometry"] = g["geometry"].simplify(simplify_tol, preserve_topology=True)
            g = _drop_bad_geoms(g)
        except Exception:
            pass
    try:
        return g.to_json()
    except Exception:
        return ""

# =============================================================================
# CLICK / DRAW
# =============================================================================
def pick_feature_id(gdf, click_latlon, id_col):
    if gdf is None or gdf.empty or not click_latlon or id_col not in gdf.columns or Point is None:
        return None
    lat, lng = click_latlon.get("lat"), click_latlon.get("lng")
    if lat is None or lng is None:
        return None
    try:
        pt = Point(lng, lat)
        idx = list(gdf.sindex.query(pt, predicate="intersects"))
        return _id_to_str(gdf.iloc[idx[0]][id_col]) if idx else None
    except Exception:
        hit = gdf[gdf.geometry.intersects(Point(lng, lat))]
        return None if hit.empty else _id_to_str(hit.iloc[0][id_col])


def add_draw_tools(m):
    if Draw is None or m is None:
        return
    Draw(export=False, position="topleft",
         draw_options={"polyline": False, "marker": False, "circle": False,
                       "circlemarker": False, "polygon": True, "rectangle": True},
         edit_options={"edit": False, "remove": True}).add_to(m)


def _extract_drawn_geometry(map_state):
    if not isinstance(map_state, dict) or shape is None:
        return None
    for cand in (map_state.get("all_drawings"), map_state.get("last_active_drawing")):
        if not cand:
            continue
        item = cand[-1] if isinstance(cand, list) else cand
        if isinstance(item, dict):
            geom = item.get("geometry", item)
            try:
                return shape(geom)
            except Exception:
                pass
    return None


def select_features_by_geometry(gdf, geom, id_col, state_key):
    if gdf is None or gdf.empty or geom is None or id_col not in gdf.columns:
        return
    try:
        hits = gdf[gdf.geometry.intersects(geom)]
    except Exception:
        return
    ids = ensure_set_of_str(hits[id_col].tolist())
    st.session_state[state_key] = ensure_set_of_str(st.session_state.get(state_key, set())) | ids


def _pick_id_from_last_object(map_state, id_col):
    obj = (map_state or {}).get("last_object_clicked")
    if not isinstance(obj, dict):
        return None
    props = obj.get("properties") if isinstance(obj.get("properties"), dict) else obj
    return _id_to_str(props.get(id_col)) if isinstance(props, dict) else None


def parse_tooltip_id(tooltip):
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


def _click_signature(picked_id, click):
    try:
        return f"{picked_id}|{float(click['lat']):.7f}|{float(click['lng']):.7f}"
    except Exception:
        return f"{picked_id}|"

# =============================================================================
# FOLIUM
# =============================================================================
def make_base_map(center=(-23.55, -46.63), zoom=11):
    if folium is None:
        return None
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(tiles=BASEMAP_URL, attr=BASEMAP_ATTR, name="Mapa base",
                     overlay=False, control=False, max_zoom=19, max_native_zoom=16).add_to(m)
    try:
        for name, z in (("parent_fill", 610), ("thematic", 625), ("detail_shapes", 640)):
            folium.map.CustomPane(name, z_index=z).add_to(m)
        # camadas fixas não capturam cliques (antes bloqueavam a seleção de polígonos)
        folium.map.CustomPane("fixed_layers", z_index=660, pointer_events=False).add_to(m)
        folium.map.CustomPane("boundaries", z_index=680, pointer_events=False).add_to(m)
        folium.map.CustomPane("basemap_labels", z_index=690, pointer_events=False).add_to(m)
        folium.map.CustomPane("labels", z_index=700).add_to(m)
    except Exception:
        pass
    folium.TileLayer(tiles=BASEMAP_LABELS_URL, attr=BASEMAP_ATTR, name="Rótulos do mapa base",
                     overlay=True, control=True, show=True, max_zoom=19, max_native_zoom=16,
                     pane="basemap_labels").add_to(m)
    return m


def _mk_tooltip(id_col, prefix):
    if GeoJsonTooltip is None:
        return None
    return GeoJsonTooltip(fields=[id_col], aliases=[prefix], sticky=True,
                          labels=True, localize=True, max_width=320)


def add_fixed_layers(m):
    if m is None or not st.session_state.get("show_fixed_layers", True):
        return
    for key, s in FIXED_LAYER_STYLE.items():
        src = _find_local_file(s["file"])
        if src is None:
            continue
        cache_key = f"fixed:{key}:{src.name}"
        geojson = _session_geojson_get(cache_key)
        if not geojson:
            g = _drop_bad_geoms(read_gdf_geojson(str(src)))
            if g is None or g.empty:
                continue
            geojson = _simplify_to_geojson(g, 0.0, [])
            _session_geojson_set(cache_key, geojson)
        if not geojson:
            continue
        fg = folium.FeatureGroup(name=s["name"], show=True)

        def _fx_style(feat, c=s["color"], w=s["weight"], f=s["fill"], o=s["fill_opacity"], d=s.get("dash")):
            gtype = ((feat or {}).get("geometry") or {}).get("type", "")
            if "Point" in gtype:
                return {"color": c, "weight": 1.0, "opacity": 0.9, "fill": True,
                        "fillColor": "#ffffff", "fillOpacity": 1.0, "radius": STATION_RADIUS}
            return {"color": c, "weight": w, "opacity": 1.0, "dashArray": d,
                    "lineCap": LINE_CAP, "lineJoin": LINE_JOIN,
                    "fill": f, "fillColor": c, "fillOpacity": o if f else 0.0}

        folium.GeoJson(
            data=geojson, pane="fixed_layers", smooth_factor=SMOOTH_FACTOR,
            marker=folium.CircleMarker(radius=STATION_RADIUS, weight=1, fill=True,
                                       fill_color="#ffffff", fill_opacity=1.0),
            style_function=_fx_style,
        ).add_to(fg)
        fg.add_to(m)


def add_boundary_overlay(m, gdf, *, color=PB_BLACK, weight=1.6, simplify_tol=0.0, cache_key=None):
    if m is None or gdf is None or gdf.empty:
        return
    key = cache_key or f"bnd:{len(gdf)}:{simplify_tol}"
    geojson = _session_geojson_get(key)
    if not geojson:
        geojson = _simplify_to_geojson(gdf, simplify_tol, [])
        _session_geojson_set(key, geojson)
    if not geojson:
        return
    folium.GeoJson(data=geojson, pane="boundaries", smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {"color": color, "weight": weight, "opacity": 1.0,
                                   "fill": False, "fillOpacity": 0.0,
                                   "lineCap": LINE_CAP, "lineJoin": LINE_JOIN}).add_to(m)


def add_thematic_to_map(m, key, g, legend):
    cfg = THEMATIC_LAYERS[key]
    if g is None or g.empty:
        st.info(f"{cfg['title']}: nenhuma feição nas isócronas selecionadas.")
        return
    if cfg["kind"] == "points":
        pts = g.geometry.representative_point()
        if len(pts) > MAX_POINTS:
            st.caption(f"Exibindo {MAX_POINTS} de {len(pts)} árvores.")
            pts = pts.iloc[:MAX_POINTS]
        fg = folium.FeatureGroup(name=cfg["title"], show=True)
        for p in pts:
            folium.CircleMarker(location=[p.y, p.x], radius=3, color=cfg["color"], weight=1,
                                fill=True, fill_color=cfg["color"], fill_opacity=0.9,
                                pane="thematic").add_to(fg)
        fg.add_to(m)
    else:
        geojson = _simplify_to_geojson(g, 0.0, ["__color", "__tt", "__leg"])
        if not geojson:
            return
        w = cfg.get("weight", 0.6)

        def _style(f, w=w):
            c = (f.get("properties") or {}).get("__color", "#888888")
            return {"color": c, "weight": w, "opacity": 1.0, "fillColor": c,
                    "fillOpacity": 0.6, "lineCap": LINE_CAP, "lineJoin": LINE_JOIN}

        tt = GeoJsonTooltip(fields=["__tt", "__leg"], aliases=[f"{cfg['title']}:", "Legenda:"],
                            sticky=True, labels=True) if GeoJsonTooltip else None
        folium.GeoJson(data=geojson, name=cfg["title"], pane="thematic",
                       smooth_factor=SMOOTH_FACTOR, style_function=_style, tooltip=tt).add_to(m)
    add_legend(m, cfg["title"], legend)


def _format_label_multiline(text):
    txt = str(text or "").strip()
    if not txt:
        return ""
    for sep in (" - ", " – ", "/"):
        if sep in txt:
            parts = [p.strip() for p in txt.split(sep) if p.strip()]
            if len(parts) >= 2:
                return "<br>".join(html.escape(p) for p in parts[:2])
    return html.escape(txt)


def add_labels_on_map(m, gdf, label_col, *, font_size=12, color="#000000", weight="700"):
    if m is None or gdf is None or gdf.empty or label_col not in gdf.columns:
        return
    g = gdf[gdf.geometry.notna()]
    pts = g.geometry.representative_point()
    for idx, row in g.iterrows():
        label = row.get(label_col)
        if pd.isna(label):
            continue
        txt_html = _format_label_multiline(label)
        pt = pts.loc[idx]
        if not txt_html or pt is None or pt.is_empty:
            continue
        folium.Marker(location=[pt.y, pt.x], interactive=False, icon=folium.DivIcon(
            icon_size=(150, 36), icon_anchor=(75, 18),
            html=f"""<div style="font-family:Roboto,Arial,sans-serif;font-size:{font_size}px;
                color:{color};font-weight:{weight};text-align:center;white-space:nowrap;line-height:1.1;
                text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff,0 0 3px #fff;
                pointer-events:none;">{txt_html}</div>""")).add_to(m)


def add_parent_fill(m, gdf, name, *, pane="parent_fill", fill_color=PB_BROWN,
                    fill_opacity=PARENT_FILL_OPACITY, stroke_color=PB_BLACK,
                    stroke_weight=PARENT_STROKE_WEIGHT, stroke_opacity=PARENT_STROKE_OPACITY,
                    dash_array=PARENT_STROKE_DASH, simplify_tol=0.0, cache_key=None):
    if m is None or gdf is None or gdf.empty:
        return
    key = cache_key or f"parent:{name}:{simplify_tol}:{len(gdf)}"
    geojson = _session_geojson_get(key)
    if not geojson:
        geojson = _simplify_to_geojson(gdf, simplify_tol, [])
        _session_geojson_set(key, geojson)
    if not geojson:
        return
    fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(data=geojson, pane=pane, smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {"color": stroke_color, "weight": stroke_weight,
            "opacity": stroke_opacity, "dashArray": dash_array, "lineCap": LINE_CAP,
            "lineJoin": LINE_JOIN, "fillColor": fill_color, "fillOpacity": fill_opacity}).add_to(fg)
    fg.add_to(m)


def _add_selected_overlay(m, gdf, id_col, sel, name, pane, simplify_tol,
                          selected_color, selected_weight, fill_color, selected_fill_opacity):
    if not sel:
        return
    sel_gdf = gdf[gdf[id_col].isin(list(sel))][[id_col, "geometry"]].copy()
    if sel_gdf.empty:
        return
    sel_gdf[id_col] = sel_gdf[id_col].map(_id_to_str)
    geojson_sel = _simplify_to_geojson(sel_gdf, simplify_tol, [id_col])
    if not geojson_sel:
        return
    fg_sel = folium.FeatureGroup(name=f"{name} (selecionados)", show=True)
    folium.GeoJson(data=geojson_sel, pane=pane, smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {"color": selected_color, "weight": selected_weight,
            "opacity": 1.0, "lineCap": LINE_CAP, "lineJoin": LINE_JOIN,
            "fillColor": fill_color, "fillOpacity": selected_fill_opacity}).add_to(fg_sel)
    fg_sel.add_to(m)


def add_polygons_selectable(m, gdf, name, id_col, *, tooltip_col=None, selected_ids=None,
                            pane="detail_shapes", base_color=PB_BLACK, base_weight=1.0,
                            fill_color="#ffffff", fill_opacity=0.10, selected_color=PB_BLACK,
                            selected_weight=2.2, selected_fill_opacity=0.26, tooltip_prefix="ID: ",
                            simplify_tol=0.0, cache_key=None):
    if m is None or gdf is None or gdf.empty or id_col not in gdf.columns:
        return
    tooltip_col = tooltip_col if tooltip_col in gdf.columns else id_col
    keep = list(dict.fromkeys([id_col, tooltip_col]))
    key = cache_key or f"base:{name}:{id_col}:{tooltip_col}:{simplify_tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[keep + ["geometry"]].copy()
        for c in keep:
            mini[c] = mini[c].map(_id_to_str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol, keep)
        _session_geojson_set(key, geojson_base)
    if not geojson_base:
        return
    fg_base = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(data=geojson_base, pane=pane, smooth_factor=SMOOTH_FACTOR,
        style_function=lambda _f: {"color": base_color, "weight": base_weight, "opacity": 1.0,
            "lineCap": LINE_CAP, "lineJoin": LINE_JOIN, "fillColor": fill_color, "fillOpacity": fill_opacity},
        highlight_function=lambda _f: {"color": PB_BLACK, "weight": base_weight + 1.0,
            "opacity": 1.0, "fillOpacity": min(fill_opacity + 0.10, 0.40)},
        tooltip=_mk_tooltip(tooltip_col, tooltip_prefix)).add_to(fg_base)
    fg_base.add_to(m)
    _add_selected_overlay(m, gdf, id_col, ensure_set_of_str(selected_ids), name, pane, simplify_tol,
                          selected_color, selected_weight, fill_color, selected_fill_opacity)


def add_polygons_selectable_colored(m, gdf, name, id_col, fill_color_col, *, selected_ids=None,
                                    tooltip_col=None, pane="detail_shapes", base_color=PB_BLACK,
                                    base_weight=1.0, fill_opacity=0.14, selected_color=PB_BLACK,
                                    selected_weight=2.2, selected_fill_opacity=0.28,
                                    tooltip_prefix="ID: ", simplify_tol=0.0, cache_key=None,
                                    default_fill="#ffffff"):
    if m is None or gdf is None or gdf.empty or id_col not in gdf.columns or fill_color_col not in gdf.columns:
        return
    tooltip_col = tooltip_col if tooltip_col in gdf.columns else id_col
    keep = list(dict.fromkeys([id_col, fill_color_col, tooltip_col]))
    key = cache_key or f"baseC:{name}:{id_col}:{tooltip_col}:{fill_color_col}:{simplify_tol}:{len(gdf)}"
    geojson_base = _session_geojson_get(key)
    if not geojson_base:
        mini = gdf[keep + ["geometry"]].copy()
        mini[id_col] = mini[id_col].map(_id_to_str)
        mini[tooltip_col] = mini[tooltip_col].map(_id_to_str)
        mini[fill_color_col] = mini[fill_color_col].astype(str)
        geojson_base = _simplify_to_geojson(mini, simplify_tol, keep)
        _session_geojson_set(key, geojson_base)
    if not geojson_base:
        return

    def _style(f):
        fc = ((f or {}).get("properties") or {}).get(fill_color_col, default_fill)
        if not fc or str(fc).lower() in ("nan", "none"):
            fc = default_fill
        return {"color": base_color, "weight": base_weight, "opacity": 1.0,
                "lineCap": LINE_CAP, "lineJoin": LINE_JOIN, "fillColor": fc, "fillOpacity": fill_opacity}

    fg_base = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(data=geojson_base, pane=pane, smooth_factor=SMOOTH_FACTOR, style_function=_style,
        highlight_function=lambda _f: {"color": PB_BLACK, "weight": base_weight + 1.0,
            "opacity": 1.0, "fillOpacity": min(fill_opacity + 0.10, 0.95)},
        tooltip=_mk_tooltip(tooltip_col, tooltip_prefix)).add_to(fg_base)
    fg_base.add_to(m)
    _add_selected_overlay(m, gdf, id_col, ensure_set_of_str(selected_ids), name, pane, simplify_tol,
                          selected_color, selected_weight, "#ffffff", selected_fill_opacity)

# =============================================================================
# HELPERS PÓS-ISÓCRONA
# =============================================================================
def _ds_codigo_of_distrito(d):
    """Retorna o ds_codigo do distrito clicado (Distritos.parquet)."""
    g_dist = read_layer("dist")
    if g_dist is None or g_dist.empty or DS_CODIGO not in g_dist.columns or d is None:
        return None
    row = g_dist[g_dist[DIST_ID].map(_norm_key) == _norm_key(d)]
    return first_non_null_value(row, DS_CODIGO)


def _isos_of_distrito(g_iso, d):
    """Filtra isocronas.parquet pelo ds_codigo do distrito clicado."""
    if g_iso is None or g_iso.empty or DS_CODIGO not in g_iso.columns or d is None:
        return _empty(g_iso)
    cod = _ds_codigo_of_distrito(d)
    if cod is None:
        return _empty(g_iso)
    return g_iso[g_iso[DS_CODIGO].map(_norm_key) == _norm_key(cod)].copy()


def build_post_iso_data():
    iso_ids = ensure_set_of_str(st.session_state.get("selected_iso_ids", set()))
    distrito_id = _id_to_str(st.session_state.get("selected_distrito_id"))
    out = {"iso_ids": iso_ids, "g_parent": None, "g_censo": None,
           "g_od": None, "g_quadra": None, "g_lote": None, "quadra_id_col": QUADRA_UID}
    if not iso_ids:
        return out
    g_iso = read_layer("iso")
    if g_iso is not None:
        out["g_parent"] = subset_by_id_multi(g_iso, ISO_ID, iso_ids)
    g_censo = read_layer("censo")
    if g_censo is not None:
        out["g_censo"] = get_censo_subset_for_isos(g_censo, iso_ids)
    post_view_now = st.session_state.get("post_iso_view", "quadra")
    g_od = read_layer("od")
    if g_od is not None:
        if ISO_ID in g_od.columns:
            out["g_od"] = subset_by_parent_multi(g_od, ISO_ID, iso_ids)
            if post_view_now == "od":
                out["g_od"] = attach_table(out["g_od"], OD_ID, "od")
        else:
            st.warning(f"ZonasOD sem '{ISO_ID}'. Colunas: {list(g_od.columns)}")
    post_view = st.session_state.get("post_iso_view", "quadra")
    if post_view == "quadra" or st.session_state.get("selected_quadra_ids"):
        g_quad = read_layer("quadra")
        if g_quad is not None:
            censo_ids = ensure_set_of_str(st.session_state.get("selected_censo_ids", set()))
            g_quad_show = get_quadras_subset_for_mode(g_quad, iso_ids=iso_ids, filter_censo_ids=censo_ids)
            id_col = QUADRA_UID if QUADRA_UID in g_quad_show.columns else QUADRA_ID
            out["g_quadra"] = g_quad_show
            out["quadra_id_col"] = id_col
    if post_view == "lote" and distrito_id is not None:
        g_lote = read_lotes_by_distrito(distrito_id)
        if g_lote is not None:
            if ISO_ID in g_lote.columns:
                out["g_lote"] = attach_table(get_lotes_subset_for_isos(g_lote, iso_ids), LOTE_ID, "iptu")
            else:
                st.warning(f"Lotes do distrito '{distrito_id}' sem '{ISO_ID}'.")
    return out


def _post_cfg(data):
    return {"censo": (data.get("g_censo"), CENSO_ID, "selected_censo_ids"),
            "od": (data.get("g_od"), OD_ID, "selected_od_ids"),
            "lote": (data.get("g_lote"), LOTE_ID, "selected_lote_ids")}

# =============================================================================
# EVENTOS
# =============================================================================
def _register_click(picked, click):
    if not picked:
        return False
    sig = _click_signature(picked, click)
    if sig == st.session_state.get("last_click_sig", ""):
        return False
    st.session_state["last_click_sig"] = sig
    return True


def _click_point(map_state):
    for k in ("last_clicked", "last_object_clicked"):
        c = (map_state or {}).get(k)
        if isinstance(c, dict) and c.get("lat") is not None and c.get("lng") is not None:
            return {"lat": c["lat"], "lng": c["lng"]}
    return None


def consume_map_event(level, map_state):
    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip")
    click = _click_point(map_state)

    if level in ("subpref", "distrito"):
        id_col = SUBPREF_ID if level == "subpref" else DIST_ID
        picked = None
        if click:
            if level == "subpref":
                g = read_layer("subpref")
                picked = pick_feature_id(g, click, SUBPREF_ID) if g is not None else None
            else:
                sp = _id_to_str(st.session_state.get("selected_subpref_id"))
                g = read_layer("dist")
                if g is not None and sp:
                    picked = pick_feature_id(subset_by_parent(g, DIST_PARENT, sp), click, DIST_ID)
        picked = picked or _pick_id_from_last_object(map_state, id_col)
        if not _register_click(picked, click):
            return
        if level == "subpref":
            reset_to("distrito", clear_click_sig=False)
            st.session_state["selected_subpref_id"] = picked
        else:
            reset_to("isocrona", clear_click_sig=False)
            st.session_state["selected_distrito_id"] = picked
        return

    if level == "isocrona":
        picked = None
        if click:
            d = _id_to_str(st.session_state.get("selected_distrito_id"))
            g = read_layer("iso")
            if g is not None and d:
                picked = pick_feature_id(_isos_of_distrito(g, d), click, ISO_ID)
        picked = picked or _pick_id_from_last_object(map_state, ISO_ID) or parse_tooltip_id(tooltip_raw)
        if _register_click(picked, click):
            _toggle_in_set("selected_iso_ids", picked)
        return

    if level == "quadra":
        post_view = st.session_state.get("post_iso_view", "quadra")
        data = build_post_iso_data()
        if post_view == "quadra":
            g_show = data.get("g_quadra")
            id_col = data.get("quadra_id_col", QUADRA_UID)
            picked = _pick_id_from_last_object(map_state, id_col)
            if not picked and click and g_show is not None:
                picked = pick_feature_id(g_show, click, id_col)
            if _register_click(picked, click):
                _toggle_in_set("selected_quadra_ids", picked)
            return
        cfg = _post_cfg(data)
        if post_view in cfg:
            g_show, id_col, state_key = cfg[post_view]
            picked = _pick_id_from_last_object(map_state, id_col) or parse_tooltip_id(tooltip_raw)
            if not picked and click and g_show is not None:
                picked = pick_feature_id(g_show, click, id_col)
            if _register_click(picked, click):
                _toggle_in_set(state_key, picked)


def consume_draw_selection(level, map_state):
    if not st.session_state.get("selection_draw_mode"):
        return
    geom = _extract_drawn_geometry(map_state)
    if geom is None:
        return
    sig = str(getattr(geom, "wkt", ""))
    if not sig or sig == st.session_state.get("last_draw_sig", ""):
        return
    st.session_state["last_draw_sig"] = sig
    if level == "isocrona":
        g = read_layer("iso")
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if g is not None and d:
            select_features_by_geometry(_isos_of_distrito(g, d), geom, ISO_ID, "selected_iso_ids")
        return
    if level == "quadra":
        data = build_post_iso_data()
        post_view = st.session_state.get("post_iso_view", "quadra")
        if post_view == "quadra":
            select_features_by_geometry(data.get("g_quadra"), geom,
                                        data.get("quadra_id_col", QUADRA_UID), "selected_quadra_ids")
            return
        cfg = _post_cfg(data)
        if post_view in cfg:
            g_show, id_col, state_key = cfg[post_view]
            select_features_by_geometry(g_show, geom, id_col, state_key)

# =============================================================================
# UI
# =============================================================================
def _variables_for_level(level):
    return {"subpref": ["Subprefeituras"], "distrito": ["Distritos"],
            "isocrona": ["Isócronas", "Isócronas (classes)"],
            "quadra": ["Quadras", "Cluster"]}.get(level, ["Nível"])


def ensure_variable_for_level(level):
    opts = _variables_for_level(level)
    if st.session_state.get("variable") not in opts:
        st.session_state["variable"] = opts[0]


def bounds_center_zoom(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    center = ((miny + maxy) / 2, (minx + maxx) / 2)
    dx = max(maxx - minx, maxy - miny)
    z = 16 if dx < 0.01 else 15 if dx < 0.03 else 14 if dx < 0.08 else 13 if dx < 0.15 else 12 if dx < 0.30 else 11
    return center, z


def set_view_to_gdf(gdf, bump=0, zmax=18):
    if gdf is None or gdf.empty:
        return
    try:
        center, zoom = bounds_center_zoom(gdf)
        st.session_state["view_center"] = center
        st.session_state["view_zoom"] = min(zoom + bump, zmax)
    except Exception:
        pass


def _fit_selected_isos():
    iso_ids = ensure_set_of_str(st.session_state.get("selected_iso_ids"))
    g_iso = read_layer("iso")
    if iso_ids and g_iso is not None:
        set_view_to_gdf(subset_by_id_multi(g_iso, ISO_ID, iso_ids))


def _fit_selected_post_level():
    data = build_post_iso_data()
    post_view = st.session_state.get("post_iso_view", "quadra")
    if post_view == "quadra":
        g, id_col, sk, bump, zmax = data.get("g_quadra"), data.get("quadra_id_col"), "selected_quadra_ids", 1, 19
    else:
        g, id_col, sk = _post_cfg(data)[post_view]
        bump, zmax = (1, 20) if post_view == "lote" else (0, 18)
    ids = ensure_set_of_str(st.session_state.get(sk))
    if g is not None and ids and id_col in g.columns:
        set_view_to_gdf(subset_by_id_multi(g, id_col, ids), bump=bump, zmax=zmax)


def _thematic_enabled():
    lvl = st.session_state.get("level", "subpref")
    return lvl in ("isocrona", "quadra") and bool(ensure_set_of_str(st.session_state.get("selected_iso_ids")))


def _go_detailed():
    mark_ui_action()
    st.session_state["post_iso_view"] = "quadra"
    st.session_state["level"] = "quadra"
    st.session_state["last_level"] = None


def control_panel():
    ss = st.session_state
    lvl = ss.get("level", "subpref")

    render_info_buttons(lvl)
    st.subheader("Navegação", anchor=False)
    avail = _available_levels()
    ss["nav_level"] = lvl if lvl in avail else avail[0]
    st.selectbox("Nível", options=avail, format_func=lambda x: LEVEL_LABELS[x],
                 key="nav_level", on_change=_on_nav_change)
    st.button("Reset", type="primary", use_container_width=True,
              on_click=lambda: (mark_ui_action(), reset_to("subpref")))
    st.divider()

    st.subheader("Variável", anchor=False)
    ensure_variable_for_level(lvl)
    st.selectbox("Variável", options=_variables_for_level(lvl), key="variable", on_change=mark_ui_action)
    st.divider()

    st.subheader("Ações e seleção", anchor=False)
    ok_iso = bool(ensure_set_of_str(ss.get("selected_iso_ids")))
    if lvl == "isocrona":
        st.button("Ajustar às isócronas selecionadas", use_container_width=True, disabled=not ok_iso,
                  on_click=lambda: (mark_ui_action(), _fit_selected_isos()))
        st.button("Avançar para Visualização detalhada", type="primary", use_container_width=True,
                  disabled=not ok_iso, on_click=_go_detailed)
        st.caption("Selecione uma ou mais isócronas antes de avançar.")

    st.selectbox("Visualização pós-isócronas", options=["quadra", "lote", "censo", "od"],
                 format_func=lambda x: {"quadra": "Quadras", "lote": "Lotes",
                                        "censo": "Setor censitário", "od": "Zonas OD"}[x],
                 key="post_iso_view", disabled=(lvl != "quadra"), on_change=mark_ui_action)
    pv = ss.get("post_iso_view", "quadra")
    if lvl == "quadra" and pv in ("od", "lote"):
        src = "od" if pv == "od" else "iptu"
        opts = ["(nenhuma)"] + ext_variables(src)
        if ss.get(f"var_{src}") not in opts:
            ss[f"var_{src}"] = opts[0]
        st.selectbox("Variável OD (soma por Zona_D)" if src == "od" else "Variável IPTU (por lote)",
                     options=opts, key=f"var_{src}", on_change=mark_ui_action)
    if lvl == "quadra":
        st.button("Ajustar ao selecionado", use_container_width=True,
                  on_click=lambda: (mark_ui_action(), _fit_selected_post_level()))

    tem_ok = _thematic_enabled()
    if not tem_ok:
        ss["thematic_layer"] = "Nenhum"
    st.selectbox("Mapa temático", options=["Nenhum"] + list(THEMATIC_LAYERS.keys()),
                 format_func=lambda k: "Nenhum" if k == "Nenhum" else THEMATIC_LAYERS[k]["title"],
                 key="thematic_layer", disabled=not tem_ok, on_change=mark_ui_action)
    if not tem_ok:
        st.caption("Disponível após selecionar isócronas.")
    st.divider()

    st.checkbox("Habilitar seleção por caixa/laço", key="selection_draw_mode", on_change=mark_ui_action)
    st.checkbox("Camadas fixas (metrô, trem, rios, verdes)", key="show_fixed_layers", on_change=mark_ui_action)
    st.divider()

    st.subheader("Downloads", anchor=False)
    ev, et = ss.get("_export_view"), ss.get("_export_thematic")
    st.download_button("CSV — visualização em tela", data=(ev[1] if ev else b""),
                       file_name=f"{ev[0] if ev else 'visualizacao'}.csv", mime="text/csv",
                       use_container_width=True, disabled=not ev)
    st.download_button("CSV — mapa temático", data=(et[1] if et else b""),
                       file_name=f"{et[0] if et else 'tematico'}.csv", mime="text/csv",
                       use_container_width=True, disabled=not et)
    mh = ss.get("_export_map_html")
    st.download_button("Mapa (HTML interativo)", data=(mh or ""), file_name="planbairros_mapa.html",
                       mime="text/html", use_container_width=True, disabled=not mh)
    st.caption("Imagem PNG: use o botão 📷 no canto superior esquerdo do mapa.")

# =============================================================================
# MAP RENDER
# =============================================================================
def render_map_panel():
    ss = st.session_state
    level = ss.get("level", "subpref")
    ensure_variable_for_level(level)
    title, m = "", None
    tem_key = ss.get("thematic_layer", "Nenhum")
    tem_on = tem_key != "Nenhum" and _thematic_enabled()
    fo = (lambda x: 0.0) if tem_on else (lambda x: x)
    ss["_export_view"] = None
    ss["_export_thematic"] = None

    def _new_map():
        return make_base_map(center=ss["view_center"], zoom=ss["view_zoom"])

    if level == "subpref":
        title = "Subprefeituras"
        g_sub = read_layer("subpref")
        if g_sub is None or g_sub.empty:
            return
        if SUBPREF_ID not in g_sub.columns:
            st.error(f"subprefeitura.parquet sem '{SUBPREF_ID}'. Colunas: {list(g_sub.columns)}"); return
        if ss.get("last_level") != "subpref":
            set_view_to_gdf(g_sub); ss["last_level"] = "subpref"
        m = _new_map()
        ttip = "sp_nome" if "sp_nome" in g_sub.columns else SUBPREF_ID
        tol = SIMPLIFY_TOL_BY_LEVEL["subpref"]
        add_polygons_selectable(m, g_sub, "Subprefeituras", SUBPREF_ID, tooltip_col=ttip,
            fill_opacity=0.06, selected_fill_opacity=0.0, tooltip_prefix="Subpref: ",
            simplify_tol=tol, cache_key=f"subpref:{tol}")
        if "sp_nome" in g_sub.columns:
            add_labels_on_map(m, g_sub, "sp_nome", font_size=13)
        set_export("_export_view", "subprefeituras", g_sub)

    elif level == "distrito":
        sp = _id_to_str(ss.get("selected_subpref_id"))
        g_dist, g_sub = read_layer("dist"), read_layer("subpref")
        if g_dist is None or g_sub is None or sp is None:
            return
        g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
        title = f"Distritos ({label_or_id(g_parent, label_col='sp_nome', fallback_col=SUBPREF_ID)})"
        g_show = subset_by_parent(g_dist, DIST_PARENT, sp)
        if ss.get("last_level") != "distrito":
            set_view_to_gdf(g_show if not g_show.empty else g_parent); ss["last_level"] = "distrito"
        m = _new_map()
        add_parent_fill(m, g_parent, "Subpref selecionada (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"], cache_key=f"parent:subpref:{sp}")
        ttip = "ds_nome" if "ds_nome" in g_show.columns else DIST_ID
        add_polygons_selectable(m, g_show, "Distritos", DIST_ID, tooltip_col=ttip,
            fill_opacity=0.06, tooltip_prefix="Distrito: ", simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"dist:sp:{sp}")
        if "ds_nome" in g_show.columns:
            add_labels_on_map(m, g_show, "ds_nome", font_size=12)
        add_boundary_overlay(m, g_parent, weight=2.0, cache_key=f"bnd:sp:{sp}")
        set_export("_export_view", "distritos", g_show)

    elif level == "isocrona":
        d = _id_to_str(ss.get("selected_distrito_id"))
        sel_ids = ensure_set_of_str(ss.get("selected_iso_ids"))
        g_iso, g_dist = read_layer("iso"), read_layer("dist")
        if g_iso is None or g_dist is None or d is None:
            return
        for c in (DS_CODIGO, ISO_ID):
            if c not in g_iso.columns:
                st.error(f"isocronas sem '{c}'. Colunas: {list(g_iso.columns)}"); return
        if DS_CODIGO not in g_dist.columns:
            st.error(f"Distritos sem '{DS_CODIGO}'. Colunas: {list(g_dist.columns)}"); return
        g_parent_dist = subset_by_id(g_dist, DIST_ID, d)
        title = f"Isócronas ({label_or_id(g_parent_dist, label_col='ds_nome', fallback_col=DIST_ID)})"
        if sel_ids:
            title += f" — selecionadas: {len(sel_ids)}"
        g_show_iso = _isos_of_distrito(g_iso, d)
        if g_show_iso.empty:
            st.warning(
                f"Nenhuma isócrona para o ds_codigo do distrito clicado: {_ds_codigo_of_distrito(d)!r} | "
                f"Distritos: {g_dist[DS_CODIGO].dropna().unique()[:8].tolist()} | "
                f"Isócronas: {g_iso[DS_CODIGO].dropna().unique()[:8].tolist()}")
        if ss.get("last_level") != "isocrona":
            set_view_to_gdf(g_show_iso if not g_show_iso.empty else g_parent_dist)
            ss["last_level"] = "isocrona"
        m = _new_map()
        add_parent_fill(m, g_parent_dist, "Distrito selecionado (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"], cache_key=f"parent:dist:{d}")
        g_viz = g_show_iso.copy()
        pairs = g_viz[ISO_CLASS_COL].map(iso_label_color).tolist() if ISO_CLASS_COL in g_viz.columns \
            else [("Sem classe", ISO_DEFAULT_COLOR)] * len(g_viz)
        g_viz["__iso_label"] = [p[0] for p in pairs]
        g_viz["__iso_color"] = [p[1] for p in pairs]
        tol = SIMPLIFY_TOL_BY_LEVEL["isocrona"]
        if ss.get("variable") == "Isócronas (classes)":
            add_polygons_selectable_colored(m, g_viz, "Isócronas", ISO_ID, "__iso_color",
                selected_ids=sel_ids, tooltip_col=ISO_ID, fill_opacity=fo(ISO_FILL_OPACITY_CLASSES),
                selected_fill_opacity=0.0, tooltip_prefix="Isócrona: ", simplify_tol=tol,
                cache_key=f"isoVIZ:dist:{d}", default_fill=ISO_DEFAULT_COLOR)
            legend = sorted({(l, c) for l, c in pairs})
            if not tem_on:
                add_legend(m, "Isócronas (classes)", legend)
        else:
            add_polygons_selectable(m, g_show_iso, "Isócronas", ISO_ID, selected_ids=sel_ids,
                tooltip_col=ISO_ID, fill_opacity=fo(ISO_FILL_OPACITY_DEFAULT), selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ", simplify_tol=tol, cache_key=f"iso:dist:{d}")
        add_boundary_overlay(m, g_parent_dist, weight=2.0, cache_key=f"bnd:dist:{d}")
        set_export("_export_view", "isocronas_selecionadas",
                   subset_by_id_multi(g_viz, ISO_ID, sel_ids) if sel_ids else None)

    elif level == "quadra":
        iso_ids = ensure_set_of_str(ss.get("selected_iso_ids"))
        if not iso_ids:
            return
        post_view = ss.get("post_iso_view", "quadra")
        data = build_post_iso_data()
        g_parent = data.get("g_parent")
        lbl_map = {"quadra": "Quadras", "lote": "Lotes", "censo": "Setor censitário", "od": "Zonas OD"}
        title = f"{lbl_map.get(post_view, 'Quadras')} — filtrado pelas isócronas selecionadas"
        pv_map = {"quadra": data.get("g_quadra"), "lote": data.get("g_lote"),
                  "censo": data.get("g_censo"), "od": data.get("g_od")}
        tgt = pv_map.get(post_view)
        target = tgt if tgt is not None and not tgt.empty else g_parent
        if ss.get("last_level") != "quadra":
            set_view_to_gdf(target); ss["last_level"] = "quadra"
        m = _new_map()
        iso_key = "|".join(sorted(iso_ids))
        if g_parent is not None and not g_parent.empty:
            add_parent_fill(m, g_parent, "Isócronas selecionadas (sombra)",
                fill_opacity=fo(PARENT_FILL_OPACITY), simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"parent:iso:{iso_key}:{fo(1)}")

        view_gdf, sel_key, id_for_sel = tgt, None, None
        if post_view == "quadra":
            g_quad = data.get("g_quadra")
            id_col_map = data.get("quadra_id_col", QUADRA_UID)
            sel_key, id_for_sel = "selected_quadra_ids", id_col_map
            if g_quad is not None and not g_quad.empty:
                g_quad_viz = attach_quadras_csv(g_quad)
                view_gdf = g_quad_viz
                ttip = QUADRA_ID if QUADRA_ID in g_quad_viz.columns else id_col_map
                if CLUSTER_COL in g_quad_viz.columns:
                    g_quad_viz["__cluster_color"] = g_quad_viz[CLUSTER_COL].apply(_coerce_int).apply(cluster_color)
                if ss.get("variable") == "Cluster" and "__cluster_color" in g_quad_viz.columns:
                    add_polygons_selectable_colored(m, g_quad_viz, "Quadras", id_col_map, "__cluster_color",
                        selected_ids=ss.get("selected_quadra_ids"), tooltip_col=ttip,
                        fill_opacity=fo(0.9), selected_fill_opacity=0.0, tooltip_prefix="Quadra: ",
                        cache_key=f"quad-cl:{iso_key}:{fo(1)}", default_fill=CLUSTER_NULL_COLOR)
                    if not tem_on:
                        add_legend(m, "Tipos de áreas urbanas", [(CLUSTER_LABELS[k], v) for k, v in CLUSTER_COLOR_MAP.items()]
                                   + [("Sem dado", CLUSTER_NULL_COLOR)])
                else:
                    add_polygons_selectable(m, g_quad, "Quadras", id_col_map, tooltip_col=ttip,
                        selected_ids=ss.get("selected_quadra_ids"), fill_opacity=fo(0.06),
                        selected_fill_opacity=0.0, tooltip_prefix="Quadra: ",
                        cache_key=f"quadB:{iso_key}:{fo(1)}")
            else:
                st.warning("Nenhuma quadra encontrada para as isócronas selecionadas.")
        else:
            styles = {"lote": ("Lotes", "#b7d7a8", 0.18, "Lote: ", "lote"),
                      "censo": ("Setor censitário", "#7aa6c2", 0.10, "Setor: ", "censo"),
                      "od": ("Zonas OD", "#d9b26f", 0.16, "Zona OD: ", "od")}
            name, fill, op, prefix, tol_k = styles[post_view]
            g_show, id_col, sel_key = _post_cfg(data)[post_view]
            id_for_sel = id_col
            src = {"od": "od", "lote": "iptu"}.get(post_view)
            var = ss.get(f"var_{src}") if src else None
            ck_base = f"{post_view}:{_id_to_str(ss.get('selected_distrito_id'))}:{iso_key}:{fo(1)}"
            drawn = False
            if g_show is not None and not g_show.empty and var and var != "(nenhuma)" and var in g_show.columns:
                drawn = add_choropleth(m, g_show, id_col, var, name, prefix, ss.get(sel_key),
                                       cache_key=f"{ck_base}:ch:{var}", opacity=fo(0.75))
            if drawn:
                pass
            elif g_show is not None and not g_show.empty:
                add_polygons_selectable(m, g_show, name, id_col, tooltip_col=id_col,
                    selected_ids=ss.get(sel_key), fill_color=fill, fill_opacity=fo(op),
                    selected_fill_opacity=0.0, tooltip_prefix=prefix,
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL[tol_k],
                    cache_key=f"{post_view}:{_id_to_str(ss.get('selected_distrito_id'))}:{iso_key}:{fo(1)}")
            else:
                st.warning(f"Nenhum(a) {name.lower()} encontrado(a) para as isócronas selecionadas.")

        if view_gdf is not None and not view_gdf.empty:
            sel = ensure_set_of_str(ss.get(sel_key)) if sel_key else set()
            if sel and id_for_sel in view_gdf.columns:
                view_gdf = subset_by_id_multi(view_gdf, id_for_sel, sel)
            set_export("_export_view", f"{post_view}_isocronas", view_gdf)

    # --- Mapa temático (abaixo dos limites) ---
    if m is not None and tem_on:
        iso_ids = ensure_set_of_str(ss.get("selected_iso_ids"))
        g_iso = read_layer("iso")
        g_iso_sel = subset_by_id_multi(g_iso, ISO_ID, iso_ids) if g_iso is not None else None
        gt = read_thematic(tem_key)
        if gt is not None:
            gf = filter_thematic(tem_key, gt, iso_ids, g_iso_sel)
            gc, legend = classify_thematic(tem_key, gf) if gf is not None and not gf.empty else (gf, [])
            add_thematic_to_map(m, tem_key, gc, legend)
            title += f" | {THEMATIC_LAYERS[tem_key]['title']}"
            set_export("_export_thematic", f"{tem_key}_isocronas", gc)
        if g_iso_sel is not None and not g_iso_sel.empty:
            add_boundary_overlay(m, g_iso_sel, weight=2.0, cache_key=f"bnd:iso:{'|'.join(sorted(iso_ids))}")

    if m is None:
        return
    add_fixed_layers(m)
    if ss.get("selection_draw_mode", False):
        add_draw_tools(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    add_print_button(m)

    st.markdown(f"### {title}")
    st_folium(m, height=780, use_container_width=True, key=f"{MAP_KEY}_{level}",
              returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_tooltip",
                                "all_drawings", "last_active_drawing"])
    if level == "quadra" and ss.get("post_iso_view") in ("od", "lote"):
        src = "od" if ss.get("post_iso_view") == "od" else "iptu"
        diag = ss.get(f"_diag_{src}")
        with st.expander("Diagnóstico de vínculos", expanded=False):
            if not diag:
                st.caption("Sem diagnóstico ainda.")
            else:
                if "erro" in diag:
                    st.error(diag["erro"])
                elif diag.get("feicoes_vinculadas", 0) == 0:
                    st.warning("Nenhuma feição vinculada — verifique a grafia/formato das chaves abaixo.")
                st.json(diag)
    try:
        ss["_export_map_html"] = m.get_root().render()
    except Exception:
        ss["_export_map_html"] = None
    ss["_map_level_rendered"] = level

# =============================================================================
# APP
# =============================================================================
# =============================================================================
# TEXTOS
# =============================================================================
FAPESP = ("**Apoio FAPESP:** Processos nº 2023/10015-7, 2021/04751-7, 2024/07947-8, 2024/08609-9, "
          "2024/16700-6, 2024/19977-9, 2025/11843-6 e 2026/09424-8.")

TXT_WELCOME = f"""
Bem-vindo ao **Painel Interativo de Dados de apoio à elaboração de Planos de Bairro**, ferramenta integrante do Projeto PlanBairros.

O PlanBairros foi desenvolvido na Escola Politécnica da Universidade de São Paulo (EPUSP), em parceria com a Faculdade de Arquitetura e Urbanismo da Universidade de São Paulo (FAUUSP), em cooperação técnica com a Secretaria Municipal de Urbanismo e Licenciamento (SMUL) e com apoio da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP).

Um dos principais resultados do projeto é o Guia para Elaboração de Planos de Bairro no Município de São Paulo, que apresenta conceitos, métodos e orientações para apoiar a elaboração desses planos.

Este painel complementa o Guia, oferecendo dados e informações territorializadas para apoiar diferentes etapas do processo, como a definição da área de abrangência, a leitura do território e seu monitoramento. Por meio de mapas, gráficos e tabelas interativos, é possível explorar diferentes aspectos do território e compreender como eles se distribuem na escala local.

As informações apresentadas constituem uma base de apoio à elaboração dos Planos de Bairro e devem ser complementadas pela análise técnica, pelo trabalho de campo e pela participação social.

{FAPESP}
"""

TXT_SOBRE = f"""
## Sobre o projeto PlanBairros

O projeto PlanBairros, intitulado “Elaboração de instrumento para apoio e fomento ao desenvolvimento de planos de bairro no município de São Paulo”, teve como objetivo principal desenvolver um referencial conceitual e metodológico para apoiar a elaboração de Planos de Bairro no município de São Paulo. Os Planos de Bairro são instrumentos previstos no Plano Diretor Estratégico de São Paulo (PDE), com potencial para aproximar o planejamento urbano das questões e demandas da escala local.

Financiado pela Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), no âmbito do Programa de Pesquisa em Políticas Públicas (PPP), o projeto é sediado na Escola Politécnica da Universidade de São Paulo (EPUSP), desenvolvido em parceria com a Faculdade de Arquitetura e Urbanismo da Universidade de São Paulo (FAUUSP) e em cooperação técnica com a Secretaria Municipal de Urbanismo e Licenciamento (SMUL) da Prefeitura de São Paulo.

A elaboração de Planos de Bairro em São Paulo é discutida desde a década de 1960. O Plano Diretor Estratégico de 2014, revisado em 2023, prevê que a Prefeitura coordene e fomente sua elaboração. Entretanto, ainda não havia um referencial conceitual e metodológico específico que orientasse esse processo. É a partir dessa lacuna que se desenvolve o PlanBairros.

Um dos principais resultados do projeto é o Guia para Elaboração de Planos de Bairro no Município de São Paulo, concebido como referencial para apoiar o desenvolvimento, a implementação e o monitoramento desses planos. Sua elaboração foi fundamentada em estudos bibliográficos, históricos, institucionais, regulatórios e normativos, bem como no levantamento e na sistematização de dados abertos e de informações disponibilizadas pela Prefeitura de São Paulo. Também foram coletados dados primários em áreas de estudo e realizadas atividades participativas com gestores públicos, especialistas e população.

Como suporte à aplicação do Guia, o PlanBairros desenvolveu este Painel Interativo de Dados, que reúne e organiza variáveis relacionadas ao planejamento na escala de bairro. A ferramenta permite consultar e visualizar informações territoriais por meio de mapas, gráficos e tabelas, apoiando as atividades de leitura, análise e comunicação que integram a elaboração de um Plano de Bairro.

Os dados podem ser explorados em diferentes recortes territoriais, como setores censitários, quadras e lotes, conforme a escala em que cada informação está disponível. O painel reúne informações relacionadas, entre outros temas, à mobilidade e acessibilidade, equipamentos sociais, verde e meio ambiente, infraestrutura urbana e uso e ocupação do solo.

A ferramenta constitui uma base inicial para a leitura do território, auxiliando na identificação de características, padrões e questões que precisam ser aprofundadas. Essa leitura deve ser complementada pela análise técnica, pelo trabalho de campo e pela participação social ao longo do processo de elaboração do Plano de Bairro.

{FAPESP}
"""

TXT_USO = """
## Como utilizar o Painel Interativo de Dados

O uso do Painel Interativo de Dados acontece em dois momentos: primeiro, define-se uma área de abrangência para o Plano de Bairro; depois, selecionam-se os temas e dados que se deseja visualizar para conhecer e analisar o território.

### Defina a área de abrangência

1. Comece selecionando no mapa a Subprefeitura e, em seguida, o Distrito onde está localizado o bairro ou território de interesse. A partir dessa seleção, o painel apresentará uma tela específica para a definição da área de abrangência.
2. Nessa tela, você encontrará polígonos territoriais previamente definidos, que servem como referência inicial para a construção do recorte do Plano de Bairro.
   Os polígonos principais representam áreas caminháveis calculadas a partir de pontos de referência, considerando simultaneamente limites de 10 minutos de caminhada e 800 metros de distância pela rede. Sua configuração incorpora características da rede de circulação, da topografia e a presença de barreiras urbanas, podendo, por isso, apresentar diferentes formas e extensões no território.
3. Além dos polígonos principais, o mapa apresenta áreas de transição, de menor dimensão. Elas permitem realizar ajustes mais detalhados no recorte, incorporando trechos que façam parte da dinâmica cotidiana do bairro, mesmo quando não estão integralmente incluídos nos polígonos principais.
4. A seleção é feita diretamente no mapa. Escolha os polígonos principais que correspondem ao território de interesse e, quando necessário, acrescente áreas de transição. É possível combinar diferentes polígonos até chegar a uma área que se aproxime do bairro que se pretende analisar.
5. A área de abrangência resulta da combinação das áreas selecionadas e passa a constituir a referência territorial para a consulta e a visualização dos dados no painel.
6. Esse recorte não deve ser entendido como um limite definitivo do bairro. Ele constitui uma primeira aproximação técnica, que deverá ser discutida e ajustada ao longo da elaboração do Plano de Bairro, considerando também o conhecimento, as práticas cotidianas e a percepção da população sobre o território.

### Escolha e explore os dados

Com a área de abrangência definida, selecione os temas e variáveis que deseja visualizar. O painel reúne informações sobre diferentes aspectos do bairro, que podem ser consultadas por meio de mapas, gráficos e tabelas.

Os dados são apresentados de acordo com a escala territorial disponível para cada informação, podendo incluir lotes, quadras, setores censitários, zonas OD e outros recortes territoriais. Dessa forma, é possível observar diferenças dentro da própria área de abrangência, identificar padrões e reconhecer questões que precisam ser aprofundadas.

Essa primeira leitura também permite identificar informações que não estão disponíveis no painel ou que necessitam de maior detalhamento, orientando levantamentos complementares por meio de trabalho de campo e processos participativos.

Os resultados podem ser registrados por meio de imagens e exportados em PDF, facilitando sua utilização em reuniões, atividades participativas, análises e relatórios.

As informações disponíveis no painel constituem uma base inicial para a leitura do território e devem ser interpretadas em conjunto com a análise técnica, o trabalho de campo e o conhecimento da população, incorporado por meio dos processos participativos.
"""

TXT_ISO = """
### Entenda como as áreas de referência foram definidas

As áreas de abrangência apresentadas no painel foram desenvolvidas por meio de um método de análise geoespacial baseado em redes, concebido para representar áreas acessíveis a pé e apoiar a delimitação de unidades territoriais na escala de bairro.

O método utiliza a quadra urbana como unidade espacial de referência e considera a rede de circulação de pedestres, a topografia e a presença de barreiras que condicionam os deslocamentos a pé, como rios, ferrovias, vias expressas e grandes áreas de acesso restrito. Travessias formalmente existentes, como pontes, viadutos e passagens, são consideradas para preservar a conectividade da rede.

A análise parte de centralidades funcionais, inicialmente identificadas a partir da concentração de destinos de viagens associados a atividades de comércio e serviços. A partir desses pontos, são calculadas áreas caminháveis via isócronas, considerando simultaneamente um limite de 10 minutos de caminhada e 800 metros de distância pela rede.

O cálculo considera uma velocidade de referência de 4,8 km/h em terreno plano, ajustada de acordo com a inclinação do percurso. Dessa forma, trechos mais íngremes aumentam o tempo necessário para caminhar e podem reduzir a extensão efetivamente alcançada. Os 800 metros representam, portanto, um limite máximo de distância percorrida pela rede, e não um raio fixo aplicado uniformemente ao território.

As barreiras territoriais também condicionam a formação dessas áreas. Quando rios, ferrovias, vias expressas ou outras grandes descontinuidades não apresentam travessias disponíveis, a expansão da área caminhável é interrompida. Isso permite gerar polígonos que respondem às condições físicas e à configuração efetiva do território, em vez de círculos definidos apenas por distância geométrica.

O procedimento é realizado de forma iterativa. Após a definição das primeiras áreas caminháveis, novos pontos de referência são gerados nas porções do território ainda não classificadas, e o processo é repetido até alcançar a cobertura do tecido urbano analisado.

Como resultado, são produzidos dois componentes territoriais:

- **Áreas principais** — correspondem às isócronas de escala de bairro, associadas às áreas caminháveis estruturadas a partir dos pontos de referência;
- **Áreas de transição** — correspondem aos espaços intermediários resultantes das etapas subsequentes do processo, situados entre as áreas principais e que não representam, por si só, novas centralidades funcionais.

A combinação de uma área principal com áreas de transição adjacentes permite construir uma área de influência mais ampla e contínua, oferecendo flexibilidade para a definição da área de abrangência do Plano de Bairro.

Os limites resultantes não devem, portanto, ser entendidos como limites administrativos ou definitivos, mas como uma referência técnica inicial, que pode ser ajustada a partir da análise local e dos processos participativos. Esse princípio é denominado no método de limites flexíveis (*flexible borders*).

**Saiba mais**

BERTHOLDO, Emílio; PONTES DE AQUINO, Aida Paula; MARINS, Karin Regina de Castro. A graph-based geospatial framework for delineating neighborhood planning units with barrier-aware isochrones. *Cities*, v. 179, 107557, 2026. DOI: [10.1016/j.cities.2026.107557](http://www.doi.org/10.1016/j.cities.2026.107557).
"""

TXT_CLUSTER = """
### Tipos de áreas urbanas

Para apoiar a leitura do território, áreas com características urbanas semelhantes foram agrupadas em cinco tipos. A classificação considera aspectos relacionados à densidade populacional, aos usos do solo, às características das edificações, ao valor da terra e ao acesso ao transporte público.

Os tipos representam diferentes padrões predominantes de ocupação urbana e ajudam a reconhecer semelhanças e diferenças entre as áreas da cidade. Não correspondem a categorias oficiais de zoneamento ou divisão territorial e devem ser utilizados como informação de apoio à leitura e à análise do território.

**Tipo 1 — Áreas periféricas de alta densidade**
Áreas predominantemente residenciais, com grande concentração de moradores e baixos valores da terra. Apresentam maior presença de edificações classificadas nos indicadores utilizados como de menor qualidade, tanto baixas quanto verticalizadas, e acesso intermediário ao transporte público. Entre os cinco tipos, é o que apresenta maior densidade populacional.

**Tipo 2 — Áreas mistas de densidade intermediária**
Áreas em que a moradia se combina com atividades de comércio e serviços, com densidade populacional e valores da terra intermediários. Predominam edificações mais baixas e classificadas como de qualidade intermediária. Apresentam acesso relativamente bom ao transporte público.

**Tipo 3 — Áreas periféricas de média densidade**
Áreas predominantemente residenciais, com baixos valores da terra e densidade populacional intermediária. Apresentam alguma verticalização, predominância de edificações classificadas como de menor qualidade e menor acesso ao transporte público.

**Tipo 4 — Áreas centrais verticalizadas e de uso misto**
Áreas caracterizadas pela verticalização e pela diversidade de usos, reunindo moradia, comércio e serviços. Apresentam edificações classificadas como de maior qualidade, altos valores da terra e excelente acesso ao transporte público. Apesar da verticalização, a presença expressiva de atividades comerciais e de serviços está associada a uma densidade de moradores relativamente baixa.

**Tipo 5 — Áreas com predominância de comércio e serviços**
Áreas verticalizadas com forte predominância de atividades comerciais e de serviços e baixa presença de moradores. Caracterizam-se por edifícios altos, elevados valores da terra e grande proximidade do transporte público.

### Entenda como os tipos de áreas urbanas foram definidos

A tipologia foi desenvolvida por meio de uma análise quantitativa das características urbanas do município de São Paulo, buscando identificar áreas que apresentam padrões semelhantes.

Foram integrados indicadores referentes a três dimensões principais: ambiente construído, acessibilidade e contexto sociodemográfico. Após a padronização dos dados, foram aplicados métodos de redução de dimensionalidade, por meio da Análise de Componentes Principais (PCA), e diferentes algoritmos de clusterização, que permitem identificar grupos de áreas com características semelhantes.

A definição dos agrupamentos considera, além da similaridade entre as características analisadas, critérios de validação e coerência espacial. O procedimento permite reconhecer padrões territoriais que nem sempre coincidem com limites administrativos e que podem contribuir para a compreensão das dinâmicas urbanas na escala do bairro.

Os cinco tipos apresentados no painel constituem, portanto, uma síntese analítica das características predominantes do território. Eles não correspondem a categorias de zoneamento, divisões administrativas ou limites oficiais de bairros.

A tipologia deve ser interpretada em conjunto com as demais informações disponíveis no painel e complementada pela análise técnica, pelo trabalho de campo e pelo conhecimento e percepção da população, incorporados ao longo do processo participativo de elaboração do Plano de Bairro.

**Saiba mais**

PONTES DE AQUINO, Aida Paula; BERTHOLDO, Emílio; MORAES, Gabriel Maggio de; MARINS, Karin Regina de Castro. A clustering framework proposal for defining neighborhood-scale dynamics: Evidence from São Paulo, Brazil. *Socio-Economic Planning Sciences*, v. 105, 102457, 2026. DOI: [10.1016/j.seps.2026.102457](http://www.doi.org/10.1016/j.seps.2026.102457).
"""

TXT_LCZ = """
### Explicação das legendas de LCZ

| LCZ | Classe | Definição / legenda |
|---|---|---|
| LCZ 1 | Alto-compacto | Edificações altas implantadas em arranjo compacto de maior adensamento. Os edifícios possuem mais de 10 pavimentos. Predominam áreas impermeáveis com pouca ou nenhuma vegetação. Materiais predominantes: concreto, pedra, aço e vidros |
| LCZ 2 | Médio-compacto | Edificações médias implantadas em arranjo compacto de maior adensamento. Os edifícios possuem de 3 a 9 pavimentos. Predominam áreas impermeáveis com pouca ou nenhuma vegetação. Materiais predominantes: concreto, pedra, tijolos e materiais cerâmicos |
| LCZ 3 | Baixo-compacto | Edificações baixas implantadas em arranjo compacto de maior adensamento. Os edifícios possuem de 1 a 3 pavimentos. Predominam áreas impermeáveis com pouca ou nenhuma vegetação. Materiais predominantes: concreto, pedra, tijolos e materiais cerâmicos |
| LCZ 4 | Alto-aberto | Edificações altas implantadas em arranjo aberto de menor adensamento. Os edifícios possuem mais de 10 pavimentos. Predominam áreas permeáveis e vegetação. Materiais predominantes: concreto, pedra, vidros e aço |
| LCZ 5 | Médio-aberto | Edifícios de altura média implantados em arranjo aberto de menor adensamento. Os edifícios possuem de 3 a 9 pavimentos. Predominam áreas permeáveis e vegetação. Materiais predominantes: concreto, pedra, vidros e aço |
| LCZ 6 | Baixo-aberto | Edifícios baixos e espaçados implantados em arranjo aberto pouco adensado. Os edifícios possuem de 1 a 3 pavimentos. Presença significativa de vegetação e espaços livres. Materiais predominantes: concreto, tijolos, madeira, materiais cerâmicos e pedras |
| LCZ 7 | Baixo-precário | Edifícios leves de baixa altura, com baixa inércia térmica, implantadas em áreas densamente construídas, pouco consolidadas e com edifícios de 1 pavimento. Pouca ou nenhuma vegetação arbórea e cobertura do solo compacta. Materiais predominantes: madeira, palha e metal corrugado |
| LCZ 8 | Baixo-grande | Edificações de baixa altura em arranjos abertos, como galpões e estruturas comerciais ou industriais, implantadas em espaços amplos e predominantemente pavimentados. Os edifícios possuem de 1 a 3 pavimentos. Materiais predominantes: aço, concreto, pedra ou metal |
| LCZ 9 | Ocupação esparsa | Composição esparsa de edifícios de altura baixa ou média em meio a ambientes naturais com abundância de áreas permeáveis |
| LCZ 10 | Indústria pesada | Estruturas industriais de altura baixa e média. Cobertura do solo predominantemente impermeável ou compactada. Materiais predominantes: aço, concreto ou metal |
| LCZ A | Veg. arbórea densa | Áreas com cobertura arbórea esparsa decídua ou perene. Cobertura do solo predominantemente permeável, com vegetação herbácea. Exemplo: florestas naturais ou cultivadas, parques urbanos |
| LCZ B | Veg. arbórea esparsa | Áreas com cobertura arbórea esparsa decídua ou perene, nas quais as árvores estão distribuídas entre áreas abertas ou vegetação rasteira, com cobertura do solo predominantemente permeável. Exemplo: florestas naturais ou cultivadas, parques urbanos |
| LCZ C | Vegetação arbustiva | Áreas dominadas por vegetação arbustiva, com árvores de pequeno porte e cobertura do solo predominantemente permeável. Exemplo: áreas arbustivas naturais ou de cultivo agrícola |
| LCZ D | Vegetação herbácea | Áreas predominantemente cobertas por vegetação rasteira. Exemplo: gramíneas, pastagens, áreas agrícolas, parques urbanos e campos |
| LCZ E | Rocha ou pavimento | Áreas dominadas por superfícies impermeáveis e pavimentadas, com pouca ou nenhuma vegetação. Exemplo: estacionamentos, pátios e grandes vias |
| LCZ F | Solo exposto | Áreas com predominância de solo exposto e areia, apresentando pouca ou nenhuma vegetação. Exemplo: desertos ou áreas agrícolas |
| LCZ G | Água | Superfícies predominantemente ocupadas por corpos d’água. Exemplo: rios, lagos, represas e grandes reservatórios |
"""


def _info_pop(title, body):
    with st.popover(f"ⓘ {title}", use_container_width=True):
        st.markdown(body)


def render_info_buttons(lvl):
    ss = st.session_state
    if lvl == "isocrona":
        _info_pop("Sobre as áreas de referência", TXT_ISO)
    if lvl == "quadra" and ss.get("post_iso_view", "quadra") == "quadra" and ss.get("variable") == "Cluster":
        _info_pop("Sobre os tipos de áreas urbanas", TXT_CLUSTER)
    if ss.get("thematic_layer") == "lcz" and _thematic_enabled():
        _info_pop("Sobre as zonas climáticas (LCZ)", TXT_LCZ)


if hasattr(st, "dialog"):
    @st.dialog("Painel Interativo de Dados — PlanBairros", width="large")
    def welcome_dialog():
        st.markdown(TXT_WELCOME)
        if st.button("Fechar", type="primary", use_container_width=True):
            st.rerun()
else:
    welcome_dialog = None


def main():
    init_state()
    inject_css()
    render_header()
    if gpd is None or folium is None or st_folium is None:
        st.error("Este app requer `geopandas`, `folium` e `streamlit-folium` (veja requirements.txt).")
        return

    ss = st.session_state
    ui_sig = int(ss.get("_ui_action_sig", 0))
    ui_action = ui_sig != int(ss.get("_ui_action_sig_seen", 0))
    ss["_ui_action_sig_seen"] = ui_sig
    if ui_action:
        ss["last_click_sig"] = ""; ss["last_draw_sig"] = ""

    cur_level = ss.get("level", "subpref")
    map_state_prev = ss.get(f"{MAP_KEY}_{cur_level}") or {}
    if (not ui_action) and ss.get("_map_level_rendered") == cur_level and isinstance(map_state_prev, dict):
        consume_map_event(cur_level, map_state_prev)
        consume_draw_selection(cur_level, map_state_prev)

    sanitize_level_state()

    if not ss.get("_welcome_seen"):
        ss["_welcome_seen"] = True
        if welcome_dialog is not None:
            welcome_dialog()

    tab_painel, tab_sobre, tab_uso = st.tabs(["Painel", "Sobre o projeto PlanBairros", "Como utilizar"])
    with tab_painel:
        left, right = st.columns([4, 1], gap="large")
        with right:
            st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
            control_panel()
            st.markdown("</div>", unsafe_allow_html=True)
        with left:
            st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
            render_map_panel()
            st.markdown("</div>", unsafe_allow_html=True)
    with tab_sobre:
        st.markdown(TXT_SOBRE)
    with tab_uso:
        st.markdown(TXT_USO)


main()
