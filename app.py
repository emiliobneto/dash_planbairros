from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import base64
import html
import json
import re
import shutil

import streamlit as st
import pandas as pd  # type: ignore

try:
    import geopandas as gpd  # type: ignore
    import folium  # type: ignore
    from folium.features import GeoJsonTooltip  # type: ignore
    from folium.plugins import Draw  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    from shapely.geometry import Point, shape  # type: ignore
except Exception:
    gpd = None  # type: ignore
    folium = None  # type: ignore
    GeoJsonTooltip = None  # type: ignore
    Draw = None  # type: ignore
    st_folium = None  # type: ignore
    Point = None  # type: ignore
    shape = None  # type: ignore

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
PB_BLACK = "#000000"

CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"

SMOOTH_FACTOR = 1.0
LINE_CAP = "round"
LINE_JOIN = "round"

PARENT_FILL_OPACITY = 0.20
PARENT_STROKE_OPACITY = 1.0
PARENT_STROKE_WEIGHT = 1.0
PARENT_STROKE_DASH = None

SIMPLIFY_TOL_BY_LEVEL = {
    "subpref": 0.0,
    "distrito": 0.0,
    "isocrona": 0.0,
    "censo": 0.0,
    "od": 0.0,
    "quadra": 0.0,
    "lote": 0.0,
}

ISO_FILL_OPACITY_DEFAULT = 0.05
ISO_FILL_OPACITY_CLASSES = 0.05

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
# VISUALIZAÇÕES
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
OD_ID = "od_id"
QUADRA_ID = "quadra_id"
QUADRA_UID = "quadra_uid"
CENSO_ID = "censo_id"
LOTE_ID = "lote_id"

DIST_PARENT = SUBPREF_ID
ISO_PARENT = DIST_ID
OD_PARENT = ISO_ID
CENSO_PARENT = ISO_ID
LOTE_PARENT = ISO_ID

LAYER_ID_COLS = {
    "subpref": [SUBPREF_ID],
    "dist": [DIST_ID, DIST_PARENT],
    "iso": [ISO_ID, ISO_PARENT, SUBPREF_ID],
    "censo": [CENSO_ID, CENSO_PARENT, QUADRA_ID, ISO_ID],
    "od": [OD_ID, OD_PARENT, ISO_ID],
    "quadra": [QUADRA_ID, ISO_ID, CENSO_ID, QUADRA_UID],
    "lote": [LOTE_ID, LOTE_PARENT, DIST_ID],
}

LOCAL_FILENAMES = {
    "subpref": "Subprefeitura.parquet",
    "dist": "Distrito.parquet",
    "iso": "Isocronas.parquet",
    "censo": "Setorcensitario.parquet",
    "od": "ZonasOD.parquet",
    "quadra": "Quadras.parquet",
}

# =============================================================================
# LOTES
# =============================================================================
LOTES_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/17-lA2P_D4oV1joysDf7BOAgp358IcoEG?usp=drive_link"
LOTES_SECRET_KEY = "PB_LOTES_FOLDER_URL"

LOTES_LINKS_BY_DISTRITO = {
    "1": "https://drive.google.com/file/d/1Kbn6RXKXoxdpcdTbI9txBBSf63Yd14zZ/view?usp=drive_link",
    "2": "https://drive.google.com/file/d/1q8oaYurUmEyJIgluOmos-nDytAtLur7Q/view?usp=drive_link",
    "3": "https://drive.google.com/file/d/15Gg4GVDabqZwzY1ubIwhhPZXXU4hkrs8/view?usp=drive_link",
    "4": "https://drive.google.com/file/d/1OAbjgydEp2E5UhWqHwCkKj_6EziDMAYL/view?usp=drive_link",
    "5": "https://drive.google.com/file/d/1jiruqIpGIGhU7q0Uu4wCSl_xUsLdPnZY/view?usp=drive_link",
    "6": "https://drive.google.com/file/d/1IcYBxz1aa-1Aoelsm6_Vf3UM_fQ_y4Iv/view?usp=drive_link",
    "7": "https://drive.google.com/file/d/1us8VgARE1VLjFEWLMfH4qjOadiJ4Y0vj/view?usp=drive_link",
    "8": "https://drive.google.com/file/d/1yGeLMS5zYUKJGi1i5mlvvV4BgJmvEZgW/view?usp=drive_link",
    "9": "https://drive.google.com/file/d/1ueN0KUmZ5ijd5wK087dVK0Fq9ATob5pS/view?usp=drive_link",
    "10": "https://drive.google.com/file/d/1KCwDfaqVBMEoFm4pqAYGPEcYPW9IDIir/view?usp=drive_link",
    "11": "https://drive.google.com/file/d/1g7wg3aHO4r2UpuEPcNG9qCgqygR4PLvY/view?usp=drive_link",
    "12": "https://drive.google.com/file/d/1mG4zs2HqmzTaeFwovJdw-nt11G27L7CC/view?usp=drive_link",
    "13": "https://drive.google.com/file/d/1BlpnBTDQsBpfPXL_xhom7TtNkt3w9k9I/view?usp=drive_link",
    "14": "https://drive.google.com/file/d/1iBkLh3atFZG61jORGXNcwYjM7c_wTPXU/view?usp=drive_link",
    "15": "https://drive.google.com/file/d/1Kx3ttk0Tmow0aJqPhnI-B0p-mpasMsxp/view?usp=drive_link",
    "16": "https://drive.google.com/file/d/1IyFlPR1nGRfbSPn8J3yrySJAWFkI5L2q/view?usp=drive_link",
    "17": "https://drive.google.com/file/d/1_kmHNaY0k5zl6Xk1dSGVHrUDtGYWaCMj/view?usp=drive_link",
    "18": "https://drive.google.com/file/d/15jngv3COgPLIwizOmbQhJuEDlwBSF3M5/view?usp=drive_link",
    "19": "https://drive.google.com/file/d/1IlvxySwpW3OoRktzooJ2bk4XGkuhcis8/view?usp=drive_link",
    "20": "https://drive.google.com/file/d/1KcCz5bIl1cMcE_7eFqLXN0zA4QAZ5Q0U/view?usp=drive_link",
    "21": "https://drive.google.com/file/d/1wU7EBRnMbJNXTk0J8d0uENE-FGrGfNRc/view?usp=drive_link",
    "22": "https://drive.google.com/file/d/1jjfwx59Cni8eJzqiUEixhuP1rcra2m79/view?usp=drive_link",
    "23": "https://drive.google.com/file/d/1XWhKjd1sm78LgYYf_64-LgsOu4E-_gxl/view?usp=drive_link",
    "24": "https://drive.google.com/file/d/1JtBX5QB2rzL5Be4tYTHZD5jW0bnNMca2/view?usp=drive_link",
    "25": "https://drive.google.com/file/d/1Mnbq4eqLlPZRbcsktuUrJQ0P_1llI8a3/view?usp=drive_link",
    "26": "https://drive.google.com/file/d/1gCCKQLKBRLCcWHfuMrMTuW3LR_jIBXCF/view?usp=drive_link",
    "27": "https://drive.google.com/file/d/1SO2TkRNpFCQ15tH5yPixlsngNpu4Pqxv/view?usp=drive_link",
    "28": "https://drive.google.com/file/d/1e-G0lIXe1gn8iK2HMn5ZwJdRMGObjN15/view?usp=drive_link",
    "31": "https://drive.google.com/file/d/1Sp7RXDbxjLVAK1VwkF2q4SBfu1-drm16/view?usp=drive_link",
    "32": "https://drive.google.com/file/d/16rw5zxnJi3nxBDFZJyjp8PQZzRB9K7LM/view?usp=drive_link",
    "33": "https://drive.google.com/file/d/1DYhc60NvClbNDCCr-tHYyyaO_1k_SZQL/view?usp=drive_link",
    "34": "https://drive.google.com/file/d/16EN1W9H_EbbT2TbhLknJykODdAgfjB4w/view?usp=drive_link",
    "35": "https://drive.google.com/file/d/1QrPXvd23iX2duiTgTDmpu_nNFTyDuIS5/view?usp=drive_link",
    "37": "https://drive.google.com/file/d/1pGeH3Mrfg6xJfFmiKJBgRvHxdAB7N1KU/view?usp=drive_link",
    "38": "https://drive.google.com/file/d/1XI2nZIcdOd2hv9MZzVLuYonCEg_s79IC/view?usp=drive_link",
    "39": "https://drive.google.com/file/d/1WsKRq-sqC5_-zWg0tpVnfjRUyLYoojUB/view?usp=drive_link",
    "40": "https://drive.google.com/file/d/1SdoU-K7AmzSoRln3BCuAPxHWEfNi292m/view?usp=drive_link",
    "41": "https://drive.google.com/file/d/1TT9nkYdA8Dw2mwci599053KpTT-_2sml/view?usp=drive_link",
    "42": "https://drive.google.com/file/d/1EqjCjnfW80EaM2F5PaG22qEb3WSspFsf/view?usp=drive_link",
    "43": "https://drive.google.com/file/d/1s2P3XpxfmGsIjIjaRoC8FXoF5h3TDjGQ/view?usp=drive_link",
    "44": "https://drive.google.com/file/d/1fUCV4U34DJaPRF-s1oVdX67DM-Z7Rln_/view?usp=drive_link",
    "45": "https://drive.google.com/file/d/1UeUQyeAckTabd4BmiHJJqClsgKarfEhL/view?usp=drive_link",
    "46": "https://drive.google.com/file/d/1SAMCChCPxWqejoXudY8XFID8jd5o_ngU/view?usp=drive_link",
    "47": "https://drive.google.com/file/d/1oN_6xqyk7HS1sieoEKk8nttjy8-xS_yu/view?usp=drive_link",
    "48": "https://drive.google.com/file/d/1lGEaNPhtHOnSJYJsB7nA1lkaeIPrNNFz/view?usp=drive_link",
    "49": "https://drive.google.com/file/d/1jB4pIZStVa-vwuvOSDdiWnsCJEpvaa9t/view?usp=drive_link",
    "50": "https://drive.google.com/file/d/1dfGxJv0utpr7E-vdjQQprAz_Ipx5ZJwZ/view?usp=drive_link",
    "51": "https://drive.google.com/file/d/1URtL1odgNGjh2nI2dD3lnjzJiBbBjqh4/view?usp=drive_link",
    "52": "https://drive.google.com/file/d/1dPCx22YRyEf1jx0N8Bn8OVedFrETA9VD/view?usp=drive_link",
    "53": "https://drive.google.com/file/d/1CMRsOPSVKXOTOfDQj202IMJX6aGOfF3V/view?usp=drive_link",
    "54": "https://drive.google.com/file/d/17zVEIEUQrefj9BCsHT200hC984qL6L36/view?usp=drive_link",
    "55": "https://drive.google.com/file/d/1oQUwYO0TgW3ryY5pPG3ixcJ4TdpvT5hg/view?usp=drive_link",
    "56": "https://drive.google.com/file/d/1WKXOCAX1SmBa9_N8a9zNxhIO1E93pMId/view?usp=drive_link",
    "57": "https://drive.google.com/file/d/12RJ7NktS6SoqQ4MsJxsqKt2Zh3qy3iH1/view?usp=drive_link",
    "58": "https://drive.google.com/file/d/1wKXvQNTcYmYt5sSEUKdvCsgFNp9m0Q5k/view?usp=drive_link",
    "59": "https://drive.google.com/file/d/1DY6qMT1MD7EddZd5fj3SBCTJNWyjuqNX/view?usp=drive_link",
    "60": "https://drive.google.com/file/d/1zyf9ouOfgoW5mns5N7unQrriQJ6xjdUi/view?usp=drive_link",
    "61": "https://drive.google.com/file/d/15zON1tja5ou3y7L8d8nu-fdc4XYGlnqb/view?usp=drive_link",
    "62": "https://drive.google.com/file/d/1VmzWAtjAFlM1DC-tKG0Dw1-otrVkdpti/view?usp=drive_link",
    "63": "https://drive.google.com/file/d/1TH5wYLMeBeprot6-bPFZyh_dVhLGLR08/view?usp=drive_link",
    "64": "https://drive.google.com/file/d/1ScM1ebCH51wTi0mmMTSIPahoZofm_xuL/view?usp=drive_link",
    "65": "https://drive.google.com/file/d/1qbWAHdca8ZSx4G7OUnMXQlMsg9neBCW8/view?usp=drive_link",
    "66": "https://drive.google.com/file/d/1ASSj1YdIvUe64BfAzWQ9IbDEumof615i/view?usp=drive_link",
    "67": "https://drive.google.com/file/d/1GKCozATY0I8UVxhnjdpco3-VBPS27Zr7/view?usp=drive_link",
    "68": "https://drive.google.com/file/d/1cy7nAHarikbvVXyXAco6EIDrsvZzlV8F/view?usp=drive_link",
    "69": "https://drive.google.com/file/d/1eiJHdo0GjqECrqYeBeS5xzXCKOkNGJab/view?usp=drive_link",
    "70": "https://drive.google.com/file/d/1pyA-Dh0P5AN3-ip9MfSu22ebInsr5R7h/view?usp=drive_link",
    "71": "https://drive.google.com/file/d/1PA4ovneuLD2qHHwv6Qx6DQxbKIyuLYux/view?usp=drive_link",
    "72": "https://drive.google.com/file/d/1aXnU47uzXtW1ZcD9SUg9nHJubVBaz3az/view?usp=drive_link",
    "73": "https://drive.google.com/file/d/1p7HZEr4s_RgAe57hMDXj5-4dpXlsHMt9/view?usp=drive_link",
    "74": "https://drive.google.com/file/d/1dfP9qK9eGnvo0hK--l0lnThxACfIZ9_f/view?usp=drive_link",
    "75": "https://drive.google.com/file/d/1prPh84jY3BYkRMTBcFtAAPeREOP9EQwe/view?usp=drive_link",
    "76": "https://drive.google.com/file/d/1VVcJajqLHrEr8XDAfVF3_QlvGuAqjuSd/view?usp=drive_link",
    "77": "https://drive.google.com/file/d/1Pw-1y6ZFgFzEDzKDdI8VdOg4O-hJHajy/view?usp=drive_link",
    "78": "https://drive.google.com/file/d/12F-MQZFQFjZPAcVKz4Y__T7fcNlyjTpO/view?usp=drive_link",
    "79": "https://drive.google.com/file/d/1RYbBaPOp2CWYfj8ltEDI_y84FdxOxlEW/view?usp=drive_link",
    "80": "https://drive.google.com/file/d/1BHy41a8XZhNitEqRm1sL3n7SontFMVka/view?usp=drive_link",
    "81": "https://drive.google.com/file/d/1Yo9gPsjPwjvTPAz_nfCGNVF5ruxFcwVU/view?usp=drive_link",
    "82": "https://drive.google.com/file/d/1mCGAXLnLPMmo7ZGxjwpPV-qES8S_0XKg/view?usp=drive_link",
    "83": "https://drive.google.com/file/d/1eggbh76-bq54M1ve_MkxrS-SfH_Uwohk/view?usp=drive_link",
    "84": "https://drive.google.com/file/d/1ZJy1KOwMiE2eL14Bgo7iU-G3Hc_BhC6x/view?usp=drive_link",
    "85": "https://drive.google.com/file/d/1fxfK10uzPHO_Q77myUDsI91gr7_e19Ut/view?usp=drive_link",
    "86": "https://drive.google.com/file/d/1OdqJ9IKe2oAN1VE3a94rt1JAwcn-16Ji/view?usp=drive_link",
    "87": "https://drive.google.com/file/d/1SodUj07iUFiC7oddWNvB6GRzHMZ9eigS/view?usp=drive_link",
    "88": "https://drive.google.com/file/d/1ZppypRlP-AbK5gQqRkh_LavTbblq4016/view?usp=drive_link",
    "89": "https://drive.google.com/file/d/1oQkZUWR5LbffNMHShbiXxSEDR3CFcTNM/view?usp=drive_link",
    "90": "https://drive.google.com/file/d/17dAYRSJciVVBIhmTtI_FMbmZBZcXQxBM/view?usp=drive_link",
    "91": "https://drive.google.com/file/d/10ceUaLciuAGkRMJNo7VHVD7ajVdSOtfj/view?usp=drive_link",
    "92": "https://drive.google.com/file/d/1szxz5749c2WcXkXTiV4ZYn_oeuIFi4ke/view?usp=drive_link",
    "93": "https://drive.google.com/file/d/1dOlYZzidB3Yoo5mjmoDSUC65dpWogoox/view?usp=drive_link",
    "94": "https://drive.google.com/file/d/17dfihWuDJsFnhwpE4y_ow493z1IXncCf/view?usp=drive_link",
    "95": "https://drive.google.com/file/d/1R9m8HCQTOYqSuSlT2FiWSBzCQ2Ry9jE5/view?usp=drive_link",
    "96": "https://drive.google.com/file/d/1yjd8bnRuSrsfGTpZY3DwEuiZJs5DXqfs/view?usp=drive_link",
}

# =============================================================================
# NORMALIZAÇÃO
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
    DIST_ID: _mk_aliases(DIST_ID) | {"id_distrito", "dist_id", "codigo_distrito", "cd_distrito"},
    ISO_ID: _mk_aliases(ISO_ID),
    OD_ID: _mk_aliases(OD_ID) | {"OD_ID", "zona_od", "zonaod", "id_od", "od", "od_id"},
    QUADRA_ID: _mk_aliases(QUADRA_ID),
    CENSO_ID: _mk_aliases(CENSO_ID) | {"cendo_id", "CENDO_ID", "setor_id", "id_setor", "codigo_setor", "cd_setor"},
    LOTE_ID: _mk_aliases(LOTE_ID) | {"id_lote", "codigo_lote", "cd_lote", "lote"},
}


def standardize_columns(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
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
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v).strip()
    s = str(v).strip()
    if s == "":
        return None
    if s.endswith(".0"):
        core = s[:-2]
        if core.replace("-", "").isdigit():
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


def normalize_id_cols(gdf: "gpd.GeoDataFrame", cols: Iterable[str]) -> "gpd.GeoDataFrame":
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    for c in cols:
        if c in g.columns:
            g[c] = g[c].map(_id_to_str)
    return g


def ensure_set_of_str(value: Any, *, drop_empty: bool = True) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        items: Iterable[Any] = value
    else:
        items = (value,)
    out: Set[str] = set()
    for x in items:
        s = _id_to_str(x)
        if s is None:
            continue
        if drop_empty and s == "":
            continue
        out.add(s)
    return out


def first_non_null_value(gdf: "gpd.GeoDataFrame", col: str) -> Optional[str]:
    if gdf is None or gdf.empty or col not in gdf.columns:
        return None
    try:
        vals = gdf[col].dropna()
        if len(vals) == 0:
            return None
        v = vals.iloc[0]
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def label_or_id(
    gdf: "gpd.GeoDataFrame",
    *,
    label_col: str,
    fallback_col: str,
    fallback_prefix: str = "",
) -> str:
    label = first_non_null_value(gdf, label_col)
    if label:
        return label
    fallback = first_non_null_value(gdf, fallback_col)
    if fallback:
        return f"{fallback_prefix}{fallback}"
    return fallback_prefix.strip() or ""


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


def choose_quadra_parent_col(
    g_quad: "gpd.GeoDataFrame",
    *,
    preferred: str = CENSO_ID,
    fallback: str = ISO_ID,
) -> Optional[str]:
    if g_quad is None or getattr(g_quad, "empty", True):
        return None
    if preferred in g_quad.columns and g_quad[preferred].notna().any():
        return preferred
    if fallback in g_quad.columns:
        return fallback
    return None


def get_quadras_subset_for_mode(
    g_quad: "gpd.GeoDataFrame",
    *,
    iso_ids: Set[str],
    filter_censo_ids: Set[str],
) -> "gpd.GeoDataFrame":
    if g_quad is None or g_quad.empty:
        return g_quad.iloc[0:0].copy()

    iso_ids = ensure_set_of_str(iso_ids)
    filter_censo_ids = ensure_set_of_str(filter_censo_ids)
    parent_col = choose_quadra_parent_col(g_quad, preferred=CENSO_ID, fallback=ISO_ID)

    if parent_col == CENSO_ID and filter_censo_ids:
        return subset_by_id_multi(g_quad, CENSO_ID, filter_censo_ids)

    if ISO_ID in g_quad.columns and iso_ids:
        return subset_by_parent_multi(g_quad, ISO_ID, iso_ids)

    return g_quad.iloc[0:0].copy()


def get_censo_subset_for_isos(g_censo: "gpd.GeoDataFrame", iso_ids: Set[str]) -> "gpd.GeoDataFrame":
    if g_censo is None or g_censo.empty:
        return g_censo.iloc[0:0].copy()
    if CENSO_PARENT in g_censo.columns:
        return subset_by_parent_multi(g_censo, CENSO_PARENT, iso_ids)
    if ISO_ID in g_censo.columns:
        return subset_by_parent_multi(g_censo, ISO_ID, iso_ids)
    return g_censo.iloc[0:0].copy()


def get_lotes_subset_for_isos(g_lote: "gpd.GeoDataFrame", iso_ids: Set[str]) -> "gpd.GeoDataFrame":
    if g_lote is None or g_lote.empty:
        return g_lote.iloc[0:0].copy()
    if ISO_ID not in g_lote.columns:
        return g_lote.iloc[0:0].copy()
    return subset_by_parent_multi(g_lote, ISO_ID, iso_ids)

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
            font-family: 'Roboto', Arial, sans-serif;
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
MAP_KEY = "map_view"


def init_state() -> None:
    st.session_state.setdefault("level", "subpref")
    st.session_state.setdefault("last_level", None)

    st.session_state.setdefault("selected_subpref_id", None)
    st.session_state.setdefault("selected_distrito_id", None)

    st.session_state.setdefault("selected_iso_ids", set())
    st.session_state.setdefault("selected_censo_ids", set())
    st.session_state.setdefault("selected_od_ids", set())
    st.session_state.setdefault("selected_quadra_ids", set())
    st.session_state.setdefault("selected_lote_ids", set())

    st.session_state.setdefault("view_center", (-23.55, -46.63))
    st.session_state.setdefault("view_zoom", 11)

    st.session_state.setdefault("last_click_sig", "")
    st.session_state.setdefault("last_draw_sig", "")

    st.session_state.setdefault("_geojson_cache", {})
    st.session_state.setdefault("_geojson_cache_order", [])
    st.session_state.setdefault("_layer_cache", {})
    st.session_state.setdefault("_layer_cache_meta", {})

    st.session_state.setdefault("_ui_action_sig", 0)
    st.session_state.setdefault("_ui_action_sig_seen", 0)

    st.session_state.setdefault("_map_level_rendered", None)
    st.session_state.setdefault("_quadra_id_col_map", QUADRA_UID)

    st.session_state.setdefault("variable", None)
    st.session_state.setdefault("selection_draw_mode", False)
    st.session_state.setdefault("post_iso_view", "quadra")


def mark_ui_action() -> None:
    st.session_state["_ui_action_sig"] = int(st.session_state.get("_ui_action_sig", 0)) + 1


def _geojson_cache_reset() -> None:
    st.session_state["_geojson_cache"] = {}
    st.session_state["_geojson_cache_order"] = []


def reset_post_iso_state() -> None:
    st.session_state["post_iso_view"] = "quadra"
    st.session_state["selected_censo_ids"] = set()
    st.session_state["selected_od_ids"] = set()
    st.session_state["selected_quadra_ids"] = set()
    st.session_state["selected_lote_ids"] = set()


def reset_to(level: str, *, clear_click_sig: bool = True) -> None:
    st.session_state["level"] = level
    if clear_click_sig:
        st.session_state["last_click_sig"] = ""
        st.session_state["last_draw_sig"] = ""
    _geojson_cache_reset()

    if level == "subpref":
        st.session_state["selected_subpref_id"] = None
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        reset_post_iso_state()
        st.session_state["view_center"] = (-23.55, -46.63)
        st.session_state["view_zoom"] = 11
        st.session_state["last_level"] = None

    elif level == "distrito":
        st.session_state["selected_distrito_id"] = None
        st.session_state["selected_iso_ids"] = set()
        reset_post_iso_state()

    elif level == "isocrona":
        st.session_state["selected_iso_ids"] = set()
        reset_post_iso_state()

    elif level == "quadra":
        reset_post_iso_state()


def _prev_level(level: str) -> Optional[str]:
    if level == "subpref":
        return None
    if level == "distrito":
        return "subpref"
    if level == "isocrona":
        return "distrito"
    if level == "quadra":
        return "isocrona"
    return None


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


def sanitize_level_state() -> None:
    lvl = st.session_state.get("level", "subpref")

    if lvl == "distrito" and _id_to_str(st.session_state.get("selected_subpref_id")) is None:
        reset_to("subpref")
        return

    if lvl in ("isocrona", "quadra") and _id_to_str(st.session_state.get("selected_distrito_id")) is None:
        reset_to("distrito")
        return

    if lvl == "quadra":
        iso_ids = st.session_state.get("selected_iso_ids", set()) or set()
        if not iso_ids:
            reset_to("isocrona")
            return

# =============================================================================
# DRIVE / IO
# =============================================================================
SECRETS_KEYS = {
    "subpref": "PB_SUBPREF_FILE_ID",
    "dist": "PB_DISTRITO_FILE_ID",
    "iso": "PB_ISOCRONAS_FILE_ID",
    "censo": "PB_CENSO_FILE_ID",
    "od": "PB_OD_FILE_ID",
    "quadra": "PB_QUADRAS_FILE_ID",
}

FALLBACK_URLS = {
    "subpref": "https://drive.google.com/file/d/1vPY34cQLCoGfADpyOJjL9pNCYkVrmSZA/view?usp=drive_link",
    "dist": "https://drive.google.com/file/d/1K-t2BiSHN_D8De0oCFxzGdrEMhnGnh10/view?usp=drive_link",
    "iso": "https://drive.google.com/file/d/1rSTVu_i-z07vKLbG3ElUNchWvvKih3xJ/view?usp=drive_link",
    "censo": "https://drive.google.com/file/d/1APp7fxT2mgTpegVisVyQwjTRWOPz6Rgn/view?usp=drive_link",
    "od": "https://drive.google.com/file/d/18yFCikpYxSvH8sqh8qULq-nMFRo2CqL7/view?usp=drive_link",
    "quadra": "https://drive.google.com/file/d/1Ivy2PyGHqFgIxSMoK3N9oik2wr5v912U/view?usp=drive_link",
}


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
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)
    m = re.search(r"([a-zA-Z0-9_-]{20,})", raw)
    return m.group(1) if m else ""


def _drive_download_candidates(file_id_or_url: str) -> List[str]:
    raw = (file_id_or_url or "").strip()
    file_id = extract_drive_id(raw)
    urls: List[str] = []

    if file_id:
        urls.append(f"https://drive.google.com/uc?export=download&id={file_id}")
        urls.append(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")

    if raw.lower().startswith("http"):
        urls.append(raw)

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _looks_like_html(path: Path) -> bool:
    try:
        head = path.read_bytes()[:2048].lower()
        return b"<!doctype html" in head or b"<html" in head or b"<head" in head
    except Exception:
        return False


def download_drive_file(file_id_or_url: str, dst: Path, label: str = "") -> Path:
    import requests

    raw = (file_id_or_url or "").strip()
    file_id = extract_drive_id(raw)
    if not raw and not file_id:
        raise RuntimeError(f"FILE_ID inválido: não foi possível extrair ID de: {file_id_or_url!r}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0 and not _looks_like_html(dst):
        return dst

    session = requests.Session()
    ui_label = label or dst.name
    candidates = _drive_download_candidates(file_id_or_url)
    last_error = None

    for base_url in candidates:
        try:
            resp = session.get(base_url, stream=True, allow_redirects=True, timeout=120)

            if "drive.google.com/uc" in base_url and file_id:
                token = None
                for k, v in resp.cookies.items():
                    if k.startswith("download_warning"):
                        token = v
                        break
                if token:
                    confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    resp = session.get(confirm_url, stream=True, allow_redirects=True, timeout=120)

            if resp.status_code != 200:
                last_error = RuntimeError(
                    f"Download falhou para '{ui_label}' (HTTP {resp.status_code}). URL={getattr(resp, 'url', base_url)}"
                )
                continue

            total = int(resp.headers.get("Content-Length", 0) or 0)
            chunk = 1024 * 1024
            downloaded = 0
            prog = st.progress(0, text=f"Baixando {ui_label}…")

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

            if dst.stat().st_size <= 0:
                dst.unlink(missing_ok=True)
                last_error = RuntimeError(f"Download de '{ui_label}' resultou em arquivo vazio.")
                continue

            if _looks_like_html(dst):
                dst.unlink(missing_ok=True)
                last_error = RuntimeError(
                    f"Download de '{ui_label}' retornou HTML. Verifique permissões e ID do Drive."
                )
                continue

            return dst

        except Exception as e:
            try:
                if dst.exists() and _looks_like_html(dst):
                    dst.unlink(missing_ok=True)
            except Exception:
                pass
            last_error = e

    raise RuntimeError(str(last_error) if last_error else f"Falha ao baixar '{ui_label}'.")


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
    return p.exists() and p.stat().st_size > 0 and not _looks_like_html(p)


def ensure_local_layer(layer_key: str, *, force_redownload: bool = False) -> Path:
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

    try:
        return download_drive_file(raw, dst, label=dst.name)
    except Exception as e:
        raise RuntimeError(
            f"Falha ao obter '{dst.name}'. "
            f"Coloque o arquivo manualmente em '{dst}' ou corrija o link/ID do Google Drive. "
            f"Detalhe: {e}"
        )

# =============================================================================
# LOTES / IO ESPECÍFICO
# =============================================================================
def get_lotes_folder_raw() -> str:
    raw_ui = str(st.session_state.get("drive_lotes_folder_raw", "")).strip()
    if raw_ui:
        return raw_ui

    raw_secret = _get_secret(LOTES_SECRET_KEY)
    if raw_secret:
        return raw_secret

    return LOTES_DRIVE_FOLDER_URL


def extract_drive_folder_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""

    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw) and "http" not in raw.lower():
        return raw

    return ""


def lotes_local_dir() -> Path:
    p = DATA_CACHE_DIR / "lotes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def lotes_repo_dir() -> Path:
    p = REPO_ROOT / "lotes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def lote_filename_for_distrito(distrito_id: Any) -> str:
    did = _id_to_str(distrito_id) or ""
    return f"Distrito_{did}.parquet"


def lote_local_path_for_distrito(distrito_id: Any) -> Path:
    return lotes_local_dir() / lote_filename_for_distrito(distrito_id)


def _download_url_from_file_id(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def local_lote_candidates(distrito_id: Any) -> List[Path]:
    filename = lote_filename_for_distrito(distrito_id)
    return [
        REPO_ROOT / filename,
        REPO_ROOT / "data" / filename,
        lotes_repo_dir() / filename,
        DATA_CACHE_DIR / filename,
        lotes_local_dir() / filename,
    ]


def try_copy_lote_from_local_sources(distrito_id: Any, dst: Path) -> Optional[Path]:
    for p in local_lote_candidates(distrito_id):
        if p.exists() and p.stat().st_size > 0 and not _looks_like_html(p):
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
            return dst if dst.exists() else p
    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def list_drive_folder_files(folder_id: str) -> Dict[str, str]:
    import requests

    folder_id = (folder_id or "").strip()
    if not folder_id:
        return {}

    urls = [
        f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing",
        f"https://drive.google.com/drive/u/0/folders/{folder_id}",
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    found: Dict[str, str] = {}

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code != 200:
                continue

            html_txt = resp.text

            matches = re.findall(r'\["([a-zA-Z0-9_-]{20,})","([^"]+\.parquet)"', html_txt)
            for file_id, name in matches:
                found[name] = file_id

            matches2 = re.findall(r'\["([^"]+\.parquet)","([a-zA-Z0-9_-]{20,})"', html_txt)
            for name, file_id in matches2:
                found[name] = file_id

            if found:
                return found

        except Exception:
            continue

    return found


def find_lote_file_id_in_folder(distrito_id: Any) -> str:
    did = _id_to_str(distrito_id)
    if not did:
        return ""

    folder_raw = get_lotes_folder_raw()
    folder_id = extract_drive_folder_id(folder_raw)
    if not folder_id:
        return ""

    target_name = lote_filename_for_distrito(did)
    files_map = list_drive_folder_files(folder_id)
    return files_map.get(target_name, "")


def ensure_local_lote_file(distrito_id: Any, *, force_redownload: bool = False) -> Path:
    did = _id_to_str(distrito_id)
    if not did:
        raise RuntimeError("distrito_id inválido para carregar arquivo de lotes.")

    dst = lote_local_path_for_distrito(did)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if force_redownload:
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass

    if dst.exists() and dst.stat().st_size > 0 and not _looks_like_html(dst):
        return dst

    copied = try_copy_lote_from_local_sources(did, dst)
    if copied is not None and copied.exists():
        return copied

    direct_url = str(LOTES_LINKS_BY_DISTRITO.get(did, "")).strip()
    if direct_url:
        try:
            return download_drive_file(direct_url, dst, label=lote_filename_for_distrito(did))
        except Exception as e:
            raise RuntimeError(
                f"Falha ao baixar o arquivo de lotes do distrito '{did}' pelo link direto consolidado. "
                f"Detalhe: {e}"
            )

    filename = lote_filename_for_distrito(did)
    file_id = find_lote_file_id_in_folder(did)

    if file_id:
        try:
            return download_drive_file(_download_url_from_file_id(file_id), dst, label=filename)
        except Exception as e:
            raise RuntimeError(
                f"Falha ao baixar o arquivo de lotes '{filename}' a partir do Google Drive. Detalhe: {e}"
            )

    raise RuntimeError(
        f"Não foi possível localizar o arquivo de lotes '{filename}'. "
        f"O app procura primeiro localmente em: "
        f"'{REPO_ROOT}', '{REPO_ROOT / 'data'}', '{lotes_repo_dir()}', '{DATA_CACHE_DIR}' e '{lotes_local_dir()}'. "
        f"Depois tenta o link consolidado por distrito e, por fim, a pasta do Google Drive configurada em: "
        f"{get_lotes_folder_raw()} . "
        f"Verifique se o arquivo existe com esse nome exato."
    )

# =============================================================================
# READ / FILTER
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
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
        if QUADRA_ID in g.columns:
            g[QUADRA_ID] = g[QUADRA_ID].map(lambda x: normalize_quadra_id(x, 6))
        if ISO_ID in g.columns:
            g[ISO_ID] = g[ISO_ID].map(_id_to_str)
        if ISO_ID in g.columns and QUADRA_ID in g.columns:
            g[QUADRA_UID] = [make_quadra_uid(i, q) for i, q in zip(g[ISO_ID], g[QUADRA_ID])]

    cache[layer_key] = g
    cache_meta[layer_key] = meta
    st.session_state["_layer_cache"] = cache
    st.session_state["_layer_cache_meta"] = cache_meta
    return g


def read_lotes_by_distrito(distrito_id: Any) -> Optional["gpd.GeoDataFrame"]:
    did = _id_to_str(distrito_id)
    if not did:
        return None

    cache_key = f"lote__{did}"

    try:
        p = ensure_local_lote_file(did)
    except Exception as e:
        st.error(str(e))
        return None

    try:
        meta = (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
    except Exception:
        meta = (str(p), 0.0, 0)

    cache: Dict[str, Any] = st.session_state.get("_layer_cache", {})
    cache_meta: Dict[str, Any] = st.session_state.get("_layer_cache_meta", {})

    if cache_key in cache and cache_meta.get(cache_key) == meta:
        g_cached = cache.get(cache_key)
        if g_cached is not None:
            return g_cached

    g = read_gdf_parquet(str(p))
    if g is None or g.empty:
        st.warning(f"Arquivo de lotes do distrito '{did}' vazio ou inválido.")
        return None

    g = standardize_columns(g)
    g = _drop_bad_geoms(g)
    g = normalize_id_cols(g, LAYER_ID_COLS.get("lote", []))

    cache[cache_key] = g
    cache_meta[cache_key] = meta
    st.session_state["_layer_cache"] = cache
    st.session_state["_layer_cache_meta"] = cache_meta
    return g

# =============================================================================
# CSV CLUSTER
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
    except Exception:
        st.warning(f"Não foi possível baixar {QUADRAS_CSV_FILENAME} do Drive.")
        return dst


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
# GEOJSON CACHE
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
    if gdf is None or gdf.empty:
        return ""

    keep_cols = keep_cols or []
    keep_cols = [c for c in keep_cols if c in gdf.columns]

    cols = keep_cols + ["geometry"]
    try:
        g = gdf[cols].copy()
    except Exception:
        return ""

    g = _drop_bad_geoms(g)

    try:
        return g.to_json()
    except Exception:
        return ""

# =============================================================================
# CLICK / DRAW
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


def add_draw_tools(m) -> None:
    if folium is None or Draw is None or m is None:
        return

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "marker": False,
            "circle": False,
            "circlemarker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": False, "remove": True},
    ).add_to(m)


def _extract_drawn_geometry(map_state: Dict[str, Any]):
    if not isinstance(map_state, dict):
        return None

    candidates = [
        map_state.get("all_drawings"),
        map_state.get("last_active_drawing"),
        map_state.get("last_drawing"),
    ]

    for cand in candidates:
        if not cand:
            continue

        if isinstance(cand, list) and len(cand) > 0:
            last = cand[-1]
            geom = last.get("geometry") if isinstance(last, dict) else None
            if geom and shape is not None:
                try:
                    return shape(geom)
                except Exception:
                    pass

        if isinstance(cand, dict):
            geom = cand.get("geometry", cand)
            if geom and shape is not None:
                try:
                    return shape(geom)
                except Exception:
                    pass

    return None


def select_features_by_geometry(
    gdf: "gpd.GeoDataFrame",
    geom,
    id_col: str,
    selection_state_key: str,
    mode: str = "add",
) -> None:
    if gdf is None or gdf.empty or geom is None or id_col not in gdf.columns:
        return

    try:
        hits = gdf[gdf.geometry.intersects(geom)]
    except Exception:
        return

    ids = {_id_to_str(v) for v in hits[id_col].tolist()}
    ids = {v for v in ids if v is not None}

    if mode == "replace":
        st.session_state[selection_state_key] = ids
        return

    current = ensure_set_of_str(st.session_state.get(selection_state_key, set()))
    st.session_state[selection_state_key] = current | ids


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

# =============================================================================
# FOLIUM
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
        folium.map.CustomPane("labels", z_index=700).add_to(m)
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

# =============================================================================
# LABELS
# =============================================================================
def _format_label_multiline(text: Any) -> str:
    if text is None:
        return ""
    txt = str(text).strip()
    if not txt:
        return ""

    for sep in [" - ", " – ", "/"]:
        if sep in txt:
            parts = [p.strip() for p in txt.split(sep) if str(p).strip()]
            if len(parts) >= 2:
                return "<br>".join(html.escape(p) for p in parts[:2])

    return html.escape(txt)


def add_labels_on_map(
    m,
    gdf: "gpd.GeoDataFrame",
    label_col: str,
    *,
    font_size: int = 12,
    color: str = "#000000",
    weight: str = "700",
) -> None:
    if folium is None or gdf is None or gdf.empty:
        return
    if label_col not in gdf.columns:
        return

    try:
        g = gdf.copy()
        g = g[g.geometry.notna()].copy()
        if g.empty:
            return

        points = g.geometry.representative_point()

        for idx, row in g.iterrows():
            label = row.get(label_col)
            if pd.isna(label):
                continue

            txt_html = _format_label_multiline(label)
            if not txt_html:
                continue

            pt = points.loc[idx]
            if pt is None or getattr(pt, "is_empty", False):
                continue

            folium.Marker(
                location=[pt.y, pt.x],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 18),
                    html=f"""
                    <div style="
                        font-family: Roboto, Arial, sans-serif;
                        font-size: {font_size}px;
                        color: {color};
                        font-weight: {weight};
                        text-align: center;
                        white-space: nowrap;
                        line-height: 1.1;
                        text-shadow:
                            -1px -1px 0 #ffffff,
                             1px -1px 0 #ffffff,
                            -1px  1px 0 #ffffff,
                             1px  1px 0 #ffffff,
                             0px  0px 3px #ffffff;
                        pointer-events: none;
                    ">
                        {txt_html}
                    </div>
                    """
                ),
            ).add_to(m)
    except Exception:
        pass


def add_parent_fill(
    m,
    gdf: "gpd.GeoDataFrame",
    name: str,
    *,
    pane: str = "parent_fill",
    fill_color: str = PB_BROWN,
    fill_opacity: float = PARENT_FILL_OPACITY,
    stroke_color: str = PB_BLACK,
    stroke_weight: float = PARENT_STROKE_WEIGHT,
    stroke_opacity: float = PARENT_STROKE_OPACITY,
    dash_array: Optional[str] = PARENT_STROKE_DASH,
    simplify_tol: float = 0.0,
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
    base_color: str = PB_BLACK,
    base_weight: float = 1.0,
    fill_color: str = "#ffffff",
    fill_opacity: float = 0.10,
    selected_color: str = PB_BLACK,
    selected_weight: float = 2.2,
    selected_fill_opacity: float = 0.26,
    tooltip_prefix: str = "ID: ",
    simplify_tol: float = 0.0,
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
            "opacity": 1.0,
            "lineCap": LINE_CAP,
            "lineJoin": LINE_JOIN,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda _f: {
            "color": PB_BLACK,
            "weight": base_weight + 1.0,
            "opacity": 1.0,
            "fillOpacity": min(fill_opacity + 0.10, 0.40),
        },
        tooltip=tooltip_base,
    ).add_to(fg_base)
    fg_base.add_to(m)

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
                        "opacity": 1.0,
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
    base_color: str = PB_BLACK,
    base_weight: float = 1.0,
    fill_opacity: float = 0.14,
    selected_color: str = PB_BLACK,
    selected_weight: float = 2.2,
    selected_fill_opacity: float = 0.28,
    tooltip_prefix: str = "ID: ",
    simplify_tol: float = 0.0,
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
            "opacity": 1.0,
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
        highlight_function=lambda _f: {
            "color": PB_BLACK,
            "weight": base_weight + 1.0,
            "opacity": 1.0,
            "fillOpacity": min(fill_opacity + 0.10, 1.0),
        },
        tooltip=tooltip_base,
    ).add_to(fg_base)
    fg_base.add_to(m)

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
                        "opacity": 1.0,
                        "lineCap": LINE_CAP,
                        "lineJoin": LINE_JOIN,
                        "fillColor": "#ffffff",
                        "fillOpacity": selected_fill_opacity,
                    },
                ).add_to(fg_sel)
                fg_sel.add_to(m)

# =============================================================================
# HELPERS PÓS-ISÓCRONA
# =============================================================================
def build_post_iso_data() -> Dict[str, Any]:
    iso_ids = ensure_set_of_str(st.session_state.get("selected_iso_ids", set()))
    distrito_id = _id_to_str(st.session_state.get("selected_distrito_id"))

    out: Dict[str, Any] = {
        "iso_ids": iso_ids,
        "g_parent": None,
        "g_censo": None,
        "g_od": None,
        "g_quadra": None,
        "g_lote": None,
        "quadra_id_col": QUADRA_UID,
    }

    if not iso_ids:
        return out

    g_iso = read_layer("iso")
    if g_iso is not None:
        out["g_parent"] = subset_by_id_multi(g_iso, ISO_ID, iso_ids)

    g_censo = read_layer("censo")
    if g_censo is not None:
        out["g_censo"] = get_censo_subset_for_isos(g_censo, iso_ids)

    g_od = read_layer("od")
    if g_od is not None:
        if ISO_ID not in g_od.columns:
            st.warning(f"ZonasOD.parquet sem coluna '{ISO_ID}'. Colunas encontradas: {list(g_od.columns)}")
        else:
            out["g_od"] = subset_by_parent_multi(g_od, ISO_ID, iso_ids)

    g_quad = read_layer("quadra")
    if g_quad is not None:
        censo_ids = ensure_set_of_str(st.session_state.get("selected_censo_ids", set()))
        g_quad_show = get_quadras_subset_for_mode(
            g_quad,
            iso_ids=iso_ids,
            filter_censo_ids=censo_ids,
        )
        id_col = QUADRA_UID if QUADRA_UID in g_quad_show.columns else QUADRA_ID
        if id_col not in g_quad_show.columns:
            id_col = QUADRA_ID if QUADRA_ID in g_quad_show.columns else QUADRA_UID
        out["g_quadra"] = g_quad_show
        out["quadra_id_col"] = id_col
        st.session_state["_quadra_id_col_map"] = id_col

    if distrito_id is not None:
        g_lote = read_lotes_by_distrito(distrito_id)
        if g_lote is not None:
            if ISO_ID not in g_lote.columns:
                st.warning(f"Arquivo de lotes do distrito '{distrito_id}' sem coluna '{ISO_ID}'.")
            else:
                out["g_lote"] = get_lotes_subset_for_isos(g_lote, iso_ids)

    return out

# =============================================================================
# EVENTOS
# =============================================================================
def consume_map_event(level: str, map_state: Dict[str, Any], allow_click: bool = True) -> None:
    if not allow_click:
        return

    tooltip_raw = (map_state or {}).get("last_object_clicked_tooltip") or None
    click = (map_state or {}).get("last_clicked") if isinstance((map_state or {}).get("last_clicked"), dict) else None

    if level in ("subpref", "distrito"):
        id_col = SUBPREF_ID if level == "subpref" else DIST_ID
        picked = _pick_id_from_last_object(map_state, id_col) or parse_tooltip_id(tooltip_raw)

        if not picked and isinstance(click, dict):
            if level == "subpref":
                g = read_layer("subpref")
                if g is not None:
                    picked = pick_feature_id(g, click, SUBPREF_ID)
            else:
                sp = _id_to_str(st.session_state.get("selected_subpref_id"))
                if sp is not None:
                    g = read_layer("dist")
                    if g is not None:
                        picked = pick_feature_id(subset_by_parent(g, DIST_PARENT, sp), click, DIST_ID)

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
        picked = _pick_id_from_last_object(map_state, ISO_ID) or parse_tooltip_id(tooltip_raw)

        if not picked and isinstance(click, dict):
            d = _id_to_str(st.session_state.get("selected_distrito_id"))
            if d is not None:
                g = read_layer("iso")
                if g is not None:
                    g_show = subset_by_parent(g, ISO_PARENT, d)
                    if g_show.empty and DIST_ID in g.columns:
                        g2 = g.copy()
                        g2[DIST_ID] = g2[DIST_ID].astype(str).str.strip()
                        g_show = g2[g2[DIST_ID] == str(d).strip()].copy()
                    picked = pick_feature_id(g_show, click, ISO_ID)

        if not picked:
            return

        sig = _click_signature(picked, click)
        if sig == st.session_state.get("last_click_sig", ""):
            return
        st.session_state["last_click_sig"] = sig
        _toggle_in_set("selected_iso_ids", picked)
        return

    if level == "quadra":
        post_view = st.session_state.get("post_iso_view", "quadra")
        data = build_post_iso_data()

        if post_view == "quadra":
            g_show = data.get("g_quadra")
            id_col_map = data.get("quadra_id_col", QUADRA_UID)
            picked = None

            obj = (map_state or {}).get("last_object_clicked") or None
            if isinstance(obj, dict):
                props = obj.get("properties") if isinstance(obj.get("properties"), dict) else obj
                if isinstance(props, dict):
                    if id_col_map == QUADRA_UID:
                        picked = _id_to_str(props.get(QUADRA_UID)) or make_quadra_uid(props.get(ISO_ID), props.get(QUADRA_ID))
                    else:
                        picked = _id_to_str(props.get(id_col_map)) or _id_to_str(props.get(QUADRA_ID))

            if not picked and isinstance(click, dict) and g_show is not None and id_col_map in g_show.columns:
                picked = pick_feature_id(g_show, click, id_col_map)

            if not picked:
                return

            sig = _click_signature(picked, click)
            if sig == st.session_state.get("last_click_sig", ""):
                return
            st.session_state["last_click_sig"] = sig
            _toggle_in_set("selected_quadra_ids", picked)
            return

        if post_view == "censo":
            g_show = data.get("g_censo")
            picked = _pick_id_from_last_object(map_state, CENSO_ID) or parse_tooltip_id(tooltip_raw)
            if not picked and isinstance(click, dict) and g_show is not None:
                picked = pick_feature_id(g_show, click, CENSO_ID)
            if not picked:
                return
            sig = _click_signature(picked, click)
            if sig == st.session_state.get("last_click_sig", ""):
                return
            st.session_state["last_click_sig"] = sig
            _toggle_in_set("selected_censo_ids", picked)
            return

        if post_view == "od":
            g_show = data.get("g_od")
            picked = _pick_id_from_last_object(map_state, OD_ID) or parse_tooltip_id(tooltip_raw)
            if not picked and isinstance(click, dict) and g_show is not None:
                picked = pick_feature_id(g_show, click, OD_ID)
            if not picked:
                return
            sig = _click_signature(picked, click)
            if sig == st.session_state.get("last_click_sig", ""):
                return
            st.session_state["last_click_sig"] = sig
            _toggle_in_set("selected_od_ids", picked)
            return

        if post_view == "lote":
            g_show = data.get("g_lote")
            picked = _pick_id_from_last_object(map_state, LOTE_ID) or parse_tooltip_id(tooltip_raw)
            if not picked and isinstance(click, dict) and g_show is not None and LOTE_ID in g_show.columns:
                picked = pick_feature_id(g_show, click, LOTE_ID)
            if not picked:
                return
            sig = _click_signature(picked, click)
            if sig == st.session_state.get("last_click_sig", ""):
                return
            st.session_state["last_click_sig"] = sig
            _toggle_in_set("selected_lote_ids", picked)
            return


def consume_draw_selection(level: str, map_state: Dict[str, Any]) -> None:
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
        if g is None or d is None:
            return
        g_show = subset_by_parent(g, ISO_PARENT, d)
        if g_show.empty and DIST_ID in g.columns:
            g2 = g.copy()
            g2[DIST_ID] = g2[DIST_ID].astype(str).str.strip()
            g_show = g2[g2[DIST_ID] == str(d).strip()].copy()
        select_features_by_geometry(g_show, geom, ISO_ID, "selected_iso_ids", mode="add")
        return

    if level == "quadra":
        data = build_post_iso_data()
        post_view = st.session_state.get("post_iso_view", "quadra")

        if post_view == "quadra":
            g_show = data.get("g_quadra")
            id_col_map = data.get("quadra_id_col", QUADRA_UID)
            if g_show is not None and id_col_map in g_show.columns:
                select_features_by_geometry(g_show, geom, id_col_map, "selected_quadra_ids", mode="add")
            return

        if post_view == "censo":
            g_show = data.get("g_censo")
            if g_show is not None:
                select_features_by_geometry(g_show, geom, CENSO_ID, "selected_censo_ids", mode="add")
            return

        if post_view == "od":
            g_show = data.get("g_od")
            if g_show is not None:
                select_features_by_geometry(g_show, geom, OD_ID, "selected_od_ids", mode="add")
            return

        if post_view == "lote":
            g_show = data.get("g_lote")
            if g_show is not None and LOTE_ID in g_show.columns:
                select_features_by_geometry(g_show, geom, LOTE_ID, "selected_lote_ids", mode="add")
            return

# =============================================================================
# UI
# =============================================================================
def _variables_for_level(level: str) -> List[str]:
    if level == "subpref":
        return ["Subprefeituras"]
    if level == "distrito":
        return ["Distritos"]
    if level == "isocrona":
        return ["Isócronas", "Isócronas (classes)"]
    if level == "quadra":
        return ["Quadras", "Cluster", "Setor censitário", "Zonas OD", "Lotes"]
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
    iso_ids = ensure_set_of_str(st.session_state.get("selected_iso_ids", set()))
    if not iso_ids:
        return
    g_iso = read_layer("iso")
    if g_iso is None:
        return
    set_view_to_gdf(subset_by_id_multi(g_iso, ISO_ID, iso_ids), bump=0, zmax=18)


def _fit_selected_post_level() -> None:
    data = build_post_iso_data()
    post_view = st.session_state.get("post_iso_view", "quadra")

    if post_view == "quadra":
        ids = ensure_set_of_str(st.session_state.get("selected_quadra_ids", set()))
        g = data.get("g_quadra")
        id_col = data.get("quadra_id_col", QUADRA_UID)
        if g is not None and ids and id_col in g.columns:
            set_view_to_gdf(subset_by_id_multi(g, id_col, ids), bump=1, zmax=19)
        return

    if post_view == "censo":
        ids = ensure_set_of_str(st.session_state.get("selected_censo_ids", set()))
        g = data.get("g_censo")
        if g is not None and ids:
            set_view_to_gdf(subset_by_id_multi(g, CENSO_ID, ids), bump=0, zmax=18)
        return

    if post_view == "od":
        ids = ensure_set_of_str(st.session_state.get("selected_od_ids", set()))
        g = data.get("g_od")
        if g is not None and ids:
            set_view_to_gdf(subset_by_id_multi(g, OD_ID, ids), bump=0, zmax=18)
        return

    if post_view == "lote":
        ids = ensure_set_of_str(st.session_state.get("selected_lote_ids", set()))
        g = data.get("g_lote")
        if g is not None and ids and LOTE_ID in g.columns:
            set_view_to_gdf(subset_by_id_multi(g, LOTE_ID, ids), bump=1, zmax=20)
        return


def _on_post_iso_view_change() -> None:
    mark_ui_action()


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
    st.subheader("Ações e seleção", anchor=False)

    if lvl == "isocrona":
        ok_iso = len(ensure_set_of_str(st.session_state.get("selected_iso_ids", set()))) > 0

        st.button(
            "Ajustar às isócronas selecionadas",
            use_container_width=True,
            disabled=not ok_iso,
            on_click=lambda: (mark_ui_action(), _fit_selected_isos()),
        )

        st.button(
            "Avançar para Visualização detalhada",
            type="primary",
            use_container_width=True,
            disabled=not ok_iso,
            on_click=lambda: (
                mark_ui_action(),
                st.session_state.__setitem__("post_iso_view", "quadra"),
                st.session_state.__setitem__("level", "quadra"),
                st.session_state.__setitem__("last_level", None),
            ),
        )

        st.caption("Selecione uma ou mais isócronas antes de avançar.")

    if lvl == "quadra":
        st.radio(
            "Visualização pós-isócronas",
            options=["quadra", "lote", "censo", "od"],
            format_func=lambda x: {
                "quadra": "Quadras",
                "lote": "Lotes",
                "censo": "Setor censitário",
                "od": "Zonas OD",
            }[x],
            key="post_iso_view",
            horizontal=False,
            on_change=_on_post_iso_view_change,
        )

        st.button(
            "Ajustar ao selecionado",
            use_container_width=True,
            on_click=lambda: (mark_ui_action(), _fit_selected_post_level()),
        )

    st.checkbox(
        "Habilitar seleção por caixa/laço",
        key="selection_draw_mode",
        on_change=mark_ui_action,
    )

# =============================================================================
# MAP RENDER
# =============================================================================
def render_map_panel() -> None:
    level = st.session_state.get("level", "subpref")
    ensure_variable_for_level(level)

    title = ""
    m = None

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
        tooltip_subpref_col = "sp_nome" if "sp_nome" in g_sub.columns else SUBPREF_ID

        add_polygons_selectable(
            m,
            g_sub,
            "Subprefeituras",
            SUBPREF_ID,
            tooltip_col=tooltip_subpref_col,
            selected_ids=set(),
            fill_opacity=0.06,
            selected_fill_opacity=0.0,
            tooltip_prefix="Subpref: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["subpref"],
            cache_key=f"subpref:{SIMPLIFY_TOL_BY_LEVEL['subpref']}",
        )

        if "sp_nome" in g_sub.columns:
            add_labels_on_map(
                m,
                g_sub,
                "sp_nome",
                font_size=13,
                color=PB_BLACK,
                weight="700",
            )

    elif level == "distrito":
        sp = _id_to_str(st.session_state.get("selected_subpref_id"))
        if sp is None:
            reset_to("subpref")
            return

        g_dist = read_layer("dist")
        g_sub = read_layer("subpref")
        if g_dist is None or g_sub is None:
            st.stop()

        g_parent = subset_by_id(g_sub, SUBPREF_ID, sp)
        subpref_nome = label_or_id(g_parent, label_col="sp_nome", fallback_col=SUBPREF_ID)
        title = f"Distritos ({subpref_nome})"

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

        tooltip_dist_col = "ds_nome" if "ds_nome" in g_show.columns else DIST_ID

        add_polygons_selectable(
            m,
            g_show,
            "Distritos",
            DIST_ID,
            tooltip_col=tooltip_dist_col,
            selected_ids=set(),
            fill_opacity=0.06,
            tooltip_prefix="Distrito: ",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"dist:sp:{sp}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

        if "ds_nome" in g_show.columns:
            add_labels_on_map(
                m,
                g_show,
                "ds_nome",
                font_size=12,
                color=PB_BLACK,
                weight="700",
            )

    elif level == "isocrona":
        d = _id_to_str(st.session_state.get("selected_distrito_id"))
        if d is None:
            reset_to("distrito")
            return

        sel_n = len(st.session_state.get("selected_iso_ids", set()) or set())

        g_iso = read_layer("iso")
        g_dist = read_layer("dist")
        if g_iso is None or g_dist is None:
            st.stop()

        g_parent_dist = subset_by_id(g_dist, DIST_ID, d)
        distrito_nome = label_or_id(g_parent_dist, label_col="ds_nome", fallback_col=DIST_ID)
        title = f"Isócronas ({distrito_nome})"
        if sel_n > 0:
            title = f"{title} — selecionadas: {sel_n}"

        if DIST_ID not in g_iso.columns:
            st.error(f"Isocronas.parquet não contém '{DIST_ID}'. Colunas: {list(g_iso.columns)}")
            st.stop()

        if ISO_ID not in g_iso.columns:
            st.error(f"Isocronas.parquet não contém '{ISO_ID}'. Colunas: {list(g_iso.columns)}")
            st.stop()

        g_show_iso = subset_by_parent(g_iso, ISO_PARENT, d)
        if g_show_iso.empty and DIST_ID in g_iso.columns:
            g_iso2 = g_iso.copy()
            g_iso2[DIST_ID] = g_iso2[DIST_ID].astype(str).str.strip()
            g_show_iso = g_iso2[g_iso2[DIST_ID] == str(d).strip()].copy()

        if st.session_state.get("last_level") != "isocrona":
            set_view_to_gdf(g_show_iso if not g_show_iso.empty else g_parent_dist, bump=0)
            st.session_state["last_level"] = "isocrona"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        add_parent_fill(
            m,
            g_parent_dist,
            "Distrito selecionado (sombra)",
            simplify_tol=SIMPLIFY_TOL_BY_LEVEL["distrito"],
            cache_key=f"parent:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['distrito']}",
        )

        g_show_viz = g_show_iso.copy()
        if ISO_CLASS_COL in g_show_viz.columns:
            pairs = g_show_viz[ISO_CLASS_COL].map(iso_label_color)
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
                fill_opacity=ISO_FILL_OPACITY_CLASSES,
                selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"isoVIZ:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
                default_fill=ISO_DEFAULT_COLOR,
            )
        else:
            add_polygons_selectable(
                m,
                g_show_iso,
                "Isócronas",
                ISO_ID,
                selected_ids=st.session_state.get("selected_iso_ids", set()),
                tooltip_col=ISO_ID,
                fill_opacity=ISO_FILL_OPACITY_DEFAULT,
                selected_fill_opacity=0.0,
                tooltip_prefix="Isócrona: ",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"iso:dist:{d}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
            )

    elif level == "quadra":
        iso_ids = ensure_set_of_str(st.session_state.get("selected_iso_ids"))
        if not iso_ids:
            reset_to("isocrona")
            return

        post_view = st.session_state.get("post_iso_view", "quadra")
        data = build_post_iso_data()
        g_parent = data.get("g_parent")

        label_map = {
            "quadra": "Quadras",
            "lote": "Lotes",
            "censo": "Setor censitário",
            "od": "Zonas OD",
        }
        title = f"{label_map.get(post_view, 'Quadras')} — filtrado pelas isócronas selecionadas"

        target = g_parent
        if post_view == "quadra":
            gq = data.get("g_quadra")
            if gq is not None and not gq.empty:
                target = gq
        elif post_view == "lote":
            gl = data.get("g_lote")
            if gl is not None and not gl.empty:
                target = gl
        elif post_view == "censo":
            gc = data.get("g_censo")
            if gc is not None and not gc.empty:
                target = gc
        elif post_view == "od":
            go = data.get("g_od")
            if go is not None and not go.empty:
                target = go

        if st.session_state.get("last_level") != "quadra":
            if target is not None and not getattr(target, "empty", True):
                set_view_to_gdf(target, bump=0)
            st.session_state["last_level"] = "quadra"

        m = make_carto_map(center=st.session_state["view_center"], zoom=st.session_state["view_zoom"])

        if g_parent is not None and not g_parent.empty:
            add_parent_fill(
                m,
                g_parent,
                "Isócronas selecionadas (sombra)",
                simplify_tol=SIMPLIFY_TOL_BY_LEVEL["isocrona"],
                cache_key=f"parent:iso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['isocrona']}",
            )

        if post_view == "quadra":
            g_quad = data.get("g_quadra")
            id_col_map = data.get("quadra_id_col", QUADRA_UID)
            if g_quad is not None and not g_quad.empty:
                g_quad_viz = attach_quadras_csv(g_quad)
                if CLUSTER_COL in g_quad_viz.columns:
                    g_quad_viz["__cluster_code"] = g_quad_viz[CLUSTER_COL].apply(_coerce_int)
                    g_quad_viz["__cluster_color"] = g_quad_viz["__cluster_code"].apply(cluster_color)

                if st.session_state.get("variable") == "Cluster" and "__cluster_color" in g_quad_viz.columns:
                    add_polygons_selectable_colored(
                        m,
                        g_quad_viz,
                        "Quadras",
                        id_col_map,
                        fill_color_col="__cluster_color",
                        selected_ids=st.session_state.get("selected_quadra_ids", set()),
                        tooltip_col=QUADRA_ID if QUADRA_ID in g_quad_viz.columns else id_col_map,
                        fill_opacity=0.9,
                        selected_fill_opacity=0.0,
                        tooltip_prefix="Quadra: ",
                        simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                        cache_key=f"quad-overlay:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
                        default_fill=CLUSTER_NULL_COLOR,
                    )
                else:
                    add_polygons_selectable(
                        m,
                        g_quad,
                        "Quadras",
                        id_col_map,
                        tooltip_col=QUADRA_ID if QUADRA_ID in g_quad.columns else id_col_map,
                        selected_ids=st.session_state.get("selected_quadra_ids", set()),
                        fill_color="#ffffff",
                        fill_opacity=0.06,
                        selected_fill_opacity=0.0,
                        tooltip_prefix="Quadra: ",
                        simplify_tol=SIMPLIFY_TOL_BY_LEVEL["quadra"],
                        cache_key=f"quadB-overlay:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['quadra']}",
                    )
            else:
                st.warning("Nenhuma quadra encontrada para as isócronas selecionadas.")

        elif post_view == "lote":
            g_lote = data.get("g_lote")
            if g_lote is not None and not g_lote.empty:
                add_polygons_selectable(
                    m,
                    g_lote,
                    "Lotes",
                    LOTE_ID,
                    tooltip_col=LOTE_ID if LOTE_ID in g_lote.columns else ISO_ID,
                    selected_ids=st.session_state.get("selected_lote_ids", set()),
                    fill_color="#b7d7a8",
                    fill_opacity=0.18,
                    base_color=PB_BLACK,
                    base_weight=1.0,
                    selected_fill_opacity=0.0,
                    tooltip_prefix="Lote: ",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["lote"],
                    cache_key=f"lote-postiso:{_id_to_str(st.session_state.get('selected_distrito_id'))}:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['lote']}",
                )
            else:
                st.warning("Nenhum lote encontrado para as isócronas selecionadas no distrito atual.")

        elif post_view == "censo":
            g_censo = data.get("g_censo")
            if g_censo is not None and not g_censo.empty:
                add_polygons_selectable(
                    m,
                    g_censo,
                    "Setor censitário",
                    CENSO_ID,
                    tooltip_col=CENSO_ID,
                    selected_ids=st.session_state.get("selected_censo_ids", set()),
                    fill_color="#7aa6c2",
                    fill_opacity=0.10,
                    selected_fill_opacity=0.0,
                    tooltip_prefix="Setor: ",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["censo"],
                    cache_key=f"censo-postiso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['censo']}",
                )
            else:
                st.warning("Nenhum setor censitário encontrado para as isócronas selecionadas.")

        elif post_view == "od":
            g_od = data.get("g_od")
            if g_od is not None and not g_od.empty:
                add_polygons_selectable(
                    m,
                    g_od,
                    "Zonas OD",
                    OD_ID,
                    tooltip_col=OD_ID,
                    selected_ids=st.session_state.get("selected_od_ids", set()),
                    fill_color="#d9b26f",
                    fill_opacity=0.16,
                    base_color=PB_BLACK,
                    base_weight=1.0,
                    selected_fill_opacity=0.0,
                    tooltip_prefix="Zona OD: ",
                    simplify_tol=SIMPLIFY_TOL_BY_LEVEL["od"],
                    cache_key=f"od-postiso:{'|'.join(sorted(list(iso_ids)))}:{SIMPLIFY_TOL_BY_LEVEL['od']}",
                )
            else:
                st.warning("Nenhuma zona OD encontrada para as isócronas selecionadas.")

    if m is not None and st.session_state.get("selection_draw_mode", False):
        add_draw_tools(m)

    st.markdown(f"### {title}")

    if st_folium is None:
        st.error("Falha ao importar `streamlit_folium`.")
        return

    _ = st_folium(
        m,
        height=780,
        use_container_width=True,
        key=MAP_KEY,
        returned_objects=[
            "last_clicked",
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "all_drawings",
            "last_active_drawing",
        ],
    )
    st.session_state["_map_level_rendered"] = level

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

    ui_sig = int(st.session_state.get("_ui_action_sig", 0))
    ui_seen = int(st.session_state.get("_ui_action_sig_seen", 0))
    ui_action = ui_sig != ui_seen
    st.session_state["_ui_action_sig_seen"] = ui_sig

    if ui_action:
        st.session_state["last_click_sig"] = ""
        st.session_state["last_draw_sig"] = ""

    cur_level = st.session_state.get("level", "subpref")
    rendered_level = st.session_state.get("_map_level_rendered")
    map_state_prev = st.session_state.get(MAP_KEY, {}) or {}

    allow_click = (not ui_action) and (rendered_level == cur_level)

    if allow_click and isinstance(map_state_prev, dict) and map_state_prev:
        consume_map_event(cur_level, map_state_prev, allow_click=True)
        consume_draw_selection(cur_level, map_state_prev)

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
