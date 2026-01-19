import streamlit as st
import pandas as pd
from pathlib import Path
import re
from unicodedata import normalize as _ud_norm

try:
    import geopandas as gpd
except Exception:
    gpd = None

BASE_DIR = Path("dash_planbairros")
ADM_DIR = BASE_DIR / "limites_administrativos"
DENS_DIR = BASE_DIR / "densidade"

def _slug(s: str) -> str:
    s2 = _ud_norm("NFKD", str(s)).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z0-9]+", "", s2.strip().lower())

def _first_parquet_matching(folder: Path, name_candidates: list[str]):
    if not folder.exists():
        return None
    wanted = {_slug(n) for n in name_candidates}
    for fp in folder.glob("*.parquet"):
        if _slug(fp.stem) in wanted:
            return fp
    return None

ADMIN_NAME_MAP = {
    "Distritos": ["Distritos"],
    "SetoresCensitarios2023": ["SetoresCensitarios2023", "SetoresCensitarios"],
    "ZonasOD2023": ["ZonasOD2023", "ZonasOD"],
    "Subprefeitura": ["Subprefeitura", "Subprefeituras", "subprefeitura"],
}

@st.cache_data(show_spinner=False)
def load_admin_layer(layer_name: str):
    if gpd is None:
        st.info("Geopandas não disponível.")
        return None
    path = _first_parquet_matching(ADM_DIR, ADMIN_NAME_MAP.get(layer_name, [layer_name]))
    if path is None:
        st.warning(f"Parquet não encontrado para '{layer_name}' em {ADM_DIR}.")
        return None
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        # fallback: pandas + reconstrução de geometria
        try:
            pdf = pd.read_parquet(path)
        except Exception as exc:
            st.warning(f"Falha ao ler {path.name}: {exc}")
            return None
        geom_col = next((c for c in pdf.columns if c.lower() in ("geometry", "geom", "wkb", "wkt")), None)
        if geom_col is None:
            st.warning(f"{path.name}: coluna de geometria não encontrada.")
            return None
        from shapely import wkb, wkt
        vals = pdf[geom_col]
        if vals.dropna().astype(str).str.startswith("POLY").any():
            geo = vals.dropna().apply(wkt.loads)
        else:
            geo = vals.dropna().apply(lambda b: wkb.loads(b, hex=isinstance(b, str)))
        gdf = gpd.GeoDataFrame(pdf.drop(columns=[geom_col]), geometry=geo)
    try:
        gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

DENSITY_CANDIDATES = ["Densidade", "Densidade2023", "Desindade", "Desindade2023", "HabitacaoPrecaria"]

@st.cache_data(show_spinner=False)
def load_density():
    # 1) dash_planbairros/densidade.parquet
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        try:
            return pd.read_parquet(direct)
        except Exception as exc:
            st.warning(f"Falha ao ler {direct}: {exc}")
            return None
    # 2) dentro da pasta densidade/
    p = _first_parquet_matching(DENS_DIR, DENSITY_CANDIDATES)
    if p:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            st.warning(f"Falha ao ler {p}: {exc}")
            return None
    # 3) também procurar na pasta de limites (pelo seu screenshot)
    p2 = _first_parquet_matching(ADM_DIR, DENSITY_CANDIDATES)
    if p2:
        try:
            return pd.read_parquet(p2)
        except Exception as exc:
            st.warning(f"Falha ao ler {p2}: {exc}")
            return None
    st.info("Dados de densidade não encontrados (pasta 'densidade' ou 'limites_administrativos').")
    return None
