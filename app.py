# densidade: aceita possíveis nomes/typos
DENSITY_CANDIDATES = ["Densidade", "Densidade2023", "Desindade", "Desindade2023", "HabitacaoPrecaria"]

@st.cache_data(show_spinner=False)
def load_density() -> Optional[pd.DataFrame]:
    # 1) arquivo direto na raiz: dash_planbairros/densidade.parquet
    direct = DENS_DIR.with_suffix(".parquet")
    if direct.exists():
        try:
            return pd.read_parquet(direct)
        except Exception as exc:
            st.warning(f"Falha ao ler {direct}: {exc}")
            return None

    # 2) procurar na pasta densidade/
    p = _first_parquet_matching(DENS_DIR, DENSITY_CANDIDATES)
    if p is not None:
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            st.warning(f"Falha ao ler {p}: {exc}")
            return None

    # 3) ***novo***: procurar também na pasta de limites_administrativos/
    p2 = _first_parquet_matching(ADM_DIR, DENSITY_CANDIDATES)
    if p2 is not None:
        try:
            return pd.read_parquet(p2)
        except Exception as exc:
            st.warning(f"Falha ao ler {p2}: {exc}")
            return None

    st.info("Dados de densidade não encontrados (procurei em 'dash_planbairros/densidade' e '.../limites_administrativos').")
    return None
