# =============================================================================
# VALIDATION (unicidade + não-nulos + FK)
# =============================================================================
LAYER_SPECS = {
    "subpref": {"id": SUBPREF_ID, "parent": None,        "parent_layer": None,     "parent_id": None},
    "dist":    {"id": DIST_ID,    "parent": DIST_PARENT, "parent_layer": "subpref", "parent_id": SUBPREF_ID},
    "iso":     {"id": ISO_ID,     "parent": ISO_PARENT,  "parent_layer": "dist",    "parent_id": DIST_ID},
    "quadra":  {"id": QUADRA_ID,  "parent": QUADRA_PARENT,"parent_layer": "iso",    "parent_id": ISO_ID},
    "lote":    {"id": LOTE_ID,    "parent": LOTE_PARENT, "parent_layer": "quadra",  "parent_id": QUADRA_ID},
    "censo":   {"id": CENSO_ID,   "parent": CENSO_PARENT,"parent_layer": "iso",     "parent_id": ISO_ID},
}

# Limite para checagem de integridade referencial (evita estourar memória em layers enormes)
MAX_FK_UNIQUES = 200_000


def _ensure_col(gdf: "gpd.GeoDataFrame", col: str, layer_key: str) -> bool:
    if col not in gdf.columns:
        st.error(f"[{layer_key}] Coluna obrigatória ausente: '{col}'")
        return False
    return True


def _validate_unique_non_null(gdf: "gpd.GeoDataFrame", col: str, layer_key: str) -> bool:
    """Regra: ID único e sem nulos."""
    s = gdf[col]
    n_null = int(s.isna().sum())
    if n_null > 0:
        st.error(f"[{layer_key}] '{col}' tem {n_null:,} nulos. IDs devem ser não-nulos.")
        return False

    # duplicados
    dup_mask = s.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        # quantos IDs repetidos
        n_dup_ids = int(s[dup_mask].nunique())
        examples = s[dup_mask].astype(str).head(5).tolist()
        st.error(
            f"[{layer_key}] '{col}' NÃO é único: {n_dup_ids:,} IDs repetidos "
            f"({n_dup_rows:,} linhas). Ex: {examples}"
        )
        return False

    return True


def _validate_fk_non_null(gdf: "gpd.GeoDataFrame", fk_col: str, layer_key: str) -> bool:
    """Regra: FK do pai sem nulos (exceto topo)."""
    s = gdf[fk_col]
    n_null = int(s.isna().sum())
    if n_null > 0:
        st.error(f"[{layer_key}] FK '{fk_col}' tem {n_null:,} nulos. Camada filha precisa de pai.")
        return False
    return True


def _validate_fk_integrity(
    child: "gpd.GeoDataFrame",
    child_fk_col: str,
    parent: "gpd.GeoDataFrame",
    parent_id_col: str,
    child_layer: str,
    parent_layer: str,
) -> bool:
    """
    Opcional: garante que todo FK existe no pai.
    - faz check por valores únicos, com limite.
    """
    if parent is None or parent.empty:
        st.warning(f"[{child_layer}] Não foi possível validar integridade FK: camada pai '{parent_layer}' vazia.")
        return True

    child_uni = pd.unique(child[child_fk_col].dropna())
    parent_uni = pd.unique(parent[parent_id_col].dropna())

    # Evita explodir memória em layers muito grandes
    if len(child_uni) > MAX_FK_UNIQUES:
        st.warning(
            f"[{child_layer}] FK integrity skip: muitos valores únicos em '{child_fk_col}' "
            f"({len(child_uni):,} > {MAX_FK_UNIQUES:,})."
        )
        return True

    parent_set = set(parent_uni.tolist())
    missing = [v for v in child_uni.tolist() if v not in parent_set]
    if missing:
        st.error(
            f"[{child_layer}] FK inválida: '{child_fk_col}' contém valores que NÃO existem em "
            f"'{parent_layer}.{parent_id_col}'. Ex: {missing[:10]}"
        )
        return False

    return True


def validate_layer_contract(
    layer_key: str,
    gdf: "gpd.GeoDataFrame",
    parent_gdf: Optional["gpd.GeoDataFrame"] = None,
    strict_fk_integrity: bool = False,
) -> bool:
    """
    Aplica as regras:
    - ID único (*_id) e sem nulos
    - FK sem nulos (exceto topo)
    - (opcional) integridade referencial
    """
    spec = LAYER_SPECS.get(layer_key)
    if spec is None:
        return True

    id_col = spec["id"]
    parent_col = spec["parent"]
    parent_layer = spec["parent_layer"]
    parent_id_col = spec["parent_id"]

    if not _ensure_col(gdf, id_col, layer_key):
        return False
    if not _validate_unique_non_null(gdf, id_col, layer_key):
        return False

    if parent_col:
        if not _ensure_col(gdf, parent_col, layer_key):
            return False
        if not _validate_fk_non_null(gdf, parent_col, layer_key):
            return False

        if strict_fk_integrity and parent_gdf is not None and parent_layer and parent_id_col:
            if not _ensure_col(parent_gdf, parent_id_col, parent_layer):
                return False
            if not _validate_fk_integrity(gdf, parent_col, parent_gdf, parent_id_col, layer_key, parent_layer):
                return False

    return True
