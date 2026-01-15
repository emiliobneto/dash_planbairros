# -*- coding: utf-8 -*-
"""
PlanBairros — Front‑end Streamlit (refatorado e organizado)

• Cabeçalho com logo + nome do projeto
• 4 abas (placeholders) para conteúdos futuros
• Abaixo das abas: painel esquerdo com filtros/controles
• Centro/Direita: visualizações (mapa e gráfico de barras, ambos placeholders)
• Paleta de cores e identidade via CSS injetado
• Estrutura em funções + cache p/ manter o app estável e legível
• Fallback para versões antigas do Streamlit (sem argumento gap em st.columns)
• Opção de upload de CSV para testar os componentes (lat, lon, valor, nome)

Como usar:
1) Salve como app.py (ou outro nome).
2) Rode com: streamlit run app.py
3) Opcional: crie uma pasta ./assets/ com um arquivo de logo (logo_todos.jpg).
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# Configuração global da página
# ============================================================
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Paleta da marca
# ============================================================
PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",
    "teal": "#6FA097",
    "navy": "#14407D",
}

# ============================================================
# Helpers de estilo e assets
# ============================================================

def inject_css() -> None:
    """Injeta CSS de identidade visual e aplica **Roboto** como fonte global."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            :root {{
                --pb-amarelo: {PB_COLORS['amarelo']};
                --pb-verde:   {PB_COLORS['verde']};
                --pb-laranja: {PB_COLORS['laranja']};
                --pb-telha:   {PB_COLORS['telha']};
                --pb-teal:    {PB_COLORS['teal']};
                --pb-navy:    {PB_COLORS['navy']};
            }}
            /* Fonte global */
            html, body, .stApp {{
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-header {{
                background: var(--pb-navy);
                color: #fff;
                border-radius: 18px;
                padding: 20px 24px;
                min-height: 110px;
                display: flex;
                align-items: center;
                gap: 16px;
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            
            .pb-title {{
                font-size: 2.4rem;
                line-height: 1.15;
                font-weight: 700;
                letter-spacing: .5px;
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-subtitle {{
                opacity: .95;
                margin-top: 4px;
                font-size: 1.1rem;
                font-weight: 400;
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .stTabs [data-baseweb=\"tab-list\"] button[role=\"tab\"] {{
                background: transparent;
                border-bottom: 3px solid transparent;
                color: #2b2b2b;
                font-weight: 600;
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .stTabs [data-baseweb=\"tab-list\"] button[aria-selected=\"true\"] {{
                border-bottom: 3px solid var(--pb-teal) !important;
                color: var(--pb-navy) !important;
            }}
            .pb-card {{
                background: #ffffff;
                border: 1px solid rgba(20,64,125,.10);
                box-shadow: 0 1px 2px rgba(0,0,0,.04);
                border-radius: 16px;
                padding: 16px;
                font-family: 'Roboto', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
            }}
            .pb-card h4 {{
                margin: 0 0 8px 0;
                font-size: 1.05rem;
                color: var(--pb-navy);
            }}
            footer {{visibility: hidden;}}
            #MainMenu {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_path() -> Optional[str]:
    """Retorna o primeiro caminho de logo existente, se houver."""
    candidates = [
        "assets/logo_todos.jpg",
        "assets/logo_paleta.jpg",
        "logo_todos.jpg",
        "logo_paleta.jpg",
        "/mnt/data/logo_todos.jpg",
        "/mnt/data/logo_paleta.jpg",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


# ============================================================
# Componentes de UI
# ============================================================

def build_header(logo_path: Optional[str]) -> None:
    """Renderiza o cabeçalho com logo e título."""
    with st.container():
        col1, col2 = st.columns([1, 7])
        with col1:
            if logo_path:
                st.image(logo_path, width=140)  # logo levemente menor para equilibrar o header
            else:
                st.write("")  # espaçamento
        with col2:
            st.markdown(
                """
                <div class="pb-header">
                    <div style="display:flex;flex-direction:column">
                        <div class="pb-title">PlanBairros</div>
                        <div class="pb-subtitle">Plataforma de visualização e planejamento em escala de bairro</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_tabs() -> None:
    """Abas placeholder (conteúdo ainda a definir)."""
    aba1, aba2, aba3, aba4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
    for i, aba in enumerate([aba1, aba2, aba3, aba4], start=1):
        with aba:
            st.markdown(
                f"**Conteúdo da Aba {i}** — este espaço será preenchido posteriormente com textos, imagens ou tabelas."
            )
    st.write("")


def build_controls() -> Tuple[str, str, str, str, Optional[pd.DataFrame]]:
    """Painel de filtros/controles à esquerda.

    Retorna as escolhas do usuário e um DataFrame opcional enviado por upload.
    """
    st.markdown("<div class='pb-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Configurações</h4>", unsafe_allow_html=True)

    variavel = st.selectbox(
        "Variáveis",
        ["População", "Habitação", "Mobilidade", "Uso do Solo", "Renda"],
        index=0,
        help="Selecione o tema principal que deseja analisar.",
        key="pb_variavel",
    )

    metrica = st.selectbox(
        "Métricas",
        ["Quantidade", "Percentual", "Densidade", "Índice (z-score)"],
        index=0,
        help="Forma de cálculo/escala a ser aplicada nas visualizações.",
        key="pb_metrica",
    )

    detalhe = st.selectbox(
        "Detalhes",
        ["Bairro", "Distrito", "Subprefeitura", "Área de ponderação"],
        index=0,
        help="Nível de desagregação espacial.",
        key="pb_detalhe",
    )

    info = st.selectbox(
        "Informações",
        ["Notas metodológicas", "Glossário", "Fontes", "Contato"],
        index=0,
        help="Selecione o tipo de informação auxiliar.",
        key="pb_info",
    )

    st.divider()
    uploaded = st.file_uploader(
        "Opcional: envie um CSV com colunas lat, lon, valor, nome",
        type=["csv"],
        help="O arquivo será usado para alimentar o mapa e o gráfico. Caso não envie, dados de demonstração serão utilizados.",
    )

    df_upload: Optional[pd.DataFrame] = None
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Não foi possível ler o CSV: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

    return variavel, metrica, detalhe, info, df_upload


# ============================================================
# Dados (placeholders) e visualizações
# ============================================================

@st.cache_data(show_spinner=False)
def generate_demo_data(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gera dados de demonstração para o gráfico e o mapa."""
    random.seed(seed)
    cat = ["A", "B", "C", "D", "E"]
    bar_df = pd.DataFrame(
        {"categoria": cat, "valor": [random.randint(10, 60) for _ in cat]}
    )

    map_df = pd.DataFrame(
        {
            "lat": [-23.5505, -23.62, -23.48, -23.57, -23.60],
            "lon": [-46.6333, -46.70, -46.55, -46.60, -46.64],
            "valor": [10, 25, 18, 32, 22],
            "nome": ["Ponto 1", "Ponto 2", "Ponto 3", "Ponto 4", "Ponto 5"],
        }
    )
    return bar_df, map_df


def plot_map(df: pd.DataFrame) -> None:
    """Renderiza um mapa de pontos com escala contínua de cores.

    Espera colunas: lat, lon, valor, nome
    """
    st.subheader("Mapa")
    mfig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="valor",
        size="valor",
        hover_name="nome",
        zoom=9,
        height=540,
        color_continuous_scale=[PB_COLORS["amarelo"], PB_COLORS["teal"], PB_COLORS["navy"]],
    )
    mfig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    mfig.update_layout(font=dict(family="Roboto, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"))
    st.plotly_chart(mfig, use_container_width=True)


def plot_bar(df: pd.DataFrame) -> None:
    """Renderiza um gráfico de barras simples.

    Espera colunas: categoria, valor
    """
    st.subheader("Gráfico")
    bfig = px.bar(df, x="categoria", y="valor", text="valor", height=540)
    bfig.update_traces(marker_color=PB_COLORS["laranja"])  # cor da marca
    bfig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    bfig.update_layout(font=dict(family="Roboto, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"))
    st.plotly_chart(bfig, use_container_width=True)


# ============================================================
# Layout principal
# ============================================================

def main() -> None:
    inject_css()
    logo_path = get_logo_path()

    # Header
    build_header(logo_path)
    st.write("")

    # Abas de topo
    build_tabs()

    # Grade abaixo das abas (com fallback para versões antigas)
    try:
        left, center, right = st.columns([1, 2, 2], gap="large")
    except TypeError:  # versões antigas sem suporte a gap
        left, center, right = st.columns([1, 2, 2])

    # Painel de filtros na esquerda
    with left:
        variavel, metrica, detalhe, info, df_upload = build_controls()

    # Dados base (upload ou demo) — mantemos coerência entre mapa e barras
    bar_df_demo, map_df_demo = generate_demo_data()

    # Se o usuário enviar CSV com as 4 colunas esperadas, sincronizamos gráfico e mapa
    if df_upload is not None and set(["lat", "lon", "valor", "nome"]).issubset(df_upload.columns):
        map_df = df_upload.copy()
        # Para o gráfico, agregamos por nome como exemplo simples
        bar_df = (
            df_upload.rename(columns={"nome": "categoria"})[["categoria", "valor"]]
            .groupby("categoria", as_index=False)
            .sum()
        )
    else:
        map_df, bar_df = map_df_demo, bar_df_demo

    # Área central e direita: visualizações
    with center:
        plot_map(map_df)

    with right:
        plot_bar(bar_df)

    # Rodapé com badges de status
    st.write("")
    with st.container():
        col_a, col_b, col_c = st.columns([1, 1, 1])
        col_a.markdown(
            f"<div style='background:{PB_COLORS['verde']};border-radius:10px;padding:10px;text-align:center;color:#17323f'>Identidade visual pronta</div>",
            unsafe_allow_html=True,
        )
        col_b.markdown(
            f"<div style='background:{PB_COLORS['amarelo']};border-radius:10px;padding:10px;text-align:center;color:#17323f'>Layout responsivo</div>",
            unsafe_allow_html=True,
        )
        col_c.markdown(
            f"<div style='background:{PB_COLORS['teal']};border-radius:10px;padding:10px;text-align:center;color:white'>Pronto para acoplar dados</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()







