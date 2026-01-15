
from pathlib import Path
import random

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="PlanBairros",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Paleta da marca
# ------------------------------------------------------------
PB_COLORS = {
    "amarelo": "#F4DD63",
    "verde": "#B1BF7C",
    "laranja": "#D58243",
    "telha": "#C65534",
    "teal": "#6FA097",
    "navy": "#14407D",
}

# ------------------------------------------------------------
# CSS de identidade visual
# ------------------------------------------------------------
st.markdown(
    f"""
    <style>
        :root {{
            --pb-amarelo: {PB_COLORS['amarelo']};
            --pb-verde:   {PB_COLORS['verde']};
            --pb-laranja: {PB_COLORS['laranja']};
            --pb-telha:   {PB_COLORS['telha']};
            --pb-teal:    {PB_COLORS['teal']};
            --pb-navy:    {PB_COLORS['navy']};
        }}
        .pb-header {{
            background: var(--pb-navy);
            color: #fff;
            border-radius: 16px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .pb-title {{
            font-size: 2rem;
            line-height: 1.2;
            font-weight: 700;
            letter-spacing: .5px;
        }}
        .pb-subtitle {{
            opacity: .9;
            margin-top: 2px;
            font-size: .95rem;
        }}
        .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
            background: transparent;
            border-bottom: 3px solid transparent;
            color: #2b2b2b;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 3px solid var(--pb-teal) !important;
            color: var(--pb-navy) !important;
        }}
        .pb-card {{
            background: #ffffff;
            border: 1px solid rgba(20,64,125,.10);
            box-shadow: 0 1px 2px rgba(0,0,0,.04);
            border-radius: 16px;
            padding: 16px;
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

# ------------------------------------------------------------
# Logo – busca melhor esforço
# ------------------------------------------------------------
LOGO_CANDIDATES = [
    "assets/logo_todos.jpg",
    "assets/logo_paleta.jpg",
    "logo_todos.jpg",
    "logo_paleta.jpg",
    "/mnt/data/logo_todos.jpg",
    "/mnt/data/logo_paleta.jpg",
]

logo_path = next((p for p in LOGO_CANDIDATES if Path(p).exists()), None)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
with st.container():
    col1, col2 = st.columns([1, 7])
    with col1:
        if logo_path:
            st.image(logo_path, use_column_width=True)
        else:
            st.write("")
    with col2:
        st.markdown(
            """
            <div class="pb-header">
                <div style="display:flex;flex-direction:column">
                    <div class="pb-title">PlanBairros</div>
                    <div class="pb-subtitle">Plataforma de visualização e planejamento em nível de bairro</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("
")

# ------------------------------------------------------------
# Abas
# ------------------------------------------------------------
aba1, aba2, aba3, aba4 = st.tabs(["Aba 1", "Aba 2", "Aba 3", "Aba 4"])
for i, aba in enumerate([aba1, aba2, aba3, aba4], start=1):
    with aba:
        st.markdown(
            f"**Conteúdo da Aba {i}** — este espaço será preenchido posteriormente com textos, imagens ou tabelas."
        )

st.write("
")

# ------------------------------------------------------------
# Layout principal sob as abas
# ------------------------------------------------------------
left, center, right = st.columns([1, 2, 2], gap="large")

with left:
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

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Dados de demonstração (placeholders)
# ------------------------------------------------------------
cat = ["A", "B", "C", "D", "E"]
bar_df = pd.DataFrame({"categoria": cat, "valor": [random.randint(10, 60) for _ in cat]})

map_df = pd.DataFrame(
    {
        "lat": [-23.5505, -23.62, -23.48, -23.57, -23.60],
        "lon": [-46.6333, -46.70, -46.55, -46.60, -46.64],
        "valor": [10, 25, 18, 32, 22],
        "nome": ["Ponto 1", "Ponto 2", "Ponto 3", "Ponto 4", "Ponto 5"],
    }
)

with center:
    st.subheader("Mapa (placeholder)")
    mfig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="valor",
        size="valor",
        hover_name="nome",
        zoom=9,
        height=540,
    )
    mfig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorscale=[[0, PB_COLORS["amarelo"]], [0.5, PB_COLORS["teal"]], [1, PB_COLORS["navy"]]],
    )
    st.plotly_chart(mfig, use_container_width=True)

with right:
    st.subheader("Gráfico (placeholder)")
    bfig = px.bar(bar_df, x="categoria", y="valor", text="valor", height=540)
    bfig.update_traces(marker_color=PB_COLORS["laranja"])  # cor da marca
    bfig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(bfig, use_container_width=True)

# ------------------------------------------------------------
# Footer (opcional)
# ------------------------------------------------------------
st.write("
")
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

# Dicas de tema (opcional):
# Em .streamlit/config.toml
# [theme]
# primaryColor = "#6FA097"
# backgroundColor = "#FFFFFF"
# secondaryBackgroundColor = "#F7F9F7"
# textColor = "#1F2937"
# font = "sans serif"
