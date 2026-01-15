from pathlib import Path
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
bfig.update_traces(marker_color=PB_COLORS["laranja"]) # cor da marca
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
