# PlanBairros — Streamlit front‑end

Interface web para visualização e planejamento em nível de bairro.
Inclui identidade visual (cores/Roboto), abas para conteúdos, filtros laterais,
mapa interativo com **seleção espacial** (desenho de polígono/retângulo) e
resumo automático de estatísticas das quadras selecionadas.

---

## ✨ Principais recursos
- **Header** com logo e identidade visual.
- **Abas** para organizar conteúdos (texto, tabelas, etc.).
- **Painel de filtros** (variável, métrica, detalhe, info).
- **Mapa Plotly** (pontos) quando não há camada vetorial.
- **Mapa Folium + Draw** quando há GeoJSON de quadras:
  - Desenhe um polígono/retângulo no mapa.
  - A aplicação seleciona as quadras que **intersectam** a geometria desenhada.
  - Exibe um **resumo estatístico** das colunas numéricas (soma, média, mediana, min, máx).
- **Gráfico de barras** (placeholder) sincronizado com os dados em uso.

---

## 🗂 Estrutura recomendada do repositório
```
your-repo/
├─ app.py
├─ requirements.txt                 # ou requirements_planbairros.txt
├─ runtime.txt                      # opcional (ex.: python-3.11.8)
├─ .streamlit/
│  └─ config.toml                   # tema do app (opcional)
└─ assets/
   └─ logo_todos.jpg                # logo opcional
```

### `.streamlit/config.toml` (opcional)
```toml
[theme]
primaryColor = "#6FA097"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F7F9F7"
textColor = "#1F2937"
font = "sans serif"  # Roboto é aplicado via CSS no app
```

> Observação: a fonte **Roboto** é importada no CSS do próprio `app.py`. 
Se quiser fontes locais (sem Google Fonts), adicione os arquivos `.woff2` em `assets/fonts/`
e troque o `@import` por `@font-face`.

---

## 📦 Dependências
Use **um** destes arquivos (renomeie para `requirements.txt` no deploy):

**Versão mínima (sem seleção espacial):**
```
streamlit>=1.30,<2.0
pandas>=2.1,<3.0
plotly>=5.18,<6.0
```

**Versão com seleção espacial (Folium/GeoPandas):**
```
streamlit>=1.30,<2.0
pandas>=2.1,<3.0
plotly>=5.18,<6.0
streamlit-folium>=0.18,<0.20
folium>=0.15,<0.17
shapely>=2.0,<3.0
geopandas>=0.14,<1.0
```

> Opcional: crie `runtime.txt` com a versão de Python, por exemplo:
```
python-3.11.8
```

---

## ▶️ Executando localmente
1. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Rode o app:
   ```bash
   streamlit run app.py
   ```
3. Acesse o endereço exibido no terminal (por padrão, `http://localhost:8501`).

---

## ☁️ Deploy no Streamlit Community Cloud (GitHub)
1. Suba `app.py`, `requirements.txt` (e opcionalmente `.streamlit/config.toml`, `runtime.txt`) no GitHub.
2. Em **Streamlit Cloud → New app**, selecione o repositório e informe:
   - *Main file path*: `app.py`
3. Deploy. O serviço instalará as dependências e iniciará o app.

---

## 📄 Formatos & CRS
- **CSV de pontos** (para o mapa Plotly): colunas obrigatórias `lat`, `lon`, `valor`, `nome`.
- **GeoJSON** de quadras/polígonos (para o mapa com seleção):
  - **WGS84 / EPSG:4326** (latitude/longitude). Se vier sem `crs`, o app assume 4326.
  - Se vier em outro SRC, o app converte para 4326 antes de exibir.
- **Cálculo de áreas/distâncias**: se necessário, reprojete para um SRC métrico (ex.: SIRGAS/UTM 23S — EPSG:31983), calcule e depois volte para 4326 para visualizar.

---

## 🧭 Como usar no app
1. (Opcional) Faça upload de um **CSV** para preencher o mapa e o gráfico com os seus pontos.
2. (Opcional) Faça upload de um **GeoJSON (4326)** de quadras para habilitar a **seleção espacial**.
3. No mapa Folium, clique no ícone do **lápis** (Draw) e desenhe um **retângulo** ou **polígono**.
4. Veja as **estatísticas resumidas** (Soma, Média, Mediana, Mín, Máx) das colunas numéricas das quadras selecionadas.

---

## ⚙️ Customização rápida
- **Cores da marca**: altere o dicionário `PB_COLORS` em `app.py`.
- **Tipografia**: Roboto aplicada via CSS (`inject_css()`); troque por outra fonte se quiser.
- **Largura do logo**: em `build_header()`, ajuste `st.image(..., width=140)`.
- **Lógica de seleção**: atualmente usa `intersects`. Troque por `within`/`contains` se fizer mais sentido.

---

## 🩺 Solução de problemas
- **Erro em `<style>` dentro de f-strings**: chaves de CSS precisam ser **escapadas**: use `{{` e `}}` no CSS. As interpolações Python (`{PB_COLORS[...]}`) ficam com **chaves simples**.

- **GeoJSON sem CRS**: o app assume EPSG:4326. Se as geometrias aparecerem deslocadas, verifique/projete a camada na origem.

- **Performance com muitas geometrias**: simplifique polígonos, filtre por área/zoom, ou avalie pré-processar no backend.

- **Porta/host no Cloud**: não defina porta no código; o Streamlit Cloud gerencia automaticamente.


---

## Licença
Defina a licença do projeto (ex.: MIT).

## Autor
Coloque créditos/contato do time PlanBairros.
