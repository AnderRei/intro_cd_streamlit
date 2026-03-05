import streamlit as st
import pandas as pd

#RQUIVO = "/workspaces/intro_cd_streamlit/StudentPerformanceFactors.csv"
ARQUIVO = r"C:\Users\ander\OneDrive\Documents\GitHub\intro_cd_streamlit\StudentPerformanceFactors.csv"

# 1) Configurações iniciais da página
st.set_page_config(page_title="Painel - Desempenho dos Estudantes", layout="wide")
st.title("Painel simples: Student Performance Factors")

# 2) Carregamento do CSV (com cache para não recarregar toda hora)
@st.cache_data
def carregar_dados(caminho: str) -> pd.DataFrame:
    return pd.read_csv(caminho)

df = carregar_dados(ARQUIVO)

# 3) Visão geral do DataFrame
st.subheader("Visão geral dos dados")
st.write("Amostra (primeiras linhas):")
st.dataframe(df.head(20), width='stretch')


# 4) Identificação automática de colunas numéricas e categóricas
colunas_numericas = df.select_dtypes(include="number").columns.tolist()
colunas_categoricas = []

for c in df.columns:
    if c not in colunas_numericas:
        colunas_categoricas.append(c)

# 5) Filtros simples no menu lateral (sidebar)
st.sidebar.header("Filtros (opcional)")

df_filtrado = df.copy()

# 5.1) Filtro por UMA coluna categórica (se existir)
if len(colunas_categoricas) > 0:
    col_cat = st.sidebar.selectbox("Filtrar por coluna categórica", colunas_categoricas)
    valores = ["(Todos)"] + sorted(df[col_cat].dropna().astype(str).unique().tolist())
    valor_escolhido = st.sidebar.selectbox("Valor", valores)

    if valor_escolhido != "(Todos)":
        df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == valor_escolhido]

# 5.2) Filtro por intervalo em UMA coluna numérica (se existir)
if len(colunas_numericas) > 0:
    col_num = st.sidebar.selectbox("Filtrar por coluna numérica", colunas_numericas)
    minimo = float(df[col_num].min())
    maximo = float(df[col_num].max())

    faixa = st.sidebar.slider(
        "Intervalo",
        min_value=minimo,
        max_value=maximo,
        value=(minimo, maximo),
    )

    df_filtrado = df_filtrado[(df_filtrado[col_num] >= faixa[0]) & (df_filtrado[col_num] <= faixa[1])]

# 6) Métricas rápidas (nativas do Streamlit)
st.subheader("Métricas rápidas")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas (total)", f"{len(df):,}")
c2.metric("Linhas (filtrado)", f"{len(df_filtrado):,}")
c3.metric("Colunas", f"{df.shape[1]}")

# 7) Visualização nativa: st.bar_chart e st.line_chart
st.subheader("Gráficos nativos do Streamlit")

# 7.1) Bar chart: média de uma coluna numérica por uma coluna categórica
st.write("**Gráfico de barras:** média de uma métrica numérica por categoria")

if len(colunas_numericas) > 0 and len(colunas_categoricas) > 0:
    cat_bar = st.selectbox("Categoria (eixo X)", colunas_categoricas, key="cat_bar")
    num_bar = st.selectbox("Métrica numérica (média)", colunas_numericas, key="num_bar")

    # Agrupa por categoria e calcula a média
    tabela_bar = (
        df_filtrado.groupby(cat_bar)[num_bar]
        .mean()
        .sort_values(ascending=False)
        .to_frame()
    )

    st.bar_chart(tabela_bar)  # nativo
else:
    st.info("Não há colunas suficientes (categóricas e numéricas) para gerar o gráfico de barras.")

# 7.2) Line chart: tendência (ordenada) de uma coluna numérica
st.write("**Gráfico de linha:** série ordenada de uma coluna numérica (útil para ver tendência/variação)")

if len(colunas_numericas) > 0:
    num_line = st.selectbox("Coluna numérica para linha", colunas_numericas, key="num_line")

    # Ordena e cria uma série simples para plotar
    serie = df_filtrado[[num_line]].dropna().sort_values(by=num_line).reset_index(drop=True)

    st.line_chart(serie)  # nativo
else:
    st.info("Não há colunas numéricas para gerar o gráfico de linha.")

# 8) Histograma simples com bins (usando value_counts, exibido com st.bar_chart)
st.subheader("Distribuição (histograma simplificado)")

if len(colunas_numericas) > 0:
    num_hist = st.selectbox("Coluna numérica para distribuição", colunas_numericas, key="num_hist")
    bins = st.slider("Quantidade de faixas (bins)", min_value=5, max_value=50, value=20)

    # Cria faixas (bins)
    faixas = pd.cut(df_filtrado[num_hist].dropna(), bins=bins)

    # Conta quantos valores caíram em cada faixa
    contagem = faixas.value_counts().sort_index()

    # Converte as faixas (Interval) para texto, para o Streamlit/Altair aceitar
    contagem.index = contagem.index.astype(str)

    # Transforma em DataFrame (st.bar_chart funciona melhor assim)
    contagem_df = contagem.to_frame(name="contagem")

    st.bar_chart(contagem_df)
else:
    st.info("Não há colunas numéricas para mostrar distribuição.")


# 9) Correlação

st.subheader("Correlação (numéricas)")

if len(colunas_numericas) >= 2:
    # Calcula a matriz de correlação (Pearson por padrão)
    matriz_corr = df_filtrado[colunas_numericas].corr(numeric_only=True)

    st.write("Matriz de correlação (tabela):")
    st.dataframe(matriz_corr, width="stretch")

    st.write("Heatmap simples (tabela colorida):")
    st.dataframe(
        matriz_corr.style.background_gradient(),  # colore valores mais altos/baixos
        width="stretch"
    )
else:
    st.info("São necessárias pelo menos 2 colunas numéricas para calcular correlação.")



# 9) K-Means

st.subheader("🧠 K-means (clusterização)")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if len(colunas_numericas) >= 2:
    # Usuário escolhe 2 variáveis para o gráfico e para clusterizar
    x_col = st.selectbox("Eixo X (numérica)", colunas_numericas, index=0, key="k_x")
    y_col = st.selectbox("Eixo Y (numérica)", colunas_numericas, index=1, key="k_y")

    k = st.slider("Número de clusters (k)", min_value=2, max_value=10, value=3)

    # Seleciona as duas colunas e remove linhas com NA nelas
    dados = df_filtrado[[x_col, y_col]].dropna().copy()

    if len(dados) >= k:
        # Padroniza (muito recomendado para K-means)
        scaler = StandardScaler()
        X = scaler.fit_transform(dados[[x_col, y_col]])

        # Treina o K-means
        modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
        dados["Cluster"] = modelo.fit_predict(X).astype(str)  # str para virar categoria no gráfico

        st.write("Amostra com cluster atribuído:")
        st.dataframe(dados.head(20), width="stretch")

        # Scatter nativo do Streamlit
        st.write("Dispersão (cada cor = um cluster):")
        st.scatter_chart(
            dados,
            x=x_col,
            y=y_col,
            color="Cluster"  # colore por cluster
        )
    else:
        st.warning("Após os filtros, há poucas linhas para rodar o K-means com esse k.")
else:
    st.info("São necessárias pelo menos 2 colunas numéricas para rodar K-means.")