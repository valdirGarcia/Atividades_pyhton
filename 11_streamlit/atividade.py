import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



st.title('Atividade Streamlit EBAC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

st.subheader('Raw data')
st.write(data)

st.subheader('Number of pickups by hour')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

st.subheader('Map of all pickups')

st.map(data)

st.subheader('Map of all pickups')
st.map(data)

hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

st.subheader('Raw data')
st.write(data)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    
    
st.title("Dashboard Interativo de Dados 📊")

# Introdução
st.markdown("""
Este aplicativo permite que você:
- Faça upload de arquivos CSV.
- Visualize e explore os dados.
- Crie gráficos interativos.
""")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])
if uploaded_file is not None:
    # Carregar dados
    data = pd.read_csv(uploaded_file)

    # Exibir uma amostra dos dados
    st.header("📋 Dados Carregados")
    st.write(data.head())

    # Mostrar informações básicas
    st.subheader("📊 Informações Básicas")
    st.write("Número de linhas:", data.shape[0])
    st.write("Número de colunas:", data.shape[1])

    # Permitir seleção de colunas numéricas para análise
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) > 0:
        st.subheader("📈 Análise Gráfica")

        # Escolher colunas para o gráfico
        x_col = st.selectbox("Escolha a coluna para o eixo X", numeric_columns)
        y_col = st.selectbox("Escolha a coluna para o eixo Y", numeric_columns)

        # Gráfico de dispersão
        st.write("### Gráfico de Dispersão")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)

        # Histograma
        st.write("### Histograma")
        selected_col = st.selectbox("Escolha uma coluna para o histograma", numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(data[selected_col], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Seu arquivo não contém colunas numéricas para análise.")

else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
    
    
    
# Configuração inicial
st.title("Análise Exploratória - Base Titanic 🚢")

# Carregar a base de dados
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

data = load_data()

# Mostrar o conjunto de dados
st.subheader("Visão Geral dos Dados")
st.write(data.head())
st.write("Shape dos dados:", data.shape)

# Opções de filtro
st.sidebar.header("Filtros")
classe = st.sidebar.multiselect(
    "Selecione as classes",
    options=data['class'].unique(),
    default=data['class'].unique()
)

sexo = st.sidebar.multiselect(
    "Selecione os gêneros",
    options=data['sex'].unique(),
    default=data['sex'].unique()
)

# Filtrar os dados
filtered_data = data[
    (data['class'].isin(classe)) &
    (data['sex'].isin(sexo))
]

st.subheader("Dados Filtrados")
st.write(filtered_data)

# Estatísticas descritivas
st.subheader("Estatísticas Descritivas")
st.write(filtered_data.describe())

# Gráficos interativos
st.subheader("Gráficos Interativos")

# Contagem por classe
st.markdown("**Distribuição por Classe**")
fig, ax = plt.subplots()
sns.countplot(data=filtered_data, x="class", hue="sex", ax=ax)
st.pyplot(fig)

# Distribuição da idade
st.markdown("**Distribuição da Idade**")
fig, ax = plt.subplots()
sns.histplot(filtered_data, x="age", kde=True, hue="sex", ax=ax)
st.pyplot(fig)

# Análise de sobrevivência
st.markdown("**Taxa de Sobrevivência por Classe**")
survival_rate = (
    filtered_data.groupby("class")["survived"]
    .mean()
    .reset_index()
    .rename(columns={"survived": "survival_rate"})
)
st.bar_chart(survival_rate.set_index("class"))

# Gráfico de dispersão (Idade vs Tarifa)
st.markdown("**Idade vs Tarifa (Scatter Plot)**")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x="age", y="fare", hue="sex", ax=ax)
st.pyplot(fig)

# Observações adicionais
st.sidebar.info("Use os filtros para explorar os dados por classe e gênero.")

    

