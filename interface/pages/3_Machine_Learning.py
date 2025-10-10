import sys
import os

# --- CORREÇÃO ROBUSTA DO CAMINHO DE IMPORTAÇÃO ---
def find_project_root(marker_file='README.md'):
    current_path = os.path.dirname(os.path.abspath(__file__))
    while current_path != os.path.dirname(current_path):
        if os.path.exists(os.path.join(current_path, marker_file)):
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError(f"Não foi possível encontrar a raiz do projeto com o marcador '{marker_file}'")

try:
    project_root = find_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except FileNotFoundError:
    sys.exit(1)
# ------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from scripts import database

# Configuração da página
st.set_page_config(page_title="Machine Learning", layout="wide")

# Título e descrição
st.title("🤖 3. Machine Learning: Previsão de Sucesso de Produtos")
st.markdown("""
Configure e treine um modelo de Machine Learning para prever se um novo produto será um sucesso de vendas.
Um produto é considerado **bem-sucedido** se ele possui um rating médio alto e está no quartil superior de volume de vendas.
""")

# --- Verificação se o banco de dados tem dados ---
try:
    count_df = database.execute_query("SELECT COUNT(*) as total_records FROM sales")
    if count_df['total_records'].iloc[0] == 0:
        st.warning("O banco de dados está vazio. Por favor, vá para a página '📤 Upload de Dados' para carregar os dados.")
        st.stop()
except Exception as e:
    st.error("Não foi possível conectar ao banco de dados.")
    st.stop()

# --- Preparação dos Dados (Feature Engineering) ---
st.header("🔧 Configuração do Modelo")

@st.cache_data
def load_and_prepare_data():
    query = """
    SELECT item_id, category, brand, AVG(preco) as preco_medio, AVG(custo) as custo_medio, SUM(quantidade_vendida) as total_vendido, AVG(rating) as rating_medio
    FROM sales GROUP BY item_id, category, brand
    """
    df = database.execute_query(query)
    
    rating_threshold = 4.0
    sales_threshold = df['total_vendido'].quantile(0.75)
    
    df['is_successful'] = ((df['rating_medio'] > rating_threshold) & (df['total_vendido'] > sales_threshold)).astype(int)
    
    return df, rating_threshold, sales_threshold

df_model, rating_thr, sales_thr = load_and_prepare_data()

st.info(f"**Definição de 'Sucesso':** Rating Médio > {rating_thr} E Volume de Vendas > {sales_thr:.0f} (Quartil 75%)")

# --- Interface de Configuração na Sidebar ---
st.sidebar.header("Parâmetros do Modelo")

categorical_features = ['category', 'brand']
numeric_features = ['preco_medio', 'custo_medio']
all_features = categorical_features + numeric_features

selected_features = st.sidebar.multiselect("Selecione as Features:", options=all_features, default=all_features)

model_name = st.sidebar.selectbox("Escolha o Algoritmo:", ["Árvore de Decisão", "Random Forest", "KNN"])

st.sidebar.subheader("Ajuste de Hiperparâmetros")
params = {}
if model_name in ["Árvore de Decisão", "Random Forest"]:
    params['max_depth'] = st.sidebar.slider("Profundidade Máxima (max_depth):", 2, 20, 5)
    if model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Número de Árvores (n_estimators):", 50, 200, 100)
elif model_name == "KNN":
    params['n_neighbors'] = st.sidebar.slider("Número de Vizinhos (n_neighbors):", 3, 15, 5)

# --- Lógica de Treinamento ---
if st.button("🚀 Treinar Modelo", type="primary"):
    if not selected_features:
        st.error("Por favor, selecione pelo menos uma feature.")
        st.stop()

    with st.spinner("Treinando o modelo... Isso pode levar alguns instantes."):
        X = df_model[selected_features]
        y = df_model['is_successful']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        preprocessor = ColumnTransformer(
            transformers=[('num', 'passthrough', [f for f in selected_features if f in numeric_features]), ('cat', OneHotEncoder(handle_unknown='ignore'), [f for f in selected_features if f in categorical_features])]
        )

        if model_name == "Árvore de Decisão":
            model = DecisionTreeClassifier(**params, random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(**params, random_state=42)
        else:
            model = KNeighborsClassifier(**params)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        model_path = "modelos/modelo_treinado.pkl"
        os.makedirs("modelos", exist_ok=True)
        joblib.dump(pipeline, model_path)
        
        st.session_state['model_trained'] = True
        st.session_state['accuracy'] = accuracy
        st.session_state['report'] = report
        st.session_state['confusion_matrix'] = cm
        st.session_state['pipeline'] = pipeline
        st.session_state['selected_features'] = selected_features

    st.success("Modelo treinado e salvo com sucesso!")

# --- Exibição dos Resultados ---
if 'model_trained' in st.session_state and st.session_state['model_trained']:
    st.header("📊 Resultados do Treinamento")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Acurácia do Modelo", f"{st.session_state['accuracy']:.2%}")
    with col2:
        report_df = pd.DataFrame(st.session_state['report']).transpose()
        st.dataframe(report_df.round(2))

    st.subheader("Matriz de Confusão")
    cm = st.session_state['confusion_matrix']
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predito", y="Verdadeiro", color="Contagem"), x=['Não Sucesso', 'Sucesso'], y=['Não Sucesso', 'Sucesso'])
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # --- Seção de Predição ---
    st.header("🔮 Fazer uma Nova Predição")
    st.write("Insira os dados de um novo produto para prever seu sucesso.")

    pipeline = st.session_state['pipeline']
    features = st.session_state['selected_features']
    
    input_data = {}
    for feature in features:
        if feature in ['category', 'brand']:
            unique_vals = df_model[feature].unique()
            input_data[feature] = st.selectbox(f"{feature.capitalize()}:", unique_vals)
        else:
            input_data[feature] = st.number_input(f"{feature.capitalize()}:", value=df_model[feature].mean())

    if st.button("Prever Sucesso do Produto"):
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0]

        if prediction == 1:
            st.success(f"🎉 O produto foi classificado como **BEM-SUCEDIDO** com {prediction_proba[1]*100:.1f}% de confiança.")
        else:
            st.error(f"😞 O produto foi classificado como **NÃO BEM-SUCEDIDO** com {prediction_proba[0]*100:.1f}% de confiança.")
