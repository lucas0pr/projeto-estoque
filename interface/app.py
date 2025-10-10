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
from scripts import database

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Estoque e Rentabilidade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título e Introdução ---
st.title("📊 Dashboard de Análise de Estoque e Rentabilidade")
st.markdown("""
Bem-vindo ao sistema de análise de dados e previsão de sucesso para produtos eletrônicos.
Esta ferramenta foi desenvolvida para fornecer insights acionáveis sobre vendas, rentabilidade e giro de estoque, 
utilizando técnicas de Machine Learning para apoiar a tomada de decisões.
""")

st.markdown("---")

# --- Métricas-Chave (KPIs) Globais ---
st.header("📈 Visão Geral do Negócio")

try:
    # Query para calcular os KPIs globais
    query_kpis = """
    SELECT
        SUM(preco * quantidade_vendida) as receita_total,
        SUM(custo * quantidade_vendida) as custo_total,
        SUM((preco - custo) * quantidade_vendida) as lucro_total,
        SUM(quantidade_vendida) as itens_vendidos,
        COUNT(DISTINCT item_id) as produtos_unicos
    FROM sales;
    """
    kpis_df = database.execute_query(query_kpis)
    
    # Exibe as métricas em colunas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Receita Total", f"${kpis_df['receita_total'].iloc[0]:,.2f}")
    col2.metric("Custo Total", f"${kpis_df['custo_total'].iloc[0]:,.2f}")
    col3.metric("Lucro Total", f"${kpis_df['lucro_total'].iloc[0]:,.2f}")
    col4.metric("Itens Vendidos", f"{kpis_df['itens_vendidos'].iloc[0]:,}")
    col5.metric("Produtos Únicos", f"{kpis_df['produtos_unicos'].iloc[0]:,}")

except Exception as e:
    st.error("Não foi possível carregar as métricas. O banco de dados está inicializado?")
    st.info("Vá para a página de **Upload de Dados** para carregar os dados e começar.")

st.markdown("---")

# --- Navegação Rápida ---
st.header("🚀 Navegação Rápida")

st.markdown("Escolha uma das seções abaixo para uma análise detalhada:")

# Usa colunas para criar botões lado a lado
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📤 Upload de Dados", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Upload.py")

with col2:
    if st.button("📈 Análise e Visualização", use_container_width=True):
        st.switch_page("pages/2_Analise.py")

with col3:
    if st.button("🤖 Machine Learning", use_container_width=True):
        st.switch_page("pages/3_Machine_Learning.py")


st.markdown("---")

# --- Status do Sistema ---
st.header("🔧 Status do Sistema")

try:
    count_df = database.execute_query("SELECT COUNT(*) as total_records FROM sales")
    total_records = count_df['total_records'].iloc[0]
    
    if total_records > 0:
        st.success("✅ Banco de dados conectado e populado.")
        st.metric("Registros na Tabela 'sales'", f"{total_records:,}")
        
        # Mostra a data da última transação
        last_date_df = database.execute_query("SELECT MAX(timestamp) as last_date FROM sales")
        last_date = last_date_df['last_date'].iloc[0]
        st.write(f"Última data de registro no dataset: {last_date}")
    else:
        st.warning("⚠️ O banco de dados está vazio. Nenhuma análise pode ser realizada.")
        st.info("Por favor, faça o upload de um arquivo CSV na página de **Upload de Dados**.")

except Exception as e:
    st.error("❌ Falha ao conectar ao banco de dados.")
    st.code(e) # Mostra o erro para debug

# --- Rodapé ---
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido para a disciplina de Tópicos Especiais em Software.")