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
import plotly.express as px
from scripts import database

# Configuração da página
st.set_page_config(page_title="Análise e Visualização", layout="wide")

# Título e descrição
st.title("📈 2. Análise e Visualização de Dados")
st.markdown("Explore os dados de vendas através de gráficos interativos para obter insights sobre rentabilidade e giro de estoque.")

# --- Verificação se o banco de dados tem dados ---
try:
    count_df = database.execute_query("SELECT COUNT(*) as total_records FROM sales")
    if count_df['total_records'].iloc[0] == 0:
        st.warning("O banco de dados está vazio. Por favor, vá para a página '📤 Upload de Dados' para carregar os dados.")
        st.stop()
except Exception as e:
    st.error("Não foi possível conectar ao banco de dados. Verifique se a tabela 'sales' existe.")
    st.stop()

# --- Filtros Interativos ---
st.sidebar.header("Filtros")

categories_df = database.execute_query("SELECT DISTINCT category FROM sales ORDER BY category")
categories_list = categories_df['category'].tolist()
categories_list.insert(0, "Todas")

selected_category = st.sidebar.selectbox(
    "Selecione uma Categoria:",
    options=categories_list
)

# --- Lógica de Filtro na Query ---
where_clause = ""
if selected_category != "Todas":
    where_clause = f"WHERE category = '{selected_category}'"

# --- 1. Métricas Principais (KPIs) ---
st.header("📊 Métricas Principais")

query_kpis = f"""
SELECT
    SUM(preco * quantidade_vendida) as receita_total,
    SUM(custo * quantidade_vendida) as custo_total,
    SUM((preco - custo) * quantidade_vendida) as lucro_total,
    SUM(quantidade_vendida) as itens_vendidos
FROM sales
{where_clause};
"""
kpis_df = database.execute_query(query_kpis)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Receita Total", f"${kpis_df['receita_total'].iloc[0]:,.2f}")
col2.metric("Custo Total", f"${kpis_df['custo_total'].iloc[0]:,.2f}")
col3.metric("Lucro Total", f"${kpis_df['lucro_total'].iloc[0]:,.2f}", delta=f"{(kpis_df['lucro_total'].iloc[0] / kpis_df['receita_total'].iloc[0] * 100):.2f}%")
col4.metric("Itens Vendidos", f"{kpis_df['itens_vendidos'].iloc[0]:,}")

st.markdown("---")

# --- 2. Análise de Rentabilidade ---
st.header("💰 Análise de Rentabilidade")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Receita Total por Categoria")
    query_receita_cat = f"""
    SELECT category, SUM(preco * quantidade_vendida) as receita_total
    FROM sales {where_clause}
    GROUP BY category ORDER BY receita_total DESC LIMIT 10;
    """
    receita_cat_df = database.execute_query(query_receita_cat)
    fig_receita_cat = px.bar(receita_cat_df, x='receita_total', y='category', orientation='h', title="Top 10 Categorias por Receita", labels={'receita_total': 'Receita Total ($)', 'category': 'Categoria'})
    fig_receita_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_receita_cat, use_container_width=True)

with col2:
    st.subheader("Lucro Total por Marca (Top 10)")
    query_lucro_brand = f"""
    SELECT brand, SUM((preco - custo) * quantidade_vendida) as lucro_total
    FROM sales {where_clause}
    GROUP BY brand ORDER BY lucro_total DESC LIMIT 10;
    """
    lucro_brand_df = database.execute_query(query_lucro_brand)
    fig_lucro_brand = px.bar(lucro_brand_df, x='lucro_total', y='brand', orientation='h', title="Top 10 Marcas por Lucro", labels={'lucro_total': 'Lucro Total ($)', 'brand': 'Marca'})
    fig_lucro_brand.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_lucro_brand, use_container_width=True)

st.markdown("---")

# --- 3. Análise de Giro e Volume ---
st.header("📦 Análise de Giro e Volume")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Volume de Vendas Mensal")
    query_vendas_tempo = f"""
    SELECT strftime('%Y-%m', timestamp) as mes, SUM(quantidade_vendida) as total_vendido
    FROM sales {where_clause}
    GROUP BY mes ORDER BY mes;
    """
    vendas_tempo_df = database.execute_query(query_vendas_tempo)
    fig_vendas_tempo = px.line(vendas_tempo_df, x='mes', y='total_vendido', title="Tendência de Vendas ao Longo do Tempo", labels={'mes': 'Mês', 'total_vendido': 'Quantidade Vendida'})
    st.plotly_chart(fig_vendas_tempo, use_container_width=True)

with col2:
    st.subheader("Relação: Preço vs. Quantidade Vendida")
    query_preco_qtd = f"""
    SELECT item_id, brand, AVG(preco) as preco_medio, SUM(quantidade_vendida) as total_vendido
    FROM sales {where_clause}
    GROUP BY item_id, brand ORDER BY total_vendido DESC;
    """
    preco_qtd_df = database.execute_query(query_preco_qtd)
    fig_preco_qtd = px.scatter(preco_qtd_df, x='preco_medio', y='total_vendido', color='brand', title="Preço Médio vs. Volume de Vendas por Produto", labels={'preco_medio': 'Preço Médio ($)', 'total_vendido': 'Total Vendido'}, hover_data=['brand'])
    st.plotly_chart(fig_preco_qtd, use_container_width=True)

# --- 4. Tabela de Produtos Mais Rentáveis ---
st.header("🏆 Tabela de Produtos Mais Rentáveis")
query_top_produtos = f"""
SELECT item_id, category, brand, SUM(preco * quantidade_vendida) as receita_total, SUM((preco - custo) * quantidade_vendida) as lucro_total, SUM(quantidade_vendida) as unidades_vendidas
FROM sales {where_clause}
GROUP BY item_id, category, brand ORDER BY lucro_total DESC LIMIT 20;
"""
top_produtos_df = database.execute_query(query_top_produtos)
st.dataframe(top_produtos_df, use_container_width=True)