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
import time

# Importa nossos módulos personalizados
from scripts import database
from scripts.enriquecer_dataset import enriquecer_e_salvar

# Configuração da página
st.set_page_config(page_title="Upload de Dados", layout="wide")

# Título e descrição
st.title("📤 1. Upload e Ingestão de Dados")
st.markdown("""
Carregue o arquivo CSV bruto (`amazon_electronics.csv`) para iniciar o processo.
O sistema irá:
1. ** Enriquecer** os dados com informações de preço, custo e quantidade vendida.
2. ** Salvar** o arquivo enriquecido na pasta `dados/processed/`.
3. ** Carregar** todos os dados no banco de dados SQLite, substituindo os dados anteriores.
""")

st.info("⚠️ **Atenção:** Certifique-se de que o arquivo CSV baixado do Kaggle seja o `amazon_electronics.csv`.")

# --- Área de Upload ---
uploaded_file = st.file_uploader(
    "Escolha o arquivo CSV bruto",
    type="csv",
    help="Faça o upload do arquivo 'amazon_electronics.csv'."
)

# --- Lógica de Processamento ---
if uploaded_file is not None:
    st.success("Arquivo carregado com sucesso!")
    st.write("Nome do arquivo:", uploaded_file.name)
    
    if st.button("🚀 Processar e Carregar no Banco", type="primary"):
        
        progress_bar = st.progress(0, text="Iniciando processo...")
        time.sleep(1)

        # Passo 1: Salvar o arquivo bruto na pasta 'dados/raw'
        try:
            progress_bar.progress(20, text="Salvando arquivo bruto...")
            raw_path = os.path.join("dados", "raw", uploaded_file.name)
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"✅ Arquivo salvo em `{raw_path}`.")
        except Exception as e:
            st.error(f"Erro ao salvar o arquivo bruto: {e}")
            st.stop()

        # Passo 2: Enriquecer os dados
        try:
            progress_bar.progress(40, text="Enriquecendo dados (isso pode levar um momento)...")
            processed_path = os.path.join("dados", "processed", "amazon_electronics_enriquecido.csv")
            df_enriquecido = enriquecer_e_salvar(raw_path, processed_path)
            if df_enriquecido is None:
                st.error("Ocorreu um erro durante o enriquecimento dos dados. Verifique o console.")
                st.stop()
            st.write("✅ Dados enriquecidos com sucesso!")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o enriquecimento dos dados: {e}")
            st.stop()

        # Passo 3: Carregar no banco de dados
        try:
            progress_bar.progress(70, text="Carregando dados no banco de dados SQLite...")
            database.create_table()
            database.insert_data_from_csv(processed_path)
            st.write("✅ Dados carregados no banco de dados com sucesso!")
        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar os dados no banco: {e}")
            st.stop()
        
        progress_bar.progress(100, text="Processo concluído!")
        st.balloons()
        st.success("🎉 **Processo finalizado com sucesso!** Os dados estão prontos para análise na página 'Análise e Visualização'.")

# --- Seção de Status do Banco de Dados ---
st.markdown("---")
st.header("📊 Status Atual do Banco de Dados")

try:
    count_df = database.execute_query("SELECT COUNT(*) as total_records FROM sales")
    total_records = count_df['total_records'].iloc[0]
    
    st.metric("Total de Registros na Tabela 'sales'", f"{total_records:,}")
    
    if total_records > 0:
        st.subheader("Amostra dos Dados no Banco (5 primeiros registros)")
        sample_df = database.execute_query("SELECT item_id, category, brand, preco, quantidade_vendida FROM sales LIMIT 5")
        st.dataframe(sample_df, use_container_width=True)

        with st.expander("Ver Resumo Estatístico dos Dados no Banco"):
            stats_df = database.execute_query("SELECT preco, custo, quantidade_vendida FROM sales")
            st.write(stats_df.describe())

except Exception as e:
    st.warning("O banco de dados ainda não foi inicializado. Faça o upload de um arquivo para começar.")