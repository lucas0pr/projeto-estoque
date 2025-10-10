import sqlite3
import pandas as pd

DB_NAME = "estoque.db"

def get_connection():
    """Cria e retorna uma conexão com o banco de dados SQLite."""
    conn = sqlite3.connect(DB_NAME)
    # Permite acessar as colunas pelo nome (como dicionário)
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    """Cria a tabela de vendas se ela não existir."""
    sql_create_table = """
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL,
        user_id INTEGER,
        rating REAL,
        timestamp TEXT,
        model_attr TEXT,
        category TEXT,
        brand TEXT,
        year INTEGER,
        user_attr TEXT,
        split TEXT,
        preco REAL,
        custo REAL,
        quantidade_vendida INTEGER
    );
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql_create_table)
    conn.commit()
    conn.close()

def insert_data_from_csv(csv_path):
    """Lê um CSV e insere os dados na tabela 'sales'."""
    conn = get_connection()
    # Lê o CSV em um DataFrame
    df = pd.read_csv(csv_path)
    # Insere o DataFrame no banco de dados. 'replace' apaga a tabela e recria.
    df.to_sql('sales', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Dados do arquivo {csv_path} inseridos com sucesso na tabela 'sales'.")

def execute_query(query):
    """Executa uma query SQL e retorna os resultados como uma lista de dicionários."""
    conn = get_connection()
    # Usa pandas para facilitar a leitura dos resultados
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Exemplo de uso (pode ser removido ou comentado depois)
if __name__ == '__main__':
    create_table()
    # Exemplo de como inserir dados (depois de gerar o arquivo enriquecido)
    # insert_data_from_csv("../dados/processed/amazon_electronics_enriquecido.csv")