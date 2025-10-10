import pandas as pd
import numpy as np
import random
import os

# --- 1. CONFIGURAÇÃO DAS PREMISSAS (Mantido igual) ---
PRECO_POR_CATEGORIA = {
    'Portable Audio & Video': (30, 400), 'Camera & Photo': (100, 1500), 'Accessories & Supplies': (5, 80),
    'Headphones': (20, 350), 'TV & Home Theater': (200, 2500), 'Computers': (250, 3000),
    'Car Electronics': (50, 800), 'Office Electronics': (40, 600), 'Home Audio & Theater': (80, 1200),
    'Cell Phones & Accessories': (15, 900), 'Musical Instruments': (50, 1500), 'Electronics': (40, 1000),
    'Wearable Technology': (80, 500), 'Security & Surveillance': (60, 400), 'Binary Clock': (15, 50)
}
MARGEM_LUCRO_POR_CATEGORIA = {
    'Portable Audio & Video': (0.4, 0.6), 'Camera & Photo': (0.3, 0.5), 'Accessories & Supplies': (0.5, 0.7),
    'Headphones': (0.4, 0.6), 'TV & Home Theater': (0.2, 0.4), 'Computers': (0.15, 0.35),
    'Car Electronics': (0.35, 0.55), 'Office Electronics': (0.4, 0.6), 'Home Audio & Theater': (0.35, 0.55),
    'Cell Phones & Accessories': (0.4, 0.6), 'Musical Instruments': (0.3, 0.5), 'Electronics': (0.35, 0.55),
    'Wearable Technology': (0.3, 0.5), 'Security & Surveillance': (0.4, 0.6), 'Binary Clock': (0.5, 0.7)
}

# --- 2. FUNÇÕES DE GERAÇÃO (Mantido igual) ---
def gerar_preco(row):
    categoria = row['category']
    preco_min, preco_max = PRECO_POR_CATEGORIA.get(categoria, (20, 200))
    fator_rating = 1 + (row['rating'] - 3) * 0.05
    preco_base = random.uniform(preco_min, preco_max)
    return round(preco_base * fator_rating, 2)

def gerar_custo(row):
    preco = row['preco']
    categoria = row['category']
    margem_min, margem_max = MARGEM_LUCRO_POR_CATEGORIA.get(categoria, (0.3, 0.5))
    margem_lucro = random.uniform(margem_min, margem_max)
    return round(preco * margem_lucro, 2)

def gerar_quantidade_vendida(row, popularidade_produto):
    preco = row['preco']
    popularidade = popularidade_produto[row['item_id']]
    if preco < 50: multiplicador_base = 5
    elif preco < 200: multiplicador_base = 3
    elif preco < 800: multiplicador_base = 1.5
    else: multiplicador_base = 1.0
    media_quantidade = popularidade * multiplicador_base * 0.1
    quantidade = np.random.poisson(media_quantidade)
    return max(1, quantidade)

# --- 3. FUNÇÃO PRINCIPAL DE ENRIQUECIMENTO (A NOVA PARTE) ---
def enriquecer_e_salvar(path_raw, path_processed):
    """
    Lê um CSV bruto, enriquece com dados sintéticos e salva em um novo arquivo.
    Retorna o DataFrame enriquecido.
    """
    print(f"Lendo o arquivo original: {path_raw}...")
    try:
        df = pd.read_csv(path_raw)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{path_raw}' não foi encontrado.")
        return None

    print("Arquivo lido com sucesso!")
    df.dropna(subset=['category', 'rating'], inplace=True)

    print("\nGerando colunas de 'preco' e 'custo'...")
    df['preco'] = df.apply(gerar_preco, axis=1)
    df['custo'] = df.apply(gerar_custo, axis=1)
    print("Colunas 'preco' e 'custo' geradas.")

    print("\nCalculando a popularidade de cada produto...")
    popularidade_produto = df.groupby('item_id').size()
    print("Popularidade calculada.")

    print("Gerando coluna de 'quantidade_vendida'...")
    df['quantidade_vendida'] = df.apply(gerar_quantidade_vendida, axis=1, args=(popularidade_produto,))
    print("Coluna 'quantidade_vendida' gerada.")
    
    print(f"\nSalvando o dataset enriquecido em: {path_processed}")
    os.makedirs(os.path.dirname(path_processed), exist_ok=True)
    df.to_csv(path_processed, index=False)
    print("Processo concluído com sucesso!")
    
    return df

# --- 4. BLOCO DE EXECUÇÃO (Mantido para uso via terminal) ---
if __name__ == "__main__":
    # Define caminhos relativos para quando o script é executado diretamente
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'dados', 'raw', 'amazon_electronics.csv')
    processed_path = os.path.join(base_dir, 'dados', 'processed', 'amazon_electronics_enriquecido.csv')
    
    enriquecer_e_salvar(raw_path, processed_path)