# Análise de Estoque e Rentabilidade com Machine Learning

Uma aplicação web interativa desenvolvida em Python que permite o upload de dados de vendas, realiza análises visuais detalhadas e utiliza modelos de Machine Learning para prever o sucesso de produtos, com foco em gestão de estoque e rentabilidade.

## 📋 Tema e Dataset

**Tema:** Análise de Estoque e Rentabilidade de Produtos Eletrônicos.

**Fonte dos Dados:** O projeto utiliza o dataset [Amazon Electronics Products Sales](https://www.kaggle.com/datasets/edusanketdk/electronics) disponível no Kaggle. Este dataset contém informações sobre avaliações de usuários para produtos eletrônicos da Amazon.

## ⚠️ Geração de Dados Sintéticos (Metodologia)

O dataset original, embora rico em informações de popularidade (`rating` e `timestamp`), não possuía dados financeiros e de volume essenciais para o tema proposto, como:
*   Preço de Venda (`price`)
*   Custo do Produto (`cost`)
*   Quantidade Vendida por Transação (`quantity_sold`)

Para contornar essa limitação e atender aos objetivos do trabalho, foi desenvolvido um processo de **enriquecimento de dados**. O script `scripts/enriquecer_dataset.py` foi criado para gerar esses campos de forma lógica e realista, com base nas seguintes premissas:

1.  **Preço (`preco`):** Gerado a partir de faixas de preço médias definidas para cada `category` de produto. Produtos em categorias como "Computers" e "TV & Home Theater" recebem preços maiores, enquanto "Accessories & Supplies" recebem preços menores. Uma pequena variação foi aplicada com base no `rating` do produto.
2.  **Custo (`custo`):** Calculado como uma porcentagem do `preco`, simulando a margem de lucro. A margem de lucro também varia por categoria, sendo maior para acessórios e menor para itens de alto valor.
3.  **Quantidade Vendida (`quantidade_vendida`):** Simulada com base na popularidade do produto (número de avaliações) e no seu preço. Produtos mais baratos e mais populares tendem a ter maiores quantidades vendidas por transação.

**Todos os dados gerados são sintéticos e servem unicamente para fins acadêmicos, permitindo a aplicação das técnicas de análise e modelagem solicitadas.**

## 📁 Estrutura do Projeto

O projeto foi organizado de forma modular para facilitar a manutenção e o entendimento, conforme solicitado.

projeto-estoque/
├── dados/ # Armazenamento dos arquivos de dados
│ ├── raw/ # Dados brutos, originais
│ └── processed/ # Dados enriquecidos e processados
├── interface/ # Código da aplicação web (Streamlit)
│ ├── app.py # Arquivo principal da aplicação
│ └── pages/ # Páginas separadas da aplicação
├── modelos/ # Modelos de Machine Learning treinados
├── scripts/ # Scripts de suporte e backend
│ ├── database.py # Funções de interação com o banco SQLite
│ └── enriquecer_dataset.py # Script para geração de dados sintéticos
├── .gitignore # Arquivos ignorados pelo versionamento
├── README.md # Documentação do projeto
└── requirements.txt # Dependências do projeto


## 🚀 Como Executar o Projeto

Siga os passos abaixo para reproduzir o ambiente e executar a aplicação localmente.

1.  **Clonar o Repositório:**
    ```bash
    git clone <https://github.com/lucas0pr/projeto-estoque.git>
    cd projeto-estoque
    ```

2.  **Criar Ambiente Virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Preparar os Dados:**
    *   Coloque o arquivo `amazon_electronics.csv` baixado do Kaggle na pasta `dados/raw/`.
    *   Execute o script de enriquecimento para gerar o dataset final:
        ```bash
        python scripts/enriquecer_dataset.py
        ```
    *   Isso criará o arquivo `amazon_electronics_enriquecido.csv` na pasta `dados/processed/`.

5.  **Executar a Aplicação:**
    ```bash
    streamlit run interface/app.py
    ```

    A aplicação estará disponível no seu navegador, geralmente em `http://localhost:8501`.

## 🛠️ Funcionalidades da Aplicação

A aplicação está dividida em três módulos principais, acessíveis pelo menu lateral:

1.  **Upload de Dados:** Permite ao usuário carregar um arquivo CSV, que é processado e armazenado em um banco de dados SQLite, garantindo flexibilidade e re-treinamento dinâmico.
2.  **Análise e Visualização:** Um dashboard interativo com gráficos (barras, linhas, mapas) que exploram o desempenho de vendas, rentabilidade por categoria, popularidade de marcas e giro de produtos.
3.  **Machine Learning:** Interface para configurar, treinar e avaliar modelos de classificação (ex: Árvore de Decisão, Random Forest) para prever o "sucesso" de um produto com base em suas características.

## 🧠 Tecnologias Utilizadas

-   **Python:** Linguagem principal.
-   **Streamlit:** Framework para a criação da aplicação web interativa.
-   **Pandas:** Manipulação e análise de dados.
-   **SQLite:** Banco de dados leve para armazenamento local dos dados.
-   **Scikit-learn:** Biblioteca para implementação dos algoritmos de Machine Learning.
-   **Plotly:** Criação de gráficos interativos e dinâmicos.
-   **NumPy:** Cálculos numéricos e geração de dados sintéticos.

## 📝 Considerações Finais e Limitações

-   A principal limitação do projeto é o uso de dados sintéticos para métricas financeiras. Embora baseados em lógica, eles não representam valores reais.
-   O modelo de Machine Learning utiliza um proxy de "sucesso" (baseado em volume de vendas e rating) que é uma simplificação da complexidade do mercado.
-   Como trabalho futuro, a aplicação poderia ser conectada a um banco de dados de uma empresa real ou integrada a APIs de e-commerce para obter dados em tempo real.
