# Previsão de Preços de Veículos — Concessionária

> **Projeto de Portfolio | Fevereiro 2026**  
> Regressão Linear (simples e múltipla) para prever o preço de veículos baseado em características reais.

---

##  Resultados Rápidos

| Métrica | Simples (1 var) | Múltipla (8 vars) |
|---------|:---:|:---:|
| **R²**  | 0.0832 | **0.5745** |
| **RMSE** | R$ 9.938,90 | **R$ 6.771,06** |
| **MAE** | — | **R$ 4.702,67** |

---

## Estrutura do Projeto

```
projeto_regressao_linear/
├── dados_concessionaria.csv          # Dataset com 10.000 registros
├── notebook_previsao_precos.ipynb    # Notebook Jupyter completo
├── app_streamlit.py                  # Dashboard interativo (Streamlit)
├── README.md                         # Este arquivo
└── plots/                            # Gráficos gerados
    ├── 01_distribuicao_preco.png
    ├── 02_preco_por_marca.png
    ├── 03_preco_vs_ano.png
    ├── 04_preco_vs_km.png
    ├── 05_boxplot_condicao.png
    ├── 06_preco_combustivel.png
    ├── 07_importancia_features.png
    ├── 08_real_vs_predito.png
    ├── 09_residuos.png
    ├── 10_regressao_simples.png
    ├── 11_comparacao_r2.png
    └── 12_heatmap_correlacoes.png
```

---

## Tecnologias Usadas

| Biblioteca | Uso |
|-----------|-----|
| `pandas` | Manipulação e análise de dados |
| `numpy` | Operações numéricas |
| `matplotlib` | Visualização de dados |
| `seaborn` | Heatmaps e gráficos estatísticos |
| `scikit-learn` | Modelos ML, split, métricas, scaler |
| `streamlit` | Dashboard web interativo |

---

## Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/projeto-regressao-linear.git
cd projeto-regressao-linear
```

### 2. Instale as dependências
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Execute o Notebook
```bash
jupyter notebook notebook_previsao_precos.ipynb
```

### 4. Execute o Dashboard Streamlit
```bash
streamlit run app_streamlit.py
```
O dashboard abertrá automaticamente em `http://localhost:8501`

---

## 📖 Etapas do Projeto

### Geração do Dataset
- 10.000 registros sintéticos com dados reais de concessionária
- 12 marcas de carros (populares e premium)
- Variáveis: Marca, Modelo, Ano, Cor, Combustível, Potência, Portas, Quilômetros, Condição, Garantia, Cidade, Estado
- Preço calculado com função econômica realista (ano, km, marca, condição, combustível, etc.) + ruído gaussiano

### EDA (Análise Exploradora)
- Distribuição do preço (skewed à direita)
- Preço médio por marca — premium vs popular
- Correlações entre variáveis numéricas
- Análise por condição do veículo e tipo de combustível

### Pré-Processamento
- Remoção de colunas não-predictivas (ID, Modelo, Cor, Cidade, Estado)
- LabelEncoder para variáveis categóricas (Marca, Combustível, Condição)
- StandardScaler para padronização das features
- Split 80/20 (treino/teste)

### Modelagem
- **Regressão Linear Simples:** 1 variável (Quilômetros) → R² = 0.0832
- **Regressão Linear Múltipla:** 8 variáveis → R² = 0.5745
- Melhoria de ~590% no R² ao adicionar mais variáveis

### Avaliação
- Métricas: R², RMSE, MAE
- Análise de resíduos (distribuição normal, homocedasticidade)
- Comparação visual: Real vs Predito

### Deploy — Dashboard Streamlit
- Previsão em tempo real com formulário interativo
- EDA interativa com filtros
- Download de dados filtrados

---

## Insights Principais

1. **Marca** é a variável com maior impacto positivo no preço — Mercedes, BMW e Audi chefiam
2. **Quilômetros** tem impacto negativo direto — cada 10.000 km reduz ~R$ 1.500 no preço médio
3. **Condição** é fator decisivo: veículos "Novo" costam até 2.5x mais que "Usado - Regular"
4. **Veículos elétricos e híbridos** apresentam preço médio 12-18% maior
5. A regressão múltipla captura interações entre variáveis que a simples ignora

---

## Melhorias Futuras

- [ ] One-Hot Encoding ao invés de LabelEncoder
- [ ] Ridge / Lasso para regularização
- [ ] Random Forest / Gradient Boosting para relações não-lineares
- [ ] Feature engineering: idade do carro (2025 − Ano), km/ano
- [ ] Cross-validation (K-Fold)
- [ ] Deploy no Streamlit Cloud

---

## Autor

**[Ivo dos Santos Viana Junior]**  
📧 ivojuniorviana@email.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ivo-dos-santos-viana-j%C3%BAnior-1b3893198/) | [GitHub](https://github.com/Juniorsantosviana/Analises-Ciencias-de-Dados/tree/main/Analises%20e%20Ciencias%20de%20Dados%20Janeiro)()

---

*Projeto desenvolvido para portfolio de estágio — Fevereiro 2026*
