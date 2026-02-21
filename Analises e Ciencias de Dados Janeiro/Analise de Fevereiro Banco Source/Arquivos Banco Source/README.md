# Banco Source — Análise de Crédito

Projeto de análise exploratória e modelagem preditiva para avaliação de crédito. O objetivo é identificar padrões de inadimplência e construir modelos que apoiem decisões mais justas e eficientes na concessão de crédito.

---

## Estrutura do projeto

```
banco_source/
├── Banco_Source.ipynb      # Notebook principal
├── requirements.txt        # Dependências
└── README.md
```

---

## Sobre o dataset

O dataset contém 10.000 registros de clientes com as seguintes variáveis:

| Variável              | Tipo        | Descrição                              |
|-----------------------|-------------|----------------------------------------|
| `score_credito`       | Numérica    | Pontuação de crédito do cliente        |
| `renda_mensal`        | Numérica    | Renda mensal em reais                  |
| `idade`               | Numérica    | Idade do cliente                       |
| `meses_relacionamento`| Numérica    | Tempo de relacionamento com o banco    |
| `num_produtos`        | Numérica    | Quantidade de produtos contratados     |
| `historico_pagamentos`| Categórica  | Classificação do histórico (Bom/Regular/Ruim) |
| `inadimplente`        | Binária     | 1 = inadimplente, 0 = adimplente (target) |

---

## O que está no notebook

1. Geração e preparação dos dados
2. Análise exploratória — distribuições, estatísticas descritivas, correlações
3. Visualizações — histogramas, boxplots, heatmap, dispersão
4. Modelagem preditiva — Regressão Logística e Random Forest
5. Avaliação — matriz de confusão, curva ROC, métricas

---

## Como executar

```bash
# Clone o repositório
git clone https://github.com/Juniorsantosviana/Analises-Ciencias-de-Dados/tree/main
cd banco-source

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter notebook Banco_Source.ipynb
```

---


- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn

---

## Resultados obtidos

Os modelos foram treinados com validação cruzada (5-fold). O Random Forest apresentou melhor desempenho geral, com AUC-ROC de aproximadamente 0.97 e acurácia de 94% no conjunto de teste.

---

## Licença

MIT
