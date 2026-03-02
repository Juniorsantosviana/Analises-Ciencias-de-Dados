# Analise Exploratoria de Dados - Loja Imaginaria de Eletro

**Periodo:** Janeiro de 2025 a Janeiro de 2026  
**Registros:** 10.000 transacoes de vendas  
**Autor:** Ivo dos Santos Viana Junior  
**LinkedIn:** https://www.linkedin.com/in/ivo-dos-santos-viana-j%C3%BAnior-1b3893198/  
**GitHub:** https://github.com/Juniorsantosviana/Analises-Ciencias-de-Dados

---

## Sobre o Projeto

Este projeto e uma analise exploratoria de dados (EDA) de uma loja ficticia de eletrodomesticos e eletronicos. O objetivo foi simular um cenario realista de varejo com 10.000 registros de vendas ao longo de um ano completo, cobrindo sazonalidade, diversidade de produtos, regioes geograficas, canais de venda e comportamento de clientes.

A analise segue o fluxo classico de trabalho de um analista de dados: entendimento e qualidade dos dados, analise univariada, bivariada, correlacoes, insights e conclusoes de negocio.

---

## Estrutura do Repositorio

```
analise exploratoria de dados Loja Imaginaria de Eletro Marco/
│
├── analise_exploratoria_loja_eletro.ipynb   # Notebook principal com toda a analise
├── dataset.csv                               # Base de dados gerada (10.000 registros)
├── requirements.txt                          # Dependencias do projeto
└── README.md                                 # Este arquivo
```

---

## Dataset

O arquivo `dataset.csv` contem 22 colunas:

| Coluna | Descricao |
|---|---|
| id_venda | Identificador unico da venda |
| data_venda | Data da transacao |
| mes / ano / trimestre | Derivados da data |
| dia_semana | Dia da semana da venda |
| categoria | Categoria do produto |
| produto | Nome do produto |
| marca | Fabricante |
| preco_unitario | Preco de venda unitario |
| quantidade | Itens vendidos na transacao |
| desconto_percentual | Desconto aplicado (%) |
| valor_desconto | Valor monetario do desconto |
| valor_total | Receita liquida da venda |
| custo_unitario | Custo de aquisicao do produto |
| margem_bruta | Lucro bruto da venda |
| forma_pagamento | Metodo de pagamento |
| regiao | Regiao geografica do cliente |
| cidade | Cidade do cliente |
| canal_venda | Canal onde a compra foi realizada |
| avaliacao_cliente | Nota do cliente (1 a 5, pode ser nulo) |
| devolvido | Se o produto foi devolvido (0/1) |

**Categorias:** Televisores, Refrigeradores, Lavadoras, Ar Condicionado, Micro-ondas, Smartphones, Notebooks, Tablets, Fogoes, Aspiradores  
**Regioes:** Sudeste, Sul, Nordeste, Centro-Oeste, Norte  
**Canais:** Loja Fisica, E-commerce, App Mobile, Televendas  
**Pagamentos:** Cartao Credito, Cartao Debito, PIX, Boleto, Financiamento

---

## Topicos Analisados no Notebook

- Qualidade e consistencia dos dados
- KPIs principais do negocio (faturamento, margem, ticket medio, taxa de devolucao)
- Evolucao mensal e sazonal das vendas
- Performance por dia da semana
- Analise por categoria de produto (faturamento e margem)
- Analise por marca
- Comparativo entre canais de venda
- Distribuicao geografica por regiao e cidade
- Formas de pagamento
- Politica de descontos
- Avaliacoes e devolucoes de clientes
- Mapa de correlacao entre variaveis
- Heatmap de sazonalidade categoria x mes
- Top 15 produtos
- Conclusoes e insights de negocio

---

## Como Executar

1. Clone o repositorio ou baixe os arquivos na mesma pasta
2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Abra o notebook:

```bash
jupyter notebook analise_exploratoria_loja_eletro.ipynb
```

4. Execute todas as celulas em ordem (Kernel > Restart & Run All)

---

## Requisitos

- Python 3.8+
- Ver `requirements.txt` para as bibliotecas necessarias

---

## Notas

Os dados sao completamente ficticios e gerados com `numpy.random` e `pandas` para fins educacionais e de portfólio. Qualquer semelhanca com dados reais e coincidencia.

---

*Projeto desenvolvido como parte do portfólio de Ciencia de Dados.*  
*Pasta: analise exploratoria de dados Loja Imaginaria de Eletro Marco*
