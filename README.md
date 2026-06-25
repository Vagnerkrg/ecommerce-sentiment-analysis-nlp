# 🎯 Classificação Automatizada de Sentimentos em E-commerce com NLP

[![Python](https://shields.io)](https://python.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)
[![NLTK](https://shields.io)](https://nltk.org)

Este projeto implementa um pipeline completo de Processamento de Linguagem Natural (NLP) e Machine Learning para classificar e extrair inteligência de negócio a partir de avaliações de clientes de e-commerce. O modelo final alcançou uma **acurácia recorde de 91.85%**, operando com alta eficiência e pronto para integração em produção.

---

## 📊 Evolução Técnica e Engenharia de Atributos

O projeto foi construído de forma iterativa, aplicando técnicas incrementais para otimização do classificador linear (*Regressão Logística*):

1.  **Baseline (79.82%):** Vetorização inicial utilizando *Bag of Words* cru.
2.  **Limpeza e Normalização (83.75%):** Remoção de pontuações, caracteres especiais, conversão para caixa baixa (`.lower()`) e filtragem de *Stopwords*.
3.  **Stemming (85.11%):** Extração de radicais morfológicos utilizando o algoritmo `RSLPStemmer`, reduzindo drasticamente a esparsidade da matriz de termos.
4.  **TF-IDF + N-grams (91.85%):** Substituição da contagem simples por ponderação estatística e introdução de Bigramas (`ngram_range=(1,2)`) para capturar contextos sequenciais e negações.

> 🧠 **Insight de Otimização:** Testes comparativos provaram que limitar o dicionário a **1000 features** entrega a mesma acurácia do que recursos ilimitados, garantindo um modelo leve, rápido e computacionalmente eficiente para produção.

---

## 🗂️ Estrutura do Repositório

*   `Dados/dataset_avaliacoes.csv`: Base de dados contendo o corpus de resenhas textuais.
*   `analise_sentimentos.ipynb`: Notebook Jupyter contendo o desenvolvimento, testes e gráficos.
*   `modelo_regressao_logistica.pkl`: Binário serializado do modelo preditivo (*Gerado via Joblib*).
*   `tfidf_vectorizer.pkl`: Binário serializado do vetorizador calibrado de 1000 features (*Gerado via Joblib*).
*   `.gitignore`: Configuração de segurança para impedir o upload de binários pesados (`*.pkl`).

---

## 🎯 Diagnóstico de Negócio (Actionable Insights)

Ao inspecionar os coeficientes lineares do modelo (`.nlargest()` e `.nsmallest()`), a Inteligência Artificial mapeou as principais alavancas de satisfação e gargalos operacionais da empresa:

*   **🟢 Logística Expressa (`rap`: +4.10):** A agilidade na entrega é o principal vetor para acionar reviews de 5 estrelas.
*   **🔴 Falha Estrutural (`defeit`: -3.03 / `fragil`: -3.03):** O maior gargalo de insatisfação está atrelado a produtos que chegam danificados, gerando um vazamento de caixa severo com processos de **devolução (`devolv`: -2.93)** e **estornos de dinheiro (`dinh`: -2.70)**.

---

## 🚀 Como Executar e Carregar o Modelo

Como o modelo foi persistido usando a biblioteca `joblib`, você pode carregar o cérebro da IA instantaneamente em qualquer script Python para fazer inferências em tempo real, sem a necessidade de reprocessar o dataset:

```python
import joblib

# Carregar os componentes de produção
tfidf_1000 = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')

# Exemplo de inferência instantânea
nova_frase = ["O produto chegou quebrado, péssima experiência!"]
vetorizado = tfidf_1000.transform(nova_frase)
predicao = regressao_logistica.predict(vetorizado)

print(f"Sentimento previsto: {predicao[0]}") # Output: 'negativo'
```
