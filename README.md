# 🧠 E-commerce Sentiment Analysis with NLP & Machine Learning

Este projeto implementa um pipeline completo de Processamento de Linguagem Natural (NLP) e Machine Learning para análise de sentimentos em avaliações de clientes de e-commerce, com foco em extração de insights de negócio e classificação automática de texto em ambiente produtivo.

O modelo final atingiu **91.85% de acurácia**, demonstrando alta capacidade preditiva e eficiência para aplicações reais.

---

## 📊 Evolução do Pipeline e Engenharia de Atributos

O modelo foi desenvolvido de forma incremental, com refinamento contínuo das técnicas de vetorização e pré-processamento textual:

### 1. Baseline (79.82%)
- Vetorização inicial utilizando **Bag of Words**
- Modelo linear como referência de performance

### 2. Limpeza e Normalização (83.75%)
- Remoção de pontuação e caracteres especiais
- Conversão para caixa baixa (`.lower()`)
- Filtragem de stopwords

### 3. Stemming (85.11%)
- Aplicação do `RSLPStemmer`
- Redução da variabilidade morfológica
- Diminuição da esparsidade da matriz de termos

### 4. TF-IDF + N-grams (91.85%)
- Aplicação de ponderação estatística (TF-IDF)
- Inclusão de bigramas (`ngram_range=(1,2)`)
- Captura de contexto e negações
- Limitação do vocabulário para **1000 features**, garantindo eficiência computacional

> 🧠 **Insight técnico:** A redução controlada do espaço vetorial manteve a performance do modelo, tornando-o mais leve, rápido e adequado para produção.

---

## 🗂️ Estrutura do Repositório

- `Dados/dataset_avaliacoes.csv` → Base de dados com reviews de clientes
- `analise_sentimentos.ipynb` → Notebook principal com EDA, modelagem e testes
- `modelo_regressao_logistica.pkl` → Modelo treinado (Joblib)
- `tfidf_vectorizer.pkl` → Vetorizador TF-IDF treinado (1000 features)
- `.gitignore` → Exclusão de arquivos binários e temporários

---

## 🎯 Insights de Negócio (Interpretabilidade do Modelo)

A análise dos coeficientes do modelo permitiu identificar os principais fatores que influenciam a satisfação dos clientes:

### 🟢 Fatores Positivos
- **Entrega rápida (`rap`)** → Forte correlação com avaliações positivas (+4.10)

### 🔴 Fatores Negativos
- **Produtos defeituosos (`defeit`, `fragil`)** → Principal causa de insatisfação (-3.03)
- **Problemas logísticos (`devolv`)** → Impacto direto na experiência do cliente (-2.93)
- **Estornos e reembolsos (`dinh`)** → Indicador de falhas operacionais (-2.70)

---

## 🚀 Execução do Modelo em Produção

O modelo foi serializado utilizando **Joblib**, permitindo inferência instantânea sem necessidade de reprocessamento do dataset.

```python
import joblib

# Carregamento dos componentes do pipeline
tfidf = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('modelo_regressao_logistica.pkl')

# Exemplo de inferência
texto = ["O produto chegou quebrado e a entrega atrasou muito"]
X = tfidf.transform(texto)
pred = model.predict(X)

print(f"Sentimento previsto: {pred[0]}")
