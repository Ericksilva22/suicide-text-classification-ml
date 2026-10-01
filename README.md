# 🧠 Classificação de Textos sobre Suicídio — Machine Learning Clássico

> Trabalho de IA Generativa — Comparação de 3 modelos de ML para detecção de textos relacionados a suicídio.

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de **classificação binária de textos** (suicide vs. non-suicide) utilizando três algoritmos de Machine Learning clássico. O objetivo é comparar o desempenho de abordagens distintas — probabilística, baseada em margens e ensemble — sob condições experimentais idênticas.

**Dataset:** [Suicide Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) — 232.011 sentenças coletadas do Reddit.

## 🏗️ Modelos Implementados

| Modelo | Algoritmo | Acurácia | F1-Score | AUC-ROC |
|--------|-----------|----------|----------|---------|
| **Naive Bayes** | MultinomialNB | 91.89% | 0.9214 | 0.9766 |
| **SVM** ⭐ | LinearSVC + CalibratedClassifierCV | **93.94%** | **0.9391** | **0.9836** |
| **Random Forest** | RandomForestClassifier (200 árvores) | 89.90% | 0.9005 | 0.9603 |

> ⭐ O **SVM** obteve o melhor desempenho em todas as métricas.

## 🔧 Pipeline

```
Texto bruto → Limpeza → TF-IDF (50k features, uni+bigramas) → Modelo → Predição
```

1. **Pré-processamento:** Lowercase, remoção de URLs/menções/hashtags/caracteres especiais
2. **Vetorização:** TF-IDF com 50.000 features, unigramas + bigramas, sublinear TF
3. **Treinamento:** 80% treino / 20% teste (divisão estratificada, `random_state=42`)
4. **Avaliação:** Acurácia, Precisão, Recall, F1-Score, AUC-ROC

## 📁 Estrutura do Projeto

```
├── eda_analise.py           # Análise exploratória dos dados (EDA)
├── naive_bayes.py           # Modelo 1: Naive Bayes
├── svm.py                   # Modelo 2: SVM (LinearSVC)
├── random_forest.py         # Modelo 3: Random Forest
├── comparacao_modelos.py    # Comparação dos 3 modelos
├── requirements.txt         # Dependências Python
└── resultados/              # Gráficos e relatórios gerados
    ├── nb_*.png / .txt      # Resultados Naive Bayes
    ├── svm_*.png / .txt     # Resultados SVM
    ├── rf_*.png / .txt      # Resultados Random Forest
    └── comparacao_*.png     # Gráficos comparativos
```

## 🚀 Como Executar

### 1. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
```

- **Windows:**
```bash
.venv\Scripts\activate
```

- **Linux/macOS:**
```bash
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar o dataset

Baixe o arquivo `Suicide_Detection.csv` do [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) e coloque na raiz do projeto.

### 4. Executar os modelos

```bash
# Análise exploratória
python eda_analise.py

# Treinar cada modelo individualmente
python naive_bayes.py
python svm.py
python random_forest.py

# Gerar comparação final
python comparacao_modelos.py
```

Os resultados (gráficos e métricas) serão salvos na pasta `resultados/`.

## 📊 Resultados Visuais

Após a execução, a pasta `resultados/` conterá:

- **Matrizes de confusão** de cada modelo
- **Curvas ROC** de cada modelo
- **Gráficos de métricas por classe** (precision, recall, f1)
- **Top 20 features mais importantes** (Random Forest)
- **Gráficos comparativos**: barras agrupadas, radar chart e tempos de execução

## 🛠️ Tecnologias

- **Python**
- **scikit-learn** — Modelos, vetorização TF-IDF e métricas
- **Pandas / NumPy** — Manipulação de dados
- **Matplotlib / Seaborn** — Visualizações

## 📝 Licença

Este projeto foi desenvolvido para fins acadêmicos.
