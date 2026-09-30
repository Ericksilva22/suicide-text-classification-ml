
# Trabalho de IA Generativa - Classificação de Textos sobre Suicídio
# Modelo 2: SVM (Support Vector Machine)
# Dataset: Suicide_Detection.csv


import sys
import os
import re
import time
import warnings

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)


# 1. CARREGAMENTO DOS DADOS

print("MODELO: SVM (Support Vector Machine - LinearSVC)")
print("=" * 70)

print("\n[1/6] Carregando dataset...")
df = pd.read_csv("Suicide_Detection.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"  → Dimensão do dataset: {df.shape[0]} sentenças, {df.shape[1]} colunas")
print(f"  → Colunas: {df.columns.tolist()}")
print(f"  → Distribuição das classes:")
print(f"    - suicide:     {(df['class'] == 'suicide').sum():>7,}")
print(f"    - non-suicide: {(df['class'] == 'non-suicide').sum():>7,}")
print(f"   Valores ausentes: {df.isnull().sum().sum()}")
print(f"   Textos duplicados: {df.duplicated(subset='text').sum()}")


# 2. PRÉ-PROCESSAMENTO DOS TEXTOS

print("\n[2/6] Pré-processando textos...")

def limpar_texto(texto):
    """
    Pipeline de limpeza de texto:
    - Converte para minúsculas
    - Remove URLs
    - Remove menções (@user)
    - Remove hashtags
    - Remove caracteres especiais e números
    - Remove espaços extras
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#\w+', '', texto)
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df['text_clean'] = df['text'].apply(limpar_texto)

antes = len(df)
df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)
depois = len(df)
print(f"  → Textos removidos por ficarem vazios: {antes - depois}")
print(f"  → Tamanho após limpeza: {depois}")


# 3. CODIFICAÇÃO DOS RÓTULOS

print("\n[3/6] Codificando rótulos...")

df['label'] = (df['class'] == 'suicide').astype(int)
print(f"  → suicide → 1, non-suicide → 0")
print(f"  → Distribuição: {dict(df['label'].value_counts())}")

# 4. DIVISÃO DOS DADOS (Treino / Teste)

print("\n[4/6] Dividindo dados em treino e teste...")

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"   Treino: {len(X_train):,} amostras")
print(f"   Teste:  {len(X_test):,} amostras")
print(f"   Proporção classes treino: { dict(y_train.value_counts()) }")
print(f"   Proporção classes teste:  { dict(y_test.value_counts()) }")

# 5. VETORIZAÇÃO TF-IDF

print("\n[5/6] Vetorizando textos com TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=50000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words='english'
)

t0 = time.time()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
t_tfidf = time.time() - t0

print(f"   Vocabulário: {len(tfidf.vocabulary_):,} termos")
print(f"   Dimensão treino: {X_train_tfidf.shape}")
print(f"   Dimensão teste:  {X_test_tfidf.shape}")
print(f"   Tempo de vetorização: {t_tfidf:.2f}s")

# 6. TREINAMENTO DO MODELO - SVM (LinearSVC com Calibração)

print("\n[6/6] Treinando modelo SVM (LinearSVC com CalibratedClassifierCV)...")
print("  → LinearSVC é escolhido por ser eficiente em alta dimensionalidade")
print("  → CalibratedClassifierCV permite obter probabilidades para AUC-ROC")

# LinearSVC não fornece predict_proba nativamente,
# então usamos CalibratedClassifierCV para calibrar as probabilidades
svc_base = LinearSVC(
    C=1.0,                  # Parâmetro de regularização
    max_iter=10000,         # Iterações máximas
    class_weight='balanced', # Balancear pesos das classes (precaução)
    random_state=42,
    dual='auto'
)

# Calibração com Platt scaling (sigmoid) usando 5-fold CV
modelo = CalibratedClassifierCV(
    estimator=svc_base,
    cv=5,
    method='sigmoid'
)

t0 = time.time()
modelo.fit(X_train_tfidf, y_train)
t_treino = time.time() - t0
print(f"   Hiperparâmetros: C=1.0, max_iter=10000, class_weight='balanced'")
print(f"   Calibração: CalibratedClassifierCV(cv=5, method='sigmoid')")
print(f"   Tempo de treinamento: {t_treino:.2f}s")

# 7. AVALIAÇÃO DO MODELO

print("\n" + "=" * 70)
print("RESULTADOS - SVM (LinearSVC)")
print("=" * 70)

# Predições
t0 = time.time()
y_pred = modelo.predict(X_test_tfidf)
y_prob = modelo.predict_proba(X_test_tfidf)[:, 1]
t_pred = time.time() - t0

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n  Acurácia:  {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precisão:  {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  Tempo de predição: {t_pred:.2f}s")

# Classification Report
print(f"\n{'─'*50}")
print("Classification Report:")
print('─'*50)
report = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'])
print(report)

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"{'─'*50}")
print("Matriz de Confusão:")
print(f"  TN (Verdadeiros Negativos): {tn:>6,}")
print(f"  FP (Falsos Positivos):      {fp:>6,}")
print(f"  FN (Falsos Negativos):      {fn:>6,}")
print(f"  TP (Verdadeiros Positivos): {tp:>6,}")

# 8. GERAÇÃO DE GRÁFICOS

print(f"\n{'─'*50}")
print("Gerando gráficos...")

os.makedirs("resultados", exist_ok=True)

# --- Gráfico 1: Matriz de Confusão ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Oranges',
    xticklabels=['non-suicide', 'suicide'],
    yticklabels=['non-suicide', 'suicide'],
    ax=ax, cbar_kws={'label': 'Contagem'},
    annot_kws={'size': 14}
)
ax.set_xlabel('Classe Predita', fontsize=12)
ax.set_ylabel('Classe Real', fontsize=12)
ax.set_title('Matriz de Confusão — SVM (LinearSVC)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/svm_matriz_confusao.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Salvo: resultados/svm_matriz_confusao.png")

# --- Gráfico 2: Curva ROC ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#FF5722', lw=2, label=f'SVM (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatório (AUC = 0.5)')
ax.set_xlabel('Taxa de Falso Positivo (FPR)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiro Positivo (TPR)', fontsize=12)
ax.set_title('Curva ROC — SVM (LinearSVC)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resultados/svm_curva_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/svm_curva_roc.png")

# --- Gráfico 3: Métricas por classe ---
report_dict = classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide'], output_dict=True)
classes = ['non-suicide', 'suicide']
metricas = ['precision', 'recall', 'f1-score']
valores = {m: [report_dict[c][m] for c in classes] for m in metricas}

x = np.arange(len(classes))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 6))
for i, (metrica, vals) in enumerate(valores.items()):
    bars = ax.bar(x + i * width, vals, width, label=metrica.capitalize())
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Classe', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Métricas por Classe — SVM (LinearSVC)', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resultados/svm_metricas_classe.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Salvo: resultados/svm_metricas_classe.png")

# 9. SALVAR RESULTADOS EM ARQUIVO

with open('resultados/svm_resultados.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RESULTADOS - SVM (LinearSVC + CalibratedClassifierCV)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset: Suicide_Detection.csv\n")
    f.write(f"Total de amostras: {len(df):,}\n")
    f.write(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}\n")
    f.write(f"Vocabulário TF-IDF: {len(tfidf.vocabulary_):,} termos\n\n")
    f.write(f"Hiperparâmetros:\n")
    f.write(f"  C (regularização): 1.0\n")
    f.write(f"  max_iter: 10000\n")
    f.write(f"  class_weight: balanced\n")
    f.write(f"  Calibração: CalibratedClassifierCV(cv=5, method='sigmoid')\n")
    f.write(f"  TF-IDF max_features: 50000\n")
    f.write(f"  TF-IDF ngram_range: (1, 2)\n")
    f.write(f"  TF-IDF min_df: 2 | max_df: 0.95\n\n")
    f.write(f"Métricas:\n")
    f.write(f"  Acurácia:  {acc:.4f}\n")
    f.write(f"  Precisão:  {prec:.4f}\n")
    f.write(f"  Recall:    {rec:.4f}\n")
    f.write(f"  F1-Score:  {f1:.4f}\n")
    f.write(f"  AUC-ROC:   {auc:.4f}\n\n")
    f.write(f"Tempos:\n")
    f.write(f"  Vetorização TF-IDF: {t_tfidf:.2f}s\n")
    f.write(f"  Treinamento: {t_treino:.2f}s\n")
    f.write(f"  Predição: {t_pred:.2f}s\n\n")
    f.write(f"Matriz de Confusão:\n")
    f.write(f"  TN={tn:,}  FP={fp:,}\n")
    f.write(f"  FN={fn:,}  TP={tp:,}\n\n")
    f.write(f"Classification Report:\n{report}\n")

print("   Salvo: resultados/svm_resultados.txt")

print(f"\n{'=' * 70}")
print("SVM concluído com sucesso!")
print(f"{'=' * 70}")
