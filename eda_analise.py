# ==============================================================================
# Trabalho de IA Generativa - Análise Exploratória dos Dados (EDA)
# Questão 2: Entendimento dos Dados
# Dataset: Suicide_Detection.csv
# ==============================================================================

import sys
import os
import warnings

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================================================================
# CARREGAMENTO DOS DADOS
# ==============================================================================
print("=" * 70)
print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("Dataset: Suicide_Detection.csv")
print("=" * 70)

df = pd.read_csv("Suicide_Detection.csv")

# ==============================================================================
# a) Dimensão do dataset (quantidade de sentenças)
# ==============================================================================
print("\n" + "─" * 70)
print("a) DIMENSÃO DO DATASET")
print("─" * 70)
print(f"  → Número total de sentenças (linhas): {df.shape[0]:,}")
print(f"  → Número de colunas (atributos):      {df.shape[1]}")
print(f"  → Colunas presentes: {df.columns.tolist()}")

# ==============================================================================
# b) Descrição de cada atributo/coluna
# ==============================================================================
print("\n" + "─" * 70)
print("b) DESCRIÇÃO DE CADA ATRIBUTO")
print("─" * 70)
print(f"\n  Colunas encontradas: {df.columns.tolist()}\n")
for col in df.columns:
    tipo = df[col].dtype
    n_unicos = df[col].nunique()
    n_nulos = df[col].isnull().sum()
    print(f"  Coluna: '{col}'")
    print(f"    - Tipo: {tipo}")
    print(f"    - Valores únicos: {n_unicos:,}")
    print(f"    - Valores nulos: {n_nulos}")
    
    if col == 'Unnamed: 0':
        print(f"    - Descrição: Índice sequencial gerado automaticamente pelo salvamento do CSV.")
        print(f"    - Faixa: [{df[col].min()} ... {df[col].max()}]")
        print(f"    → POUCO RELEVANTE: é apenas um identificador numérico sem valor analítico.")
    elif col == 'text':
        tamanhos = df[col].astype(str).str.len()
        palavras = df[col].astype(str).str.split().str.len()
        print(f"    - Descrição: Texto da sentença/postagem (variável preditora principal).")
        print(f"    - Comprimento médio (caracteres): {tamanhos.mean():.1f}")
        print(f"    - Comprimento mediano (caracteres): {tamanhos.median():.1f}")
        print(f"    - Comprimento mín/máx (caracteres): {tamanhos.min()} / {tamanhos.max()}")
        print(f"    - Nº médio de palavras: {palavras.mean():.1f}")
        print(f"    - Nº mediano de palavras: {palavras.median():.1f}")
        print(f"    - Nº mín/máx de palavras: {palavras.min()} / {palavras.max()}")
    elif col == 'class':
        print(f"    - Descrição: Variável-alvo (classe). Indica se o texto é relacionado a suicídio ou não.")
        print(f"    - Valores: {df[col].unique().tolist()}")
    else:
        print(f"    - Primeiros valores: {df[col].head(3).tolist()}")
    print()

# ==============================================================================
# c) Distribuição da variável-alvo
# ==============================================================================
print("─" * 70)
print("c) DISTRIBUIÇÃO DA VARIÁVEL-ALVO ('class')")
print("─" * 70)

dist = df['class'].value_counts()
dist_pct = df['class'].value_counts(normalize=True) * 100

print(f"\n  Classe             Contagem     Percentual")
print(f"  {'─'*45}")
for classe in dist.index:
    print(f"  {classe:<18} {dist[classe]:>8,}     {dist_pct[classe]:>6.2f}%")
print(f"  {'─'*45}")
print(f"  {'Total':<18} {dist.sum():>8,}     {100.00:>6.2f}%")

# ==============================================================================
# d) O conjunto de dados é balanceado ou desbalanceado?
# ==============================================================================
print("\n" + "─" * 70)
print("d) BALANCEAMENTO DO DATASET")
print("─" * 70)

razao = dist.min() / dist.max()
diff_abs = abs(dist.iloc[0] - dist.iloc[1])
diff_pct = abs(dist_pct.iloc[0] - dist_pct.iloc[1])

print(f"\n  Razão minoria/maioria: {razao:.4f} ({razao*100:.2f}%)")
print(f"  Diferença absoluta entre classes: {diff_abs:,}")
print(f"  Diferença percentual: {diff_pct:.2f} pontos percentuais")

if razao >= 0.8:
    print(f"\n  → CONCLUSÃO: O dataset é BALANCEADO.")
    print(f"    A proporção entre as classes é próxima de 50/50.")
    print(f"    Não há necessidade de técnicas de balanceamento (oversampling/undersampling).")
elif razao >= 0.5:
    print(f"\n  → CONCLUSÃO: O dataset é LEVEMENTE DESBALANCEADO.")
    print(f"    A diferença existe mas não é severa.")
else:
    print(f"\n  → CONCLUSÃO: O dataset é DESBALANCEADO.")
    print(f"    Técnicas de balanceamento podem ser necessárias.")

# ==============================================================================
# e) Atributos pouco relevantes ou redundantes
# ==============================================================================
print("\n" + "─" * 70)
print("e) ATRIBUTOS POUCO RELEVANTES OU REDUNDANTES")
print("─" * 70)

if 'Unnamed: 0' in df.columns:
    print(f"\n  → A coluna 'Unnamed: 0' é um ÍNDICE NUMÉRICO SEQUENCIAL (0 a {df['Unnamed: 0'].max():,}).")
    print(f"    É um artefato do salvamento do CSV com pandas (df.to_csv sem index=False).")
    print(f"    NÃO possui nenhum valor preditivo ou informativo para a tarefa de classificação.")
    print(f"    RECOMENDAÇÃO: Deve ser removida antes do treinamento.")
    
    # Verificar se é realmente sequencial
    is_sequential = (df['Unnamed: 0'].diff().dropna() == 1).all()
    print(f"    É sequencial: {'Sim' if is_sequential else 'Não'}")
    print(f"    Correlação com a classe: Nenhuma (é um índice arbitrário).")
else:
    print(f"\n  → O dataset possui apenas 2 colunas relevantes ('text' e 'class').")
    print(f"    Não há atributos redundantes.")

# ==============================================================================
# f) Informações importantes observadas durante a exploração
# ==============================================================================
print("\n" + "─" * 70)
print("f) INFORMAÇÕES IMPORTANTES OBSERVADAS")
print("─" * 70)

# Valores nulos
nulos_total = df.isnull().sum().sum()
print(f"\n  [1] Valores Nulos:")
for col in df.columns:
    n = df[col].isnull().sum()
    print(f"      - '{col}': {n} ({n/len(df)*100:.2f}%)")
print(f"      Total: {nulos_total}")

# Duplicatas
dup_texto = df.duplicated(subset='text').sum()
dup_completa = df.duplicated().sum()
print(f"\n  [2] Duplicatas:")
print(f"      - Textos duplicados: {dup_texto:,} ({dup_texto/len(df)*100:.2f}%)")
print(f"      - Linhas completamente duplicadas: {dup_completa:,}")

# Estatísticas de comprimento dos textos
df['__n_chars'] = df['text'].astype(str).str.len()
df['__n_words'] = df['text'].astype(str).str.split().str.len()

print(f"\n  [3] Estatísticas de Comprimento dos Textos:")
print(f"      Caracteres:")
print(f"        - Média:   {df['__n_chars'].mean():.1f}")
print(f"        - Mediana: {df['__n_chars'].median():.1f}")
print(f"        - Desvio:  {df['__n_chars'].std():.1f}")
print(f"        - Mín:     {df['__n_chars'].min()}")
print(f"        - Máx:     {df['__n_chars'].max()}")
print(f"      Palavras:")
print(f"        - Média:   {df['__n_words'].mean():.1f}")
print(f"        - Mediana: {df['__n_words'].median():.1f}")
print(f"        - Desvio:  {df['__n_words'].std():.1f}")
print(f"        - Mín:     {df['__n_words'].min()}")
print(f"        - Máx:     {df['__n_words'].max()}")

# Comparar comprimento por classe
print(f"\n  [4] Comprimento Médio por Classe:")
for classe in df['class'].unique():
    subset = df[df['class'] == classe]
    print(f"      - {classe}:")
    print(f"          Palavras: média={subset['__n_words'].mean():.1f}, mediana={subset['__n_words'].median():.1f}")
    print(f"          Caracteres: média={subset['__n_chars'].mean():.1f}, mediana={subset['__n_chars'].median():.1f}")

# Textos muito curtos
curtos = (df['__n_words'] <= 3).sum()
print(f"\n  [5] Textos Muito Curtos (≤ 3 palavras): {curtos:,} ({curtos/len(df)*100:.2f}%)")

# Textos muito longos (outliers)
q99 = df['__n_words'].quantile(0.99)
longos = (df['__n_words'] > q99).sum()
print(f"  [6] Textos Muito Longos (> P99 = {q99:.0f} palavras): {longos:,}")

# Amostra de textos
print(f"\n  [7] Exemplos de Textos:")
print(f"\n      --- Classe 'suicide' ---")
sample_suicide = df[df['class'] == 'suicide']['text'].head(2)
for i, txt in enumerate(sample_suicide):
    truncated = txt[:150] + "..." if len(str(txt)) > 150 else txt
    print(f"      [{i+1}] {truncated}")

print(f"\n      --- Classe 'non-suicide' ---")
sample_non = df[df['class'] == 'non-suicide']['text'].head(2)
for i, txt in enumerate(sample_non):
    truncated = txt[:150] + "..." if len(str(txt)) > 150 else txt
    print(f"      [{i+1}] {truncated}")

# ==============================================================================
# GERAÇÃO DE GRÁFICOS DA EDA
# ==============================================================================
print(f"\n{'─'*70}")
print("GERANDO GRÁFICOS DA EDA...")
print("─" * 70)

os.makedirs("resultados", exist_ok=True)

# --- Gráfico 1: Distribuição da variável-alvo ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Barplot
cores = ['#4CAF50', '#F44336']
bars = axes[0].bar(dist.index, dist.values, color=cores, edgecolor='white', linewidth=1.5)
for bar, val, pct in zip(bars, dist.values, dist_pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Classe', fontsize=12)
axes[0].set_ylabel('Quantidade', fontsize=12)
axes[0].set_title('Distribuição das Classes', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, dist.max() * 1.15)

# Pieplot
axes[1].pie(dist.values, labels=dist.index, autopct='%1.1f%%', colors=cores,
            startangle=90, explode=(0.03, 0.03), textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Proporção das Classes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/eda_distribuicao_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/eda_distribuicao_classes.png")

# --- Gráfico 2: Distribuição do comprimento dos textos ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de palavras
for classe, cor in zip(['non-suicide', 'suicide'], cores):
    subset = df[df['class'] == classe]['__n_words']
    axes[0].hist(subset, bins=80, alpha=0.6, color=cor, label=classe, density=True)
axes[0].set_xlabel('Número de Palavras', fontsize=12)
axes[0].set_ylabel('Densidade', fontsize=12)
axes[0].set_title('Distribuição do Nº de Palavras por Classe', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_xlim(0, df['__n_words'].quantile(0.98))
axes[0].grid(True, alpha=0.3)

# Boxplot de palavras por classe
bp = axes[1].boxplot(
    [df[df['class'] == 'non-suicide']['__n_words'], df[df['class'] == 'suicide']['__n_words']],
    tick_labels=['non-suicide', 'suicide'], patch_artist=True,
    boxprops=dict(linewidth=1.5),
    medianprops=dict(color='black', linewidth=2),
    flierprops=dict(marker='.', markersize=2, alpha=0.3)
)
for patch, cor in zip(bp['boxes'], cores):
    patch.set_facecolor(cor)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Número de Palavras', fontsize=12)
axes[1].set_title('Boxplot do Nº de Palavras por Classe', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('resultados/eda_comprimento_textos.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/eda_comprimento_textos.png")

# --- Gráfico 3: Distribuição do comprimento (caracteres) ---
fig, ax = plt.subplots(figsize=(10, 5))
for classe, cor in zip(['non-suicide', 'suicide'], cores):
    subset = df[df['class'] == classe]['__n_chars']
    ax.hist(subset, bins=80, alpha=0.6, color=cor, label=classe, density=True)
ax.set_xlabel('Número de Caracteres', fontsize=12)
ax.set_ylabel('Densidade', fontsize=12)
ax.set_title('Distribuição do Nº de Caracteres por Classe', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, df['__n_chars'].quantile(0.98))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resultados/eda_comprimento_caracteres.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/eda_comprimento_caracteres.png")

# Limpar colunas auxiliares
df.drop(columns=['__n_chars', '__n_words'], inplace=True)

print(f"\n{'=' * 70}")
print("ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
print(f"{'=' * 70}")
