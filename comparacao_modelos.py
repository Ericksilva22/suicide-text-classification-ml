# ==============================================================================
# Trabalho de IA Generativa - Classificação de Textos sobre Suicídio
# Script de Comparação entre os 3 Modelos
# Gera gráficos comparativos e tabela resumo
# ==============================================================================
# EXECUTAR APÓS rodar os 3 scripts individuais:
#   python naive_bayes.py
#   python svm.py
#   python random_forest.py
# ==============================================================================

import os
import re
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# 1. LEITURA DOS RESULTADOS


def extrair_metricas(caminho):
    """Extrai métricas de um arquivo de resultados."""
    metricas = {}
    with open(caminho, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    for metrica in ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']:
        match = re.search(rf'{metrica}:\s+([\d.]+)', conteudo)
        if match:
            metricas[metrica] = float(match.group(1))
    
    # Extrair tempos
    for tempo in ['Treinamento', 'Predição', 'Vetorização TF-IDF']:
        match = re.search(rf'{tempo}:\s+([\d.]+)s', conteudo)
        if match:
            metricas[f'Tempo_{tempo.split()[0]}'] = float(match.group(1))
    
    # Extrair matriz de confusão
    match_tn = re.search(r'TN=([\d,]+)', conteudo)
    match_fp = re.search(r'FP=([\d,]+)', conteudo)
    match_fn = re.search(r'FN=([\d,]+)', conteudo)
    match_tp = re.search(r'TP=([\d,]+)', conteudo)
    if all([match_tn, match_fp, match_fn, match_tp]):
        metricas['TN'] = int(match_tn.group(1).replace(',', ''))
        metricas['FP'] = int(match_fp.group(1).replace(',', ''))
        metricas['FN'] = int(match_fn.group(1).replace(',', ''))
        metricas['TP'] = int(match_tp.group(1).replace(',', ''))
    
    return metricas


print("=" * 70)
print("COMPARAÇÃO ENTRE OS 3 MODELOS")
print("=" * 70)

modelos = {
    'Naive Bayes': 'resultados/nb_resultados.txt',
    'SVM': 'resultados/svm_resultados.txt',
    'Random Forest': 'resultados/rf_resultados.txt'
}

resultados = {}
for nome, caminho in modelos.items():
    if os.path.exists(caminho):
        resultados[nome] = extrair_metricas(caminho)
        print(f"\n  ✓ {nome}: resultados carregados")
    else:
        print(f"\n  ✗ {nome}: arquivo não encontrado ({caminho})")
        print(f"    Execute primeiro: python {caminho.replace('resultados/', '').replace('_resultados.txt', '.py')}")

if len(resultados) < 3:
    print("\n⚠ Nem todos os modelos foram executados. Execute os 3 scripts primeiro.")
    print("  python naive_bayes.py")
    print("  python svm.py")
    print("  python random_forest.py")
    exit(1)


# 2. TABELA COMPARATIVA

print("\n" + "=" * 70)
print("TABELA COMPARATIVA")
print("=" * 70)

header = f"{'Métrica':<15} | {'Naive Bayes':>12} | {'SVM':>12} | {'Random Forest':>14} | {'Melhor':>14}"
print(header)
print("─" * len(header))

metricas_comparar = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']

for metrica in metricas_comparar:
    vals = {nome: res.get(metrica, 0) for nome, res in resultados.items()}
    melhor = max(vals, key=vals.get)
    nb_v = vals.get('Naive Bayes', 0)
    svm_v = vals.get('SVM', 0)
    rf_v = vals.get('Random Forest', 0)
    print(f"{metrica:<15} | {nb_v:>12.4f} | {svm_v:>12.4f} | {rf_v:>14.4f} | {melhor:>14}")

# Tempos
print("─" * len(header))
for tempo_key, tempo_label in [('Tempo_Treinamento', 'Treino (s)'), ('Tempo_Predição', 'Predição (s)')]:
    vals = {nome: res.get(tempo_key, 0) for nome, res in resultados.items()}
    melhor = min(vals, key=vals.get)  # Menor tempo é melhor
    nb_v = vals.get('Naive Bayes', 0)
    svm_v = vals.get('SVM', 0)
    rf_v = vals.get('Random Forest', 0)
    print(f"{tempo_label:<15} | {nb_v:>12.2f} | {svm_v:>12.2f} | {rf_v:>14.2f} | {melhor:>14}")


# 3. GRÁFICOS COMPARATIVOS

print("\n" + "─" * 50)
print("Gerando gráficos comparativos...")

os.makedirs("resultados", exist_ok=True)
cores = ['#2196F3', '#FF5722', '#4CAF50']  # Azul, Laranja, Verde
nomes_modelos = list(resultados.keys())

# --- Gráfico 1: Barras comparativas de métricas ---
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(metricas_comparar))
width = 0.25

for i, (nome, res) in enumerate(resultados.items()):
    valores = [res.get(m, 0) for m in metricas_comparar]
    bars = ax.bar(x + i * width, valores, width, label=nome, color=cores[i],
                  edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Métrica', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Comparação de Métricas — Naive Bayes vs SVM vs Random Forest',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metricas_comparar, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resultados/comparacao_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/comparacao_metricas.png")

# --- Gráfico 2: Radar/Spider chart ---
metricas_radar = metricas_comparar
num_vars = len(metricas_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, (nome, res) in enumerate(resultados.items()):
    valores = [res.get(m, 0) for m in metricas_radar]
    valores += valores[:1]
    ax.plot(angles, valores, 'o-', linewidth=2, label=nome, color=cores[i])
    ax.fill(angles, valores, alpha=0.1, color=cores[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metricas_radar, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Perfil de Desempenho — Comparação entre Modelos',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)
plt.tight_layout()
plt.savefig('resultados/comparacao_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/comparacao_radar.png")

# --- Gráfico 3: Tempos de execução ---
tempos_treino = [resultados[n].get('Tempo_Treinamento', 0) for n in nomes_modelos]
tempos_pred = [resultados[n].get('Tempo_Predição', 0) for n in nomes_modelos]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(nomes_modelos))
width = 0.35

bars1 = ax.bar(x - width/2, tempos_treino, width, label='Treinamento', color='#42A5F5', edgecolor='white')
bars2 = ax.bar(x + width/2, tempos_pred, width, label='Predição', color='#EF5350', edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}s', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Modelo', fontsize=13)
ax.set_ylabel('Tempo (segundos)', fontsize=13)
ax.set_title('Tempo de Execução — Comparação entre Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(nomes_modelos, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resultados/comparacao_tempos.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/comparacao_tempos.png")

# --- Gráfico 4: Falsos Positivos vs Falsos Negativos ---
fig, ax = plt.subplots(figsize=(10, 6))
fps = [resultados[n].get('FP', 0) for n in nomes_modelos]
fns = [resultados[n].get('FN', 0) for n in nomes_modelos]

x = np.arange(len(nomes_modelos))
width = 0.35

bars1 = ax.bar(x - width/2, fps, width, label='Falsos Positivos (FP)', color='#FFA726', edgecolor='white')
bars2 = ax.bar(x + width/2, fns, width, label='Falsos Negativos (FN)', color='#AB47BC', edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Modelo', fontsize=13)
ax.set_ylabel('Quantidade de Erros', fontsize=13)
ax.set_title('Falsos Positivos vs Falsos Negativos — Comparação', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(nomes_modelos, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('resultados/comparacao_erros.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Salvo: resultados/comparacao_erros.png")

# 4. SALVAR RELATÓRIO COMPARATIVO

with open('resultados/comparacao_final.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("RELATÓRIO COMPARATIVO — 3 MODELOS DE CLASSIFICAÇÃO\n")
    f.write("Dataset: Suicide_Detection.csv (Detecção de Suicídio)\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("TABELA COMPARATIVA DE MÉTRICAS\n")
    f.write("─" * 70 + "\n")
    f.write(f"{'Métrica':<15} | {'Naive Bayes':>12} | {'SVM':>12} | {'Random Forest':>14}\n")
    f.write("─" * 70 + "\n")
    for metrica in metricas_comparar:
        vals = {nome: res.get(metrica, 0) for nome, res in resultados.items()}
        f.write(f"{metrica:<15} | {vals['Naive Bayes']:>12.4f} | {vals['SVM']:>12.4f} | {vals['Random Forest']:>14.4f}\n")
    f.write("─" * 70 + "\n\n")
    
    f.write("TEMPOS DE EXECUÇÃO\n")
    f.write("─" * 70 + "\n")
    for nome in nomes_modelos:
        t_treino = resultados[nome].get('Tempo_Treinamento', 0)
        t_pred = resultados[nome].get('Tempo_Predição', 0)
        f.write(f"  {nome:<15}: Treino={t_treino:.2f}s | Predição={t_pred:.2f}s\n")
    f.write("\n")
    
    f.write("ANÁLISE DE ERROS\n")
    f.write("─" * 70 + "\n")
    for nome in nomes_modelos:
        fp = resultados[nome].get('FP', 0)
        fn = resultados[nome].get('FN', 0)
        f.write(f"  {nome:<15}: FP={fp:>5,} | FN={fn:>5,} | Total Erros={fp+fn:>6,}\n")
    f.write("\n")
    
    # Determinar melhor modelo por cada métrica
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("─" * 70 + "\n")
    for metrica in metricas_comparar:
        vals = {nome: res.get(metrica, 0) for nome, res in resultados.items()}
        melhor = max(vals, key=vals.get)
        f.write(f"  {metrica:<15}: {melhor} ({vals[melhor]:.4f})\n")
    f.write("\n")

print("  → Salvo: resultados/comparacao_final.txt")

print(f"\n{'=' * 70}")
print("Comparação concluída com sucesso!")
print(f"{'=' * 70}")
