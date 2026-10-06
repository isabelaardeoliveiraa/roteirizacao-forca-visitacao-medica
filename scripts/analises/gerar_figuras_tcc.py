# -*- coding: utf-8 -*-
"""
Figuras do TCC, no estilo exigido pelo Manual USP/Esalq (Tabela 8):
  sem linhas de grade, sem borda, sem preenchimento, sem titulo no grafico,
  eixos em linha solida preta de 1,5 pt, fonte Arial 11 ou menor, cor preta.

Saida: saida/resultados/figuras_tcc/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = 'saida/resultados/figuras_tcc'
os.makedirs(DIR, exist_ok=True)

CINZA = '#595959'   # tons neutros: a norma nao proibe cor, mas o documento e P&B
CLARO = '#BFBFBF'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def eixos(ax):
    """Aplica a norma: sem grade, sem borda superior/direita, eixos pretos 1,5 pt."""
    ax.grid(False)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    for lado in ('left', 'bottom'):
        ax.spines[lado].set_linewidth(1.5)
        ax.spines[lado].set_color('black')
    ax.tick_params(colors='black', width=1.0)
    ax.set_title('')
    ax.set_facecolor('none')
    return ax


def salvar(fig, nome):
    fig.patch.set_alpha(0)
    caminho = os.path.join(DIR, nome)
    fig.savefig(caminho, transparent=False, facecolor='white')
    plt.close(fig)
    print('  [ok]', caminho)


# ------------------------------------------------------- Figura 1: Silhouette
def figura1():
    k = pd.read_csv('saida/resultados/kmeans_avaliacao.csv')
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(k.silhouette, bins=22, color=CINZA, edgecolor='white', linewidth=0.6)
    m = k.silhouette.mean()
    topo = ax.get_ylim()[1]
    ax.set_ylim(0, topo * 1.18)          # folga para a anotacao nao colidir
    ax.axvline(m, color='black', linestyle='--', linewidth=1.2)
    ax.annotate('média {:.3f}'.format(m), xy=(m, topo * 1.10),
                xytext=(6, 0), textcoords='offset points', fontsize=9, color='black')
    ax.set_xlabel('Coeficiente de silhueta')
    ax.set_ylabel('Número de representantes')
    eixos(ax)
    salvar(fig, 'figura1_silhouette.png')


# ------------------------------------------------------- Figura 2: zonas de um rep
def figura2(rep='Rep_024'):
    s = pd.read_csv('saida/resultados/base_medicos_score.csv')
    d = s[(s.rep_id == rep) & s.latitude.notna() & s.cluster.notna()]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    marcadores = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    tons = ['black', CINZA, CLARO]
    for i, (c, g) in enumerate(d.groupby('cluster')):
        ax.scatter(g.longitude, g.latitude, s=34,
                   marker=marcadores[i % len(marcadores)],
                   facecolor=tons[i % len(tons)], edgecolor='black',
                   linewidth=0.6, alpha=0.85,
                   label='Zona {}'.format(int(c) + 1))
    # enquadra o miolo da carteira: pontos isolados distorcem a escala
    x0, x1 = d.longitude.quantile([0.01, 0.99])
    y0, y1 = d.latitude.quantile([0.01, 0.99])
    mx, my = (x1 - x0) * 0.12 + 0.02, (y1 - y0) * 0.12 + 0.02
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y0 - my, y1 + my)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    leg = ax.legend(frameon=False, fontsize=8, loc='center left',
                    bbox_to_anchor=(1.01, 0.5))
    for t in leg.get_texts():
        t.set_color('black')
    eixos(ax)
    salvar(fig, 'figura2_zonas_representante.png')


# ------------------------------------------------------- Figura 3: ganho de sequenciamento
def figura3():
    d = pd.read_csv('saida/resultados/comparativo_sequenciamento.csv')
    faixa = pd.cut(d.n_visitas, [2, 4, 6, 8, 99], labels=['3 a 4', '5 a 6', '7 a 8', '9 ou mais'])
    g = d.groupby(faixa, observed=True).agg(o=('km_otimizado', 'sum'), r=('km_aleatorio', 'sum'))
    g = g[g.o > 0]
    x = np.arange(len(g))
    larg = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - larg / 2, g.r / 1000, larg, color=CLARO, edgecolor='black',
           linewidth=0.7, label='Ordem não otimizada')
    ax.bar(x + larg / 2, g.o / 1000, larg, color=CINZA, edgecolor='black',
           linewidth=0.7, label='Rota otimizada')
    for i, (_, row) in enumerate(g.iterrows()):
        red = 100 * (1 - row.o / row.r)
        ax.annotate('-{:.0f}%'.format(red), xy=(i, max(row.r, row.o) / 1000),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=9, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(g.index.astype(str))
    ax.set_xlabel('Visitas no dia')
    ax.set_ylabel('Distância percorrida (mil km)')
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color('black')
    eixos(ax)
    salvar(fig, 'figura3_ganho_sequenciamento.png')


# ------------------------------------------------------- Figura 4: equilibrio das zonas
def figura4():
    p = pd.read_csv('saida/resultados/comparativo_equilibrio_zonas.csv')
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(p.cv_tam, p.cv_px, s=20, facecolor='none',
               edgecolor='black', linewidth=0.7)
    lim = max(p.cv_tam.max(), p.cv_px.max()) * 1.05
    ax.plot([0, lim], [0, lim], color='black', linestyle='--', linewidth=1.0)
    ax.annotate('igualdade', xy=(lim * 0.72, lim * 0.72), xytext=(4, -12),
                textcoords='offset points', fontsize=9, color='black', rotation=45)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('CV do número de médicos por zona (%)')
    ax.set_ylabel('CV do potencial de prescrição por zona (%)')
    eixos(ax)
    salvar(fig, 'figura4_equilibrio_zonas.png')


if __name__ == '__main__':
    print('Gerando figuras em', DIR)
    figura1()
    figura2()
    figura3()
    figura4()
    print('concluido')
