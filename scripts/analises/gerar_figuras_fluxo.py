# -*- coding: utf-8 -*-
"""
Figuras estruturais do TCC, no estilo do Manual USP/Esalq:
  Figura 1 — fluxo das etapas do pipeline (abre a Metodologia)
  Figura 6 — funil de consolidacao da base (fecha os Resultados)

Sem grade, sem borda, sem preenchimento colorido, sem titulo no grafico.
Fonte Arial, cor preta. Saida em saida/resultados/figuras_tcc/
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DIR = 'saida/resultados/figuras_tcc'
os.makedirs(DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

CINZA_CLARO = '#F2F2F2'
CINZA = '#D9D9D9'


_a_centrar = []


def caixa(ax, x, y, w, h, titulo, detalhe, fundo=CINZA_CLARO,
          tam_tit=9, tam_det=7.6, neg_tit=True, neg_det=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                linewidth=1.0, edgecolor='black', facecolor=fundo))
    t1 = ax.text(x + w / 2, y + h * 0.62, titulo, ha='center', va='center',
                 fontsize=tam_tit, fontweight='bold' if neg_tit else 'normal', color='black')
    t2 = ax.text(x + w / 2, y + h * 0.30, detalhe, ha='center', va='center',
                 fontsize=tam_det, fontweight='bold' if neg_det else 'normal', color='black')
    # o bloco titulo+detalhe tem altura variavel conforme o numero de linhas;
    # a centragem exata so e possivel depois de renderizar
    _a_centrar.append((ax, [t1, t2], y + h / 2))


def centraliza_textos(fig):
    """Desloca cada par titulo/detalhe para que o bloco fique no centro da caixa."""
    fig.canvas.draw()
    for ax, textos, y_alvo in _a_centrar:
        inv = ax.transData.inverted()
        topo, base = [], []
        for t in textos:
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            (_, y0), (_, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            base.append(y0)
            topo.append(y1)
        desloc = y_alvo - (max(topo) + min(base)) / 2
        for t in textos:
            t.set_y(t.get_position()[1] + desloc)
    _a_centrar.clear()


def seta(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=11, linewidth=1.0, color='black',
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------- Figura 1: fluxo do pipeline
def figura_fluxo():
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0.82, 10.3); ax.axis('off')

    fases = [
        ('1. Preparação', 'anonimização e\nqualidade do cadastro'),
        ('2. Localização', 'geocodificação por endereço\ne validação das coordenadas'),
        ('3. Priorização', 'índice de potencial\na partir de prescrição'),
        ('4. Zoneamento', 'k-means com\nrestrição de tamanho'),
        ('5. Roteirização', 'sequenciamento por\nprogramação por restrições'),
    ]
    w, h, x = 6.0, 1.52, 2.0
    y = 8.7
    for i, (tit, det) in enumerate(fases):
        caixa(ax, x, y, w, h, tit, det, CINZA if i % 2 == 0 else CINZA_CLARO)
        if i < len(fases) - 1:
            seta(ax, x + w / 2, y, x + w / 2, y - 0.43)
        y -= h + 0.43
    centraliza_textos(fig)
    salvar(fig, 'figura1_fluxo_pipeline.png')


# ------------------------------------------------- Figura 6: funil da base
def figura_funil():
    etapas = [
        ('Médicos no cadastro', '15.288'),
        ('Com coordenada validada', '14.880'),
        ('Tarefas de visita zoneadas', '25.213'),
        ('Distribuídas em zonas semanais', '688'),
        ('Visitas sequenciadas no ciclo', '22.536'),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0.92, 9.95); ax.axis('off')
    larguras = [8.4, 7.4, 6.4, 5.4, 4.4]
    h = 1.55
    y = 8.3
    for i, ((rot, val), lw) in enumerate(zip(etapas, larguras)):
        x = (10 - lw) / 2
        caixa(ax, x, y, lw, h, val, rot, CINZA if i % 2 == 0 else CINZA_CLARO,
              tam_tit=11, tam_det=7.8)
        if i < len(etapas) - 1:
            seta(ax, 5, y, 5, y - 0.28)
        y -= h + 0.28
    centraliza_textos(fig)
    salvar(fig, 'figura6_funil_base.png')


def salvar(fig, nome):
    caminho = os.path.join(DIR, nome)
    fig.savefig(caminho, facecolor='white')
    plt.close(fig)
    print('  [ok]', caminho)


if __name__ == '__main__':
    figura_fluxo()
    figura_funil()
