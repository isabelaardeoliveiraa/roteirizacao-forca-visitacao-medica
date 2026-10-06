# -*- coding: utf-8 -*-
"""
Fluxograma da cascata de geocodificacao (Figura 2 do TCC).

Convencao de formas:
  arredondado -> inicio e fim
  retangulo   -> processo
  losango     -> decisao

Geometria: duas colunas bem separadas (X e XB) para que nenhuma forma encoste
na vizinha; losangos escalonados em altura para que o retorno do OpenCage nao
cruze a recusa do teste de aceitacao; e juncoes explicitas onde dois caminhos
se reencontram, para nao empilhar duas pontas de seta no mesmo ponto.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

DIR = 'saida/resultados/figuras_tcc'
os.makedirs(DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'figure.dpi': 300,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

BRANCO, CLARO, MEDIO = '#FFFFFF', '#F2F2F2', '#D9D9D9'
FS = 7.0

fig, ax = plt.subplots(figsize=(6.5, 7.2))
ax.axis('off')

X, XB = 32.0, 100.0        # colunas: principal e desvio
W, WB = 54.0, 46.0


def termo(x, y, w, h, txt, fundo=MEDIO):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0,rounding_size=%.1f' % (h / 2),
                                linewidth=1.0, edgecolor='black', facecolor=fundo))
    ax.text(x, y, txt, ha='center', va='center', fontsize=FS, color='black')


def proc(x, y, w, h, txt, fundo=CLARO):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='square,pad=0',
                                linewidth=1.0, edgecolor='black', facecolor=fundo))
    ax.text(x, y, txt, ha='center', va='center', fontsize=FS, color='black')


def dec(x, y, w, h, txt, fundo=BRANCO):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, linewidth=1.0, edgecolor='black', facecolor=fundo))
    ax.text(x, y, txt, ha='center', va='center', fontsize=FS - 0.4, color='black')


def linha(pts):
    for a, b in zip(pts, pts[1:]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-', linewidth=0.9,
                                     color='black', shrinkA=0, shrinkB=0))


def seta(pts):
    linha(pts[:-1])
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=9,
                                 linewidth=0.9, color='black', shrinkA=0, shrinkB=0))


def no(x, y):
    """No de juncao: marca onde dois caminhos se fundem, para que as linhas
    sem ponta de seta nao sejam lidas como trechos inacabados."""
    ax.plot([x], [y], marker='o', markersize=3.4, color='black', zorder=5)


def rot(x, y, txt, ha='left'):
    ax.text(x, y, txt, fontsize=FS - 0.6, color='black', ha=ha, va='center')


# ── inicio e origem do endereco ───────────────────────────────────────────────
termo(X, 126, 28, 8, 'Início')
seta([(X, 122), (X, 118)])

proc(X, 112, W, 11, 'Consulta ao ViaCEP\npelo código postal')
seta([(X, 106.5), (X, 102)])

dec(X, 94, W, 16, 'ViaCEP retornou\nlogradouro?')
rot(X + 1.5, 83, 'Sim')
seta([(X, 86), (X, 80)])
proc(X, 72, W, 16, 'Endereço montado com\nlogradouro oficial e\nnúmero do cadastro')

rot(X + W / 2 + 2, 96, 'Não')
seta([(X + W / 2, 94), (XB, 94), (XB, 80)])
proc(XB, 72, WB, 16, 'Endereço montado\nsomente com dados\ndo cadastro')

# juncao dos dois enderecos antes da geocodificacao
linha([(X, 64), (X, 59)])
linha([(XB, 64), (XB, 59), (X, 59)])
no(X, 59)
seta([(X, 59), (X, 55)])

# ── geocodificacao: servico aberto e, se falhar, comercial ───────────────────
proc(X, 50, W, 9, 'Consulta ao Nominatim')
seta([(X, 45.5), (X, 41)])

dec(X, 33, W, 16, 'Nominatim\nretornou coordenada?')

rot(X + W / 2 + 2, 35, 'Não')
seta([(X + W / 2, 33), (XB, 33), (XB, 24.5)])
proc(XB, 20, WB, 9, 'Consulta ao OpenCage')
seta([(XB, 15.5), (XB, 12)])
dec(XB, 4, WB + 6, 16, 'OpenCage\nretornou coordenada?')

# juncao antes do teste: Nominatim desce pela coluna, OpenCage entra pela direita
rot(X + 1.5, 16, 'Sim')
linha([(X, 25), (X, -2)])
rot(XB - (WB + 6) / 2 - 2, 6.5, 'Sim', ha='right')
linha([(XB - (WB + 6) / 2, 4), (XB - (WB + 6) / 2, -2), (X, -2)])
no(X, -2)
seta([(X, -2), (X, -6)])

# ── teste de aceitacao e desfechos ───────────────────────────────────────────
dec(X, -16, W + 2, 20, 'Aprovada na validação\ngeográfica e no\nteste municipal?')
rot(X + 1.5, -29, 'Sim')
seta([(X, -26), (X, -32)])
termo(X, -39, W, 12, 'Geo-masking de 300 m\ne registro da coordenada')

rot(X + (W + 2) / 2 + 2, -14, 'Não')
linha([(X + (W + 2) / 2, -16), (XB, -16)])   # recusa do teste de aceitacao
rot(XB + 2, -9, 'Não')
linha([(XB, -4), (XB, -16)])                 # recusa do OpenCage
no(XB, -16)
seta([(XB, -16), (XB, -32)])
termo(XB, -39, WB + 6, 12, 'Mantém logradouro\nou centro do bairro', CLARO)

ax.set_xlim(0, 132)
ax.set_ylim(-47, 132)
caminho = os.path.join(DIR, 'figura2_fluxograma_geocodificacao.png')
fig.savefig(caminho, facecolor='white')
plt.close(fig)
print('[ok]', caminho)
