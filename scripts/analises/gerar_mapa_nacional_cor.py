# -*- coding: utf-8 -*-
"""
Mapa nacional das carteiras: cada medico aparece na cor do representante
responsavel.

A versao anterior tingia os estados pelo numero de medicos e dizia pouco
alem do obvio. Colorir por representante mostra o que o trabalho de fato
organiza: o territorio repartido em carteiras. O leitor ve os nucleos de
cor, a sobreposicao nas pracas grandes e os satelites distantes que o
sequenciamento precisa acomodar.

Nao ha legenda por representante, e nao faz falta: sao 109, e a figura nao
serve para identificar um deles, mas para mostrar o mosaico.

Saida: saida/resultados/figuras_tcc/figura_mapa_nacional.png
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DIR = 'saida/resultados/figuras_tcc'
MALHA = os.path.join(DIR, '_brasil_uf.geojson')
CLUSTER = 'saida/resultados/base_medicos_clusterizada.csv'
REP_ALVO = 'Rep_024'

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
                     'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})


def main():
    c = pd.read_csv(CLUSTER, low_memory=False)
    d = (c.dropna(subset=['latitude', 'longitude', 'rep_id'])
          .drop_duplicates(subset=['medico_id', 'rep_id']))
    # mesmo criterio do zoneamento: raio mediano da carteira acima de 300 km
    def raio_mediano(g):
        la0 = float(g.latitude.median()); lo0 = float(g.longitude.median())
        return float(np.median(np.hypot((g.latitude - la0) * 111.32,
                                        (g.longitude - lo0) * 111.32
                                        * np.cos(np.radians(la0)))))
    raios = d.groupby('rep_id').apply(raio_mediano, include_groups=False)
    nacionais = set(raios[raios > 300].index)
    locais = d[~d.rep_id.isin(nacionais)]
    remotos = d[d.rep_id.isin(nacionais)]
    reps = sorted(locais.rep_id.unique())
    # ciclo de matizes bem espalhado: carteiras vizinhas tendem a cair em
    # cores distantes, o que evita que duas pracas proximas se confundam
    passo = 0.61803398875                      # razao aurea
    cor = {r: plt.cm.hsv((i * passo) % 1.0) for i, r in enumerate(reps)}

    gj = json.load(io.open(MALHA, encoding='utf-8'))
    aneis = []
    for feat in gj['features']:
        g = feat['geometry']
        partes = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for parte in partes:
            a = np.asarray(parte[0], dtype=float)
            if len(a) >= 3:
                aneis.append(a)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.add_collection(PatchCollection([MplPolygon(a, closed=True) for a in aneis],
                                      facecolor='white', edgecolor='#9a9a9a',
                                      linewidths=0.45, zorder=1))
    ax.scatter(remotos.longitude, remotos.latitude, s=2.6, c='#c3c3c3',
               alpha=0.8, linewidths=0, zorder=2, rasterized=True)
    ax.scatter(locais.longitude, locais.latitude, s=3.2,
               c=[cor[r] for r in locais.rep_id], alpha=0.78,
               linewidths=0, zorder=3, rasterized=True)

    z = locais[locais.rep_id == REP_ALVO]
    cx, cy = z.longitude.median(), z.latitude.median()
    r = float(np.quantile(np.hypot(z.longitude - cx, z.latitude - cy), 0.98)) * 1.15
    ax.add_patch(Rectangle((cx - r - 0.9, cy - r - 0.9), 2 * (r + 0.9), 2 * (r + 0.9),
                           fill=False, edgecolor='black', linewidth=1.1, zorder=5))

    ax.set_xlim(-74.5, -33.5)
    ax.set_ylim(-34.0, 6.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect(1.0 / np.cos(np.radians(-14.0)), adjustable='box')
    ax.text(0.01, 0.035, 'Uma cor por representante (%d carteiras locais)' % len(reps),
            transform=ax.transAxes, fontsize=7.5, color='#333333')
    ax.text(0.01, 0.005, 'Em cinza, os %d representantes de atuação nacional, '
            'excluídos do zoneamento' % len(nacionais),
            transform=ax.transAxes, fontsize=7.5, color='#666666')

    caminho = os.path.join(DIR, 'figura_mapa_nacional.png')
    fig.savefig(caminho, facecolor='white')
    plt.close(fig)
    print('  [ok] %s' % caminho)
    print('  %s médicos em %d carteiras locais; %s pontos de %d reps nacionais'
          % (format(len(locais), ',d'), len(reps), format(len(remotos), ',d'), len(nacionais)))


if __name__ == '__main__':
    main()
