# -*- coding: utf-8 -*-
"""
Figuras cartograficas do TCC, no estilo exigido pelo Manual USP/Esalq:
sem grade, sem borda, sem titulo, tons neutros.

  figura_mapa_nacional.png  dois paineis: (a) distribuicao dos medicos
                            geocodificados no territorio nacional, com a
                            malha das unidades federativas e a moldura que
                            marca o recorte do painel seguinte; (b) zonas
                            semanais da carteira de um representante, com
                            escala em quilometros.

A malha estadual vem de arquivo GeoJSON local (_brasil_uf.geojson), de modo
que a figura possa ser regerada sem acesso a rede.

Projecao: as coordenadas sao desenhadas em grau decimal, com a razao de
aspecto corrigida por 1/cos(latitude) para que a escala horizontal e a
vertical fiquem comparaveis na latitude do recorte.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection

DIR = 'saida/resultados/figuras_tcc'
MALHA = os.path.join(DIR, '_brasil_uf.geojson')
os.makedirs(DIR, exist_ok=True)

CINZA = '#595959'
BORDA = '#8C8C8C'
FUNDO = '#F2F2F2'

# representante do recorte: carteira concentrada numa praca urbana
REP_ALVO = 'Rep_024'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def poligonos_uf():
    """Aneis externos de cada unidade federativa."""
    with open(MALHA, encoding='utf-8') as f:
        gj = json.load(f)
    aneis = []
    for feat in gj['features']:
        g = feat['geometry']
        partes = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for parte in partes:
            externo = np.asarray(parte[0], dtype=float)
            if len(externo) >= 3:
                aneis.append(externo)
    return aneis


def limpa(ax, lat_media):
    """Mapa nao leva eixo: latitude e longitude nao informam o leitor aqui."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor('none')
    ax.set_aspect(1.0 / np.cos(np.radians(lat_media)), adjustable='box')


def escala_km(ax, km=5):
    """Barra de escala horizontal, canto inferior esquerdo."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lat = (y0 + y1) / 2
    graus = km / (111.32 * np.cos(np.radians(lat)))
    bx = x0 + (x1 - x0) * 0.06
    by = y0 + (y1 - y0) * 0.07
    ax.plot([bx, bx + graus], [by, by], color='black', lw=1.5,
            solid_capstyle='butt', zorder=6)
    ax.text(bx + graus / 2, by + (y1 - y0) * 0.022, '%d km' % km,
            ha='center', va='bottom', fontsize=8, color='black', zorder=6)


def carrega_medicos():
    b = pd.read_csv('saida/resultados/base_medicos_validada.csv')
    return b[b.latitude.notna() & b.longitude.notna()].copy()


def carrega_zonas():
    c = pd.read_csv('saida/resultados/base_medicos_clusterizada.csv', low_memory=False)
    d = c[(c.rep_id == REP_ALVO)].dropna(subset=['latitude', 'longitude'])
    return d.drop_duplicates(subset=['medico_id', 'cluster'])


def painel_nacional(ax, med, aneis, recorte):
    col = PatchCollection([MplPolygon(a, closed=True) for a in aneis],
                          facecolor=FUNDO, edgecolor=BORDA, linewidths=0.35, zorder=1)
    ax.add_collection(col)
    ax.scatter(med.longitude, med.latitude, s=1.4, c=CINZA,
               alpha=0.28, linewidths=0, zorder=2, rasterized=True)
    x0, x1, y0, y1 = recorte
    folga = 0.9  # moldura legivel na escala nacional
    ax.add_patch(Rectangle((x0 - folga, y0 - folga),
                           (x1 - x0) + 2 * folga, (y1 - y0) + 2 * folga,
                           fill=False, edgecolor='black', linewidth=0.9, zorder=5))
    ax.set_xlim(-74.5, -33.5)
    ax.set_ylim(-34.0, 6.5)
    limpa(ax, -14.0)


def janela_robusta(d, q=0.98):
    """Recorte pela massa dos pontos: pontos isolados comprimiriam o desenho."""
    cx, cy = d.longitude.median(), d.latitude.median()
    r = np.hypot(d.longitude - cx, d.latitude - cy)
    raio = float(np.quantile(r, q)) * 1.15
    fora = int((r > raio).sum())
    return (cx - raio, cx + raio, cy - raio, cy + raio), fora


def painel_zonas(ax, d):
    zonas = sorted(d.cluster.unique())
    marcas = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h']
    tons = np.linspace(0.10, 0.80, len(zonas))
    for i, z in enumerate(zonas):
        s = d[d.cluster == z]
        ax.scatter(s.longitude, s.latitude, s=26, marker=marcas[i % len(marcas)],
                   facecolor=str(tons[i]), edgecolor='black', linewidths=0.4,
                   zorder=3, label='Zona %d (%d)' % (i + 1, len(s)))
    (x0, x1, y0, y1), fora = janela_robusta(d)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    limpa(ax, float(d.latitude.median()))
    escala_km(ax, 5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), frameon=False,
              ncol=3, handletextpad=0.3, borderpad=0, columnspacing=1.0,
              labelspacing=0.35, fontsize=8)
    return fora


def main():
    med = carrega_medicos()
    aneis = poligonos_uf()
    d = carrega_zonas()
    recorte, _ = janela_robusta(d)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    painel_nacional(ax, med, aneis, recorte)
    fora = 0
    fig.patch.set_alpha(0)
    caminho = os.path.join(DIR, 'figura_mapa_nacional.png')
    fig.savefig(caminho, transparent=False, facecolor='white')
    plt.close(fig)
    print('  [ok] %s' % caminho)
    print('  (a) %d medicos geocodificados' % len(med))
    print('  painel unico; o recorte das zonas vai na figura sobre a malha viaria')


if __name__ == '__main__':
    main()
