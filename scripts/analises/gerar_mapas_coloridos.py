# -*- coding: utf-8 -*-
"""
Mapas coloridos sobre a malha viaria real, para o TCC.

Substituem os graficos de dispersao em preto e branco: sobre o fundo do
OpenStreetMap o leitor reconhece bairros e vias, e enxerga que as zonas
respeitam a geografia da cidade em vez de serem recortes abstratos.

Gera tres figuras para um representante:

  mapa_zonas_regiao.png    a carteira inteira, uma cor por zona semanal
  mapa_zonas_detalhe.png   recorte do nucleo urbano, onde a densidade e alta
  mapa_semana_rota.png     uma semana sequenciada, um simbolo por dia util,
                           com a ordem de visita numerada

Os tiles vem do servidor publico do OpenStreetMap e sao guardados em cache
local, de modo que reexecucoes nao repetem o download. A politica de uso do
servidor pede identificacao e volume modesto; ambos sao respeitados.

Atencao: as coordenadas da base passaram pelo deslocamento de protecao de
ate 300 m, de modo que os pontos no mapa nao marcam o consultorio real.
"""
import io
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DIR = 'saida/resultados/figuras_tcc'
CACHE = 'saida/resultados/_tiles_osm'
OSRM = 'https://router.project-osrm.org'
BASE = 'saida/resultados/base_medicos_clusterizada.csv'
ROTAS = 'saida/resultados/rotas_vrp.csv'
REP = 'Rep_024'
UA = {'User-Agent': 'TCC-USP-Esalq-2026 (roteirizacao academica)'}
TAM = 256

os.makedirs(DIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9,
                     'figure.dpi': 200, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})

# paleta distinguivel tambem em impressao, evitando pares vermelho/verde
CORES = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
         '#8c564b', '#17becf', '#bcbd22', '#e377c2', '#7f7f7f']


# ── geometria do trajeto pela malha viaria ────────────────────────────────
GEO_CACHE = 'saida/resultados/_osrm_geom.json'
_geom = None


def _carrega_geom():
    global _geom
    if _geom is None:
        _geom = (json.load(io.open(GEO_CACHE, encoding='utf-8'))
                 if os.path.exists(GEO_CACHE) else {})
    return _geom


def traco_viario(lats, lons):
    """Coordenadas do percurso por ruas entre os pontos, na ordem dada.

    Devolve None quando o servico nao responde, caso em que o chamador
    desenha o segmento reto e o mapa deixa de prometer o que nao mediu.
    """
    if len(lats) < 2:
        return None
    c = _carrega_geom()
    k = '|'.join('%.5f,%.5f' % (a, b) for a, b in zip(lats, lons))
    if k in c:
        return c[k]
    coords = ';'.join('%f,%f' % (lo, la) for la, lo in zip(lats, lons))
    try:
        time.sleep(1.1)
        r = requests.get('%s/route/v1/driving/%s' % (OSRM, coords),
                         params={'overview': 'full', 'geometries': 'geojson',
                                 'continue_straight': 'true'}, timeout=30)
        j = r.json() if r.ok else {}
        if j.get('code') == 'Ok' and j.get('routes'):
            g = j['routes'][0]['geometry']['coordinates']
            c[k] = g
            json.dump(c, io.open(GEO_CACHE, 'w', encoding='utf-8'))
            return g
    except Exception:
        pass
    return None


def deg2num(lat, lon, z):
    la = math.radians(lat)
    n = 2.0 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(la)) / math.pi) / 2.0 * n)


def num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def baixa_tile(z, x, y):
    caminho = os.path.join(CACHE, '%d_%d_%d.png' % (z, x, y))
    if os.path.exists(caminho):
        return Image.open(caminho).convert('RGB')
    url = 'https://tile.openstreetmap.org/%d/%d/%d.png' % (z, x, y)
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    io.open(caminho, 'wb').write(r.content)
    time.sleep(0.25)                      # cortesia com o servidor publico
    return Image.open(caminho).convert('RGB')


def mosaico(lat0, lat1, lon0, lon1, z):
    """Monta o fundo cobrindo a janela pedida e devolve a imagem e os limites."""
    x0, y0 = deg2num(lat1, lon0, z)
    x1, y1 = deg2num(lat0, lon1, z)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    larg, alt = (tx1 - tx0 + 1) * TAM, (ty1 - ty0 + 1) * TAM
    img = Image.new('RGB', (larg, alt), '#f2f2f2')
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                img.paste(baixa_tile(z, tx, ty), ((tx - tx0) * TAM, (ty - ty0) * TAM))
            except Exception:
                pass
    la_n, lo_o = num2deg(tx0, ty0, z)
    la_s, lo_l = num2deg(tx1 + 1, ty1 + 1, z)
    return img, (lo_o, lo_l, la_s, la_n)


def zoom_para(lat0, lat1, lon0, lon1, alvo_tiles=5):
    for z in range(16, 7, -1):
        x0, y0 = deg2num(lat1, lon0, z)
        x1, y1 = deg2num(lat0, lon1, z)
        if (int(x1) - int(x0) + 1) <= alvo_tiles and (int(y1) - int(y0) + 1) <= alvo_tiles:
            return z
    return 9


def janela(d, q=0.98, folga=0.12):
    cx, cy = d.longitude.median(), d.latitude.median()
    r = np.hypot(d.longitude - cx, d.latitude - cy)
    raio = float(np.quantile(r, q)) * (1 + folga)
    return cy - raio, cy + raio, cx - raio, cx + raio


def escala(ax, lon0, lon1, lat, km):
    graus = km / (111.32 * math.cos(math.radians(lat)))
    x = lon0 + (lon1 - lon0) * 0.05
    y = lat
    ax.plot([x, x + graus], [y, y], color='black', lw=2.5, solid_capstyle='butt',
            zorder=9, path_effects=None)
    ax.text(x + graus / 2, y, ' %d km' % km, ha='left', va='bottom',
            fontsize=8, color='black', zorder=9,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))


def desenha(d, lat0, lat1, lon0, lon1, nome, legenda_titulo, z=None, numerar=None,
            ligar=False):
    z = z or zoom_para(lat0, lat1, lon0, lon1)
    img, (eo, el, es, en) = mosaico(lat0, lat1, lon0, lon1, z)
    fig, ax = plt.subplots(figsize=(7.0, 7.0 * (lat1 - lat0) /
                                    ((lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2)))))
    ax.imshow(np.asarray(img), extent=[eo, el, es, en], zorder=0)
    DIAS = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
    grupos = sorted(d._grupo.unique(),
                    key=lambda g: DIAS.index(g) if g in DIAS else (99, g))
    for i, g in enumerate(grupos):
        s = d[d._grupo == g]
        if ligar and 'ordem' in s.columns:
            seq = s.sort_values('ordem')
            traco = traco_viario(seq.latitude.tolist(), seq.longitude.tolist())
            if traco:
                xs, ys = zip(*traco)
            else:
                xs, ys = seq.longitude.tolist(), seq.latitude.tolist()
            ax.plot(xs, ys, color=CORES[i % len(CORES)],
                    lw=1.6, alpha=0.85, zorder=2, solid_capstyle='round')
        ax.scatter(s.longitude, s.latitude, s=34, c=CORES[i % len(CORES)],
                   edgecolor='white', linewidths=0.7, zorder=3,
                   label='%s (%d)' % (g, len(s)))
    if numerar is not None:
        for _, r in numerar.iterrows():
            ax.annotate(str(int(r.ordem)), (r.longitude, r.latitude),
                        fontsize=7, color='black', zorder=5, weight='bold',
                        xytext=(4, 4), textcoords='offset points',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('black')
    escala(ax, lon0, lon1, lat0 + (lat1 - lat0) * 0.04,
           5 if (lon1 - lon0) > 0.08 else 1)
    ax.legend(title=legenda_titulo, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8, title_fontsize=8, labelspacing=0.5)
    ax.text(1.0, -0.012, 'Base cartográfica: © colaboradores do OpenStreetMap',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            color='#333333')
    caminho = os.path.join(DIR, nome)
    fig.savefig(caminho, facecolor='white')
    plt.close(fig)
    print('  [ok] %s  (zoom %d, %d pontos)' % (caminho, z, len(d)))


def main():
    c = pd.read_csv(BASE, low_memory=False)
    d = c[c.rep_id == REP].dropna(subset=['latitude', 'longitude'])
    d = d.drop_duplicates(subset=['medico_id', 'cluster']).copy()
    d['_grupo'] = d.cluster.map(lambda k: 'Zona %d' % (sorted(d.cluster.unique()).index(k) + 1))

    la0, la1, lo0, lo1 = janela(d)
    desenha(d, la0, la1, lo0, lo1, 'mapa_zonas_regiao.png', 'Zonas semanais')

    nucleo = janela(d, q=0.55, folga=0.25)
    desenha(d, *nucleo, nome='mapa_zonas_detalhe.png', legenda_titulo='Zonas semanais')

    r = pd.read_csv(ROTAS, low_memory=False)
    rr = r[r.rep_id == REP]
    if len(rr):
        sem = rr.semana.value_counts().idxmax()
        s = rr[rr.semana == sem].dropna(subset=['latitude', 'longitude']).copy()
        s['_grupo'] = s.dia
        la0, la1, lo0, lo1 = janela(s, q=0.97, folga=0.18)
        desenha(s, la0, la1, lo0, lo1, 'mapa_semana_rota.png',
                'Dia da semana', numerar=s, ligar=True)
        print('  semana ilustrada: %s (%d visitas)' % (sem, len(s)))


if __name__ == '__main__':
    main()
