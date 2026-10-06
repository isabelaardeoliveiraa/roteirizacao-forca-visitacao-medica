# -*- coding: utf-8 -*-
"""
Distancia rodoviaria de um dia de visitas, com cache em disco.

A avaliacao contra a pratica registrada media os dois lados em linha reta.
A sequencia de visitas e conhecida dos dois lados, de modo que nada impedia
medir por malha viaria; a escolha anterior foi de custo de processamento,
nao de metodo, e ela nao e neutra. A razao entre percurso por ruas e linha
reta varia com o regime: aproxima-se de 1 em rodovia e passa de 1,4 em
malha urbana densa. Como a pratica observada tem muitas pernas longas entre
cidades e o plano do modelo e quase todo urbano, medir em linha reta tende
a favorecer o modelo.

Este modulo resolve um dia por chamada ao endpoint /route do OSRM, somando
o percurso efetivo entre visitas consecutivas na ordem dada. O resultado e
gravado em cache pelo conjunto ordenado de coordenadas, de modo que uma
reexecucao nao repete a consulta.

Em caso de falha do servico, devolve a distancia de haversine e sinaliza,
para que a proporcao de dias sem rota real possa ser relatada.
"""
import io
import json
import os
import time

import numpy as np
import requests

URL = 'https://router.project-osrm.org'
INTERVALO = 1.1          # s entre chamadas: politica do servidor publico
TIMEOUT = 25
TENTATIVAS = 3
CACHE = 'saida/resultados/_osrm_dias.json'

_cache = None
_ultima = 0.0
_stats = {'cache': 0, 'osrm': 0, 'haversine': 0}


def _carrega():
    global _cache
    if _cache is None:
        if os.path.exists(CACHE):
            _cache = json.load(io.open(CACHE, encoding='utf-8'))
        else:
            _cache = {}
    return _cache


def salvar():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(_carrega(), io.open(CACHE, 'w', encoding='utf-8'))


def haversine_seq(lats, lons):
    la, lo = np.radians(np.asarray(lats)), np.radians(np.asarray(lons))
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return float((6371.0 * 2 * np.arcsin(np.sqrt(a))).sum())


def _chave(lats, lons):
    return '|'.join('%.5f,%.5f' % (a, b) for a, b in zip(lats, lons))


def km_dia(lats, lons):
    """Quilometragem do dia pela malha viaria, na ordem dada."""
    global _ultima
    lats, lons = list(lats), list(lons)
    if len(lats) < 2:
        return 0.0
    c = _carrega()
    k = _chave(lats, lons)
    if k in c:
        _stats['cache'] += 1
        return c[k]

    coords = ';'.join('%f,%f' % (lo, la) for la, lo in zip(lats, lons))
    for tentativa in range(TENTATIVAS):
        espera = INTERVALO - (time.monotonic() - _ultima)
        if espera > 0:
            time.sleep(espera)
        try:
            r = requests.get('%s/route/v1/driving/%s' % (URL, coords),
                             params={'overview': 'false', 'continue_straight': 'true'},
                             timeout=TIMEOUT)
            _ultima = time.monotonic()
            j = r.json() if r.ok else {}
            if j.get('code') == 'Ok' and j.get('routes'):
                km = j['routes'][0]['distance'] / 1000.0
                c[k] = km
                _stats['osrm'] += 1
                if _stats['osrm'] % 25 == 0:
                    salvar()
                    print('    OSRM %d dias resolvidos' % _stats['osrm'], flush=True)
                return km
        except Exception:
            _ultima = time.monotonic()
        if tentativa < TENTATIVAS - 1:
            time.sleep(2 ** tentativa)

    km = haversine_seq(lats, lons)
    c[k] = km
    _stats['haversine'] += 1
    return km


def resumo():
    return dict(_stats)
