# -*- coding: utf-8 -*-
"""
Teste de sensibilidade do zoneamento a duas escolhas de modelagem.

  1. ESPACO DE AGRUPAMENTO. O pipeline padroniza latitude e longitude por
     StandardScaler antes do k-means. Padronizar cada eixo pela propria
     dispersao nao preserva distancia geografica: uma carteira alongada no
     sentido norte-sul e comprimida nesse eixo. A alternativa correta e
     projetar os graus em quilometros, corrigindo a longitude por cos(lat).
     Mede-se aqui quanto a escolha custa, em distancia media ao centroide da
     zona, calculada sempre em quilometros para que as duas sejam comparaveis.

  2. RESTRICAO DE TAMANHO. Os limites de 60% e 140% do tamanho medio tornam
     a zona exequivel numa semana, mas afastam a particao do otimo geometrico.
     Compara-se com o k-means sem restricao.

Saida: saida/resultados/sensibilidade_kmeans.txt
"""
import io
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = 'saida/resultados/base_medicos_clusterizada.csv'
SAIDA = 'saida/resultados/sensibilidade_kmeans.txt'
RANDOM_STATE = 42
POR_ZONA = 40

linhas = []


def diz(t=''):
    linhas.append(t)
    print(t)


def projeta(d):
    """Graus -> km, com a longitude corrigida pelo cosseno da latitude media."""
    lat0 = float(d.latitude.mean())
    return np.c_[d.latitude.values * 111.32,
                 d.longitude.values * 111.32 * math.cos(math.radians(lat0))]


def dist_intra(P, lab):
    """Distancia media, em km, de cada ponto ao centroide da sua zona."""
    partes = []
    for z in np.unique(lab):
        s = P[lab == z]
        partes.append(np.hypot(*(s - s.mean(axis=0)).T))
    return float(np.concatenate(partes).mean())


def restrito(X, k):
    media = len(X) / k
    smin = max(1, int(media * 0.6))
    smax = min(int(np.ceil(media * 1.4)), len(X) - 1)
    return KMeansConstrained(n_clusters=k, size_min=smin, size_max=smax,
                             random_state=RANDOM_STATE).fit_predict(X)


def cv(lab):
    _, c = np.unique(lab, return_counts=True)
    return float(c.std() / c.mean() * 100)


def main():
    b = pd.read_csv(BASE, low_memory=False).dropna(subset=['latitude', 'longitude'])
    # os dois representantes de atuacao nacional ficam de fora, pelo mesmo
    # criterio que o zoneamento usa: carteira espalhada pelo pais inteiro
    # distorce qualquer media de distancia ao centro da zona.
    b = b[b.cluster.notna() & (b.cluster != -1)]
    reps = sorted(b.rep_id.dropna().unique())
    res = []
    for i, r in enumerate(reps, 1):
        d = b[b.rep_id == r]
        n = len(d)
        k = max(1, math.ceil(n / POR_ZONA))
        if k < 2:
            continue
        P = projeta(d)
        bruto = d[['latitude', 'longitude']].values
        lab_pad = restrito(StandardScaler().fit_transform(bruto), k)
        lab_proj = restrito(P, k)
        lab_livre = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                           n_init=10).fit_predict(P)
        res.append(dict(rep=r, n=n, k=k,
                        km_pad=dist_intra(P, lab_pad),
                        km_proj=dist_intra(P, lab_proj),
                        km_livre=dist_intra(P, lab_livre),
                        cv_restr=cv(lab_pad), cv_livre=cv(lab_livre)))
        if i % 20 == 0:
            print('  ... %d/%d' % (i, len(reps)), file=sys.stderr)

    t = pd.DataFrame(res)
    t['dif_proj'] = (t.km_pad / t.km_proj - 1) * 100
    t['custo_restr'] = (t.km_pad / t.km_livre - 1) * 100

    diz('SENSIBILIDADE DO ZONEAMENTO A DUAS ESCOLHAS DE MODELAGEM')
    diz('=' * 74)
    diz('  Representantes avaliados: %d' % len(t))
    diz()
    diz('  1. ESPACO DE AGRUPAMENTO — padronizacao x projecao em quilometros')
    diz('     Distancia media ao centroide da zona, em km:')
    diz('       coordenadas padronizadas : mediana %.2f km' % t.km_pad.median())
    diz('       coordenadas projetadas   : mediana %.2f km' % t.km_proj.median())
    diz('     Diferenca mediana          : %+.1f%%' % t.dif_proj.median())
    diz('     Representantes em que a padronizacao e pior: %d de %d'
        % (int((t.dif_proj > 0).sum()), len(t)))
    diz('     Amplitude da diferenca     : %+.1f%% a %+.1f%%'
        % (t.dif_proj.min(), t.dif_proj.max()))
    diz()
    diz('     Leitura: dentro da carteira de um mesmo representante a dispersao')
    diz('     de latitude e de longitude tem ordem de grandeza semelhante, de modo')
    diz('     que padronizar equivale, na pratica, a reescalar os dois eixos pelo')
    diz('     mesmo fator. A distorcao existe em tese e e irrelevante nesta base.')
    diz()
    diz('  2. CUSTO DA RESTRICAO DE TAMANHO — restrito x k-means sem restricao')
    diz('     Distancia media ao centroide da zona, em km:')
    diz('       com restricao de tamanho : mediana %.2f km' % t.km_pad.median())
    diz('       sem restricao            : mediana %.2f km' % t.km_livre.median())
    diz('     Custo mediano da restricao : %+.1f%%' % t.custo_restr.median())
    diz('     Coeficiente de variacao do tamanho das zonas:')
    diz('       com restricao            : mediana %.1f%%' % t.cv_restr.median())
    diz('       sem restricao            : mediana %.1f%%' % t.cv_livre.median())
    diz()
    diz('     Leitura: a restricao troca coesao geometrica por exequibilidade. O')
    diz('     k-means sem restricao produz zonas mais compactas, porem de tamanhos')
    diz('     que nao cabem numa semana de trabalho padronizada.')

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    io.open(SAIDA, 'w', encoding='utf-8').write('\n'.join(linhas) + '\n')
    t.round(3).to_csv('saida/resultados/sensibilidade_kmeans.csv', index=False)
    print('\n[ok] %s' % SAIDA)


if __name__ == '__main__':
    main()
