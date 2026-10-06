# -*- coding: utf-8 -*-
"""O desequilíbrio de potencial entre as zonas vem do método ou do dado?

O coeficiente de variação do volume de prescrição entre as zonas de um mesmo
representante não tem, isolado, como ser interpretado: o potencial é tão
concentrado em poucos médicos que qualquer divisão produziria zonas desiguais.
A referência construída aqui reparte a mesma carteira ao acaso, mantendo o
número de zonas e os tamanhos que o agrupamento produziu, de modo que a
diferença entre as duas medidas isola o que o critério geográfico acrescentou.

Saída: saida/resultados/cv_aleatorio.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARQ = r"saida\resultados\base_medicos_score.csv"
REPETICOES = 20
SEMENTE = 42

linhas = []


def p(t=""):
    print(t)
    linhas.append(t)


def cv(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.mean() == 0 else x.std(ddof=0) / x.mean() * 100


rng = np.random.default_rng(SEMENTE)
# cluster == -1 marca o medico sem coordenada, fora de qualquer zona
d = pd.read_csv(ARQ, low_memory=False).dropna(subset=["cluster"])
d = d[d["cluster"] != -1]
d["px"] = pd.to_numeric(d["px_total_tri"], errors="coerce").fillna(0)

real, acaso = [], []
for _, g in d.groupby("rep_id"):
    tamanhos = g.groupby("cluster").size().values
    if len(tamanhos) < 2:
        continue
    real.append(cv(g.groupby("cluster")["px"].sum().values))

    px = g["px"].values
    cortes = np.cumsum(tamanhos)[:-1]
    sims = [cv([px[parte].sum() for parte in np.split(rng.permutation(len(px)), cortes)])
            for _ in range(REPETICOES)]
    acaso.append(np.nanmean(sims))

real, acaso = np.array(real, float), np.array(acaso, float)
dif = real - acaso
ok = ~np.isnan(dif)

p("=" * 74)
p("  DESEQUILÍBRIO DE POTENCIAL: DO MÉTODO OU DO DADO?")
p("=" * 74)
p(f"  Representantes avaliados : {len(real)}")
p(f"  Repetições por carteira  : {REPETICOES}")
p("")
p("  CV do volume de prescrição entre as zonas do mesmo representante")
p(f"    zoneamento geoespacial : média {np.nanmean(real):5.1f}%   "
  f"mediana {np.nanmedian(real):5.1f}%")
p(f"    partição aleatória     : média {np.nanmean(acaso):5.1f}%   "
  f"mediana {np.nanmedian(acaso):5.1f}%")
p("")
p(f"  Diferença média          : {np.nanmean(dif):+.1f} pontos percentuais")
p(f"  Representantes em que o geoespacial supera o acaso: "
  f"{int(np.nansum(dif > 0))} de {int(ok.sum())} ({np.nanmean(dif > 0) * 100:.0f}%)")
p(f"  Wilcoxon pareado         : p = {wilcoxon(real[ok], acaso[ok]).pvalue:.2e}")
p("")
p("  Leitura: a maior parte do desequilíbrio é estrutural, já presente numa")
p("  divisão ao acaso, porque o volume de prescrição se concentra em poucos")
p("  médicos. A parcela restante foi introduzida pelo critério geográfico, que")
p("  reúne na mesma zona os prescritores de alto volume por serem vizinhos.")

Path(r"saida\resultados\cv_aleatorio.txt").write_text("\n".join(linhas), encoding="utf-8")
p("\n  Salvo em saida/resultados/cv_aleatorio.txt")
