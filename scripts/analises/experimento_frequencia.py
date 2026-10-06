# -*- coding: utf-8 -*-
"""Para onde vai a capacidade que sobra depois de cobrir a carteira inteira.

Em 12 semanas a 33 visitas semanais todo representante passa por toda a sua
carteira e ainda dispõe de folga equivalente a cerca de três quartos dela. A
cobertura integral é, portanto, premissa das três políticas comparadas aqui, e
não uma delas: o que as distingue é apenas quem recebe a visita adicional.

Rodado praça a praça, porque a mesma regra rende resultados diferentes conforme
a carteira seja adensada ou espalhada.

Saída: saida/resultados/experimento_frequencia.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
import vrp

ARQ = r"saida\resultados\base_medicos_score.csv"
HORIZONTE = 12
CAPACIDADE = 33
MIN_NA_PRACA = 20
FAIXAS = ["Foco 1", "Foco 2", "Foco 3", "sem score"]

# Teto de visitas por médico no ciclo, por faixa de potencial. O piso é sempre
# 1: nenhuma política deixa médico sem visita. A sobra é distribuída até o teto,
# na ordem indicada por 'ordem'.
POLITICAS = {
    "A. Sobra uniforme (prática atual)": {
        "teto": {"Foco 1": 9, "Foco 2": 9, "Foco 3": 9, "sem score": 9},
        "ordem": None,          # nivelamento: todos a 2, depois todos a 3...
    },
    "B. Sobra para quem prescreve": {
        "teto": {"Foco 1": 6, "Foco 2": 4, "Foco 3": 2, "sem score": 1},
        "ordem": FAIXAS,        # satura o topo antes de descer
    },
    "C. Sobra escalonada por potencial": {
        "teto": {"Foco 1": 4, "Foco 2": 3, "Foco 3": 2, "sem score": 1},
        "ordem": None,          # nivelamento respeitando o teto de cada faixa
    },
}

linhas = []


def p(t=""):
    print(t)
    linhas.append(t)


df = pd.read_csv(ARQ, low_memory=False)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["foco"] = (df["acao_estrategica"].fillna("").str.extract(r"FOCO\s*(\d)")[0]
                .map(lambda f: f"Foco {f}" if pd.notna(f) else "sem score"))
geo = df.dropna(subset=["latitude", "longitude"]).copy()
geo["_cid"] = geo["CIDADE"].astype(str).str.upper()


def alocar(cart, politica, capacidade):
    """Distribui as visitas do ciclo: uma por médico, sobra conforme a política.

    Devolve a carteira com a coluna 'visitas'. A soma nunca excede a capacidade
    do ciclo, e nunca deixa médico em zero enquanto houver capacidade.
    """
    d = cart.reset_index(drop=True).copy()
    teto = politica["teto"]
    d["teto"] = d["foco"].map(teto)

    # piso de cobertura: um por médico, até onde a capacidade alcançar
    restante = capacidade - len(d)
    if restante < 0:
        d["visitas"] = 0
        ordem = d["foco"].map({f: i for i, f in enumerate(FAIXAS)}).sort_values()
        d.loc[ordem.index[:capacidade], "visitas"] = 1
        return d
    d["visitas"] = 1

    if politica["ordem"]:
        # satura faixa a faixa, de cima para baixo
        for faixa in politica["ordem"]:
            if restante <= 0:
                break
            idx = d.index[d["foco"] == faixa]
            for _ in range(teto[faixa] - 1):
                if restante <= 0:
                    break
                elegiveis = [i for i in idx if d.at[i, "visitas"] < d.at[i, "teto"]]
                n = min(len(elegiveis), restante)
                d.loc[elegiveis[:n], "visitas"] += 1
                restante -= n
    else:
        # nivelamento: sobe todo mundo um degrau por vez, respeitando o teto
        while restante > 0:
            elegiveis = d.index[d["visitas"] < d["teto"]]
            if len(elegiveis) == 0:
                break
            # quem está mais atrás na própria escala sobe primeiro
            atraso = (d.loc[elegiveis, "visitas"] / d.loc[elegiveis, "teto"])
            elegiveis = atraso.sort_values().index
            n = min(len(elegiveis), restante)
            d.loc[elegiveis[:n], "visitas"] += 1
            restante -= n
    return d


def km_ciclo(d):
    """Quilometragem do ciclo: visitas espalhadas pelas semanas, sequência por
    vizinho mais próximo dentro de cada semana."""
    semanas = {s: [] for s in range(1, HORIZONTE + 1)}
    for i, r in d.iterrows():
        n = int(r["visitas"])
        if n <= 0:
            continue
        passo = HORIZONTE / n
        for k in range(n):
            s = min(HORIZONTE, int(k * passo) + 1)
            semanas[s].append(i)
    total = 0.0
    for s, idx in semanas.items():
        if len(idx) > 1:
            sub = d.loc[idx].sort_values(["latitude", "longitude"])
            total += vrp.distancia_sequencia(sub)
    return total


def raio_mediano(sub):
    la, lo = sub.latitude.median(), sub.longitude.median()
    dx = (sub.longitude - lo) * 111 * np.cos(np.radians(la))
    dy = (sub.latitude - la) * 111
    return float(np.sqrt(dx ** 2 + dy ** 2).median())


# ── praças: cidades-sede com pelo menos dois representantes ─────────────────
sede = geo.groupby(["_cid", "rep_id"]).size().reset_index(name="n")
sede = sede[sede["n"] >= MIN_NA_PRACA]
cand = sede.groupby("_cid")["rep_id"].agg(list).reset_index()
cand = cand[cand["rep_id"].map(len) >= 2]
pracas = [(r["_cid"], r["rep_id"], geo[geo["rep_id"].isin(r["rep_id"])])
          for _, r in cand.iterrows()]
pracas.sort(key=lambda t: -len(t[2]))
pracas = pracas[:8]

p("=" * 78)
p("  PARA ONDE VAI A SOBRA DO CICLO")
p("=" * 78)
p(f"  Horizonte: {HORIZONTE} semanas   Capacidade: {CAPACIDADE} visitas/semana")
p("  Premissa comum às três políticas: todo médico visitado ao menos uma vez.")
p("")

# folga disponível em toda a base
cart_rep = geo.groupby("rep_id").size()
cap_ciclo = CAPACIDADE * HORIZONTE
folga = (cap_ciclo - cart_rep) / cart_rep * 100
p(f"  Carteira por representante : mediana {cart_rep.median():,.0f} tarefas")
p(f"  Capacidade do ciclo        : {cap_ciclo:,} visitas")
p(f"  Sobra após cobrir a carteira: mediana {folga.median():.0f}% dela "
  f"(q1 {folga.quantile(.25):.0f}%, q3 {folga.quantile(.75):.0f}%)")
p(f"  Representantes sem folga   : {(cart_rep > cap_ciclo).sum()} de {len(cart_rep)}")

p("")
p("  Teto de visitas por médico no ciclo:")
p(f"    {'política':<36}" + "".join(f"{f:>11}" for f in FAIXAS))
for nome, pol in POLITICAS.items():
    p(f"    {nome:<36}" + "".join(f"{pol['teto'][f]:>11}" for f in FAIXAS))

p("")
p(f"  {'praça':<18}{'reps':>5}{'tarefas':>9}{'% Foco 1':>10}{'raio med km':>13}")
for nome, reps, cart in pracas:
    p(f"  {nome.title():<18}{len(reps):>5}{len(cart):>9,}"
      f"{(cart['foco'] == 'Foco 1').mean() * 100:>9.1f}%{raio_mediano(cart):>12.1f}")

# ── simulação ───────────────────────────────────────────────────────────────
tab = []
for nome, reps, cart in pracas:
    L = {"praca": nome}
    for pol_nome, pol in POLITICAS.items():
        tot, por_faixa, km, nao_vis = 0, {f: 0 for f in FAIXAS}, 0.0, 0
        nf = {f: 0 for f in FAIXAS}
        for _, c in cart.groupby("rep_id"):
            d = alocar(c, pol, cap_ciclo)
            tot += int(d["visitas"].sum())
            nao_vis += int((d["visitas"] == 0).sum())
            for f, g in d.groupby("foco"):
                por_faixa[f] += int(g["visitas"].sum())
                nf[f] += len(g)
            km += km_ciclo(d)
        L[pol_nome] = {
            "visitas": tot,
            "pct_f1": por_faixa["Foco 1"] / max(tot, 1) * 100,
            "int": {f: por_faixa[f] / max(nf[f], 1) for f in FAIXAS},
            "km": km,
            "nao_visitados": nao_vis,
        }
    tab.append(L)

cols = list(POLITICAS)


def bloco(titulo, fn, fmt):
    p("")
    p("-" * 78)
    p(f"  {titulo}")
    p("-" * 78)
    p(f"  {'praça':<18}" + "".join(f"{k.split('.')[0]:>12}" for k in cols))
    for L in tab:
        p(f"  {L['praca'].title():<18}" + "".join(fmt.format(fn(L[k])) for k in cols))


bloco("MÉDICOS SEM NENHUMA VISITA NO CICLO",
      lambda a: a["nao_visitados"], "{:>12,}")
bloco("CONTATO COM O TOPO — % das visitas entregues aos Foco 1",
      lambda a: a["pct_f1"], "{:>11.1f}%")
bloco("INTENSIDADE — visitas por médico Foco 1 no ciclo",
      lambda a: a["int"]["Foco 1"], "{:>12.1f}")
bloco("INTENSIDADE — visitas por médico sem prescrição registrada",
      lambda a: a["int"]["sem score"], "{:>12.1f}")
bloco("DESLOCAMENTO — km do ciclo (proxy)",
      lambda a: a["km"], "{:>12,.0f}")

# ── leitura ─────────────────────────────────────────────────────────────────
p("")
p("=" * 78)
p("  LEITURA")
p("=" * 78)
A, B, C = cols
sem_vis = sum(L[k]["nao_visitados"] for L in tab for k in cols)
p(f"  Médicos deixados sem visita, somadas as três políticas e as oito praças: {sem_vis}")

for alvo in (B, C):
    g = [L[alvo]["pct_f1"] / max(L[A]["pct_f1"], 1e-9) for L in tab]
    k = [(L[alvo]["km"] / max(L[A]["km"], 1e-9) - 1) * 100 for L in tab]
    i = [L[alvo]["int"]["sem score"] for L in tab]
    p("")
    p(f"  {alvo}")
    p(f"    contato com o topo, frente à prática atual : mediana {np.median(g):.1f}x "
      f"(faixa {min(g):.1f}x a {max(g):.1f}x)")
    p(f"    deslocamento, frente à prática atual       : mediana {np.median(k):+.0f}% "
      f"(faixa {min(k):+.0f}% a {max(k):+.0f}%)")
    p(f"    visitas ao médico sem prescrição registrada: mediana {np.median(i):.1f} no ciclo")

ordens = {}
for L in tab:
    o = tuple(sorted(cols, key=lambda k: -L[k]["pct_f1"]))
    ordens.setdefault(o, []).append(L["praca"].title())
p("")
p("  Ordem entre as políticas pelo contato com o topo:")
for o, ps in sorted(ordens.items(), key=lambda t: -len(t[1])):
    p(f"    {' > '.join(k.split('.')[0] for k in o)}  em {len(ps)} praças: {', '.join(ps)}")

# ── por que doze semanas ────────────────────────────────────────────────────
p("")
p("=" * 78)
p("  POR QUE DOZE SEMANAS")
p("=" * 78)
p("  O ciclo precisa comportar a carteira inteira, ou a cobertura integral")
p("  deixa de ser premissa viável. A coluna decisiva é a última.")
p("")
p(f"  {'semanas':>8}{'capacidade':>12}{'sobra':>9}{'reps que não cobrem':>22}")
for h in (4, 6, 8, 10, 12, 16):
    cap = CAPACIDADE * h
    sobra = ((cap - cart_rep) / cart_rep * 100).median()
    p(f"  {h:>8}{cap:>12,}{sobra:>8.0f}%{(cart_rep > cap).sum():>22}")
p("")
p(f"  Doze semanas é o menor horizonte em que os {len(cart_rep)} representantes")
p("  percorrem a carteira integralmente. Coincide com o trimestre a que se")
p("  referem os dados de prescrição.")

Path(r"saida\resultados\experimento_frequencia.txt").write_text(
    "\n".join(linhas), encoding="utf-8")
p("\n  Salvo em saida/resultados/experimento_frequencia.txt")
