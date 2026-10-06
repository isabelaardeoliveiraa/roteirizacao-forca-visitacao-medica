"""O que a operação REALMENTE faz — medido no histórico de visitas do CRM.

Mede o que a equipe faz de fato, em vez de assumir a meta de 8 visitas por dia.

Este script mede, no `task_report`:
  · visitas por dia efetivamente realizadas, por representante
  · duração declarada de cada visita
  · horário de início e fim da jornada
  · quantos dias úteis o representante de fato trabalha na rua

O objetivo não é validar a meta de 8, e sim descobrir qual é o número real —
e se ele é o mesmo em toda praça ou varia com a densidade da carteira.

Saída: saida/resultados/diagnostico_visitas_crm.txt (+ eco no console)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ARQ = Path(r"dados_crm\2807\crm_task_report.csv")
SAIDA = Path(r"saida\resultados\diagnostico_visitas_crm.txt")

linhas = []
def p(t=""):
    print(t)
    linhas.append(t)

def titulo(t):
    p("\n" + "=" * 70)
    p(f"  {t}")
    p("=" * 70)


t = pd.read_csv(ARQ, sep=";", dtype=str)
p(f"Registros no task_report: {len(t):,}")

# As datas vêm em ISO (2026-07-15 08:45:00). NÃO usar dayfirst aqui: com ele o
# pandas descarta 12.500 das 20.207 linhas e embaralha as demais.
t["inicio"] = pd.to_datetime(t["taskstart"], errors="coerce")
t["fim"] = pd.to_datetime(t["taskfinish"], errors="coerce")
t["data"] = t["inicio"].dt.date

titulo("O QUE EXISTE NA BASE")
p(f"  Situação da tarefa:")
for k, n in t["taskstate"].value_counts().items():
    p(f"    {str(k)[:40]:<42}{n:>7,}")
p("")
p(f"  Tipo de tarefa:")
for k, n in t["tasktype"].value_counts().head(8).items():
    p(f"    {str(k)[:40]:<42}{n:>7,}")
p("")
p(f"  Período coberto: {t['inicio'].min():%d/%m/%Y} a {t['inicio'].max():%d/%m/%Y}")
p(f"  Representantes  : {t['owner'].nunique()}")
p("")
p("  Volume por mês (o registro só engrena a partir de mai/2026):")
for k, n in t["inicio"].dt.to_period("M").value_counts().sort_index().items():
    p(f"    {k}  {n:>6,}  {'█' * int(n / 8278 * 40)}")

# Só o que foi de fato realizado conta como comportamento observado. Tarefa
# planejada e não executada diz o que se pretendia, não o que se faz.
realizadas = t[t["taskstate"].astype(str).str.contains("✅|realiz|conclu|complet",
                                                       case=False, regex=True, na=False)]
if realizadas.empty:                      # rótulo inesperado: usa tudo e avisa
    realizadas = t
    p("\n  ATENÇÃO: não foi possível identificar o rótulo de 'realizada' — "
      "a análise abaixo usa TODAS as tarefas.")
p(f"\n  Tarefas consideradas realizadas: {len(realizadas):,} "
  f"({len(realizadas)/len(t)*100:.1f}%)")

# Antes de mai/2026 o registro é esparso — dezenas de visitas por mês numa
# operação de 100 representantes. Medir comportamento ali descreve a adoção do
# CRM, não a rotina. O recorte fica nos meses de operação plena.
PERIODO = ("2026-05-01", "2026-08-01")
realizadas = realizadas[realizadas["inicio"].between(*PERIODO)]
p(f"  Recorte para medir rotina: {PERIODO[0]} a {PERIODO[1]} "
  f"-> {len(realizadas):,} visitas")


# ── Duração da visita ─────────────────────────────────────────────────────────
titulo("DURAÇÃO DA VISITA")
d = realizadas.dropna(subset=["inicio", "fim"]).copy()
d["min"] = (d["fim"] - d["inicio"]).dt.total_seconds() / 60
# Duração negativa ou absurda é erro de preenchimento, não visita longa.
val = d[(d["min"] > 0) & (d["min"] <= 240)]
p(f"  Visitas com início e fim preenchidos: {len(val):,}")
if len(val):
    p("")
    for q in [10, 25, 50, 75, 90]:
        p(f"    percentil {q:>2}: {np.percentile(val['min'], q):>6.0f} min")
    p(f"    média      : {val['min'].mean():>6.0f} min")
    p("")
    p(f"  Parâmetro usado hoje no modelo: 30 min")
    p(f"  Mediana observada             : {val['min'].median():.0f} min")


# ── Visitas por dia ───────────────────────────────────────────────────────────
titulo("VISITAS POR DIA — o que a equipe faz de fato")
dia = realizadas.dropna(subset=["data"]).groupby(["owner", "data"]).size()
p(f"  Duplas representante-dia observadas: {len(dia):,}")
p("")
for q in [10, 25, 50, 75, 90, 95]:
    p(f"    percentil {q:>2}: {np.percentile(dia, q):>5.1f} visitas/dia")
p(f"    média      : {dia.mean():>5.1f}")
p(f"    máximo     : {dia.max():>5.0f}")
p("")
p("  Distribuição:")
for n, qtd in dia.value_counts().sort_index().head(16).items():
    barra = "█" * int(qtd / max(dia.value_counts().max(), 1) * 40)
    p(f"    {n:>2} visita(s): {qtd:>5,}  {barra}")
p("")
p(f"  Meta atual do modelo: 8 visitas/dia")
p(f"  Mediana observada   : {np.median(dia):.0f}")
p(f"  Dias com 8 ou mais  : {(dia >= 8).sum():,} de {len(dia):,} "
  f"({(dia >= 8).mean()*100:.1f}%)")


# ── Jornada ───────────────────────────────────────────────────────────────────
titulo("JORNADA — quando começa e quando termina")
j = realizadas.dropna(subset=["inicio", "data"]).groupby(["owner", "data"]).agg(
    primeira=("inicio", "min"), ultima=("inicio", "max"), n=("inicio", "size"))
j["h_ini"] = j["primeira"].dt.hour + j["primeira"].dt.minute / 60
j["h_fim"] = j["ultima"].dt.hour + j["ultima"].dt.minute / 60
cheios = j[j["n"] >= 5]                    # dias de rua de verdade
p(f"  Dias com 5+ visitas: {len(cheios):,}")
if len(cheios):
    p("")
    p(f"    início mediano : {int(cheios['h_ini'].median()):02d}h"
      f"{int(cheios['h_ini'].median() % 1 * 60):02d}")
    p(f"    última visita  : {int(cheios['h_fim'].median()):02d}h"
      f"{int(cheios['h_fim'].median() % 1 * 60):02d}")
    p(f"    amplitude p10-p90: {cheios['h_ini'].quantile(.1):.1f}h "
      f"às {cheios['h_fim'].quantile(.9):.1f}h")
    p("")
    p(f"  Parâmetro usado hoje no modelo: jornada 8h–18h")


# ── Variação entre representantes ─────────────────────────────────────────────
titulo("VARIA ENTRE REPRESENTANTES? — a pergunta que decide a meta única")
por_rep = realizadas.dropna(subset=["data"]).groupby(
    ["owner", "data"]).size().groupby("owner").agg(
    mediana_dia="median", dias="size", total="sum")
ativos = por_rep[por_rep["dias"] >= 10]
p(f"  Representantes com 10+ dias registrados: {len(ativos)}")
if len(ativos):
    p("")
    for q in [10, 25, 50, 75, 90]:
        p(f"    percentil {q:>2}: {np.percentile(ativos['mediana_dia'], q):>5.1f} visitas/dia")
    p("")
    p(f"  Menor mediana individual: {ativos['mediana_dia'].min():.0f} visitas/dia")
    p(f"  Maior mediana individual: {ativos['mediana_dia'].max():.0f} visitas/dia")
    p("")
    p("  Se a variação for grande, uma meta única de 8 para todos não descreve")
    p("  a operação — a meta precisaria ser por representante ou por densidade.")
    p("")
    p("  Os 10 que mais fazem:")
    for o, r in ativos.nlargest(10, "mediana_dia").iterrows():
        p(f"    {str(o)[:34]:<36} {r['mediana_dia']:>4.0f}/dia  "
          f"({int(r['dias'])} dias, {int(r['total'])} visitas)")
    p("")
    p("  Os 10 que menos fazem:")
    for o, r in ativos.nsmallest(10, "mediana_dia").iterrows():
        p(f"    {str(o)[:34]:<36} {r['mediana_dia']:>4.0f}/dia  "
          f"({int(r['dias'])} dias, {int(r['total'])} visitas)")


# ── Ritmo semanal: o que o modelo assume x o que se registra ─────────────────
titulo("RITMO SEMANAL — a comparação que mais importa")
sem = realizadas.dropna(subset=["data"]).copy()
sem["sem"] = pd.to_datetime(sem["data"]).dt.to_period("W")
dias_sem = sem.groupby(["owner", "sem"])["data"].nunique()
vis_sem = sem.groupby(["owner", "sem"]).size()
p(f"  Duplas representante-semana: {len(vis_sem):,}")
p("")
p(f"  DIAS de campo por semana : mediana {dias_sem.median():.0f} · "
  f"média {dias_sem.mean():.1f} · p90 {dias_sem.quantile(.9):.0f}")
p(f"  VISITAS por semana       : mediana {vis_sem.median():.0f} · "
  f"média {vis_sem.mean():.1f} · p90 {vis_sem.quantile(.9):.0f}")
p("")
p("  O modelo planeja 5 dias e 40 visitas por semana.")
p("")
p("  Distribuição de dias de campo por semana:")
for n, q in dias_sem.value_counts().sort_index().items():
    p(f"    {n} dia(s): {q:>5,} ({q/len(dias_sem)*100:>5.1f}%)")


# ── Frequência e cobertura ────────────────────────────────────────────────────
titulo("FREQUÊNCIA E COBERTURA DA CARTEIRA")
freq = realizadas.dropna(subset=["contact_id"]).groupby("contact_id").size()
p(f"  Médicos com ao menos 1 visita no período: {len(freq):,}")
p(f"  Visitas por médico: mediana {freq.median():.0f} · "
  f"média {freq.mean():.1f} · máx {freq.max()}")
p("")
p("  Quantas vezes cada médico foi visitado:")
for n, q in freq.value_counts().sort_index().head(8).items():
    p(f"    {n}x: {q:>5,} médicos")
p("")
p("  O modelo visita CADA médico exatamente 1 vez por ciclo, sem distinção")
p("  de prioridade. O histórico mostra frequência desigual — o que sugere que")
p("  a priorização real acontece na FREQUÊNCIA, não na ordem da rota.")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
SAIDA.write_text("\n".join(linhas), encoding="utf-8")
p(f"\n  Relatório salvo em {SAIDA}")
