"""
Etapa 7 — Sequenciamento das visitas (OR-Tools VRP).

A unidade de planejamento é a SEMANA (5 dias × 8 visitas = 40), não a zona: as
zonas do K-Means variam de 20 a 56 médicos, e tratá-las como semanas fechadas
deixava 42% delas com dias ociosos. Aqui as zonas de cada representante são
encadeadas por proximidade e cortadas a cada 40 visitas, de modo que a emenda
sempre cai entre zonas vizinhas.

Para cada semana, o VRP distribui os médicos entre os 5 dias e define a ordem
das visitas, minimizando a distância percorrida e respeitando capacidade diária,
restrições de dia e períodos (manhã/tarde).

Distâncias vêm do OSRM (trajeto real por ruas); o haversine × fator entra apenas
como reserva se o serviço falhar — medimos que o fator fixo subestimava o
percurso em ~36%.

Semanas cuja dispersão inviabiliza a cobertura NÃO são forçadas: são
classificadas e reportadas para decisão do gestor (revisar carteira, programar
viagem ou abrir território). Ver vrp.diagnosticar_zona().

Input : saida/resultados/base_medicos_score.csv
Saídas: saida/resultados/rotas_vrp.csv     (1 linha por visita sequenciada)
        saida/resultados/vrp_avaliacao_zonas.csv   (métricas por semana)
        saida/resultados/diagnostico_carteira.csv  (semanas problemáticas)
        saida/resultados/medicos_isolados.csv      (candidatos a revisão de carteira)
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2] / "dashboard"))
import vrp

ARQ_ENTRADA = r"saida\resultados\base_medicos_score.csv"
TEMPO_POR_SEMANA = 5      # segundos de busca por semana

try:
    from lib import dispo
    REGRAS = dispo.como_dict()
except Exception:
    REGRAS = {}

print("\n" + "=" * 62)
print("  ETAPA 7 — SEQUENCIAMENTO DAS VISITAS (OR-TOOLS VRP)")
print("=" * 62)

df = pd.read_csv(ARQ_ENTRADA, dtype=str)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["zona"] = df["cluster_label"].str.extract(r"_Zona_(\d+)")[0]
df["foco"] = (df["acao_estrategica"].fillna("").str.extract(r"FOCO\s*(\d)")[0]
                .map(lambda f: f"Foco {f}" if pd.notna(f) else None))

geo = df.dropna(subset=["latitude", "longitude", "zona"])

unidades = []
for rep, d_rep in geo.groupby("rep_id"):
    for s in vrp.montar_semanas(d_rep):
        unidades.append((rep, s["semana"], s["zonas"], s["df"]))

print(f"  Representantes      : {geo['rep_id'].nunique()}")
print(f"  Zonas do K-Means    : {geo.groupby(['rep_id', 'zona']).ngroups}")
print(f"  SEMANAS a processar : {len(unidades)}")
print(f"  Restrições cadastradas: {len(REGRAS)}")
print(f"  Distância: OSRM (ruas reais), haversine como reserva")
print(f"  Busca por semana    : {TEMPO_POR_SEMANA}s\n")

rotas, avaliacao, problemas, isolados = [], [], [], []
t0 = time.time()

for i, (rep, semana, zonas_da_semana, bloco) in enumerate(unidades, 1):
    etiqueta = "+".join(str(z) for z in zonas_da_semana)
    diag = vrp.diagnosticar_zona(bloco)

    for iso in diag["isolados"]:
        isolados.append({"rep_id": rep, "semana": semana, "zonas": etiqueta, **iso})

    base_aval = {"rep_id": rep, "semana": semana, "zonas": etiqueta,
                 "n_medicos": diag["n"], "classificacao_zona": diag["classificacao"],
                 "p90_km": diag["p90_km"]}

    # Semana inviável não é forçada: seria descartar médicos em silêncio.
    if diag["classificacao"] == "inviavel_semana":
        problemas.append({
            **base_aval, "max_km": diag["max_km"],
            "horas_para_atravessar": diag["horas_para_atravessar"],
            "n_isolados": len(diag["isolados"]),
            "acao_sugerida": "Revisar carteira ou programar viagem — "
                             "dispersão incompatível com ciclo semanal",
        })
        avaliacao.append({**base_aval, "status": "inviavel_semana", "km_total": None,
                          "n_agendados": 0, "n_fora": diag["n"], "fonte_distancia": ""})
        continue

    plano, m = vrp.otimizar_zona(bloco, regras=REGRAS, tempo_limite=TEMPO_POR_SEMANA)

    if plano is None:
        avaliacao.append({**base_aval, "status": m["status"], "km_total": None,
                          "n_agendados": 0, "n_fora": len(bloco), "fonte_distancia": ""})
        continue

    for dia, b in plano.items():
        for r in b.itertuples():
            rotas.append({
                "rep_id": rep, "semana": semana, "zonas": etiqueta, "dia": dia,
                "ordem": r.ordem, "medico_id": r.medico_id,
                "papel": getattr(r, "papel", ""), "cidade": getattr(r, "CIDADE", ""),
                "bairro": getattr(r, "BAIRRO", ""),
                "latitude": r.latitude, "longitude": r.longitude,
                "chegada": vrp.hora(r.chegada_min),
                "km_acumulado": r.km_ate_aqui,
                "foco": getattr(r, "foco", None),
                "classificacao": getattr(r, "classificacao", None),
            })

    avaliacao.append({
        **base_aval, "status": m["status"], "km_total": m["km_total"],
        "n_agendados": m["n_agendados"], "n_fora": m["n_fora"],
        "fonte_distancia": m.get("fonte_distancia", ""),
        **{f"visitas_{d}": m["visitas_por_dia"][d] for d in vrp.DIAS},
    })

    if m["n_fora"] > 0:
        problemas.append({
            **base_aval, "max_km": diag["max_km"],
            "horas_para_atravessar": diag["horas_para_atravessar"],
            "n_isolados": m["n_fora"],
            "acao_sugerida": f"{m['n_fora']} médico(s) não couberam na semana — "
                             "avaliar prioridade ou dividir a zona",
        })

    if i % 50 == 0:
        print(f"    {i}/{len(unidades)} semanas | {time.time()-t0:.0f}s decorridos")

# ── Saídas ────────────────────────────────────────────────────────────────────
df_rotas = pd.DataFrame(rotas)
df_aval = pd.DataFrame(avaliacao)
df_rotas.to_csv(r"saida\resultados\rotas_vrp.csv", index=False)
df_aval.to_csv(r"saida\resultados\vrp_avaliacao_zonas.csv", index=False)

if isolados:
    pd.DataFrame(isolados).to_csv(r"saida\resultados\medicos_isolados.csv", index=False)
df_prob = pd.DataFrame(problemas)
if not df_prob.empty:
    df_prob.sort_values("p90_km", ascending=False).to_csv(
        r"saida\resultados\diagnostico_carteira.csv", index=False)

# ── Resumo ────────────────────────────────────────────────────────────────────
ok = df_aval[df_aval["status"] == "ok"]
n_osrm = int((ok["fonte_distancia"] == "osrm").sum()) if len(ok) else 0

print(f"\n{'='*62}")
print(f"  RESUMO  ({time.time()-t0:.0f}s)")
print(f"{'='*62}")
print(f"  Semanas processadas   : {len(df_aval)}")
print(f"  Semanas sequenciadas  : {len(ok)}")
print(f"  Com distância de rua  : {n_osrm} ({n_osrm/max(len(ok),1)*100:.0f}%)")
print(f"  Visitas sequenciadas  : {len(df_rotas)}")
if len(ok):
    print(f"  Distância total       : {ok['km_total'].sum():,.0f} km/ciclo")
    print(f"  Distância por semana  : {ok['km_total'].mean():.1f} km (média)")
    cheias = sum(1 for _, r in ok.iterrows()
                 if all(r.get(f"visitas_{d}", 0) > 0 for d in vrp.DIAS))
    print(f"  Semanas usando os 5 dias: {cheias} de {len(ok)} ({cheias/len(ok)*100:.0f}%)")
print(f"\n  Classificação das semanas:")
for c, n in df_aval["classificacao_zona"].value_counts().items():
    print(f"    {c:<20}: {n:>4}")
print(f"\n  DIAGNÓSTICO DE CARTEIRA")
print(f"    semanas sinalizadas : {len(df_prob)}")
print(f"    médicos isolados    : {len(isolados)}")
print(f"\n  Saídas em saida/resultados/: rotas_vrp.csv, vrp_avaliacao_zonas.csv,")
print(f"                           diagnostico_carteira.csv, medicos_isolados.csv")
