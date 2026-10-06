"""
Agrupamento geoespacial de médicos via K-Means — por representante (base do CRM).

Para cada representante, agrupa em zonas geográficas TODOS os médicos da carteira
COMBINADA (rep_facial + rep_nao_facial). k é calculado dinamicamente por rep:
  k = ceil(n / VISITAS_ZONA)  onde VISITAS_ZONA = META_DIARIA × DIAS_SEMANA = 40
Assim cada zona tem ~40 médicos → ~8 visitas/dia ao longo de 1 semana.

Reps com carteira grande recebem mais zonas (ciclo > 1 mês); reps com carteira
pequena recebem menos zonas. O ciclo completo cobre toda a carteira em k semanas.

Cada médico pode gerar até 2 "tarefas" (uma para o rep_facial, outra para o
rep_nao_facial, se forem reps diferentes) — saída em formato longo (medico x rep x papel).

Input : saida/resultados/base_medicos_final.csv  (saída do etapa_04)
Saídas: saida/resultados/base_medicos_clusterizada.csv  (1 linha por tarefa medico x rep)
        saida/resultados/kmeans_avaliacao.csv     (silhouette + k por representante)
        saida/resultados/mapa_clusters.html       (mapa interativo — todos os reps)
"""

import math
import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import folium
import warnings
warnings.filterwarnings("ignore")

# ── Configurações ──────────────────────────────────────────────────────────────
# Entrada é a base VALIDADA (etapa 4c), não a saída bruta da geocodificação:
# coordenada em município errado distorce as zonas e depois trava o VRP.
ARQUIVO          = r"saida\resultados\base_medicos_validada.csv"
META_DIARIA      = 8        # visitas/dia confirmado pelo especialista de negócio
DIAS_SEMANA      = 5        # dias úteis por semana
VISITAS_ZONA     = META_DIARIA * DIAS_SEMANA   # 40 médicos por zona
MIN_MEDICOS_REP  = 20       # reps com menos médicos que isso não são clusterizados
RANDOM_STATE     = 42

# Remoto = carteira sem base local. Detectado pelo RAIO MEDIANO (metade dos médicos
# está a mais de 300 km do centro da carteira), não por lista fixa nem pelo campo
# direction: HIBRIDO não implica dispersão (Rep_122 é HIBRIDO com raio de 12 km).
# O p90 seria o critério errado — ele pega reps com núcleo denso + satélites reais,
# que devem ser roteirizados normalmente (a zona distante vira uma viagem periódica).
RAIO_MEDIANO_REMOTO_KM = 300

# Filtro de médicos arquivados (regra definida pela área comercial):
#  - impedimento factual  -> exclui sempre, mesmo com alto potencial
#  - decisão comercial    -> exclui, salvo se Potencial de Produtos = ALTO (coluna AE)
#  - afastamento temporário -> exclui do ciclo atual (o médico retorna depois)
MOTIVOS_FACTUAIS   = ["Falecido", "Deixou o país", "Duplicado", "Aposentadoria"]
MOTIVOS_TEMPORARIOS = ["Licença maternidade", "Licença médica"]
ARQ_PRIORIZACAO    = r"entrada\00. Priorização CRMs - CONSOLIDADO.xlsx"
ARQ_CORRESP        = r"saida\resultados\correspondencia_contactid.csv"
# ───────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(ARQUIVO, dtype=str)
df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# ── Filtro de médicos arquivados ──────────────────────────────────────────────
n_antes = len(df)
motivo = df.get("motivo_inativo", pd.Series("", index=df.index)).fillna("").str.strip()

# Potencial de Produtos (coluna AE da planilha), buscado por CRM+UF
try:
    corresp = pd.read_csv(ARQ_CORRESP, dtype=str)
    xl = pd.ExcelFile(ARQ_PRIORIZACAO)
    abas = [s for s in xl.sheet_names
            if "FACIAL" in s.upper() or ("PRIORIZ" in s.upper() and "FACIAL" not in s.upper())]
    pri = pd.concat([pd.read_excel(xl, sheet_name=s, usecols=["CRM", "Potencial de Produtos"],
                                    dtype=str) for s in abas])
    pri["CRM"] = pri["CRM"].astype(str).str.strip().str.upper()
    alto_crm = set(pri.loc[pri["Potencial de Produtos"].astype(str).str.upper()
                              .str.contains("ALTO", na=False), "CRM"])
    crm_por_medico = dict(zip(corresp["medico_id"], corresp.get("crm_uf", pd.Series(dtype=str))))
    potencial_alto = df["medico_id"].map(crm_por_medico).isin(alto_crm)
except Exception as e:
    print(f"  AVISO: nao foi possivel ler o potencial ({e}). Arquivados serao todos excluidos.")
    potencial_alto = pd.Series(False, index=df.index)

e_factual   = motivo.isin(MOTIVOS_FACTUAIS)
e_temporario = motivo.isin(MOTIVOS_TEMPORARIOS)
e_comercial = (motivo != "") & ~e_factual & ~e_temporario

manter = ~(e_factual | e_temporario | (e_comercial & ~potencial_alto))
df["arquivado_mantido"] = e_comercial & potencial_alto   # sinalizados p/ o gestor

print(f"\n  FILTRO DE MEDICOS ARQUIVADOS")
print(f"    impedimento factual (excluidos)      : {int(e_factual.sum()):>5}")
print(f"    afastamento temporario (excluidos)   : {int(e_temporario.sum()):>5}")
print(f"    decisao comercial, sem alto potencial: {int((e_comercial & ~potencial_alto).sum()):>5}  excluidos")
print(f"    decisao comercial, POTENCIAL ALTO    : {int((e_comercial & potencial_alto).sum()):>5}  MANTIDOS (sinalizados)")
df = df[manter].copy()
print(f"    base: {n_antes} -> {len(df)} medicos")

# ── Monta carteira combinada (rep_facial + rep_nao_facial) ──────────────────────
# Cada médico gera 1 linha por papel (facial / nao_facial). Quando o mesmo rep
# acumula os dois papeis para o médico, ele entra só uma vez (evita duplicar
# a tarefa de visita).

df_facial = df[df["rep_facial"].notna()].copy()
df_facial["rep_id"] = df_facial["rep_facial"]
df_facial["papel"]  = "facial"

df_naofac = df[df["rep_nao_facial"].notna() &
               (df["rep_nao_facial"] != df["rep_facial"])].copy()
df_naofac["rep_id"] = df_naofac["rep_nao_facial"]
df_naofac["papel"]  = "nao_facial"

df_long = pd.concat([df_facial, df_naofac], ignore_index=True)

total         = len(df)
total_tarefas = len(df_long)
sem_coord     = df_long["latitude"].isna().sum()
df_geo        = df_long.dropna(subset=["latitude", "longitude"]).copy()

# ── Detecta reps sem base local (raio mediano da carteira) ────────────────────
REPS_REMOTOS = []
_dispersao = []
for _rep in df_geo["rep_id"].dropna().unique():
    _d = df_geo[df_geo["rep_id"] == _rep]
    if len(_d) < MIN_MEDICOS_REP:
        continue
    _clat, _clon = _d["latitude"].median(), _d["longitude"].median()
    _km = np.hypot((_d["latitude"] - _clat) * 111.32,
                   (_d["longitude"] - _clon) * 111.32 * np.cos(np.radians(_clat)))
    _raio = float(_km.median())
    _dispersao.append({"rep_id": _rep, "n": len(_d), "raio_mediano_km": round(_raio, 1)})
    if _raio > RAIO_MEDIANO_REMOTO_KM:
        REPS_REMOTOS.append(_rep)

print(f"\n  REPS SEM BASE LOCAL (raio mediano > {RAIO_MEDIANO_REMOTO_KM} km): {len(REPS_REMOTOS)}")
for _r in sorted(REPS_REMOTOS):
    _i = next(x for x in _dispersao if x["rep_id"] == _r)
    print(f"    {_r}: {_i['n']} tarefas | raio mediano {_i['raio_mediano_km']} km")
pd.DataFrame(_dispersao).sort_values("raio_mediano_km", ascending=False).to_csv(
    r"saida\resultados\dispersao_representantes.csv", index=False)

print(f"\n{'='*60}")
print(f"  K-MEANS GEOESPACIAL - POR REPRESENTANTE")
print(f"  (carteira combinada: rep_facial + rep_nao_facial)")
print(f"{'='*60}")
print(f"  Total de medicos (unicos)        : {total:>6}")
print(f"  Total de tarefas (medico x rep)  : {total_tarefas:>6}")
print(f"  Tarefas com coordenadas          : {len(df_geo):>6}  ({len(df_geo)/total_tarefas*100:.1f}%)")
print(f"  Tarefas sem coordenadas          : {sem_coord:>6}  ({sem_coord/total_tarefas*100:.1f}%)")

# ── Roda K-Means para cada representante ─────────────────────────────────────

df_long["cluster"]       = -1
df_long["cluster_label"] = pd.NA   # ex: "Rep_031_Zona_02"

resultados_aval = []
reps = sorted(df_geo["rep_id"].dropna().unique())

print(f"\n  Representantes a processar: {len(reps)}")
print(f"\n{'-'*60}")

for rep in reps:
    df_rep = df_geo[df_geo["rep_id"] == rep].copy()
    n      = len(df_rep)

    if rep in REPS_REMOTOS:
        df_long.loc[df_rep.index, "cluster_label"] = "Remoto"
        print(f"  {rep}: {n} tarefas - remoto (sem roteirizacao local)")
        continue

    if n < MIN_MEDICOS_REP:
        print(f"  {rep}: {n} tarefas - pulado (abaixo do minimo {MIN_MEDICOS_REP})")
        continue

    k = max(1, math.ceil(n / VISITAS_ZONA))   # k dinâmico: ~40 médicos por zona
    coords      = df_rep[["latitude", "longitude"]].values
    scaler      = StandardScaler()
    coords_norm = scaler.fit_transform(coords)

    # k=1: carteira pequena — zona única, sem clustering
    if k == 1:
        labels = np.zeros(n, dtype=int)
        sil    = 0.0
    else:
        # Limites de tamanho por cluster: ±40% da média, nunca exceder n
        media    = n / k
        size_min = max(1, int(media * 0.6))
        size_max = min(int(np.ceil(media * 1.4)), n - 1)

        km     = KMeansConstrained(n_clusters=k, size_min=size_min, size_max=size_max,
                                    random_state=RANDOM_STATE)
        labels = km.fit_predict(coords_norm)
        sil    = silhouette_score(coords_norm, labels) if k > 1 else 0.0

    df_long.loc[df_rep.index, "cluster"]       = labels
    df_long.loc[df_rep.index, "cluster_label"] = [f"{rep}_Zona_{str(l+1).zfill(2)}"
                                                    for l in labels]

    sizes    = pd.Series(labels).value_counts()
    cv       = sizes.std() / sizes.mean() * 100 if k > 1 else 0
    n_facial = (df_rep["papel"] == "facial").sum()
    n_naofac = (df_rep["papel"] == "nao_facial").sum()

    print(f"  {rep}: {n:>4} tarefas ({n_facial} facial + {n_naofac} nao_facial) | k={k} zonas | "
          f"Silhouette={sil:.3f} | CV={cv:.0f}% | min={sizes.min()} max={sizes.max()} med={sizes.mean():.0f} | ~{n/k/DIAS_SEMANA:.1f}/dia")

    resultados_aval.append({
        "rep_id": rep, "n_tarefas": n, "n_facial": n_facial, "n_nao_facial": n_naofac,
        "k_clusters": k, "silhouette": round(sil, 4), "cv_pct": round(cv, 1),
        "min_dia": sizes.min(), "max_dia": sizes.max(),
        "media_dia": round(sizes.mean(), 1)
    })

# ── Resumo geral ──────────────────────────────────────────────────────────────
df_aval = pd.DataFrame(resultados_aval)

print(f"\n{'='*60}")
print(f"  RESUMO GERAL")
print(f"{'='*60}")
print(f"  Representantes clusterizados : {len(df_aval)}")
print(f"  Representantes remotos       : {len(REPS_REMOTOS)}  ({', '.join(REPS_REMOTOS)})")
print(f"  Silhouette medio             : {df_aval['silhouette'].mean():.3f}")
print(f"  Silhouette minimo            : {df_aval['silhouette'].min():.3f}  ({df_aval.loc[df_aval['silhouette'].idxmin(),'rep_id']})")
print(f"  Silhouette maximo            : {df_aval['silhouette'].max():.3f}  ({df_aval.loc[df_aval['silhouette'].idxmax(),'rep_id']})")
print(f"  CV medio (equilibrio dias)   : {df_aval['cv_pct'].mean():.1f}%")

# ── Salva outputs ─────────────────────────────────────────────────────────────
df_long.to_csv(r"saida\resultados\base_medicos_clusterizada.csv", index=False)
df_aval.to_csv(r"saida\resultados\kmeans_avaliacao.csv", index=False)
print(f"\n  Base salva em       : saida/resultados/base_medicos_clusterizada.csv")
print(f"  Avaliacao salva em  : saida/resultados/kmeans_avaliacao.csv")

# ── Mapa interativo ───────────────────────────────────────────────────────────
print(f"\n  Gerando mapa interativo...")

cores = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
]

df_map = df_long.dropna(subset=["latitude","longitude","cluster_label"])
df_map = df_map[df_map["cluster_label"] != "Remoto"]
centro_lat = df_map["latitude"].mean()
centro_lon = df_map["longitude"].mean()
mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=5)

for _, row in df_map.iterrows():
    try:
        dia = int(str(row["cluster_label"]).split("_Zona_")[-1]) - 1
    except Exception:
        dia = 0
    cor = cores[dia % len(cores)]
    folium.CircleMarker(
        location=[float(row["latitude"]), float(row["longitude"])],
        radius=3,
        color=cor,
        fill=True,
        fill_color=cor,
        fill_opacity=0.6,
        tooltip=f"{row['cluster_label']} | {row['papel']} | {row.get('CIDADE','')}",
    ).add_to(mapa)

mapa.save(r"saida\resultados\mapa_clusters.html")
print(f"  Mapa salvo em       : saida/resultados/mapa_clusters.html")
print()
