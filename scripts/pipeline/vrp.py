"""
Otimização da rota semanal de uma zona via OR-Tools (VRP com janelas de tempo).

Cada zona do K-Means representa uma semana de visitação. O problema é:
distribuir os médicos da zona entre os 5 dias úteis E sequenciar as visitas
dentro de cada dia, minimizando a distância percorrida — respeitando a
capacidade diária, as restrições de dia e os períodos (manhã/tarde).

Modelagem
---------
  veículos    : 5 (um por dia útil da semana)
  depósito    : centroide da zona (nó virtual)
  capacidade  : META_DIARIA visitas por dia
  dimensão t  : tempo de deslocamento + atendimento, com janelas por período

Parâmetros calibrados com o histórico do CRM (task_report, 20.207 registros):
  · duração da visita  : 30 min (mediana observada)
  · jornada            : 8h–18h, típica 9h–16h
  · visitas/dia        : mediana 7, p75 8 → meta de 8 confirmada pelo negócio

Premissas declaradas (limitações do modelo)
-------------------------------------------
  1. O endereço de origem do representante não existe na base. Usa-se o
     centroide da zona como depósito virtual — proxy razoável, já que o rep
     atua naquela região durante a semana.
  2. A distância é euclidiana (haversine) multiplicada por FATOR_ROTA para
     aproximar o trajeto por ruas. Distância real exigiria serviço de rotas
     (OSRM/Azure Maps); é a evolução natural do modelo.
"""

import math
import time

import numpy as np
import requests
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ── Parâmetros ────────────────────────────────────────────────────────────────
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex"]
META_DIARIA      = 8      # visitas/dia (confirmado pelo negócio; p75 observado)
DURACAO_VISITA   = 30     # minutos (mediana do task_report)
VELOCIDADE_KMH   = 25     # velocidade média urbana
FATOR_ROTA       = 1.35   # haversine → trajeto por ruas (fator de desvio usual)
JORNADA_INICIO   = 8 * 60     # 08:00 em minutos
JORNADA_FIM      = 18 * 60    # 18:00
LIMITE_MANHA     = 12 * 60    # 12:00
INICIO_TARDE     = 13 * 60    # 13:00
TEMPO_LIMITE_SEG = 10     # busca por zona
COEF_EQUILIBRIO  = 50     # equilíbrio entre os dias (0 desliga) — ver otimizar_zona

# Penalidade por deixar um médico fora da semana, em METROS equivalentes.
# Calibração: uma semana típica percorre ~43 km. Penalidades muito acima disso
# tornam TODOS os médicos obrigatórios e a prioridade deixa de discriminar — foi
# o que aconteceu na primeira versão (2.000 km a 150 km), em que Foco 1 e "sem
# score" acabaram com a mesma cobertura (~92%). Os valores abaixo ficam na faixa
# do custo marginal real de incluir um médico distante, então a ordem de descarte
# passa a valer: sem score sai primeiro, Foco 1 praticamente nunca sai.
# Calibrado em amostra de 28 semanas: o piso de 60 km derrubava a cobertura geral
# para 96,4%; com 200 km ela volta a 98,5% SEM perder o gradiente de prioridade.
PENALIDADE_FOCO1     = 5_000_000   # 5.000 km — na prática, obrigatório
PENALIDADE_FOCO2     =   900_000   #   900 km
PENALIDADE_FOCO3     =   450_000   #   450 km
PENALIDADE_SEM_SCORE =   200_000   #   200 km — primeiro a ceder, sem exagero

# Ataque = potencial com share abaixo da meta (conquista); Defesa = já prescreve
# bem (manutenção). O peso extra em Ataque é mantido por coerência com a
# estratégia comercial, MAS seu efeito medido é desprezível: testado até 3,0x, a
# diferença de cobertura ficou abaixo do ruído do solver (99,1% x 99,3%).
# Motivo: a penalidade só decide algo quando a semana não cabe, e na maioria das
# zonas cabe todo mundo. Priorizar Ataque de fato exige agir ANTES da
# roteirização — na escolha de quais médicos entram no ciclo. Fica registrado
# como limitação do modelo.
# O valor 1,6 preserva a hierarquia: Foco 3 Ataque (720k) < Foco 2 Defesa (900k).
MULTIPLICADOR_ATAQUE = 1.6
MULTIPLICADOR_DEFESA = 1.0


OSRM_URL = "http://router.project-osrm.org"
OSRM_TIMEOUT = 15         # o servidor público responde em ~0,5s; 45s só prolongava a falha
OSRM_TENTATIVAS = 3       # o erro típico é limite de uso — cede com uma pausa curta
OSRM_INTERVALO = 1.1      # s entre chamadas: o servidor público pede ~1 req/s
OSRM_FALHAS_ATE_DESISTIR = 5   # falhas CONSECUTIVAS antes de desligar o serviço

_osrm_indisponivel = False
_osrm_falhas = 0          # zera a cada sucesso: uma falha isolada não derruba o lote
_osrm_ultima_chamada = 0.0


def osrm_status():
    """Estado do roteador — para o chamador reportar honestamente a fonte."""
    return {"indisponivel": _osrm_indisponivel, "falhas_consecutivas": _osrm_falhas}


def _osrm_falhou(motivo):
    """Contabiliza a falha e desliga o serviço se ela virar padrão.

    BUG CORRIGIDO (jul/2026): antes, só uma EXCEÇÃO marcava o serviço como
    indisponível. Mas o servidor público recusa por limite de uso com um JSON
    válido (`code != "Ok"`), que passava batido — resultado: 640 semanas
    retentaram, cada uma até 45 s, e 79% caíram no haversine sem aviso.
    """
    global _osrm_indisponivel, _osrm_falhas
    _osrm_falhas += 1
    if _osrm_falhas >= OSRM_FALHAS_ATE_DESISTIR:
        _osrm_indisponivel = True
        print(f"    [OSRM] desligado após {_osrm_falhas} falhas seguidas ({motivo}) — "
              f"o restante do lote usa haversine")


def matriz_osrm(lats, lons):
    """Matriz de distâncias REAIS por ruas, via OSRM (endpoint /table).

    Uma única chamada resolve a matriz inteira da semana (~0,6 s para 45 pontos).
    Retorna None se o serviço falhar — o chamador cai no haversine.

    Por que importa: medimos o desvio real contra o roteador e o fator não é
    constante — 4,2× em trechos abaixo de 100 m (o carro contorna o quarteirão),
    1,5× acima de 1 km. Um fator fixo subestimava o percurso em ~36%.
    """
    global _osrm_falhas, _osrm_ultima_chamada
    if _osrm_indisponivel or len(lats) < 2:
        return None

    coords = ";".join(f"{lo},{la}" for la, lo in zip(lats, lons))
    for tentativa in range(OSRM_TENTATIVAS):
        espera = OSRM_INTERVALO - (time.monotonic() - _osrm_ultima_chamada)
        if espera > 0:
            time.sleep(espera)
        try:
            r = requests.get(f"{OSRM_URL}/table/v1/driving/{coords}",
                             params={"annotations": "distance"}, timeout=OSRM_TIMEOUT)
            _osrm_ultima_chamada = time.monotonic()
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            motivo = None
            if r.status_code != 200:
                motivo = f"HTTP {r.status_code}"
            elif j.get("code") != "Ok":
                motivo = j.get("code", "resposta sem code")
        except Exception as e:
            _osrm_ultima_chamada = time.monotonic()
            motivo = type(e).__name__

        if motivo is None:
            _osrm_falhas = 0
            m = np.array(j["distances"], dtype=float)
            if not np.isfinite(m).all():      # trechos sem rota viável
                m = np.where(np.isfinite(m), m, matriz_distancias(lats, lons))
            return m.astype(int)

        if tentativa < OSRM_TENTATIVAS - 1:
            time.sleep(2 ** tentativa)        # backoff: 1s, 2s

    _osrm_falhou(motivo)
    return None


def rota_osrm(lats, lons):
    """Traçado REAL da rota pelas ruas, na ordem dada (endpoint /route).

    Retorna a lista de pontos [[lat, lon], ...] que desenha o caminho de fato
    percorrido — não o segmento reto entre uma visita e a seguinte. Serve para o
    mapa: mostrar linha reta ao lado de um número de km calculado por rua faz o
    trajeto parecer mais curto do que é.

    Retorna None se o serviço falhar; aí o chamador desenha os segmentos retos.
    """
    global _osrm_falhas, _osrm_ultima_chamada
    if _osrm_indisponivel or len(lats) < 2:
        return None

    coords = ";".join(f"{lo},{la}" for la, lo in zip(lats, lons))
    espera = OSRM_INTERVALO - (time.monotonic() - _osrm_ultima_chamada)
    if espera > 0:
        time.sleep(espera)
    try:
        r = requests.get(f"{OSRM_URL}/route/v1/driving/{coords}",
                         params={"overview": "full", "geometries": "geojson"},
                         timeout=OSRM_TIMEOUT)
        _osrm_ultima_chamada = time.monotonic()
        j = r.json()
        if r.status_code != 200 or j.get("code") != "Ok":
            return None
        _osrm_falhas = 0
        # o GeoJSON vem como [lon, lat]; o folium espera [lat, lon]
        return [[p[1], p[0]] for p in j["routes"][0]["geometry"]["coordinates"]]
    except Exception:
        _osrm_ultima_chamada = time.monotonic()
        return None


def matriz_distancias(lats, lons):
    """Matriz simétrica de distâncias em metros (haversine × fator de rota).

    Aproximação usada quando o roteador de ruas não está disponível.
    """
    lat = np.radians(np.asarray(lats, dtype=float))
    lon = np.radians(np.asarray(lons, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return (km * FATOR_ROTA * 1000).astype(int)


def _janela(periodo, duracao=None, jor_ini=None, jor_fim=None):
    """Janela de tempo (minutos) conforme o período em que o médico atende."""
    duracao = DURACAO_VISITA if duracao is None else duracao
    jor_ini = JORNADA_INICIO if jor_ini is None else jor_ini
    jor_fim = JORNADA_FIM if jor_fim is None else jor_fim
    if periodo == "manha":
        return (jor_ini, LIMITE_MANHA - duracao)
    if periodo == "tarde":
        return (INICIO_TARDE, jor_fim - duracao)
    return (jor_ini, jor_fim - duracao)


def otimizar_zona(df_zona, regras=None, meta_dia=META_DIARIA,
                  tempo_limite=TEMPO_LIMITE_SEG, usar_osrm=True,
                  teto_dia=None, duracao_visita=None, jornada=None,
                  coef_visitas=0):
    """Distribui e sequencia as visitas de uma zona ao longo da semana.

    df_zona : DataFrame com medico_id, latitude, longitude
    regras  : {medico_id: {'dias': set|None, 'periodo': str}} (disponibilidade)

    Parâmetros do modo de carga flexível (todos opcionais):
      teto_dia       — máximo de visitas num dia. Sendo maior que meta_dia, o
                       número de visitas do dia deixa de ser fixo: o limite passa
                       a ser o relógio, então dia com deslocamento longo rende
                       menos visitas e dia denso rende mais. A semana continua
                       com 40 médicos, então a MÉDIA de 8/dia se mantém.
      duracao_visita — minutos por visita. O padrão de 30 é o valor que o CRM
                       preenche sozinho, não uma medição: no histórico, do
                       percentil 10 ao 75 a duração é exatamente 30. A jornada
                       observada (9h às 16h para 7 visitas) implica ~64 min
                       efetivos por visita, incluindo espera e administrativo.
      jornada        — (inicio, fim) em minutos. O observado é 9h–17h30.
      coef_visitas   — penaliza a diferença de CARGA entre os dias. Sem isso, ao
                       soltar o teto o solver tende a lotar um dia e esvaziar
                       outro, porque só a distância entra no custo.

    Retorna (plano, metricas):
      plano   — {dia: DataFrame com colunas ordem, chegada_min, km_ate_aqui}
      metricas— dict com distancia total, por dia e status da solução
    """
    regras = regras or {}
    duracao_visita = DURACAO_VISITA if duracao_visita is None else duracao_visita
    jor_ini, jor_fim = jornada or (JORNADA_INICIO, JORNADA_FIM)
    df = df_zona.dropna(subset=["latitude", "longitude"]).reset_index(drop=True).copy()
    if df.empty:
        return {d: df.copy() for d in DIAS}, {"status": "vazio", "km_total": 0.0}

    n = len(df)
    # nó 0 = depósito virtual (centroide da zona); nós 1..n = médicos
    lats = [df["latitude"].mean()] + df["latitude"].tolist()
    lons = [df["longitude"].mean()] + df["longitude"].tolist()
    dist = matriz_osrm(lats, lons) if usar_osrm else None
    fonte_dist = "osrm" if dist is not None else "haversine"
    if dist is None:
        dist = matriz_distancias(lats, lons)

    n_dias = len(DIAS)
    # O teto é o que decide se o dia tem carga fixa ou flexível. Igual à meta,
    # todo dia fecha em 8. Acima dela, o relógio passa a mandar.
    capacidade = max(teto_dia or meta_dia, math.ceil(n / n_dias))
    manager = pywrapcp.RoutingIndexManager(n + 1, n_dias, 0)
    routing = pywrapcp.RoutingModel(manager)

    # ── custo: distância ──────────────────────────────────────────────────────
    def cb_dist(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])
    idx_dist = routing.RegisterTransitCallback(cb_dist)
    routing.SetArcCostEvaluatorOfAllVehicles(idx_dist)

    # ── equilíbrio entre os dias ──────────────────────────────────────────────
    # Penaliza a diferença entre o dia mais longo e o mais curto. Sem isso o
    # solver concentra todo o deslocamento num único dia — em Goiânia (Rep_006)
    # gerava uma terça de 68 km contra 1,7 km na segunda.
    # Não é só conforto do representante: quando os médicos distantes estão
    # espalhados em várias direções, distribuí-los REDUZ a distância total
    # (94,1 km → 25,7 km no mesmo caso), porque evita o zigue-zague pela
    # periferia. Onde não ajuda, o custo é desprezível.
    routing.AddDimension(idx_dist, 0, 900_000, True, "Distancia")
    if COEF_EQUILIBRIO:
        routing.GetDimensionOrDie("Distancia").SetGlobalSpanCostCoefficient(COEF_EQUILIBRIO)

    # ── capacidade: nº de visitas por dia ─────────────────────────────────────
    def cb_demanda(i):
        return 0 if manager.IndexToNode(i) == 0 else 1
    idx_dem = routing.RegisterUnaryTransitCallback(cb_demanda)
    routing.AddDimensionWithVehicleCapacity(
        idx_dem, 0, [capacidade] * n_dias, True, "Visitas")
    # Com teto solto, só a distância entra no custo — e sai mais barato lotar um
    # dia e esvaziar outro. Este coeficiente devolve o equilíbrio de CARGA sem
    # voltar a fixar o número: o dia continua livre para render menos quando o
    # deslocamento é longo.
    if coef_visitas:
        routing.GetDimensionOrDie("Visitas").SetGlobalSpanCostCoefficient(coef_visitas)

    # ── tempo: deslocamento + atendimento, com janelas por período ────────────
    def cb_tempo(i, j):
        no_i = manager.IndexToNode(i)
        metros = dist[no_i][manager.IndexToNode(j)]
        viagem = metros / 1000.0 / VELOCIDADE_KMH * 60.0
        return int(viagem + (0 if no_i == 0 else duracao_visita))
    idx_tempo = routing.RegisterTransitCallback(cb_tempo)
    routing.AddDimension(idx_tempo, jor_fim, jor_fim, False, "Tempo")
    dim_tempo = routing.GetDimensionOrDie("Tempo")

    for pos, mid in enumerate(df["medico_id"], start=1):
        ini, fim = _janela((regras.get(mid) or {}).get("periodo", "ambos"),
                           duracao_visita, jor_ini, jor_fim)
        dim_tempo.CumulVar(manager.NodeToIndex(pos)).SetRange(ini, max(ini, fim))

    for v in range(n_dias):
        dim_tempo.CumulVar(routing.Start(v)).SetRange(jor_ini, jor_fim)

    # ── restrição de dia: limita quais veículos podem atender o nó ────────────
    for pos, mid in enumerate(df["medico_id"], start=1):
        dias_ok = (regras.get(mid) or {}).get("dias")
        if not dias_ok:
            continue
        permitidos = [v for v, d in enumerate(DIAS) if d.lower()[:3] in dias_ok]
        if permitidos:
            routing.VehicleVar(manager.NodeToIndex(pos)).SetValues(permitidos)

    # ── visitas opcionais, com penalidade proporcional à prioridade ───────────
    # Uma zona pode conter mais médicos do que cabe na semana (a meta é 40; há
    # zonas com 49). Em vez de devolver "sem solução", o modelo escolhe QUEM
    # fica de fora — e a penalidade faz essa escolha recair sobre os de menor
    # prioridade. É assim que o score de priorização entra na roteirização.
    foco = df.get("foco", pd.Series([None] * n, index=df.index)).fillna("")
    acao = df.get("acao_estrategica", pd.Series([""] * n, index=df.index)).fillna("")
    for pos in range(1, n + 1):
        f = str(foco.iloc[pos - 1])
        if "1" in f:      base = PENALIDADE_FOCO1     # praticamente obrigatório
        elif "2" in f:    base = PENALIDADE_FOCO2
        elif "3" in f:    base = PENALIDADE_FOCO3
        else:             base = PENALIDADE_SEM_SCORE  # sai primeiro

        texto = str(acao.iloc[pos - 1]).upper()
        if "ATAQUE" in texto:
            base *= MULTIPLICADOR_ATAQUE
        elif "DEFESA" in texto:
            base *= MULTIPLICADOR_DEFESA
        routing.AddDisjunction([manager.NodeToIndex(pos)], int(base))

    # ── busca ─────────────────────────────────────────────────────────────────
    # PARALLEL_CHEAPEST_INSERTION lida melhor com janelas de tempo apertadas que
    # PATH_CHEAPEST_ARC, que estourava o tempo sem achar solução inicial.
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(tempo_limite)
    sol = routing.SolveWithParameters(params)

    if sol is None:
        return None, {"status": "sem_solucao", "km_total": None}

    # ── extrai o plano ────────────────────────────────────────────────────────
    # A distância reportada exclui a saída do depósito e o retorno a ele: como o
    # depósito é virtual (centroide), essas duas pernas não existem na prática.
    # Mede-se o percurso entre o 1º e o último médico do dia.
    plano, km_por_dia = {}, {}
    for v, dia in enumerate(DIAS):
        linhas, metros, ordem = [], 0, 1
        i = routing.Start(v)
        while not routing.IsEnd(i):
            no = manager.IndexToNode(i)
            prox = sol.Value(routing.NextVar(i))
            no_prox = manager.IndexToNode(prox)
            if no != 0:
                r = df.iloc[no - 1]
                linhas.append({
                    **r.to_dict(),
                    "ordem": ordem,
                    "chegada_min": sol.Min(dim_tempo.CumulVar(i)),
                    "km_ate_aqui": round(metros / 1000, 2),
                })
                ordem += 1
                if no_prox != 0:            # ignora a perna de retorno ao depósito
                    metros += dist[no][no_prox]
            i = prox
        plano[dia] = pd.DataFrame(linhas)
        km_por_dia[dia] = round(metros / 1000, 2)

    # médicos que não couberam na semana (menor prioridade primeiro)
    agendados = set()
    for b in plano.values():
        if not b.empty:
            agendados.update(b["medico_id"])
    fora = df[~df["medico_id"].isin(agendados)].copy()

    return plano, {
        "status": "ok",
        "km_total": round(sum(km_por_dia.values()), 2),
        "km_por_dia": km_por_dia,
        "visitas_por_dia": {d: len(b) for d, b in plano.items()},
        "capacidade_usada": capacidade,
        "n_agendados": len(agendados),
        "n_fora": len(fora),
        "fora": fora,
        "fonte_distancia": fonte_dist,
    }


# ── Montagem das semanas ──────────────────────────────────────────────────────
# O que é fixo na operação é a SEMANA (5 dias × 8 visitas = 40), não a zona.
# Tratar "1 zona = 1 semana" deixava 42% das semanas com dias ociosos, porque o
# K-Means gera zonas de 20 a 56 médicos (tolerância de ±40% em torno da média).
# Aqui as zonas são encadeadas por proximidade e cortadas a cada 40 visitas: a
# semana fica cheia e a emenda cai sempre entre zonas vizinhas, não distantes.

VISITAS_SEMANA = META_DIARIA * len(DIAS)    # 40


def _ordenar_zonas_por_proximidade(centroides):
    """Encadeia as zonas por vizinho-mais-próximo, começando pela mais central."""
    zonas = list(centroides)
    if len(zonas) <= 2:
        return zonas
    pts = np.array([centroides[z] for z in zonas], dtype=float)
    centro = pts.mean(axis=0)
    atual = int(np.argmin(((pts - centro) ** 2).sum(axis=1)))
    restantes, ordem = set(range(len(zonas))), []
    while restantes:
        restantes.discard(atual)
        ordem.append(zonas[atual])
        if not restantes:
            break
        livres = list(restantes)
        atual = livres[int(np.argmin(((pts[livres] - pts[atual]) ** 2).sum(axis=1)))]
    return ordem


def montar_semanas(df_rep, visitas_semana=VISITAS_SEMANA):
    """Encadeia as zonas de um representante em semanas cheias.

    Retorna lista de dicts: {semana, zonas, df}. A última semana do ciclo pode
    ficar incompleta — é o resto da carteira.
    """
    d = df_rep.dropna(subset=["latitude", "longitude", "zona"]).copy()
    if d.empty:
        return []

    centroides = {z: (b["latitude"].mean(), b["longitude"].mean())
                  for z, b in d.groupby("zona")}
    ordem = _ordenar_zonas_por_proximidade(centroides)

    # dentro da zona, mantém a coerência geográfica antes de cortar
    blocos = []
    for z in ordem:
        b = d[d["zona"] == z].copy()
        clat, clon = centroides[z]
        b["_r"] = np.hypot((b["latitude"] - clat) * 111.32,
                           (b["longitude"] - clon) * 111.32 * np.cos(np.radians(clat)))
        blocos.append(b.sort_values("_r").drop(columns="_r"))

    fila = pd.concat(blocos, ignore_index=True)
    semanas = []
    for i in range(0, len(fila), visitas_semana):
        pedaco = fila.iloc[i:i + visitas_semana].copy()
        semanas.append({
            "semana": f"Semana {len(semanas) + 1:02d}",
            "zonas": sorted(pedaco["zona"].unique(), key=str),
            "df": pedaco,
        })
    return semanas


# ── Diagnóstico de viabilidade da zona ────────────────────────────────────────
# O K-Means agrupa por proximidade RELATIVA: dentro da carteira de um rep, os
# médicos de um grupo são os mais próximos entre si. Isso não garante que sejam
# VISITÁVEIS na mesma semana — o K-Means não conhece o relógio. Um caso real:
# Rep_068/Zona 03 reúne Porto Velho e Tabatinga (AM), 1.664 km, 66 h de viagem.
# Este diagnóstico expõe esses casos para decisão do gestor (revisar carteira,
# programar viagem ou abrir território novo) em vez de deixá-los passar batido.

RAIO_DIA_KM     = 50    # até aqui o rep cobre no dia a dia
RAIO_VIAGEM_KM  = 300   # entre 50 e 300 km: exige viagem/pernoite programado


def diagnosticar_zona(df_zona):
    """Classifica a viabilidade operacional de uma zona semanal.

    Retorna dict com a classificação, as métricas de dispersão e a lista de
    médicos geograficamente isolados (candidatos a saída da carteira ou a
    remanejamento para outro representante).
    """
    d = df_zona.dropna(subset=["latitude", "longitude"])
    if len(d) < 2:
        return {"classificacao": "viavel", "n": len(d), "p90_km": 0.0,
                "max_km": 0.0, "isolados": []}

    lat, lon = d["latitude"].to_numpy(float), d["longitude"].to_numpy(float)
    clat, clon = np.median(lat), np.median(lon)
    km_centro = np.hypot((lat - clat) * 111.32,
                         (lon - clon) * 111.32 * np.cos(np.radians(clat)))

    m = matriz_distancias(lat, lon) / 1000.0
    np.fill_diagonal(m, np.inf)
    vizinho_mais_proximo = m.min(axis=1)     # distância até o colega mais próximo

    p90 = float(np.percentile(km_centro, 90))
    if p90 <= RAIO_DIA_KM:
        classe = "viavel"
    elif p90 <= RAIO_VIAGEM_KM:
        classe = "viagem_programada"
    else:
        classe = "inviavel_semana"

    # Isolado = ninguém da mesma zona por perto. Visitar sai caro por definição.
    isolados = [
        {"medico_id": d.iloc[i]["medico_id"],
         "cidade": d.iloc[i].get("CIDADE", ""),
         "km_do_centro": round(float(km_centro[i]), 1),
         "km_do_colega_mais_proximo": round(float(vizinho_mais_proximo[i]), 1)}
        for i in range(len(d)) if vizinho_mais_proximo[i] > RAIO_DIA_KM
    ]

    return {
        "classificacao": classe,
        "n": len(d),
        "p90_km": round(p90, 1),
        "max_km": round(float(km_centro.max()), 1),
        "mediana_entre_pares_km": round(float(np.median(m[np.isfinite(m)])), 1),
        "horas_para_atravessar": round(float(km_centro.max() * 2 / VELOCIDADE_KMH), 1),
        "isolados": isolados,
    }


def hora(minutos):
    """540 -> '09:00'"""
    if pd.isna(minutos):
        return ""
    return f"{int(minutos) // 60:02d}:{int(minutos) % 60:02d}"


def distancia_sequencia(df_ordenado):
    """Distância total (km) de uma sequência já ordenada — usado nas comparações."""
    if len(df_ordenado) < 2:
        return 0.0
    d = matriz_distancias(df_ordenado["latitude"].tolist(),
                          df_ordenado["longitude"].tolist())
    return round(sum(d[i][i + 1] for i in range(len(df_ordenado) - 1)) / 1000, 2)
