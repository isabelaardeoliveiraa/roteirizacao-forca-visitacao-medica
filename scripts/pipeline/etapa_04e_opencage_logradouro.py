# -*- coding: utf-8 -*-
"""
Etapa 04e — Eleva a precisão dos registros que ficaram em nível de logradouro.

Por que existe
--------------
A etapa 04b (OpenCage) só processava quem estava SEM coordenada:

    sem_coord = df[df["latitude"].isna()]

O critério era "tem coordenada?" quando deveria ser "tem coordenada boa o
bastante?". Em consequência, os 976 médicos que ficaram com precisão de
logradouro — coordenada obtida do CEP, sem número do imóvel — nunca foram
submetidos ao serviço comercial: zero deles constam do checkpoint da 04b.

Destes, 533 possuem endereço completo no cadastro e 486 haviam recebido
"sem resultado" do Nominatim na etapa 04d. Como o OpenCage opera sobre base
distinta, o que um serviço não resolve o outro pode resolver — foi assim que a
04b recuperou 642 endereços que o Nominatim havia recusado.

Regra de aceitação: idêntica à da 04d. A coordenada nova entra somente se
passar na validação geográfica E no teste municipal; reprovando, mantém-se a
coordenada de logradouro que já existia. Não pode degradar a base.

Entrada : saida/resultados/base_medicos_validada.csv
          saida/resultados/viacep_logradouros.csv   (cache da 04d)
Saída   : saida/resultados/base_medicos_validada.csv  (atualizado)
          saida/resultados/opencage_logradouro_checkpoint.csv (retomável)
          saida/resultados/opencage_logradouro_relatorio.txt
"""
import os
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent))
import validacao_geo as vg

# A chave nao fica no codigo: o repositorio do trabalho e publicavel.
# Defina antes de rodar:  $env:OPENCAGE_API_KEY = "<sua chave>"
API_KEY = os.environ.get("OPENCAGE_API_KEY")
if not API_KEY:
    raise SystemExit("Defina a variavel de ambiente OPENCAGE_API_KEY")
ARQ_BASE  = "saida/resultados/base_medicos_validada.csv"
ARQ_VIACEP = "saida/resultados/viacep_logradouros.csv"
ARQ_CONTACT = "dados_crm/2807/crm_contact_report.csv"
ARQ_CORRESP = "saida/resultados/correspondencia_contactid.csv"
ARQ_CHECK = "saida/resultados/opencage_logradouro_checkpoint.csv"
ARQ_REL   = "saida/resultados/opencage_logradouro_relatorio.txt"
SALT_PATH = Path("saida/.geomask_salt")

DELAY = 1.1
LIMITE_DIARIO = 2400
GEO_MASKING_M = 300
RAIO_MUNICIPIO_KM = 60
MIN_MEDICOS_CIDADE = 3
R_TERRA = 6371.0

SALT = SALT_PATH.read_text(encoding="utf-8").strip() if SALT_PATH.exists() else ""
if not SALT:
    raise SystemExit("Segredo do geo-masking ausente.")


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return R_TERRA * 2 * np.arcsin(np.sqrt(a))


def geo_mask(lat, lon, medico_id, metros=GEO_MASKING_M):
    rng = np.random.default_rng(zlib.crc32("{}:{}".format(SALT, medico_id).encode("utf-8")))
    dlat = (metros / 111_320) * rng.uniform(-1, 1)
    dlon = (metros / (111_320 * math.cos(math.radians(lat)))) * rng.uniform(-1, 1)
    return round(lat + dlat, 6), round(lon + dlon, 6)


def opencage(query):
    try:
        r = requests.get("https://api.opencagedata.com/geocode/v1/json",
                         params={"q": query, "key": API_KEY, "language": "pt",
                                 "countrycode": "br", "limit": 3, "no_annotations": 1},
                         timeout=10)
        d = r.json()
        saida = []
        for res in d.get("results", []):
            g = res["geometry"]
            tipo = (res.get("components", {}) or {}).get("_type")
            saida.append((g["lat"], g["lng"], tipo))
        return saida or None
    except Exception as e:
        print("    erro: {}".format(type(e).__name__))
        return None


# ── Dados ─────────────────────────────────────────────────────────────────────
print("Carregando base...")
df = pd.read_csv(ARQ_BASE)
vc = pd.read_csv(ARQ_VIACEP, dtype={"cep": str}).fillna("")
viacep = {r.cep: (r.logradouro, r.localidade, r.uf) for r in vc.itertuples()}

contact = pd.read_csv(ARQ_CONTACT, sep=";", dtype={"postcode": str}, low_memory=False)
contact["cep"] = (contact.postcode.astype(str).str.replace(r"\.0$", "", regex=True)
                         .str.replace(r"\D", "", regex=True).str.zfill(8))
corresp = pd.read_csv(ARQ_CORRESP)[["contact_id", "medico_id"]]
end = (corresp.merge(contact[["contact_id", "cep", "street", "building", "city", "region"]],
                     on="contact_id", how="left").set_index("medico_id"))
for col in ("cep", "street", "building", "city", "region"):
    df[col] = df.medico_id.map(end[col])

# referência municipal, das coordenadas vigentes, por cidade + UF
atual = df[df.latitude.notna()]
ref = (atual.groupby(["CIDADE", "region"])
            .agg(lat_ref=("latitude", "median"), lon_ref=("longitude", "median"),
                 n_cidade=("medico_id", "size")).reset_index())
df = df.merge(ref, on=["CIDADE", "region"], how="left")


def preenchido(s):
    return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.strip().str.lower() != "nan")


alvo = df[df._geo_estrategia.eq("cep") & df.latitude.notna()
          & preenchido(df.street) & preenchido(df.building)
          & preenchido(df.city) & preenchido(df.region)].copy()
print("  em precisão de logradouro, com endereço completo: {}".format(len(alvo)))

try:
    ck = pd.read_csv(ARQ_CHECK)
    cache = {r.medico_id: (r.lat, r.lon, r.tipo) for r in ck.itertuples()}
    print("  checkpoint: {} já consultados".format(len(cache)))
except FileNotFoundError:
    cache = {}


def salvar():
    pd.DataFrame([{"medico_id": k, "lat": v[0], "lon": v[1], "tipo": v[2]}
                  for k, v in cache.items()]).to_csv(ARQ_CHECK, index=False)


# ── Consulta ──────────────────────────────────────────────────────────────────
pend = [r for r in alvo.itertuples() if r.medico_id not in cache]
print("  a consultar: {} (~{:.0f} min)\n".format(len(pend), len(pend) * DELAY / 60))

consultas = 0
for i, r in enumerate(pend, 1):
    if consultas >= LIMITE_DIARIO:
        print("  LIMITE DIARIO ATINGIDO — rode novamente amanhã.")
        break
    log, loc, uf = viacep.get(r.cep, ("", "", ""))
    rua = log.strip() if log.strip() else str(r.street).strip()
    cidade = (loc or str(r.city)).strip()
    estado = (uf or str(r.region)).strip()
    q = "{}, {}, {}, {}, Brasil".format(rua, str(r.building).strip(), cidade, estado)
    res = opencage(q)
    consultas += 1
    # guarda o primeiro resultado que passa na validacao geografica
    guardado = (None, None, "sem_resultado")
    for la, lo, tipo in (res or []):
        aceito, motivo = vg.validar(la, lo, r.cep, tipo)
        if aceito:
            guardado = (la, lo, tipo)
            break
        guardado = (None, None, "validacao_geo:{}".format(motivo))
    cache[r.medico_id] = guardado
    if i % 25 == 0:
        salvar()
        print("  [{}/{}] {:.0f}%".format(i, len(pend), 100 * i / len(pend)), flush=True)
    time.sleep(DELAY)
salvar()

# ── Aceitação ─────────────────────────────────────────────────────────────────
print("\nAplicando a regra de aceitação...")
pos = {m: k for k, m in enumerate(df.medico_id)}
aceitos, motivos, movidos = 0, [], []
for mid, (la, lo, tipo) in cache.items():
    k = pos.get(mid)
    if k is None or la is None or (isinstance(la, float) and np.isnan(la)):
        motivos.append(tipo if isinstance(tipo, str) else "sem_resultado")
        continue
    linha = df.iloc[k]
    if pd.notna(linha.lat_ref) and linha.n_cidade >= MIN_MEDICOS_CIDADE:
        ref_pt = (linha.lat_ref, linha.lon_ref)
    elif pd.notna(linha.latitude):
        ref_pt = (linha.latitude, linha.longitude)
    else:
        motivos.append("sem_referencia_municipal")
        continue
    if haversine(la, lo, ref_pt[0], ref_pt[1]) > RAIO_MUNICIPIO_KM:
        motivos.append("fora_do_municipio")
        continue
    lat_m, lon_m = geo_mask(la, lo, mid)
    movidos.append(haversine(linha.latitude, linha.longitude, lat_m, lon_m))
    df.iat[k, df.columns.get_loc("latitude")] = lat_m
    df.iat[k, df.columns.get_loc("longitude")] = lon_m
    df.iat[k, df.columns.get_loc("_geo_estrategia")] = "opencage_logradouro"
    aceitos += 1

df.drop(columns=["cep", "street", "building", "city", "region",
                 "lat_ref", "lon_ref", "n_cidade"]).to_csv(ARQ_BASE, index=False)

L = ["=" * 66, "  ETAPA 04e — OPENCAGE SOBRE OS REGISTROS EM NIVEL DE LOGRADOURO", "=" * 66,
     "  candidatos            : {}".format(len(alvo)),
     "  consultas nesta rodada: {}".format(consultas),
     "  ACEITOS (subiram para precisao de endereco): {}".format(aceitos),
     "  mantiveram logradouro : {}".format(len(motivos)), "", "  Motivos:"]
for m, n in pd.Series(motivos).value_counts().items():
    L.append("    {:34s} {:5d}".format(str(m)[:34], n))
if movidos:
    mv = pd.Series(movidos)
    L += ["", "  Deslocamento dos aceitos (km): mediana {:.2f} | p90 {:.2f} | max {:.2f}".format(
        mv.median(), mv.quantile(.9), mv.max())]
com = df.latitude.notna()
L += ["", "  Cobertura final: {:,}/{:,} ({:.1f}%)".format(com.sum(), len(df), 100 * com.mean()), "",
      "  Composicao por estrategia:"]
for e, n in df[com]._geo_estrategia.value_counts().items():
    L.append("    {:24s} {:6,}  ({:4.1f}%)".format(str(e), n, 100 * n / com.sum()))
txt = "\n".join(L)
Path(ARQ_REL).write_text(txt, encoding="utf-8")
print("\n" + txt)
