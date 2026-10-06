# -*- coding: utf-8 -*-
"""
Etapa 04d — Regeocodificação a partir do ENDEREÇO COMPLETO.

Motivação
---------
A etapa 04 geocodifica CEPs únicos: consulta o ViaCEP para descobrir o logradouro
daquele CEP e manda esse logradouro ao Nominatim. Duas consequências:
  - o número do consultório nunca é usado; a resolução máxima é a rua;
  - todo médico que compartilha um CEP recebe coordenada IDÊNTICA (33,6% das
    pernas de rota ligavam pontos iguais, separados só pelo geo-masking).
Em CEP de localidade (terminado em 000, 13,4% da base) o efeito é pior: o ViaCEP
devolve logradouro vazio e o Nominatim retorna o centro do município.

Esta etapa consulta o Nominatim com o endereço completo, montado a partir de
duas fontes: o LOGRADOURO oficial do ViaCEP e o NÚMERO declarado no CRM.

Por que não usar o logradouro do próprio CRM: medido em 200 CEPs, o nome
digitado confere com o oficial em 97,3% dos casos — erro de digitação não é
problema. O problema é ABREVIAÇÃO: o CRM guarda "CONS DANTAS" onde o oficial é
"Rua Conselheiro Dantas", e "DBRENO SOARES MAIA" para "Avenida Doutor Breno
Soares Maia". O Nominatim resolve muito melhor a forma expandida.
O campo do CRM entra como reserva quando o ViaCEP não tem logradouro
(CEP de localidade, 3,5%) ou não conhece o CEP (5,5%).

Regra de aceitação — a coordenada nova entra SOMENTE se passar em dois testes:
  1. validacao_geo.validar()  — dentro do Brasil, não é centroide genérico,
     nível específico o bastante, UF compatível com o CEP;
  2. teste municipal — a menos de RAIO_MUNICIPIO_KM da mediana das coordenadas
     ATUAIS dos médicos daquela cidade. É este que pega o erro de "cidade errada"
     (rua homônima em outro município), que o teste de UF deixa passar: no teste
     de amostra, 10% dos resultados caíam a mais de 10 km e 3,9% a mais de 50 km,
     todos dentro da UF correta.
Reprovando em qualquer um dos dois, MANTÉM a coordenada antiga. A etapa é
não-regressiva por construção.

Entrada : saida/resultados/base_medicos_validada.csv
          dados_crm/2807/crm_contact_report.csv
          saida/resultados/correspondencia_contactid.csv
Saída   : saida/resultados/base_medicos_endereco.csv
          saida/resultados/regeocod_endereco_checkpoint.csv   (retomável)
          saida/resultados/regeocod_endereco_relatorio.txt
"""
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

# ── Configuração ──────────────────────────────────────────────────────────────
ARQ_BASE      = r"saida\resultados\base_medicos_validada.csv"
ARQ_CONTACT   = r"dados_crm\2807\crm_contact_report.csv"
ARQ_CORRESP   = r"saida\resultados\correspondencia_contactid.csv"
ARQ_CHECK     = r"saida\resultados\regeocod_endereco_checkpoint.csv"
ARQ_VIACEP    = r"saida\resultados\viacep_logradouros.csv"
ARQ_SAIDA     = r"saida\resultados\base_medicos_endereco.csv"
ARQ_RELATORIO = r"saida\resultados\regeocod_endereco_relatorio.txt"
SALT_PATH     = Path(r"saida\.geomask_salt")

DELAY           = 1.1      # s entre chamadas — política do Nominatim é 1 req/s
TIMEOUT         = 10
UA              = os.environ.get("GEOCODER_CONTACT",
                   "TCC-USP-Esalq-2026 (roteirizacao)")
GEO_MASKING_M   = 300
RAIO_MUNICIPIO_KM = 60     # mesmo limiar da etapa 04c
MIN_MEDICOS_CIDADE = 3     # abaixo disso a mediana da cidade não é referência confiável
CHECKPOINT_A_CADA = 25
LIMITE            = None   # None = base inteira
# ──────────────────────────────────────────────────────────────────────────────

SALT = SALT_PATH.read_text(encoding="utf-8").strip() if SALT_PATH.exists() else ""
if not SALT:
    raise SystemExit("Segredo do geo-masking ausente em {}. Rode a etapa 04 antes.".format(SALT_PATH))

R_TERRA = 6371.0


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return R_TERRA * 2 * np.arcsin(np.sqrt(a))


def geo_mask(lat, lon, medico_id, metros=GEO_MASKING_M):
    """Mesmo deslocamento determinístico das etapas 04 e 04b."""
    rng = np.random.default_rng(zlib.crc32("{}:{}".format(SALT, medico_id).encode("utf-8")))
    dlat = (metros / 111_320) * rng.uniform(-1, 1)
    dlon = (metros / (111_320 * math.cos(math.radians(lat)))) * rng.uniform(-1, 1)
    return round(lat + dlat, 6), round(lon + dlon, 6)


def nominatim(query):
    """Retorna (lat, lon, tipo) ou (None, None, motivo)."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": query, "format": "json", "limit": 1,
                                 "addressdetails": 1, "countrycodes": "br"},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, None, "http_{}".format(r.status_code)
        d = r.json()
        if not d:
            return None, None, "sem_resultado"
        x = d[0]
        return float(x["lat"]), float(x["lon"]), (x.get("addresstype") or x.get("type") or "?")
    except Exception as e:
        return None, None, "erro_{}".format(type(e).__name__)


# ── Carrega base e endereços ──────────────────────────────────────────────────
print("Carregando base e endereços...")
base = pd.read_csv(ARQ_BASE)
contact = pd.read_csv(ARQ_CONTACT, sep=";", dtype={"postcode": str}, low_memory=False)
contact["cep"] = (contact.postcode.astype(str).str.replace(r"\.0$", "", regex=True)
                         .str.replace(r"\D", "", regex=True).str.zfill(8))
corresp = pd.read_csv(ARQ_CORRESP)[["contact_id", "medico_id"]]

end = (corresp.merge(contact[["contact_id", "cep", "street", "building", "city", "region"]],
                     on="contact_id", how="left")
              .set_index("medico_id"))

df = base.copy()
for col in ("cep", "street", "building", "city", "region"):
    df[col] = df.medico_id.map(end[col])

# Referência municipal: mediana das coordenadas ATUAIS por cidade.
# Calculada uma vez, antes de qualquer alteração — é o controle estável.
atual = df[df.latitude.notna()]
# Agrupa por CIDADE + UF: cinco nomes de municipio se repetem entre estados
# (Boa Vista PB/RR, Campo Grande AL/MS, Lajeado RS/TO, Praia Grande SC/SP,
# Santo Andre PB/SP) e agrupar so pelo nome mistura pontos a milhares de km.
ref = (atual.groupby(["CIDADE", "region"])
            .agg(lat_ref=("latitude", "median"), lon_ref=("longitude", "median"),
                 n_cidade=("medico_id", "size"))
            .reset_index())
df = df.merge(ref, on=["CIDADE", "region"], how="left")
print("  {:,} médicos | {:,} cidades com referência".format(len(df), len(ref)))

# ── Logradouros oficiais do ViaCEP (um por CEP único, com cache) ──────────────
# O CRM abrevia ("CONS DANTAS"); o ViaCEP traz a forma expandida, que o
# Nominatim resolve muito melhor.
# Guarda logradouro, localidade E uf: o endereco consultado deve vir todo da
# mesma fonte. Misturar a rua do ViaCEP com a cidade do CRM monta um endereco
# inexistente sempre que os dois discordam — e o cadastro de cidade do CRM erra
# (a etapa 04c identificou 27 casos de cidade divergente).
COLS_VIACEP = ("logradouro", "localidade", "uf")
try:
    vc = pd.read_csv(ARQ_VIACEP, dtype={"cep": str}).fillna("")
    viacep = {r.cep: (r.logradouro, r.localidade, r.uf) for r in vc.itertuples()}
    print("  ViaCEP em cache: {:,} CEPs".format(len(viacep)))
except FileNotFoundError:
    viacep = {}


def salvar_viacep():
    pd.DataFrame([{"cep": k, "logradouro": v[0], "localidade": v[1], "uf": v[2]}
                  for k, v in viacep.items()]).to_csv(ARQ_VIACEP, index=False)


ceps_alvo = [c for c in df.cep.dropna().unique() if c not in viacep]
if ceps_alvo:
    print("  consultando ViaCEP para {:,} CEPs (~{:.1f} min)...".format(
        len(ceps_alvo), len(ceps_alvo) * 0.7 / 60), flush=True)
    for i, cep in enumerate(ceps_alvo, 1):
        try:
            d = requests.get("https://viacep.com.br/ws/{}/json/".format(cep), timeout=8).json()
            viacep[cep] = (("", "", "") if "erro" in d
                           else tuple((d.get(k) or "").strip() for k in COLS_VIACEP))
        except Exception:
            viacep[cep] = ("", "", "")
        if i % 200 == 0:
            salvar_viacep()
            print("    ViaCEP [{:,}/{:,}]".format(i, len(ceps_alvo)), flush=True)
        time.sleep(0.7)
    salvar_viacep()

com_log = sum(1 for c in df.cep.dropna().unique() if (viacep.get(c, ("",))[0] or "").strip())
print("  CEPs com logradouro oficial: {:,} de {:,}".format(com_log, df.cep.nunique()), flush=True)

# ── Elegibilidade à consulta por endereço ────────────────────────────────────
# Avaliada sobre o endereço JÁ ENRIQUECIDO, e não sobre os campos crus: o
# logradouro, o município e a UF saem do ViaCEP quando ele conhece o código,
# e do cadastro quando não conhece. O número é o único campo sem substituto,
# porque o código postal identifica a via e não o imóvel.
def _via_completo(c):
    v = viacep.get(c, ("", "", ""))
    return all(str(x).strip() for x in v)


tem_numero = df.building.notna() & (df.building.astype(str).str.strip() != "")
via_ok = df.cep.map(_via_completo)
cad_ok = (df.street.notna() & (df.street.astype(str).str.strip() != "")
          & df.city.notna() & (df.city.astype(str).str.strip() != "")
          & df.region.notna() & (df.region.astype(str).str.strip() != ""))
tem_endereco = tem_numero & (via_ok | cad_ok)
alvo = df[tem_endereco].copy()
if LIMITE:
    alvo = alvo.head(LIMITE)
print("  {:,} médicos elegíveis à consulta por endereço".format(len(alvo)))
print("     dos quais {:,} só o são pelo endereçamento oficial"
      .format(int((tem_numero & via_ok & ~cad_ok).sum())))

# ── Checkpoint ────────────────────────────────────────────────────────────────
try:
    ck = pd.read_csv(ARQ_CHECK)
    cache = {r.medico_id: (r.lat, r.lon, r.tipo) for r in ck.itertuples()}
    print("  checkpoint: {:,} já consultados".format(len(cache)))
except FileNotFoundError:
    cache = {}


def salvar_checkpoint():
    pd.DataFrame([{"medico_id": k, "lat": v[0], "lon": v[1], "tipo": v[2]}
                  for k, v in cache.items()]).to_csv(ARQ_CHECK, index=False)


# ── Consulta ──────────────────────────────────────────────────────────────────
novas = 0
pendentes = [r for r in alvo.itertuples() if r.medico_id not in cache]
print("  a consultar agora: {:,}  (~{:.1f} h)\n".format(len(pendentes), len(pendentes) * DELAY / 3600))

for i, r in enumerate(pendentes, 1):
    # Endereco de fonte unica: rua, cidade e UF saem todos do ViaCEP quando ele
    # conhece o CEP. So o numero vem do CRM, porque o ViaCEP nao tem numero.
    # Sem logradouro oficial (CEP de localidade ou inexistente), tudo vem do CRM.
    log, loc, uf = viacep.get(r.cep, ("", "", ""))
    if log:
        rua, cidade, estado = log, (loc or str(r.city).strip()), (uf or str(r.region).strip())
    else:
        rua, cidade, estado = str(r.street).strip(), str(r.city).strip(), str(r.region).strip()
    q = "{}, {}, {}, {}, Brasil".format(rua, str(r.building).strip(), cidade, estado)
    la, lo, tipo = nominatim(q)
    cache[r.medico_id] = (la, lo, tipo)
    novas += 1
    if i % CHECKPOINT_A_CADA == 0:
        salvar_checkpoint()
        print("  [{:,}/{:,}] {:.1f}%".format(i, len(pendentes), 100 * i / len(pendentes)), flush=True)
    time.sleep(DELAY)

salvar_checkpoint()

# ── Aplica a regra de aceitação ───────────────────────────────────────────────
print("\nAplicando os dois testes de aceitação...")
motivos = []
aceitos = 0
df["_origem_coord"] = np.where(df.latitude.notna(), "cep_anterior", "sem_coordenada")
df["_km_movido"] = np.nan

idx = df.set_index("medico_id").index
pos = {m: k for k, m in enumerate(df.medico_id)}

for mid, (la, lo, tipo) in cache.items():
    k = pos.get(mid)
    if k is None:
        continue
    linha = df.iloc[k]

    if la is None or (isinstance(la, float) and np.isnan(la)):
        motivos.append(tipo if isinstance(tipo, str) else "sem_resultado")
        continue

    ok, motivo = vg.validar(la, lo, linha.cep, tipo)
    if not ok:
        motivos.append("validacao_geo:{}".format(motivo))
        continue

    # Teste municipal. Toda coordenada precisa de UMA referencia aprovada.
    # Preferencia: a mediana do municipio (robusta com 3+ medicos).
    # Sem ela, cai para a propria coordenada anterior, que veio do codigo
    # postal e e confiavel em nivel de municipio.
    # Versao anterior deste bloco ACEITAVA quando nao havia referencia, o que
    # deixou passar 4 coordenadas erradas (ate 227 km) em cidades com 1-2
    # medicos. O padrao agora e recusar.
    if pd.notna(linha.lat_ref) and linha.n_cidade >= MIN_MEDICOS_CIDADE:
        referencia = (linha.lat_ref, linha.lon_ref)
    elif pd.notna(linha.latitude):
        referencia = (linha.latitude, linha.longitude)
    else:
        motivos.append("sem_referencia_municipal")
        continue
    if haversine(la, lo, referencia[0], referencia[1]) > RAIO_MUNICIPIO_KM:
        motivos.append("fora_do_municipio")
        continue

    lat_m, lon_m = geo_mask(la, lo, mid)
    if pd.notna(linha.latitude):
        df.iat[k, df.columns.get_loc("_km_movido")] = haversine(
            linha.latitude, linha.longitude, lat_m, lon_m)
    df.iat[k, df.columns.get_loc("latitude")] = lat_m
    df.iat[k, df.columns.get_loc("longitude")] = lon_m
    df.iat[k, df.columns.get_loc("_geo_estrategia")] = "endereco_nominatim"
    df.iat[k, df.columns.get_loc("_origem_coord")] = "endereco_novo"
    aceitos += 1

# ── Salva ─────────────────────────────────────────────────────────────────────
saida = df.drop(columns=["cep", "street", "building", "city", "region",
                         "lat_ref", "lon_ref", "n_cidade"])
saida.to_csv(ARQ_SAIDA, index=False)

# ── Relatório ─────────────────────────────────────────────────────────────────
L = []
L.append("=" * 70)
L.append("  ETAPA 04d — REGEOCODIFICACAO POR ENDERECO COMPLETO")
L.append("=" * 70)
L.append("  medicos na base                : {:,}".format(len(df)))
L.append("  com endereco para consultar    : {:,}".format(len(alvo)))
L.append("  consultas nesta rodada         : {:,}".format(novas))
L.append("  coordenadas ACEITAS (novas)    : {:,}".format(aceitos))
L.append("  mantiveram a coordenada antiga : {:,}".format(len(motivos)))
L.append("")
L.append("  Motivos de rejeicao:")
for m, n in pd.Series(motivos).value_counts().items():
    L.append("    {:34s} {:6,}".format(str(m)[:34], n))
mov = df._km_movido.dropna()
if len(mov):
    L.append("")
    L.append("  DESLOCAMENTO das coordenadas aceitas (km):")
    L.append("    mediana {:.2f} | media {:.2f} | p90 {:.2f} | max {:.2f}".format(
        mov.median(), mov.mean(), mov.quantile(.9), mov.max()))
    for lim in (0.5, 2, 10):
        L.append("    movem mais de {:4.1f} km: {:,} ({:.1f}%)".format(
            lim, (mov > lim).sum(), 100 * (mov > lim).mean()))
L.append("")
com = saida.latitude.notna()
L.append("  COBERTURA FINAL: {:,}/{:,} ({:.1f}%)".format(com.sum(), len(saida), 100 * com.mean()))
L.append("")
L.append("  Composicao por estrategia:")
for e, n in saida[com]._geo_estrategia.value_counts().items():
    L.append("    {:24s} {:6,}  ({:4.1f}%)".format(str(e), n, 100 * n / com.sum()))
txt = "\n".join(L)
Path(ARQ_RELATORIO).write_text(txt, encoding="utf-8")
print("\n" + txt)
print("\n[ok] {}".format(ARQ_SAIDA))
