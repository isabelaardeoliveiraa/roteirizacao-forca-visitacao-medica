"""
Fallback de geocodificação via OpenCage — médicos sem coordenadas (base do CRM).

Usa endereço completo (rua + número + CEP + cidade) do crm_contact_report.csv
para geocodificar os médicos que falharam na etapa 04 via CEP/Nominatim.
Aplica geo-masking de até 300m (LGPD) nos resultados.

Input : saida/resultados/base_medicos_final.csv
        saida/resultados/correspondencia_contactid.csv
        dados_crm/crm_contact_report.csv
Saída : saida/resultados/base_medicos_final.csv  (atualizado in-place)
"""

import os
import time, math, zlib, sys, requests
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import validacao_geo as vg

# ── Configurações ──────────────────────────────────────────────────────────────
# A chave nao fica no codigo: o repositorio do trabalho e publicavel.
# Defina antes de rodar:  $env:OPENCAGE_API_KEY = "<sua chave>"
API_KEY = os.environ.get("OPENCAGE_API_KEY")
if not API_KEY:
    raise SystemExit("Defina a variavel de ambiente OPENCAGE_API_KEY")
ARQ_BASE      = r"saida\resultados\base_medicos_final.csv"
ARQ_CORRESP   = r"saida\resultados\correspondencia_contactid.csv"
ARQ_CONTACT   = r"dados_crm\2807\crm_contact_report.csv"
ARQ_CHECK     = r"saida\opencage_checkpoint.csv"
SALT_PATH     = Path(r"saida\.geomask_salt")
GEO_MASKING_M = 300
DELAY_SEG     = 1.1
LIMITE_DIARIO = 2400          # free tier = 2.500/dia; margem de segurança
# ───────────────────────────────────────────────────────────────────────────────

# Mesmo segredo do etapa_04: o mascaramento precisa ser idêntico em todas as
# etapas, senão o mesmo médico teria deslocamentos diferentes conforme a origem
# da coordenada.
SALT = SALT_PATH.read_text(encoding="utf-8").strip() if SALT_PATH.exists() else ""
if not SALT:
    raise SystemExit(
        f"Segredo do geo-masking nao encontrado em {SALT_PATH}.\n"
        "Rode etapa_04_geocodificacao.py primeiro (ele gera o arquivo)."
    )


def geo_mask(lat, lon, medico_id, max_metros=GEO_MASKING_M):
    """Deslocamento determinístico por médico — igual ao do etapa_04."""
    rng = np.random.default_rng(zlib.crc32(f"{SALT}:{medico_id}".encode("utf-8")))
    dlat = (max_metros / 111_320) * rng.uniform(-1, 1)
    dlon = (max_metros / (111_320 * math.cos(math.radians(lat)))) * rng.uniform(-1, 1)
    return round(lat + dlat, 6), round(lon + dlon, 6)


def geocodar_opencage(query, cep):
    """Retorna (lat, lon, motivo_rejeicao). Aceita só resultado que passe na
    validação geográfica — sem isso, um match amplo entra como se fosse endereço."""
    url = "https://api.opencagedata.com/geocode/v1/json"
    try:
        r = requests.get(url, params={
            "q": query, "key": API_KEY,
            "language": "pt", "countrycode": "br",
            "limit": 3, "no_annotations": 1,
        }, timeout=10)
        data = r.json()
        motivo = "sem_resultado"
        for res in data.get("results", []):
            geo = res["geometry"]
            tipo = (res.get("components", {}) or {}).get("_type")
            ok, m = vg.validar(geo["lat"], geo["lng"], cep, tipo)
            if ok:
                return geo["lat"], geo["lng"], None
            motivo = m
        return None, None, motivo
    except Exception as e:
        print(f"    Erro: {e}")
        return None, None, "erro_api"

# ── Carrega base e limpa coords do OpenCage (restaura sem coordenadas) ─────────
df = pd.read_csv(ARQ_BASE, dtype=str)
df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Restaura registros geocodificados via OpenCage para None (vamos refazer com endereço)
mask_oc = df["_geo_estrategia"].isin(["opencage_cidade", "opencage_cidade_centroide"])
df.loc[mask_oc, ["latitude", "longitude", "_geo_estrategia"]] = [None, None, "cep"]
sem_coord = df[df["latitude"].isna()].copy()
print(f"\n{'='*60}")
print(f"  FALLBACK GEOCODIFICAÇÃO — OPENCAGE (endereço completo)")
print(f"{'='*60}")
print(f"  Médicos sem coordenadas : {len(sem_coord)}")

# ── Monta tabela de endereços reais ───────────────────────────────────────────
corresp  = pd.read_csv(ARQ_CORRESP)[["medico_id", "contact_id"]]
contact  = pd.read_csv(ARQ_CONTACT, sep=";")[
    ["contact_id", "street", "building", "postcode", "city"]
]
enderecos = corresp.merge(contact, on="contact_id", how="left")
enderecos["postcode"] = enderecos["postcode"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

# Índice medico_id → endereço
end_idx = enderecos.set_index("medico_id")

# ── Checkpoint (retoma de onde parou; protege a cota da API) ──────────────────
try:
    cache = pd.read_csv(ARQ_CHECK, dtype={"medico_id": str})
    resolvidos = {r.medico_id: (r.lat, r.lon, r.estrategia) for r in cache.itertuples()}
    print(f"  Checkpoint: {len(resolvidos)} medicos ja consultados anteriormente")
except FileNotFoundError:
    resolvidos = {}

# ── Geocodifica ───────────────────────────────────────────────────────────────
sucesso = sucesso_cep = sucesso_rua = falha = 0
consultas = 0
rejeicoes = []

for idx, row in sem_coord.iterrows():
    mid = row["medico_id"]

    # Reaproveita consulta anterior (nao gasta cota)
    if mid in resolvidos:
        la, lo, est = resolvidos[mid]
        if pd.notna(la):
            lat_m, lon_m = geo_mask(float(la), float(lo), mid)
            df.at[idx, "latitude"] = lat_m
            df.at[idx, "longitude"] = lon_m
            df.at[idx, "_geo_estrategia"] = est
            sucesso += 1
            sucesso_cep += est == "opencage_cep"
            sucesso_rua += est == "opencage_endereco"
        else:
            falha += 1
        continue

    if consultas >= LIMITE_DIARIO:
        print(f"\n  LIMITE DIARIO DE {LIMITE_DIARIO} CONSULTAS ATINGIDO.")
        print("  Rode novamente amanha — o checkpoint retoma daqui.")
        break

    end = end_idx.loc[mid] if mid in end_idx.index else None
    rua    = str(end["street"]   if end is not None else "").strip()
    numero = str(end["building"] if end is not None else "").strip()
    cep    = str(end["postcode"] if end is not None else "").strip()

    # Só endereço específico. O fallback "somente cidade" foi removido: ele
    # devolve o centroide do município, colocando todos os médicos da cidade no
    # mesmo ponto — foi o que gerou as coordenadas erradas de Brasília na
    # primeira rodada. Sem endereço, é mais honesto ficar sem coordenada.
    if rua and cep and len(cep) >= 8:
        query, estrategia = f"{rua}, {numero}, {cep}, Brasil", "opencage_cep"
    elif rua and str(row.get("CIDADE", "") or "").strip():
        query, estrategia = f"{rua}, {numero}, {row['CIDADE']}, Brasil", "opencage_endereco"
    else:
        falha += 1
        resolvidos[mid] = (None, None, "sem_endereco")
        continue

    lat, lon, motivo = geocodar_opencage(query, cep)
    consultas += 1
    time.sleep(DELAY_SEG)

    resolvidos[mid] = (lat, lon, estrategia if lat is not None else motivo)

    if lat is not None:
        lat_m, lon_m = geo_mask(lat, lon, mid)
        df.at[idx, "latitude"]        = lat_m
        df.at[idx, "longitude"]       = lon_m
        df.at[idx, "_geo_estrategia"] = estrategia
        sucesso += 1
        sucesso_cep += estrategia == "opencage_cep"
        sucesso_rua += estrategia == "opencage_endereco"
    else:
        falha += 1
        rejeicoes.append(motivo)

    if consultas % 25 == 0:
        pd.DataFrame([{"medico_id": k, "lat": v[0], "lon": v[1], "estrategia": v[2]}
                      for k, v in resolvidos.items()]).to_csv(ARQ_CHECK, index=False)
        print(f"    [{consultas}] consultas | {sucesso} ok | {falha} sem resultado")

pd.DataFrame([{"medico_id": k, "lat": v[0], "lon": v[1], "estrategia": v[2]}
              for k, v in resolvidos.items()]).to_csv(ARQ_CHECK, index=False)

# ── Resumo ────────────────────────────────────────────────────────────────────
total_geocod = df["latitude"].notna().sum()
print(f"\n{'='*60}")
print(f"  RESUMO")
print(f"{'='*60}")
print(f"  Consultas gastas nesta rodada        : {consultas}")
print(f"  Via CEP completo (opencage_cep)      : {sucesso_cep}")
print(f"  Via rua+cidade (opencage_endereco)   : {sucesso_rua}")
print(f"  Sem resultado                        : {falha}")
if rejeicoes:
    print("\n  Motivos de rejeicao:")
    for m, n in pd.Series(rejeicoes).value_counts().items():
        print(f"    {m}: {n}")
print(f"\n  Total geocodificados                 : {total_geocod}/{len(df)} ({total_geocod/len(df)*100:.1f}%)")

# ── Salva ─────────────────────────────────────────────────────────────────────
df["latitude"]  = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)
df.to_csv(ARQ_BASE, index=False)
print(f"\n  Base salva: {ARQ_BASE}\n")
