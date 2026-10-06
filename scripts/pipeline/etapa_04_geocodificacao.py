### Geocodificação e Geo-masking de médicos (base do CRM) a partir de CEPs
# Equivalente ao 04_geocodificacao.py, aplicado a base_medicos_corrigida.csv.
# Reaproveita o checkpoint já existente (geocodificacao_checkpoint.csv) como cache,
# já que ~98% dos CEPs coincidem entre a base antiga e a nova — evita
# re-consultar APIs para CEPs já geocodificados.

import pandas as pd
import requests
import numpy as np
import time
import zlib
import secrets
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import validacao_geo as vg

rejeitados = {}   # cep -> motivo da rejeição (relatório no fim)

df = pd.read_csv(r"saida\resultados\base_medicos_corrigida.csv", dtype={"CEP": str})

# ── ETAPA 1: Extrai CEPs únicos ──────────────────────────────
ceps_unicos = df['CEP'].dropna().unique()
print(f"Total medicos: {len(df)}")
print(f"CEPs unicos: {len(ceps_unicos)}")

# ── ETAPA 2: Geocodifica só os CEPs únicos ───────────────────
def geocodificar_cep(cep):
    try:
        cep_limpo = str(cep).replace("-","").replace(" ","").zfill(8)

        # Tenta ViaCEP primeiro para obter endereço estruturado
        resp = requests.get(
            f"https://viacep.com.br/ws/{cep_limpo}/json/",
            timeout=5
        )
        dados = resp.json()

        if "erro" in dados:
            return None, None

        # Monta endereço para Nominatim
        endereco = (
            f"{dados.get('logradouro','')}, "
            f"{dados.get('localidade','')}, "
            f"{dados.get('uf','')}, Brasil"
        )

        time.sleep(1)  # respeita limite da API

        resp_geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": endereco, "format": "json", "limit": 3,
                    "addressdetails": 1, "countrycodes": "br"},
            headers={"User-Agent": "TCC-USP-Esalq-2026"},
            timeout=5
        )
        resultados = resp_geo.json()

        # Aceita o primeiro resultado que passe na validação geográfica.
        # Sem isso, o Nominatim devolve o centroide do Brasil quando não acha
        # o logradouro — e a coordenada errada entra na base como válida.
        motivo_ultimo = "sem_resultado"
        for r in resultados or []:
            lat, lon = float(r["lat"]), float(r["lon"])
            tipo = r.get("addresstype") or r.get("type")
            ok, motivo = vg.validar(lat, lon, cep_limpo, tipo)
            if ok:
                return lat, lon
            motivo_ultimo = motivo
        rejeitados[cep_limpo] = motivo_ultimo
        return None, None

    except:
        return None, None

# Geocodifica com checkpoint a cada 10 CEPs (reaproveita checkpoint da base antiga)
resultados = {}
checkpoint_path = r"saida\geocodificacao_checkpoint.csv"

def normaliza_cep(serie):
    """CEP sempre como 8 digitos. Sem isso, o pandas le '01526020' como float e
    grava '1526020.0' no checkpoint — a chave deixa de casar e tudo e
    regeocodificado do zero."""
    return (serie.astype(str)
                 .str.replace(r"\.0$", "", regex=True)
                 .str.replace(r"\D", "", regex=True)
                 .str.zfill(8))

try:
    df_check = pd.read_csv(checkpoint_path, dtype={"cep": str})
    resultados = dict(zip(normaliza_cep(df_check['cep']),
                         zip(df_check['lat'], df_check['lon'])))
    print(f"Checkpoint carregado: {len(resultados)} CEPs ja processados")

    # O cache foi construído antes da validação existir e contém coordenadas
    # inválidas (centroide do Brasil etc). Descarta as ruins para que sejam
    # reconsultadas — as que falharem de novo caem no fallback do 04b.
    invalidas = {}
    for cep, (la, lo) in list(resultados.items()):
        if pd.isna(la) or pd.isna(lo):
            continue
        ok, motivo = vg.validar(la, lo, cep)
        if not ok:
            invalidas[cep] = motivo
            del resultados[cep]
    if invalidas:
        print(f"  {len(invalidas)} CEPs do cache reprovados na validacao e removidos:")
        for motivo, n in pd.Series(list(invalidas.values())).value_counts().items():
            print(f"    {motivo}: {n}")
except FileNotFoundError:
    print("Iniciando do zero")

ceps_unicos_norm = normaliza_cep(pd.Series(ceps_unicos)).unique()
ja_em_cache = sum(1 for c in ceps_unicos_norm if c in resultados)
print(f"CEPs ja no cache: {ja_em_cache} / {len(ceps_unicos_norm)}")

for i, cep in enumerate(ceps_unicos_norm):

    # Pula se já processou
    if cep in resultados:
        continue

    print(f"  [{i+1}/{len(ceps_unicos_norm)}] Processando CEP {cep}...", end=" ", flush=True)
    lat, lon = geocodificar_cep(cep)
    resultados[cep] = (lat, lon)
    status = f"lat={lat:.4f}" if lat else "sem resultado"
    print(status)

    # Salva checkpoint a cada 10
    if i % 10 == 0:
        df_checkpoint = pd.DataFrame([
            {'cep': k, 'lat': v[0], 'lon': v[1]}
            for k, v in resultados.items()
        ])
        df_checkpoint.to_csv(checkpoint_path, index=False)
        sucesso = sum(1 for v in resultados.values() if v[0])
        print(f"  >>> Checkpoint: {len(resultados)} processados | {sucesso} com sucesso")

# Salva checkpoint final atualizado
df_checkpoint = pd.DataFrame([
    {'cep': k, 'lat': v[0], 'lon': v[1]}
    for k, v in resultados.items()
])
df_checkpoint.to_csv(checkpoint_path, index=False)

# ── ETAPA 3: Join com base de médicos ────────────────────────
df['CEP'] = df['CEP'].astype(str).str.zfill(8)
df_ceps = df_checkpoint.rename(columns={'cep': 'CEP'})
df = df.merge(df_ceps, on='CEP', how='left')

# ── ETAPA 4: Geo-masking ─────────────────────────────────────
# O deslocamento é determinístico POR MÉDICO (reprodutibilidade metodológica) mas
# NÃO reversível por quem só tem a base publicada: a semente combina o medico_id
# com um segredo local que nunca é versionado nem entregue com o TCC.
# Sem o segredo, recalcular o deslocamento e desfazer o mascaramento é inviável —
# a proteção está na chave, não no sigilo do algoritmo.
SALT_PATH = Path(r"saida\.geomask_salt")
if SALT_PATH.exists():
    SALT = SALT_PATH.read_text(encoding="utf-8").strip()
    print(f"Geo-masking: segredo carregado de {SALT_PATH}")
else:
    SALT = secrets.token_hex(32)
    SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SALT_PATH.write_text(SALT, encoding="utf-8")
    print(f"Geo-masking: segredo NOVO gerado em {SALT_PATH}")
    print("  ATENCAO: guarde esse arquivo junto das tabelas de correspondencia.")
    print("  Se ele for perdido, as coordenadas mascaradas mudam e os resultados")
    print("  publicados deixam de ser reproduziveis. NAO versionar, NAO enviar ao TCC.")


def geo_masking(lat, lon, medico_id, raio=300):
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    semente = zlib.crc32(f"{SALT}:{medico_id}".encode("utf-8"))
    rng = np.random.default_rng(semente)
    delta_lat = (raio / 111320) * rng.uniform(-1, 1)
    delta_lon = (raio / (111320 * np.cos(np.radians(lat)))) \
                * rng.uniform(-1, 1)
    return round(lat + delta_lat, 6), round(lon + delta_lon, 6)

coords = df.apply(
    lambda row: geo_masking(row['lat'], row['lon'], row['medico_id']), axis=1
)
df['latitude']  = [c[0] for c in coords]
df['longitude'] = [c[1] for c in coords]

# ── ETAPA 5: Remove dados sensíveis ──────────────────────────
df['cep_faixa'] = df['CEP'].astype(str).str[:5] + "000"
df = df.drop(columns=['CEP', 'ENDERECO', 'lat', 'lon'])

# ── ETAPA 6: Salva base final ─────────────────────────────────
df.to_csv(r"saida\resultados\base_medicos_final.csv", index=False)

# Relatório final
total = len(df)
ok    = df['latitude'].notna().sum()
falha = total - ok

print(f"\n{'='*46}")
print(f"Total de medicos  : {total:,}")
print(f"Geocodificados    : {ok:,} ({ok/total*100:.1f}%)")
print(f"Falhas            : {falha:,} ({falha/total*100:.1f}%)")
print(f"{'='*46}")

if rejeitados:
    print(f"\nCEPs rejeitados pela validacao geografica: {len(rejeitados)}")
    for motivo, n in pd.Series(list(rejeitados.values())).value_counts().items():
        print(f"  {motivo}: {n}")
    pd.DataFrame([{"cep": c, "motivo": m} for c, m in rejeitados.items()]).to_csv(
        r"saida\resultados\geocodificacao_rejeitados.csv", index=False)
    print("  detalhe -> saida/resultados/geocodificacao_rejeitados.csv")
    print("  (esses CEPs seguem para o fallback por endereco no 04b)")
