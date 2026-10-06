"""
Etapa 4c — Validação por coerência com o município (e reparo).

Por que existe
--------------
A validação de etapa_04 (validacao_geo) usa uma caixa retangular por UF. A
caixa de SP invade o Paraná, então uma coordenada 362 km fora passou: 7 médicos
de um mesmo bairro foram parar no estado vizinho, e isso bloqueou duas
semanas inteiras no VRP por "dispersão".

O teste aqui não depende da geometria do estado. A pergunta é direta:
**a coordenada cai perto do centro do município que o CRM declara?**

O centro do município é a MEDIANA das coordenadas de todos os médicos daquela
cidade na própria base. Com dezenas a milhares de médicos por cidade, a mediana
é robusta — alguns pontos errados não a deslocam.

Como o caso é classificado
--------------------------
Quando a coordenada discorda da cidade, ainda falta saber QUEM errou. O critério
de desempate é o CEP: comparam-se os médicos que compartilham o mesmo prefixo de
5 dígitos (mesmo bairro, na prática).

  · coordenada longe da cidade E longe dos vizinhos de CEP
      -> a COORDENADA está errada. Reparo: assume a mediana dos vizinhos de CEP.
  · coordenada longe da cidade MAS junto dos vizinhos de CEP
      -> a coordenada está certa e o campo CIDADE do CRM é que está errado.
         Não se mexe na coordenada; o caso vai para correção de cadastro.
  · sem vizinhos de CEP suficientes para decidir
      -> quarentena: a coordenada é removida. Roteirizar com ponto errado é pior
         que roteirizar sem o médico, porque o erro se propaga em silêncio.

Sobre o geo-masking: as coordenadas reparadas são a mediana de pontos que já
passaram pelo deslocamento de 300 m, então o resultado continua mascarado. Não
se aplica máscara nova para não somar dois deslocamentos.

Entrada : saida/resultados/base_medicos_final.csv
Saídas  : saida/resultados/base_medicos_validada.csv   (consumida pela etapa 5)
          saida/resultados/validacao_municipio.csv             (todos os casos tratados)
          saida/resultados/cadastro_cidade_divergente.csv      (para correção no CRM)
"""

import numpy as np
import pandas as pd

ARQ_ENTRADA = "saida/resultados/base_medicos_endereco.csv"   # saida da 04d
ARQ_SAIDA   = r"saida\resultados\base_medicos_validada.csv"

RAIO_MUNICIPIO_KM = 60   # além disso, coordenada e cidade discordam
RAIO_CEP_KM       = 20   # dentro disso, coordenada concorda com os vizinhos de CEP
MIN_PARA_MEDIANA  = 3    # vizinhos necessários para a mediana valer como referência

print("\n" + "=" * 62)
print("  ETAPA 4c — VALIDAÇÃO POR MUNICÍPIO")
print("=" * 62)

df = pd.read_csv(ARQ_ENTRADA, dtype=str)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

tem_coord = df["latitude"].notna() & df["longitude"].notna()
print(f"  Médicos na base          : {len(df):,}")
print(f"  Com coordenada           : {tem_coord.sum():,}")


def km_ate(lat, lon, ref_lat, ref_lon):
    """Distância aproximada em km — suficiente para triagem nesta escala."""
    return np.hypot((lat - ref_lat) * 111.32,
                    (lon - ref_lon) * 111.32 * np.cos(np.radians(lat)))


# ── Referências: centro do município e centro do bairro (prefixo de CEP) ──────
g = df[tem_coord]
centro_cidade = g.groupby("CIDADE")[["latitude", "longitude"]].median()

df["_pref"] = df["cep_faixa"].astype(str).str[:5]
pref = g.assign(_pref=g["cep_faixa"].astype(str).str[:5]).groupby("_pref")
centro_cep = pref[["latitude", "longitude"]].median()
n_cep = pref.size()

df["_km_cidade"] = km_ate(df["latitude"], df["longitude"],
                          df["CIDADE"].map(centro_cidade["latitude"]),
                          df["CIDADE"].map(centro_cidade["longitude"]))
df["_km_cep"] = km_ate(df["latitude"], df["longitude"],
                       df["_pref"].map(centro_cep["latitude"]),
                       df["_pref"].map(centro_cep["longitude"]))
df["_n_cep"] = df["_pref"].map(n_cep).fillna(0)
df["validacao_municipio"] = ""

suspeito = tem_coord & (df["_km_cidade"] > RAIO_MUNICIPIO_KM)
print(f"  Coordenada x município discordam: {suspeito.sum():,} "
      f"({suspeito.sum()/tem_coord.sum()*100:.2f}%)")


# ── Recuperação: médicos sem coordenada cujo bairro já é conhecido ────────────
# A geocodificação procura pelo CEP COMPLETO do médico (76821405). Quando esse
# CEP específico falha nas APIs, o médico fica sem coordenada — mesmo quando
# dezenas de colegas do mesmo bairro (prefixo 76821) já foram localizados.
# Aqui esses casos herdam o centro do bairro. A precisão cai de porta para
# quarteirão, o que é adequado para decidir rota e é registrado como tal.
recuperavel = (~tem_coord) & (df["_n_cep"] >= MIN_PARA_MEDIANA)
n_rec = int(recuperavel.sum())
if n_rec:
    df.loc[recuperavel, "latitude"] = df.loc[recuperavel, "_pref"].map(centro_cep["latitude"])
    df.loc[recuperavel, "longitude"] = df.loc[recuperavel, "_pref"].map(centro_cep["longitude"])
    df.loc[recuperavel, "_geo_estrategia"] = "cep_bairro"
    df.loc[recuperavel, "validacao_municipio"] = "recuperado_pelo_bairro"
print(f"  Sem coordenada, recuperados pelo bairro: {n_rec:,}")


# ── Classificação e reparo ────────────────────────────────────────────────────
# O médico não conta como vizinho de si mesmo: com poucos pontos no prefixo, ele
# puxaria a mediana para a própria coordenada errada e o caso se auto-inocentaria.
pode_decidir = df["_n_cep"] >= (MIN_PARA_MEDIANA + 1)

coord_errada = suspeito & pode_decidir & (df["_km_cep"] > RAIO_CEP_KM)
cidade_errada = suspeito & pode_decidir & (df["_km_cep"] <= RAIO_CEP_KM)
indecidivel = suspeito & ~pode_decidir

df.loc[coord_errada, "validacao_municipio"] = "coordenada_reparada"
df.loc[cidade_errada, "validacao_municipio"] = "cadastro_cidade_divergente"
df.loc[indecidivel, "validacao_municipio"] = "quarentena_sem_referencia"

registro = df[suspeito].copy()
registro["latitude_original"] = registro["latitude"]
registro["longitude_original"] = registro["longitude"]

# reparo: assume o centro do próprio bairro (prefixo de CEP)
df.loc[coord_errada, "latitude"] = df.loc[coord_errada, "_pref"].map(centro_cep["latitude"])
df.loc[coord_errada, "longitude"] = df.loc[coord_errada, "_pref"].map(centro_cep["longitude"])

# quarentena: sem coordenada é melhor que com coordenada errada
df.loc[indecidivel, ["latitude", "longitude"]] = np.nan

registro["latitude_corrigida"] = registro["medico_id"].map(
    df.set_index("medico_id")["latitude"].to_dict())
registro["longitude_corrigida"] = registro["medico_id"].map(
    df.set_index("medico_id")["longitude"].to_dict())

print("")
print(f"  {'classificação':<32}{'médicos':>9}")
for k, n in df.loc[suspeito, "validacao_municipio"].value_counts().items():
    print(f"  {k:<32}{n:>9,}")

print("")
print("  Por estratégia de geocodificação — taxa de coordenada errada:")
base = df[tem_coord].groupby(df["_geo_estrategia"].fillna("?")).size()
err = df[coord_errada].groupby(df["_geo_estrategia"].fillna("?")).size()
for est in base.index:
    e = int(err.get(est, 0))
    print(f"    {est:<24}{e:>5} de {base[est]:>6,}  ({e/base[est]*100:>5.2f}%)")


# ── Saídas ────────────────────────────────────────────────────────────────────
cols = ["medico_id", "CIDADE", "BAIRRO", "cep_faixa", "_geo_estrategia",
        "validacao_municipio", "latitude_original", "longitude_original",
        "latitude_corrigida", "longitude_corrigida", "_km_cidade", "_km_cep", "_n_cep"]
registro[cols].sort_values("_km_cidade", ascending=False).to_csv(
    r"saida\resultados\validacao_municipio.csv", index=False)

cad = registro[registro["validacao_municipio"] == "cadastro_cidade_divergente"]
if len(cad):
    cad[["medico_id", "CIDADE", "BAIRRO", "cep_faixa", "_km_cidade"]].rename(
        columns={"CIDADE": "cidade_no_crm", "_km_cidade": "km_da_cidade_declarada"}
    ).to_csv(r"saida\resultados\cadastro_cidade_divergente.csv", index=False)

df.drop(columns=["_pref", "_km_cidade", "_km_cep", "_n_cep"]).to_csv(
    ARQ_SAIDA, index=False)

restantes = df["latitude"].notna().sum()
print("")
print(f"  Coordenadas antes        : {tem_coord.sum():,} "
      f"({tem_coord.sum()/len(df)*100:.1f}%)")
print(f"    + recuperadas pelo bairro : {n_rec:>6,}")
print(f"    - postas em quarentena    : {int(indecidivel.sum()):>6,}")
print(f"  Coordenadas depois       : {restantes:,} "
      f"({restantes/len(df)*100:.1f}%)")
print(f"\n  Saídas: {ARQ_SAIDA}")
print(f"          saida/resultados/validacao_municipio.csv")
print(f"          saida/resultados/cadastro_cidade_divergente.csv")
