"""
Corrige CEPs da base anonimizada do CRM com base no diagnóstico
(etapa_02_diagnostico.py).

Equivalente ao 03_correcao_cep.py, aplicado a base_medicos_anonimizada.csv.

Estratégia: nenhum médico fica sem coordenada — o fallback é endereço+cidade.

  ok                      → CEP mantido
  corrigir_zero_inicial   → zero adicionado à esquerda (cidade confirmada pela API)
  cep_incerto             → CEP corrigido mesmo assim; geocodificação usará endereço como fallback
  suspeito_cidade_errada  → CEP mantido, cidade corrigida pelo retorno da API
  suspeito_repetido       → CEP mantido, cidade corrigida pelo retorno da API
  cep_inexistente         → CEP mantido para registro; geocodificação usará endereço como fallback
  irrecuperavel           → CEP mantido como está; geocodificação usará endereço como fallback

Saídas:
  saida/base_medicos_corrigida.csv → input do etapa_04_geocodificacao.py
  saida/ceps_para_revisao.csv      → registros que precisam de atenção (para auditoria)
"""

import pandas as pd

BASE        = r"saida\resultados\base_medicos_anonimizada.csv"
DIAGNOSTICO = r"saida\resultados\problemas_cep.csv"
SAIDA       = r"saida\resultados\base_medicos_corrigida.csv"
REVISAO     = r"saida\resultados\ceps_para_revisao.csv"

# ── carga ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(BASE, dtype=str)

diag = pd.read_csv(DIAGNOSTICO, dtype=str)[
    ["medico_id", "_cep_norm", "_cep_acao", "_api_cidade"]
]

df = df.merge(diag, on="medico_id", how="left")

# Registros sem linha no diagnóstico = CEP ok (não entrou no arquivo de problemas)
df["_cep_acao"]   = df["_cep_acao"].fillna("ok")
df["_cep_norm"]   = df["_cep_norm"].fillna(
    df["CEP"].str.replace(r"[^0-9]", "", regex=True).str.zfill(8)
)
df["_api_cidade"] = df["_api_cidade"].fillna(pd.NA)

# ── 1. corrigir CEP ───────────────────────────────────────────────────────────
# Todos os casos recebem o CEP normalizado (_cep_norm).
# Para irrecuperáveis (ausente, zerado, curto, longo) mantemos o valor original
# pois _cep_norm não tem significado nesses casos.

mask_irr = df["_cep_acao"] == "irrecuperavel"

df["CEP_corrigido"] = df["_cep_norm"]
df.loc[mask_irr, "CEP_corrigido"] = df.loc[mask_irr, "CEP"]

# ── 2. corrigir CIDADE onde a API devolveu valor confiável ────────────────────
# Aplica para casos em que o CEP estava certo mas a cidade na base estava errada.

mask_cidade = df["_cep_acao"].isin([
    "suspeito_cidade_errada",
    "suspeito_repetido",
    "corrigir_zero_inicial",   # aproveita cidade confirmada da API
]) & df["_api_cidade"].notna()

df["CIDADE_corrigida"] = df["CIDADE"]
df.loc[mask_cidade, "CIDADE_corrigida"] = df.loc[mask_cidade, "_api_cidade"]

# ── 3. flag de estratégia de geocodificação ───────────────────────────────────
# O etapa_04_geocodificacao.py usa esse campo para escolher entre CEP ou endereço+cidade.

def estrategia_geo(acao):
    if acao in ("ok", "corrigir_zero_inicial", "suspeito_cidade_errada",
                "suspeito_repetido"):
        return "cep"           # CEP confiável → geocodifica pelo CEP
    return "endereco"          # CEP duvidoso → fallback endereço+cidade

df["_geo_estrategia"] = df["_cep_acao"].apply(estrategia_geo)

# ── resumo ────────────────────────────────────────────────────────────────────

total          = len(df)
via_cep        = (df["_geo_estrategia"] == "cep").sum()
via_endereco   = (df["_geo_estrategia"] == "endereco").sum()
cidade_corrig  = mask_cidade.sum()

print(f"\n{'='*55}")
print(f"  CORRECAO DE CEPs")
print(f"  {total:,} registros")
print(f"{'='*55}")
print(f"  Geocodificar via CEP      : {via_cep:>6}  ({via_cep/total*100:.1f}%)")
print(f"  Geocodificar via endereco : {via_endereco:>6}  ({via_endereco/total*100:.1f}%)")
print(f"  Cidades corrigidas pela API: {cidade_corrig:>5}")
print(f"\n  Distribuicao por acao do diagnostico:")
for acao, n in df["_cep_acao"].value_counts().items():
    print(f"    {acao:<30} {n:>6}")

# ── exporta base corrigida ────────────────────────────────────────────────────

df_saida = (
    df.drop(columns=["CEP", "CIDADE", "_cep_norm", "_cep_acao", "_api_cidade"])
      .rename(columns={"CEP_corrigido": "CEP", "CIDADE_corrigida": "CIDADE"})
)
df_saida.to_csv(SAIDA, index=False)
print(f"\n  Base corrigida salva em: {SAIDA}")

# ── exporta lista para auditoria ─────────────────────────────────────────────

acoes_auditoria = ["cep_incerto", "suspeito_cidade_errada", "cep_inexistente",
                   "suspeito_repetido", "irrecuperavel"]
df_rev = df[df["_cep_acao"].isin(acoes_auditoria)][
    ["medico_id", "CEP", "CEP_corrigido", "_cep_acao", "_geo_estrategia",
     "CIDADE", "CIDADE_corrigida", "_api_cidade", "BAIRRO", "ENDERECO"]
].sort_values(["_cep_acao", "CIDADE"])

df_rev.to_csv(REVISAO, index=False)
print(f"  Registros para auditoria : {len(df_rev):,}  -> {REVISAO}")
print()
