"""
Diagnóstico completo de qualidade de dados da base anonimizada do CRM.
Equivalente ao 02_diagnostico.py, aplicado a base_medicos_anonimizada.csv.

Saídas: relatório no terminal + problemas_cep.csv com cada registro classificado.
"""

import pandas as pd
import re
import unicodedata
import requests
import time

ARQUIVO           = r"saida\resultados\base_medicos_anonimizada.csv"
VALIDAR_VIA_CEP   = True   # False = pula API, classifica só pelo formato
DELAY_API         = 0.25   # segundos entre chamadas

# ── helpers ───────────────────────────────────────────────────────────────────

def normalizar_texto(t):
    if pd.isna(t) or str(t).strip() == "":
        return ""
    s = unicodedata.normalize("NFD", str(t))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip().lower()

def limpar_digitos(valor):
    if pd.isna(valor):
        return ""
    return re.sub(r"[^0-9]", "", str(valor).strip())

def classificar_cep_formato(raw):
    d = limpar_digitos(raw)
    if not d:                  return "ausente"
    if len(d) < 7:             return "curto_demais"
    if len(d) == 7:            return "7_digitos"
    if len(d) == 8:
        if d == "00000000":    return "zerado"
        return "ok"
    return "longo_demais"

def consultar_viacep(cep8):
    try:
        r = requests.get(f"https://viacep.com.br/ws/{cep8}/json/", timeout=5)
        if r.status_code != 200:
            return None
        d = r.json()
        if "erro" in d:
            return None
        return {"cidade": d.get("localidade", ""), "uf": d.get("uf", ""),
                "logradouro": d.get("logradouro", "")}
    except Exception:
        return None

# ── carga ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(ARQUIVO, dtype=str)
total = len(df)

print(f"\n{'='*62}")
print(f"  DIAGNOSTICO DE QUALIDADE - {ARQUIVO}")
print(f"  {total:,} registros")
print(f"{'='*62}")

# ── 1. nulos gerais ───────────────────────────────────────────────────────────

print("\n-- 1. CAMPOS COM VALORES AUSENTES --------------------------")
nulos = df.isnull().sum()
nulos_pct = (nulos / total * 100).round(1)
resumo = pd.DataFrame({"Ausentes": nulos, "%": nulos_pct})
resumo = resumo[resumo["Ausentes"] > 0].sort_values("Ausentes", ascending=False)
print(resumo.to_string() if not resumo.empty else "  Nenhum campo com nulos.")

# ── 2. classificação de formato do CEP ───────────────────────────────────────

print("\n-- 2. FORMATO DO CEP ----------------------------------------")
df["_digitos"]  = df["CEP"].apply(limpar_digitos)
df["_formato"]  = df["CEP"].apply(classificar_cep_formato)
df["_cep_norm"] = df.apply(
    lambda r: r["_digitos"].zfill(8) if r["_formato"] in ("ok", "7_digitos") else None,
    axis=1
)

fmt_count = df["_formato"].value_counts()
for fmt, n in fmt_count.items():
    print(f"  {fmt:<20} {n:>6}  ({n/total*100:.1f}%)")

# ── 3. repetição de CEP ───────────────────────────────────────────────────────

print("\n-- 3. REPETICAO DE CEP ---------------------------------------")
freq = df["_cep_norm"].value_counts()
df["_freq"] = df["_cep_norm"].map(freq).fillna(0).astype(int)

unicos      = (freq == 1).sum()
repetidos   = (freq  > 1).sum()
reg_rep     = (df["_freq"] > 1).sum()

print(f"  CEPs unicos (1 medico)       : {unicos:>6}")
print(f"  CEPs repetidos (>=2 medicos) : {repetidos:>6}")
print(f"  Registros com CEP repetido   : {reg_rep:>6}  ({reg_rep/total*100:.1f}%)")

bins   = [1, 2, 5, 10, 20, 50, float("inf")]
labels = ["2", "3-5", "6-10", "11-20", "21-50", ">50"]
faixas = pd.cut(freq[freq > 1], bins=bins, labels=labels, right=True)
print("\n  Distribuicao dos CEPs repetidos por faixa:")
for label, n in faixas.value_counts().sort_index().items():
    print(f"    {label:<8} medicos/CEP : {n:>5} CEPs")

# ── 4. CEPs repetidos com cidades divergentes ─────────────────────────────────

print("\n-- 4. CEPs REPETIDOS COM CIDADES DIFERENTES ------------------")
mask_rep = df["_freq"] > 1
df_rep   = df[mask_rep].copy()

def cidades_do_cep(grupo):
    return grupo["CIDADE"].dropna().apply(normalizar_texto).nunique()

diversidade = df_rep.groupby("_cep_norm").apply(cidades_do_cep)
ceps_multi_cidade = diversidade[diversidade > 1]

print(f"  CEPs com > 1 cidade associada: {len(ceps_multi_cidade)}")

if not ceps_multi_cidade.empty:
    print(f"\n  Top 15 (mais cidades distintas por CEP):")
    top = ceps_multi_cidade.sort_values(ascending=False).head(15)
    for cep, n_cidades in top.items():
        cidades = (df_rep[df_rep["_cep_norm"] == cep]["CIDADE"]
                   .dropna().unique().tolist())
        print(f"    {cep}  ->  {n_cidades} cidades: {', '.join(cidades[:6])}")

# marca na base
df["_cep_multi_cidade"] = df["_cep_norm"].isin(ceps_multi_cidade.index)

# ── 5. validação via ViaCEP (foco nos casos de risco) ─────────────────────────

df["_api_status"]  = pd.NA
df["_api_cidade"]  = pd.NA
df["_cidade_ok"]   = pd.NA
df["_cep_acao"]    = "ok"

# CEPs sem dígitos suficientes — irrecuperáveis
mask_irr = df["_formato"].isin(["ausente", "curto_demais", "zerado", "longo_demais"])
df.loc[mask_irr, "_cep_acao"] = "irrecuperavel"

# CEPs repetidos com cidades divergentes — suspeitos
df.loc[df["_cep_multi_cidade"], "_cep_acao"] = "suspeito_repetido"

# CEPs com 7 dígitos — hipótese: zero inicial perdido pelo Excel
df.loc[df["_formato"] == "7_digitos", "_cep_acao"] = "hipotese_zero_inicial"

if VALIDAR_VIA_CEP:
    print("\n-- 5. VALIDACAO VIA API VIACEP -------------------------------")

    # Consulta apenas CEPs únicos que precisam de validação
    mask_api = df["_cep_acao"].isin(["hipotese_zero_inicial", "suspeito_repetido"])
    ceps_consultar = df.loc[mask_api, "_cep_norm"].dropna().unique()
    print(f"  CEPs unicos a consultar: {len(ceps_consultar)}")

    cache = {}
    for i, cep in enumerate(ceps_consultar):
        if cep not in cache:
            cache[cep] = consultar_viacep(cep)
            time.sleep(DELAY_API)
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(ceps_consultar)}")

    def aplicar_api(row):
        if row["_cep_norm"] not in cache:
            return row["_api_status"], row["_api_cidade"], row["_cidade_ok"], row["_cep_acao"]

        resultado = cache[row["_cep_norm"]]
        if resultado is None:
            api_status = "nao_encontrado"
            api_cidade = None
            cidade_ok  = False
            acao = "cep_inexistente"
        else:
            api_status = "encontrado"
            api_cidade = resultado["cidade"]
            cidade_ok  = (normalizar_texto(row["CIDADE"]) ==
                          normalizar_texto(api_cidade))
            if row["_formato"] == "7_digitos":
                acao = "corrigir_zero_inicial" if cidade_ok else "cep_incerto"
            else:
                acao = "ok" if cidade_ok else "suspeito_cidade_errada"

        return api_status, api_cidade, cidade_ok, acao

    resultados = df[mask_api].apply(aplicar_api, axis=1, result_type="expand")
    resultados.columns = ["_api_status", "_api_cidade", "_cidade_ok", "_cep_acao"]
    df.loc[mask_api, ["_api_status", "_api_cidade", "_cidade_ok", "_cep_acao"]] = resultados

# ── 6. resumo por ação recomendada ────────────────────────────────────────────

print("\n-- 6. RESUMO - ACAO RECOMENDADA POR REGISTRO -----------------")
acoes = df["_cep_acao"].value_counts()
descricoes = {
    "ok":                     "CEP valido, cidade confirmada",
    "corrigir_zero_inicial":  "CEP 7 dig. -> adicionar zero a esquerda (cidade OK)",
    "cep_incerto":            "CEP 7 dig. -> zero inicial NAO confirmou cidade (revisar)",
    "suspeito_cidade_errada": "CEP repetido com cidade divergente da API",
    "suspeito_repetido":      "CEP repetido com multiplas cidades (sem validacao API)",
    "cep_inexistente":        "CEP nao encontrado na API ViaCEP",
    "irrecuperavel":          "Formato invalido (ausente / zerado / tamanho errado)",
}
for acao, n in acoes.items():
    desc = descricoes.get(acao, "")
    print(f"  {acao:<30} {n:>6}  -  {desc}")

# ── 7. exporta CSV com diagnóstico ────────────────────────────────────────────

saida = r"saida\resultados\problemas_cep.csv"
cols  = ["medico_id", "CEP", "_cep_norm", "_formato", "_freq",
         "_cep_multi_cidade", "_api_status", "_api_cidade",
         "_cidade_ok", "_cep_acao", "CIDADE", "BAIRRO", "ENDERECO"]
cols  = [c for c in cols if c in df.columns]

df_problemas = df[df["_cep_acao"] != "ok"][cols].sort_values(
    ["_cep_acao", "_freq"], ascending=[True, False]
)
df_problemas.to_csv(saida, index=False)
print(f"\n  Detalhes exportados para: {saida}  ({len(df_problemas):,} registros)")
print()
