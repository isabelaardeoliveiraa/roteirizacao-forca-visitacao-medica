"""
Score de Priorização — base do CRM (etapa 6).

Junta a classificação de prioridade (planilha da área comercial) com a base clusterizada,
usando nome normalizado do médico como chave de join:

  medico_id → contact (nome real, via correspondencia_contactid.csv)
            → normaliza → match com MED_NOME da planilha de priorização

A planilha tem duas abas:
  "Priorização FACIAL"  → junta com tarefas de papel == "facial"
  "Priorização Não"     → junta com tarefas de papel == "nao_facial"

Colunas adicionadas à base clusterizada:
  px_total_tri, px_crm_tri, curva_abc,
  potencial_produtos, patamar_share, classificacao, acao_estrategica

Input : saida/resultados/base_medicos_clusterizada.csv
        saida/resultados/correspondencia_contactid.csv
        entrada/00. Priorização CRMs - CONSOLIDADO - Inverno.xlsx
Saída : saida/resultados/base_medicos_score.csv
"""

import unicodedata
import numpy as np
import pandas as pd

# ── Configurações ──────────────────────────────────────────────────────────────
ARQ_CLUSTER  = r"saida\resultados\base_medicos_clusterizada.csv"
ARQ_CORRESP  = r"saida\resultados\correspondencia_contactid.csv"
# Arquivo "Geral" (CONSOLIDADO), confirmado pelo especialista de negócio em jul/2026
# como a versão a usar: é a que tem mais correspondências com a base do CRM.
# NÃO combinar com o "Inverno - TODOS profissa": para os médicos presentes nos dois,
# a classificação diverge em ~52% dos casos (curva ABC e share são relativos ao
# universo de cada arquivo), o que geraria ações comerciais contraditórias.
ARQ_EXCEL    = r"entrada\00. Priorização CRMs - CONSOLIDADO.xlsx"
ARQ_SAIDA    = r"saida\resultados\base_medicos_score.csv"

COLUNAS_SCORE = [
    "PX TOTAL Trimestre", "PX EMPRESA Trimestre",
    "Classificação Curva ABC", "Potencial de Produtos",
    "Patamar Share", "Classificação", "Ação Estratégica",
]
RENAME_SCORE = {
    "PX TOTAL Trimestre":       "px_total_tri",
    "PX EMPRESA Trimestre":   "px_crm_tri",
    "Classificação Curva ABC":  "curva_abc",
    "Potencial de Produtos":    "potencial_produtos",
    "Patamar Share":            "patamar_share",
    "Classificação":            "classificacao",
    "Ação Estratégica":         "acao_estrategica",
}
# ───────────────────────────────────────────────────────────────────────────────

def normalizar(texto):
    """Remove acentos, upper, strip e colapsa espaços múltiplos."""
    if pd.isna(texto):
        return ""
    s = str(texto).strip().upper()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.split())


print("\n" + "="*60)
print("  SCORE DE PRIORIZAÇÃO — ETAPA 6")
print("="*60)

# ── 1. Base clusterizada ───────────────────────────────────────────────────────
df = pd.read_csv(ARQ_CLUSTER, dtype=str)
print(f"\n  Base clusterizada carregada : {len(df):>6} tarefas")

# ── 2. Correspondência medico_id → nome real + CRM+UF ─────────────────────────
# O CRM (external_key, disponibilizado pelo CRM a partir da extração de 28/07)
# é a chave preferencial: identifica o médico sem ambiguidade. O nome entra só
# como fallback para quem não tem CRM cadastrado.
corresp = pd.read_csv(ARQ_CORRESP, dtype=str)
corresp["nome_norm"] = corresp["contact"].apply(normalizar)
tem_crm = "crm_uf" in corresp.columns
if tem_crm:
    corresp["crm_uf"] = corresp["crm_uf"].astype(str).str.strip().str.upper()
    corresp.loc[corresp["crm_uf"].isin(["", "NAN", "NONE"]), "crm_uf"] = pd.NA
    print(f"  Correspondência carregada   : {len(corresp):>6} médicos "
          f"({corresp['crm_uf'].notna().sum()} com CRM)")
else:
    corresp["crm_uf"] = pd.NA
    print(f"  Correspondência carregada   : {len(corresp):>6} médicos (sem CRM — join só por nome)")

# ── 3. Planilha de priorização ─────────────────────────────────────────────────
def carregar_aba(aba):
    cols = ["MED_NOME", "CRM"] + COLUNAS_SCORE
    df_aba = pd.read_excel(ARQ_EXCEL, sheet_name=aba, usecols=cols)
    df_aba = df_aba.dropna(subset=["MED_NOME"])
    df_aba["nome_norm"] = df_aba["MED_NOME"].apply(normalizar)
    df_aba["crm_uf"]    = df_aba["CRM"].astype(str).str.strip().str.upper()
    df_aba = df_aba.rename(columns=RENAME_SCORE).drop(columns=["MED_NOME", "CRM"])
    return df_aba

pri_facial  = carregar_aba("Priorização FACIAL")
pri_naofac  = carregar_aba("Priorização Não")
print(f"  Priorização FACIAL          : {len(pri_facial):>6} linhas")
print(f"  Priorização NÃO FACIAL      : {len(pri_naofac):>6} linhas")

cols_score = list(RENAME_SCORE.values())

# ── 4. Join em duas passadas: CRM primeiro, nome como fallback ────────────────
df = df.merge(corresp[["medico_id", "nome_norm", "crm_uf"]], on="medico_id", how="left")

def juntar(df_papel, pri):
    """Junta pelo CRM; o que sobrar tenta por nome normalizado."""
    por_crm  = pri.dropna(subset=["crm_uf"]).drop_duplicates("crm_uf", keep="first")
    por_nome = pri.drop_duplicates("nome_norm", keep="first").drop(columns=["crm_uf"])

    out = df_papel.merge(por_crm.drop(columns=["nome_norm"]), on="crm_uf", how="left")
    out["origem_score"] = np.where(out["classificacao"].notna(), "crm", pd.NA)

    falta = out["classificacao"].isna()
    if falta.any():
        recuperado = (out.loc[falta, ["nome_norm"]]
                        .merge(por_nome, on="nome_norm", how="left"))
        recuperado.index = out.index[falta]
        for c in cols_score:
            out.loc[falta, c] = recuperado[c]
        out.loc[falta & out["classificacao"].notna(), "origem_score"] = "nome"
    return out

df_out = pd.concat([juntar(df[df["papel"] == "facial"].copy(),     pri_facial),
                    juntar(df[df["papel"] == "nao_facial"].copy(), pri_naofac)],
                   ignore_index=True)

# Quem não tem correspondência não é "baixo potencial" — é potencial DESCONHECIDO.
# Categoria própria, ordenada abaixo das demais na priorização (decisão validada
# com o especialista de negócio): entra por último no desenho da semana, mas
# permanece auditável e não é contado como baixo prescritor.
df_out["classificacao"] = df_out["classificacao"].fillna("7. SEM PX REGISTRADO")
df_out["acao_estrategica"] = df_out["acao_estrategica"].fillna(
    "(SEM DADO) - Sem prescrição capturada no trimestre (IQVIA/CLOSE-UP)")
df_out["origem_score"] = df_out["origem_score"].fillna("sem_correspondencia")
df_out = df_out.drop(columns=["nome_norm", "crm_uf"])

# ── 5. Diagnóstico de cobertura ────────────────────────────────────────────────
# "com score" = teve correspondência na planilha. As tarefas marcadas como
# SEM PX REGISTRADO têm classificação atribuída, mas não são cobertura.
tem_score = df_out["origem_score"] != "sem_correspondencia"

print(f"\n{'-'*60}")
print(f"  COBERTURA DO JOIN")
print(f"{'-'*60}")
for papel in ["facial", "nao_facial"]:
    sub = df_out[df_out["papel"] == papel]
    com = (sub["origem_score"] != "sem_correspondencia").sum()
    print(f"  {papel:<12}: {com:>5}/{len(sub)} com score ({com/len(sub)*100:.1f}%)")

total_all = len(df_out)
print(f"  {'TOTAL':<12}: {tem_score.sum():>5}/{total_all} com score ({tem_score.sum()/total_all*100:.1f}%)")

print(f"\n  Por chave de correspondência:")
for origem, n in df_out["origem_score"].value_counts().items():
    rot = {"crm": "CRM+UF (chave preferencial)",
           "nome": "nome normalizado (fallback)",
           "sem_correspondencia": "sem correspondência"}.get(origem, origem)
    print(f"    {rot:<32}: {n:>6} ({n/total_all*100:.1f}%)")

# Distribuição das classificações
print(f"\n{'-'*60}")
print(f"  DISTRIBUIÇÃO — CLASSIFICAÇÃO")
print(f"{'-'*60}")
dist = df_out["classificacao"].value_counts(dropna=False).sort_index()
for cls, n in dist.items():
    print(f"  {str(cls):<55}: {n:>5} ({n/total_all*100:.1f}%)")

print(f"\n{'-'*60}")
print(f"  DISTRIBUIÇÃO — POTENCIAL DE PRODUTOS")
print(f"{'-'*60}")
dist2 = df_out["potencial_produtos"].value_counts(dropna=False)
for cls, n in dist2.items():
    print(f"  {str(cls):<20}: {n:>5} ({n/total_all*100:.1f}%)")

# ── 6. Salva ───────────────────────────────────────────────────────────────────
df_out.to_csv(ARQ_SAIDA, index=False)
print(f"\n  Saída salva em: {ARQ_SAIDA}")
print(f"  Colunas finais: {len(df_out.columns)}")
print()
