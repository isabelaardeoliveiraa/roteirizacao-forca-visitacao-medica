"""
Anonimização da base do CRM (novo CRM, extraído via SFTP em "dados_crm/").

Equivalente aos scripts 01 + 03b da base antiga, mas usando contact_id como
identificador estável (em vez de matching por nome) e derivando rep_facial /
rep_nao_facial a partir de contactowner_report + user_report (campo direction).

Saídas:
  saida/base_medicos_anonimizada.csv   -> input do etapa_02_diagnostico.py
  saida/correspondencia_contactid.csv  -> contact_id <-> medico_id (fica na empresa)
  saida/correspondencia_representantes.csv     -> atualizada com novos representantes (fica na empresa)
"""

import pandas as pd
import unicodedata
import re

PASTA_NOVA  = r"dados_crm\2807"   # extração de 28/07/2026 (com external_key e contact_id nas visitas)
CORRESP_REP = r"saida\correspondencia_representantes.csv"


def normaliza_nome(s):
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"^(DR|DRA|DR\.|DRA\.)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normaliza_texto(s):
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s


# ── 1. Carrega tabelas do CRM ─────────────────────────────────────────────
df_contact = pd.read_csv(f"{PASTA_NOVA}/crm_contact_report.csv", sep=";", encoding="utf-8-sig", dtype=str)
df_cowner  = pd.read_csv(f"{PASTA_NOVA}/crm_contactowner_report.csv", sep=";", encoding="utf-8-sig", dtype=str)
df_user    = pd.read_csv(f"{PASTA_NOVA}/crm_user_report.csv", sep=";", encoding="utf-8-sig", dtype=str)

total = len(df_contact)

# ── 2. Direção (FACIAL / NAO FACIAL / HIBRIDO) por representante ─────────────
df_user["user_norm"] = df_user["user_name"].apply(normaliza_nome)
direcao_por_user = dict(zip(df_user["user_norm"], df_user["direction"].apply(normaliza_texto)))

df_cowner["owner_norm"] = df_cowner["owner"].apply(normaliza_nome)
df_cowner["direcao_norm"] = df_cowner["owner_norm"].map(direcao_por_user).fillna("")

# Para owners "VAGO ..." (sem entrada em user_report), infere a direção pelo
# próprio nome da vaga (ex.: "VAGO MANAUS -  Facial", "VAGO SALVADOR 2 NAO FACIAL")
owner_txt_norm = df_cowner["owner"].apply(normaliza_texto)
sem_direcao = df_cowner["direcao_norm"] == ""
df_cowner.loc[sem_direcao & owner_txt_norm.str.contains("NAO FACIAL"), "direcao_norm"] = "NAO FACIAL"
df_cowner.loc[sem_direcao & df_cowner["direcao_norm"].eq("") & owner_txt_norm.str.contains("FACIAL"), "direcao_norm"] = "FACIAL"

is_facial    = df_cowner["direcao_norm"].str.contains("HIBRIDO") | \
               (df_cowner["direcao_norm"].str.contains("FACIAL") & ~df_cowner["direcao_norm"].str.contains("NAO FACIAL"))
is_naofacial = df_cowner["direcao_norm"].str.contains("HIBRIDO") | \
               df_cowner["direcao_norm"].str.contains("NAO FACIAL")

# Um rep facial e um rep não facial por médico (pega o primeiro em ordem alfabética
# quando há mais de um candidato — caso raro)
df_facial = (df_cowner[is_facial].sort_values(["contact_id", "owner"])
              .drop_duplicates("contact_id", keep="first")[["contact_id", "owner"]]
              .rename(columns={"owner": "rep_facial_nome"}))
df_naofac = (df_cowner[is_naofacial].sort_values(["contact_id", "owner"])
              .drop_duplicates("contact_id", keep="first")[["contact_id", "owner"]]
              .rename(columns={"owner": "rep_nao_facial_nome"}))

df_contact = df_contact.merge(df_facial, on="contact_id", how="left")
df_contact = df_contact.merge(df_naofac, on="contact_id", how="left")

# ── 3. Anonimiza representantes (reaproveita IDs já existentes) ──────────────
try:
    df_corresp_rep = pd.read_csv(CORRESP_REP, dtype=str)
except FileNotFoundError:
    df_corresp_rep = pd.DataFrame(columns=["nome_real", "rep_id"])

mapa_rep = dict(zip(df_corresp_rep["nome_real"].apply(normaliza_nome), df_corresp_rep["rep_id"]))
max_id = 0
for rid in df_corresp_rep["rep_id"].dropna():
    m = re.match(r"Rep_(\d+)", str(rid))
    if m:
        max_id = max(max_id, int(m.group(1)))

nomes_novos = sorted({
    normaliza_nome(n) for n in pd.concat([df_contact["rep_facial_nome"], df_contact["rep_nao_facial_nome"]]).dropna()
    if normaliza_nome(n) not in mapa_rep and normaliza_nome(n) != ""
})

for nome in nomes_novos:
    max_id += 1
    mapa_rep[nome] = f"Rep_{str(max_id).zfill(3)}"

df_contact["rep_facial"]     = df_contact["rep_facial_nome"].apply(normaliza_nome).map(mapa_rep)
df_contact["rep_nao_facial"] = df_contact["rep_nao_facial_nome"].apply(normaliza_nome).map(mapa_rep)

# Atualiza tabela de correspondência de representantes (mantém entradas antigas + novas)
linhas_corresp = [{"nome_real": nome, "rep_id": rid} for nome, rid in mapa_rep.items()]
# Recupera o nome "bonito" original (não normalizado) quando possível
nomes_originais = {}
for col in ["rep_facial_nome", "rep_nao_facial_nome"]:
    for nome in df_contact[col].dropna().unique():
        nomes_originais.setdefault(normaliza_nome(nome), nome)
for col in ["nome_real"]:
    pass
df_corresp_atualizada = pd.DataFrame([
    {"nome_real": nomes_originais.get(nome_norm, nome_norm), "rep_id": rid}
    for nome_norm, rid in mapa_rep.items()
]).drop_duplicates(subset="rep_id").sort_values("rep_id")
df_corresp_atualizada.to_csv(CORRESP_REP, index=False)

# ── 4. Monta base anonimizada ─────────────────────────────────────────────────
df_contact["medico_id"] = "MedicoP_" + df_contact["contact_id"].astype(str).str.zfill(5)
df_contact["ENDERECO"]  = df_contact["street"].fillna("") + df_contact["building"].apply(
    lambda b: f", {b}" if pd.notna(b) and str(b).strip() else ""
)
df_contact["ativo"] = df_contact["acrhive_reason"].isna() | (df_contact["acrhive_reason"].str.strip() == "")

# motivo do arquivamento: usado no filtro de roteirização (etapa 5) e auditável aqui
df_contact["motivo_inativo"] = df_contact["acrhive_reason"].fillna("").str.strip()

colunas_modelo = {
    "medico_id": "medico_id",
    "CEP": "postcode",
    "CIDADE": "city",
    "BAIRRO": "barrio",
    "ENDERECO": "ENDERECO",
    "specialization": "specialization",
    "company_type": "company_type",
    "ativo": "ativo",
    "motivo_inativo": "motivo_inativo",
    "rep_facial": "rep_facial",
    "rep_nao_facial": "rep_nao_facial",
}

df_anon = pd.DataFrame({novo: df_contact[antigo] for novo, antigo in colunas_modelo.items()})
df_anon.to_csv(r"saida\resultados\base_medicos_anonimizada.csv", index=False)

# ── 5. Tabela de correspondência contact_id <-> medico_id (fica na empresa) ──
# external_key = CRM+UF (ex.: RJ0515487). Identifica o médico, por isso fica aqui
# e não na base anonimizada. É a chave do join com a planilha de priorização.
cols_corresp = ["contact_id", "contact", "medico_id"]
if "external_key" in df_contact.columns:
    df_contact["crm_uf"] = df_contact["external_key"].astype(str).str.strip().str.upper()
    df_contact.loc[df_contact["crm_uf"].isin(["", "NAN", "NONE"]), "crm_uf"] = pd.NA
    cols_corresp.append("crm_uf")
    n_crm = df_contact["crm_uf"].notna().sum()
    print(f"  CRM+UF (external_key) capturado : {n_crm} ({n_crm/total*100:.1f}%)")
else:
    print("  AVISO: coluna external_key ausente — join do score usará apenas nome.")

df_contact[cols_corresp].to_csv(
    r"saida\resultados\correspondencia_contactid.csv", index=False
)

# ── Resumo ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ANONIMIZACAO - base do CRM")
print(f"{'='*55}")
print(f"  Total de medicos             : {total:>6}")
print(f"  Com rep facial atribuido      : {df_contact['rep_facial'].notna().sum():>6}")
print(f"  Com rep nao facial atribuido  : {df_contact['rep_nao_facial'].notna().sum():>6}")
print(f"  Sem rep facial (vago/sem dado): {df_contact['rep_facial'].isna().sum():>6}")
print(f"  Sem rep nao facial            : {df_contact['rep_nao_facial'].isna().sum():>6}")
print(f"  Ativos                        : {df_contact['ativo'].sum():>6}")
print(f"  Inativos (acrhive_reason)     : {(~df_contact['ativo']).sum():>6}")
print(f"  Representantes novos mapeados : {len(nomes_novos):>6}")
print(f"  Total de representantes (Rep_xxx): {len(mapa_rep):>6}")
print(f"\n  Base anonimizada salva em : saida/resultados/base_medicos_anonimizada.csv")
print(f"  Correspondencia contact_id : saida/resultados/correspondencia_contactid.csv (nao enviar ao TCC)")
print(f"  Correspondencia reps       : {CORRESP_REP} (nao enviar ao TCC)")
print()
