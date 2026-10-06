# -*- coding: utf-8 -*-
"""
Qualidade do cadastro de endereco do CRM — BASE INTEIRA, sem amostragem.

Compara o logradouro digitado no CRM com o logradouro oficial do ViaCEP para
todos os medicos cujo CEP esta no cache montado pela etapa 04d. Como o cache ja
foi construido ali, esta analise nao faz nenhuma chamada de API.

Mede dois eixos independentes, que nao devem ser somados:
  A) o NOME CENTRAL da rua confere? (depois de remover o tipo de logradouro)
  B) o TIPO de logradouro (Rua/Avenida/...) esta escrito?
Alem disso confere se a CIDADE do CRM bate com a do ViaCEP.

Saidas:
  saida/resultados/qualidade_endereco_crm.txt   — relatorio
  saida/resultados/cadastro_endereco_divergente.csv — casos para correcao no CRM
"""
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ARQ_VIACEP  = r"saida\resultados\viacep_logradouros.csv"
ARQ_CONTACT = r"dados_crm\2807\crm_contact_report.csv"
ARQ_CORRESP = r"saida\resultados\correspondencia_contactid.csv"
ARQ_TXT     = r"saida\resultados\qualidade_endereco_crm.txt"
ARQ_CSV     = r"saida\resultados\cadastro_endereco_divergente.csv"

LIMIAR_DIVERGENTE = 0.80   # abaixo disso, o nome central nao e a mesma rua

TIPOS = (r"^(rua|r|avenida|av|travessa|tv|alameda|al|praca|pca|rodovia|rod|"
         r"estrada|est|largo|lgo|via|quadra|q|conjunto|cj|setor|st|viela|"
         r"passagem|servidao|marginal|ladeira|parque|pq)\b\.?\s*")


def desacento(s):
    s = str(s or "").strip().lower()
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def nome_central(s):
    """Remove acento, pontuacao e o tipo de logradouro — sobra o nome da rua."""
    s = re.sub(r"[^a-z0-9\s]", " ", desacento(s))
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+", " ", re.sub(TIPOS, "", s)).strip()


def tem_tipo(s):
    return bool(re.match(TIPOS, re.sub(r"[^a-z0-9\s]", " ", desacento(s)).strip()))


# ── Dados ─────────────────────────────────────────────────────────────────────
vc = pd.read_csv(ARQ_VIACEP, dtype={"cep": str}).fillna("")
c = pd.read_csv(ARQ_CONTACT, sep=";", dtype={"postcode": str}, low_memory=False)
c["cep"] = (c.postcode.astype(str).str.replace(r"\.0$", "", regex=True)
              .str.replace(r"\D", "", regex=True).str.zfill(8))
corresp = pd.read_csv(ARQ_CORRESP)[["contact_id", "medico_id"]]

df = (corresp.merge(c[["contact_id", "cep", "street", "city", "region"]], on="contact_id", how="left")
             .merge(vc, on="cep", how="left"))
total_base = len(df)
df = df[df.logradouro.notna()]          # so os CEPs ja consultados
print("medicos na base: {:,} | com CEP ja consultado no ViaCEP: {:,}".format(total_base, len(df)))

tem_log = df.logradouro.astype(str).str.strip() != ""
comp = df[tem_log & df.street.notna()].copy()
comp["nc_crm"] = comp.street.map(nome_central)
comp["nc_vc"] = comp.logradouro.map(nome_central)
comp["sim"] = [SequenceMatcher(None, a, b).ratio() for a, b in zip(comp.nc_crm, comp.nc_vc)]
comp["crm_tem_tipo"] = comp.street.map(tem_tipo)
comp["vc_tem_tipo"] = comp.logradouro.map(tem_tipo)
comp["cidade_bate"] = (comp.city.map(desacento).str.strip()
                       == comp.localidade.map(desacento).str.strip())

# ── Relatorio ─────────────────────────────────────────────────────────────────
L = []
A = L.append
A("=" * 72)
A("  QUALIDADE DO CADASTRO DE ENDERECO DO CRM — BASE INTEIRA")
A("=" * 72)
A("  medicos na base                      : {:,}".format(total_base))
A("  com CEP resolvido pelo ViaCEP        : {:,} ({:.1f}%)".format(len(df), 100 * len(df) / total_base))
A("  com logradouro oficial comparavel    : {:,} ({:.1f}%)".format(len(comp), 100 * len(comp) / total_base))
A("  sem logradouro (CEP de localidade/inexistente): {:,}".format(int((~tem_log).sum())))
A("")
A("  EIXO A — o nome central da rua confere?")
faixas = [("identico (>= 0,95)", comp.sim >= 0.95),
          ("quase igual (0,80 a 0,95)", comp.sim.between(0.80, 0.95, inclusive="left")),
          ("divergente (0,50 a 0,80)", comp.sim.between(0.50, 0.80, inclusive="left")),
          ("muito diferente (< 0,50)", comp.sim < 0.50)]
for rot, m in faixas:
    A("    {:30s} {:6,}  ({:5.2f}%)".format(rot, int(m.sum()), 100 * m.mean()))
A("")
A("  EIXO B — o tipo de logradouro esta escrito? (independente do eixo A)")
A("    {:30s} {:6,}  ({:5.2f}%)".format("ViaCEP escreve", int(comp.vc_tem_tipo.sum()), 100 * comp.vc_tem_tipo.mean()))
A("    {:30s} {:6,}  ({:5.2f}%)".format("CRM escreve", int(comp.crm_tem_tipo.sum()), 100 * comp.crm_tem_tipo.mean()))
falta = comp.vc_tem_tipo & ~comp.crm_tem_tipo
A("    {:30s} {:6,}  ({:5.2f}%)".format("ViaCEP tem e CRM omite", int(falta.sum()), 100 * falta.mean()))
A("")
A("  CIDADE — o municipio do CRM bate com o do ViaCEP?")
A("    {:30s} {:6,}  ({:5.2f}%)".format("confere", int(comp.cidade_bate.sum()), 100 * comp.cidade_bate.mean()))
A("    {:30s} {:6,}  ({:5.2f}%)".format("diverge", int((~comp.cidade_bate).sum()), 100 * (~comp.cidade_bate).mean()))
A("")
A("  VARIACAO POR ESTADO (estados com 100+ medicos comparaveis)")
A("    {:22s} {:>8s} {:>10s} {:>12s}".format("estado", "medicos", "identico", "divergente"))
uf = (comp.assign(ident=comp.sim >= 0.95, diverg=comp.sim < 0.80)
          .groupby("region")
          .agg(n=("sim", "size"), ident=("ident", "mean"), diverg=("diverg", "mean")))
uf = uf[uf.n >= 100].sort_values("ident")
for e, r in uf.iterrows():
    A("    {:22s} {:8,} {:9.1f}% {:11.1f}%".format(str(e)[:22], int(r.n), 100 * r.ident, 100 * r.diverg))
if len(uf) > 1:
    A("")
    A("    amplitude: de {:.1f}% a {:.1f}% de registros identicos".format(
        100 * uf.ident.min(), 100 * uf.ident.max()))
A("")
div = comp[comp.sim < LIMIAR_DIVERGENTE]
A("  CASOS PARA CORRECAO DE CADASTRO (semelhanca < {:.2f}): {:,}".format(LIMIAR_DIVERGENTE, len(div)))
A("  Nestes o CEP e a rua digitada apontam para lugares diferentes: ou o CEP")
A("  esta errado, ou o nome da rua esta. Lista em {}".format(ARQ_CSV))
if len(div):
    A("")
    A("  Amostra:")
    for r in div.nsmallest(10, "sim").itertuples():
        A("    CEP {} sim={:.2f}".format(r.cep, r.sim))
        A("       CRM   : {}".format(str(r.street)[:58]))
        A("       ViaCEP: {}".format(str(r.logradouro)[:58]))

txt = "\n".join(L)
Path(ARQ_TXT).write_text(txt, encoding="utf-8")
div[["medico_id", "cep", "street", "logradouro", "city", "localidade", "sim"]].to_csv(ARQ_CSV, index=False)
print(txt)
print("\n[ok] {}\n[ok] {}".format(ARQ_TXT, ARQ_CSV))
