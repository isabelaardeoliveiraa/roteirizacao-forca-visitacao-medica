# -*- coding: utf-8 -*-
"""
A equipe ja prioriza por potencial? — cruzamento entre a frequencia de visita
efetivamente registrada no CRM e a classificacao de prioridade da auditoria
de mercado.

A pergunta importa porque o modelo de roteirizacao assume uma carteira plana:
visita cada medico uma vez por ciclo. Se a pratica ja concentrasse esforco nos
prescritores de maior potencial, a contribuicao do modelo seria apenas
logistica. Se nao concentra, ha espaco de ganho comercial alem do ganho de
deslocamento.

Recorte: visitas com situacao 'Realizado' entre 01/05/2026 e 31/07/2026, janela
em que o registro no CRM ja esta consolidado (antes de maio ha dezenas de
visitas por mes, contra milhares depois).

ATENCAO: datas em ISO. Usar dayfirst=True descarta 12.500 das 20.207 linhas.

Saida: saida/resultados/prioridade_observada.txt
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

TAREFAS = 'dados_crm/2807/crm_task_report.csv'
CORRESP = 'saida/resultados/correspondencia_contactid.csv'
SCORE = 'saida/resultados/base_medicos_score.csv'
SAIDA = 'saida/resultados/prioridade_observada.txt'
INICIO, FIM = '2026-05-01', '2026-08-01'

linhas = []


def diz(txt=''):
    linhas.append(txt)
    print(txt)


def carrega():
    t = pd.read_csv(TAREFAS, sep=';', low_memory=False)
    t['dt'] = pd.to_datetime(t.taskstart, errors='coerce')
    t = t[t.taskstate.astype(str).str.contains('Realizado', na=False)]
    t = t[(t.dt >= INICIO) & (t.dt < FIM)]
    corr = pd.read_csv(CORRESP)
    t = t.merge(corr[['contact_id', 'medico_id']], on='contact_id', how='left')

    sc = pd.read_csv(SCORE, low_memory=False)
    med = sc.drop_duplicates('medico_id')[
        ['medico_id', 'classificacao', 'px_total_tri', 'px_crm_tri']]
    vis = t.groupby('medico_id').size().rename('visitas')
    d = med.merge(vis, on='medico_id', how='left')
    d['visitas'] = d.visitas.fillna(0)
    d['visitado'] = d.visitas > 0
    d['share'] = np.where(d.classificacao.astype(str).str.contains('ACIMA'), 'ACIMA',
                 np.where(d.classificacao.astype(str).str.contains('ABAIXO'), 'ABAIXO', None))
    d['vol'] = d.classificacao.astype(str).str.extract(r'(ALTO|MÉDIO|BAIXO)')[0]
    return t, d


def main():
    t, d = carrega()
    diz('PRIORIZACAO OBSERVADA x POTENCIAL DE PRESCRICAO')
    diz('Visitas realizadas entre %s e %s, registradas no CRM' % (INICIO, FIM))
    diz('=' * 74)
    diz('  Visitas no recorte      : %s' % format(len(t), ',d'))
    diz('  Medicos da carteira     : %s' % format(len(d), ',d'))
    diz('  Medicos visitados ao menos uma vez: %s (%.1f%%)'
        % (format(int(d.visitado.sum()), ',d'), d.visitado.mean() * 100))
    diz()

    diz('=' * 74)
    diz('  1. INTENSIDADE DE VISITA POR FAIXA DE PRIORIDADE')
    diz('=' * 74)
    diz('  %-32s %8s %9s %9s' % ('classificacao', 'medicos', '% visit.', 'vis/med'))
    g = d.groupby('classificacao')
    for k in sorted(g.groups):
        s = g.get_group(k)
        diz('  %-32s %8s %8.1f%% %9.2f'
            % (str(k)[:32], format(len(s), ',d'), s.visitado.mean() * 100, s.visitas.mean()))
    diz()
    diz('  A ordenacao acompanha a prioridade, mas a associacao individual e fraca:')
    com = d[d.px_total_tri.fillna(0) > 0]
    rho, p = stats.spearmanr(com.px_total_tri, com.visitas)
    diz('  Spearman entre volume de prescricao e numero de visitas: %.3f (p=%.1e, n=%s)'
        % (rho, p, format(len(com), ',d')))
    diz()

    diz('=' * 74)
    diz('  2. ESFORCO x POTENCIAL — participacao de cada faixa')
    diz('=' * 74)
    tv, tp = d.visitas.sum(), d.px_total_tri.fillna(0).sum()
    diz('  %-32s %9s %8s %8s' % ('classificacao', '% visitas', '% Px', 'razao'))
    for k in sorted(g.groups):
        s = g.get_group(k)
        pv = s.visitas.sum() / tv * 100
        pp = s.px_total_tri.fillna(0).sum() / tp * 100
        diz('  %-32s %8.1f%% %7.1f%% %8s'
            % (str(k)[:32], pv, pp, ('%.2f' % (pv / pp)) if pp > 0 else '—'))
    diz()
    diz('  Razao abaixo de 1 indica faixa que recebe menos esforco do que o')
    diz('  potencial de mercado que concentra.')
    diz()

    diz('=' * 74)
    diz('  3. O ESFORCO SEGUE A OPORTUNIDADE OU A PARTICIPACAO JA CONQUISTADA?')
    diz('=' * 74)
    diz('  Media de visitas por medico, por volume de prescricao x participacao:')
    diz('  %-8s %9s %7s %9s %7s %10s'
        % ('volume', 'ABAIXO', 'n', 'ACIMA', 'n', 'p'))
    for v in ['ALTO', 'MÉDIO', 'BAIXO']:
        x = d[(d.vol == v) & (d.share == 'ACIMA')].visitas
        y = d[(d.vol == v) & (d.share == 'ABAIXO')].visitas
        _, p = stats.mannwhitneyu(x, y, alternative='greater')
        diz('  %-8s %9.2f %7d %9.2f %7d %10.1e'
            % (v, y.mean(), len(y), x.mean(), len(x), p))
    diz()
    diz('  Teste de Mann-Whitney unilateral, sobre visitas por medico no trimestre.')
    diz()
    diz('  Nos tres niveis de volume, o medico em que a empresa ja tem participacao')
    diz('  acima recebe mais visitas do que o de mesmo volume com participacao')
    diz('  abaixo. O esforco de campo acompanha a participacao consolidada, e nao')
    diz('  a oportunidade em aberto.')
    diz()
    diz('  RESSALVA DE CAUSALIDADE: a relacao tambem admite leitura inversa — a')
    diz('  participacao pode ser alta PORQUE o medico foi mais visitado. Os dados')
    diz('  disponiveis sao transversais e nao permitem separar as duas direcoes.')
    diz()
    diz('  RESSALVA DE INTERPRETACAO: a faixa sem prescricao registrada indica')
    diz('  potencial desconhecido, nao potencial nulo. A auditoria de mercado')
    diz('  cobre parcela da base, e a ausencia reflete o alcance da medicao.')

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    io.open(SAIDA, 'w', encoding='utf-8').write('\n'.join(linhas) + '\n')
    print('\n[ok] %s' % SAIDA)


if __name__ == '__main__':
    main()
