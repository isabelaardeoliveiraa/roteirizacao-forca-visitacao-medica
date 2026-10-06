# -*- coding: utf-8 -*-
"""
Avaliacao comparativa da roteirizacao — insumo da secao Resultados e Discussao do TCC.

Tres comparacoes:
  1. Ganho de sequenciamento: rota otimizada x ordem nao otimizada (mesmos medicos, mesmo dia)
  2. Baseline observado: pratica real registrada no CRM (mai-jul/2026) x rota otimizada
  3. Equilibrio de potencial entre as zonas de um mesmo representante

Todas as distancias desta analise usam haversine, para que baseline e rota otimizada
sejam medidos pela MESMA regua. A quilometragem absoluta do pipeline (etapa 07) usa OSRM.
"""
import numpy as np
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distancia_rodoviaria as dr

RNG = np.random.default_rng(42)
N_PERM = 20          # permutacoes aleatorias por dia
RAIO_TERRA_KM = 6371.0
SAIDA = 'saida/resultados/avaliacao_comparativa.txt'


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return RAIO_TERRA_KM * 2 * np.arcsin(np.sqrt(a))


def comprimento(lats, lons):
    """Soma das pernas consecutivas de um trajeto."""
    if len(lats) < 2:
        return 0.0
    return float(haversine(lats[:-1], lons[:-1], lats[1:], lons[1:]).sum())


def bloco(titulo):
    return '\n' + '=' * 74 + '\n  ' + titulo + '\n' + '=' * 74


# ---------------------------------------------------------------- 1. sequenciamento
def ganho_sequenciamento(rotas):
    linhas = [bloco('1. GANHO DE SEQUENCIAMENTO — rota otimizada x ordem nao otimizada')]
    linhas.append('  Mesmos medicos, mesmo dia, mesma zona. Muda so a ORDEM das visitas.')
    linhas.append('  Baseline = media de {} permutacoes aleatorias por dia.\n'.format(N_PERM))

    reg = []
    for (rep, sem, dia), g in rotas.groupby(['rep_id', 'semana', 'dia'], sort=False):
        g = g.sort_values('ordem')
        la, lo = g.latitude.to_numpy(), g.longitude.to_numpy()
        n = len(la)
        if n < 3:                      # com 1 ou 2 pontos nao ha o que ordenar
            continue
        km_otim = comprimento(la, lo)
        km_rand = np.mean([comprimento(la[p], lo[p])
                           for p in (RNG.permutation(n) for _ in range(N_PERM))])
        reg.append({'rep_id': rep, 'semana': sem, 'dia': dia, 'n_visitas': n,
                    'km_otimizado': km_otim, 'km_aleatorio': km_rand})

    d = pd.DataFrame(reg)
    tot_o, tot_r = d.km_otimizado.sum(), d.km_aleatorio.sum()
    d['reducao_pct'] = np.where(d.km_aleatorio > 0,
                                100 * (1 - d.km_otimizado / d.km_aleatorio), np.nan)

    linhas.append('  Dias avaliados          : {:,}'.format(len(d)))
    linhas.append('  Visitas                 : {:,}'.format(d.n_visitas.sum()))
    linhas.append('  Km ordem nao otimizada  : {:,.0f}'.format(tot_r))
    linhas.append('  Km rota otimizada       : {:,.0f}'.format(tot_o))
    linhas.append('  REDUCAO AGREGADA        : {:.1f}%'.format(100 * (1 - tot_o / tot_r)))
    linhas.append('  Reducao mediana por dia : {:.1f}%'.format(d.reducao_pct.median()))
    linhas.append('')
    linhas.append('  Reducao por tamanho do dia:')
    faixa = pd.cut(d.n_visitas, [2, 4, 6, 8, 99], labels=['3-4', '5-6', '7-8', '9+'])
    for f, gg in d.groupby(faixa, observed=True):
        linhas.append('    {:>4s} visitas  n={:5,}  reducao {:5.1f}%'.format(
            str(f), len(gg), 100 * (1 - gg.km_otimizado.sum() / gg.km_aleatorio.sum())))
    return '\n'.join(linhas), d


# ---------------------------------------------------------------- 2. pratica observada
def baseline_observado(rotas):
    linhas = [bloco('2. BASELINE OBSERVADO — pratica registrada no CRM x rota otimizada')]

    t = pd.read_csv('dados_crm/2807/crm_task_report.csv', sep=';', low_memory=False)
    # ATENCAO: datas em ISO. Usar dayfirst=True descarta 12.500 das 20.207 linhas.
    t['taskstart'] = pd.to_datetime(t.taskstart, errors='coerce')
    t = t[t.taskstart.notna()]
    t = t[t.taskstate.astype(str).str.contains('Realizad', case=False, na=False)]

    # recorte valido: o registro so engrena em mai/2026
    t = t[(t.taskstart >= '2026-05-01') & (t.taskstart < '2026-08-01')]

    corr = pd.read_csv('saida/resultados/correspondencia_contactid.csv')
    coord = pd.read_csv('saida/resultados/base_medicos_validada.csv',
                        usecols=['medico_id', 'latitude', 'longitude'])
    t = (t.merge(corr[['contact_id', 'medico_id']], on='contact_id', how='inner')
           .merge(coord, on='medico_id', how='inner'))
    t = t[t.latitude.notna()]
    t['data'] = t.taskstart.dt.date

    # so os representantes que o modelo de fato roteirizou
    repmap = pd.read_csv('saida/correspondencia_representantes.csv')
    no_modelo = set(rotas.rep_id.unique())
    repmap = repmap[repmap.rep_id.isin(no_modelo)]
    t = t.merge(repmap, left_on='owner', right_on='nome_real', how='inner')

    reg = []
    for (rep, dia), g in t.groupby(['rep_id', 'data'], sort=False):
        g = g.sort_values('taskstart')
        la, lo = g.latitude.to_numpy(), g.longitude.to_numpy()
        if len(la) < 2:
            continue
        reg.append({'rep_id': rep, 'data': dia, 'n_visitas': len(la),
                    'km': dr.km_dia(la, lo)})
    obs = pd.DataFrame(reg)
    print('    observado: %d dias medidos' % len(obs), flush=True)

    # rota otimizada, mesma regua
    reg_o = []
    for (rep, sem, dia), g in rotas.groupby(['rep_id', 'semana', 'dia'], sort=False):
        g = g.sort_values('ordem')
        la, lo = g.latitude.to_numpy(), g.longitude.to_numpy()
        if len(la) < 2:
            continue
        reg_o.append({'rep_id': rep, 'n_visitas': len(la), 'km': dr.km_dia(la, lo)})
    otim = pd.DataFrame(reg_o)
    dr.salvar()
    print('    modelo: %d dias medidos | %s' % (len(otim), dr.resumo()), flush=True)

    # o dia observado tem cauda pesada (deslocamentos aereos entre pracas).
    # o corte no p99 remove esses dias sem tocar na rotina de campo.
    corte = obs.km.quantile(0.99)
    obs_t = obs[obs.km <= corte]

    def kpv(d):
        return d.km.sum() / d.n_visitas.sum()

    linhas.append('  Recorte: visitas realizadas entre mai/2026 e jul/2026 (o registro no CRM')
    linhas.append('  so engrena em maio; antes disso ha dezenas de visitas por mes), restrito')
    linhas.append('  aos {} representantes que o modelo roteirizou.\n'.format(obs.rep_id.nunique()))
    linhas.append('  {:28s}{:>13s}{:>13s}'.format('', 'OBSERVADO', 'MODELO'))
    linhas.append('  {:28s}{:>13,}{:>13,}'.format('dias de campo', len(obs), len(otim)))
    linhas.append('  {:28s}{:>13,}{:>13,}'.format('visitas', obs.n_visitas.sum(), otim.n_visitas.sum()))
    linhas.append('  {:28s}{:>13.1f}{:>13.1f}'.format('visitas por dia (mediana)', obs.n_visitas.median(), otim.n_visitas.median()))
    linhas.append('  {:28s}{:>13.1f}{:>13.1f}'.format('km por dia (mediana)', obs.km.median(), otim.km.median()))
    linhas.append('')
    linhas.append('  KM POR VISITA, por criterio de agregacao:')
    linhas.append('  {:28s}{:>13.2f}{:>13.2f}   reducao {:.1f}%'.format(
        'media, todos os dias', kpv(obs), kpv(otim), 100 * (1 - kpv(otim) / kpv(obs))))
    linhas.append('  {:28s}{:>13.2f}{:>13.2f}   reducao {:.1f}%'.format(
        'media, sem o 1% extremo', kpv(obs_t), kpv(otim), 100 * (1 - kpv(otim) / kpv(obs_t))))
    med_o = (obs.km / obs.n_visitas).median()
    med_m = (otim.km / otim.n_visitas).median()
    linhas.append('  {:28s}{:>13.2f}{:>13.2f}   reducao {:.1f}%'.format(
        'mediana do dia', med_o, med_m, 100 * (1 - med_m / med_o)))

    # comparacao pareada: cada representante e seu proprio controle
    par = (obs.groupby('rep_id').apply(kpv, include_groups=False).rename('obs')
             .to_frame()
             .join(otim.groupby('rep_id').apply(kpv, include_groups=False).rename('mod'))
             .dropna())
    par['reducao_pct'] = 100 * (1 - par['mod'] / par['obs'])
    linhas.append('')
    linhas.append('  COMPARACAO PAREADA (cada representante e seu proprio controle):')
    linhas.append('    representantes com dado dos dois lados : {}'.format(len(par)))
    linhas.append('    reducao mediana                        : {:.1f}%'.format(par.reducao_pct.median()))
    linhas.append('    1o quartil / 3o quartil                : {:.1f}% / {:.1f}%'.format(
        par.reducao_pct.quantile(.25), par.reducao_pct.quantile(.75)))
    linhas.append('    representantes com piora (reducao < 0) : {}'.format(int((par.reducao_pct < 0).sum())))
    linhas.append('')
    linhas.append('  O QUE ESTE NUMERO MEDE — e o que ele NAO mede:')
    linhas.append('  A comparacao contrapoe o dia planejado pelo modelo ao dia efetivamente')
    linhas.append('  realizado. O ganho vem de duas fontes somadas: o zoneamento, que agrupa')
    linhas.append('  medicos proximos no mesmo dia, e o sequenciamento, isolado no item 1.')
    linhas.append('  Nao e um teste controlado: o modelo escolhe QUAIS medicos visitar junto,')
    linhas.append('  enquanto o historico registra as escolhas que os representantes fizeram.')
    linhas.append('')
    linhas.append('  REGUA: os dois lados foram medidos pela malha viaria real, somando o')
    linhas.append('  percurso por ruas entre visitas consecutivas, na ordem em que ocorreram.')
    linhas.append('')
    linhas.append('  LIMITACAO: o historico nao registra o ponto de partida do representante,')
    linhas.append('  entao a quilometragem observada e medida entre visitas consecutivas, sem')
    linhas.append('  o deslocamento de casa ate a primeira visita nem o retorno. A rota do')
    linhas.append('  modelo foi medida do mesmo modo, para manter a comparacao simetrica.')
    return '\n'.join(linhas), obs, otim


# ---------------------------------------------------------------- 3. equilibrio de potencial
def equilibrio_potencial():
    linhas = [bloco('3. EQUILIBRIO DE POTENCIAL ENTRE AS ZONAS DE UM MESMO REPRESENTANTE')]
    s = pd.read_csv('saida/resultados/base_medicos_score.csv')
    # cluster == -1 marca o medico sem coordenada, que nao entrou em zona
    # alguma. Agrupar por cluster sem excluir esse valor cria uma zona
    # fantasma por representante, de tamanho arbitrario e potencial quase
    # nulo, que inflava os tres coeficientes de variacao.
    s = s[s.cluster.notna() & (s.cluster != -1)].copy()
    s['px'] = pd.to_numeric(s.px_total_tri, errors='coerce').fillna(0)
    s['foco1'] = s.classificacao.astype(str).str.startswith(('1.', '2.'))

    zona = (s.groupby(['rep_id', 'cluster'])
              .agg(n=('medico_id', 'size'), px=('px', 'sum'), foco1=('foco1', 'sum'))
              .reset_index())

    def cv(x):
        m = x.mean()
        return 100 * x.std(ddof=0) / m if m > 0 else np.nan

    porrep = (zona.groupby('rep_id')
                  .agg(zonas=('cluster', 'size'), cv_tam=('n', cv),
                       cv_px=('px', cv), cv_f1=('foco1', cv))
                  .reset_index())
    porrep = porrep[porrep.zonas > 1]

    linhas.append('  O k-means equilibra TAMANHO de zona. A pergunta e se o potencial comercial')
    linhas.append('  tambem ficou equilibrado, ou se o modelo criou zonas ricas e zonas pobres.')
    linhas.append('  Metrica: coeficiente de variacao entre as zonas do mesmo representante.\n')
    linhas.append('  Representantes avaliados : {}'.format(len(porrep)))
    linhas.append('  CV do numero de medicos  : {:5.1f}%  (mediana {:.1f}%)'.format(porrep.cv_tam.mean(), porrep.cv_tam.median()))
    linhas.append('  CV do potencial (Px)     : {:5.1f}%  (mediana {:.1f}%)'.format(porrep.cv_px.mean(), porrep.cv_px.median()))
    linhas.append('  CV de medicos Foco 1/2   : {:5.1f}%  (mediana {:.1f}%)'.format(porrep.cv_f1.mean(), porrep.cv_f1.median()))
    linhas.append('')
    linhas.append('  Leitura: quanto maior a diferenca entre o CV de potencial e o CV de tamanho,')
    linhas.append('  mais o equilibrio geografico deixou de garantir equilibrio comercial.')
    pior = porrep.nlargest(5, 'cv_px')[['rep_id', 'zonas', 'cv_tam', 'cv_px']]
    linhas.append('\n  Representantes com maior desequilibrio de potencial:')
    for _, r in pior.iterrows():
        linhas.append('    {}  zonas={:2d}  CV tamanho={:5.1f}%  CV potencial={:6.1f}%'.format(
            r.rep_id, int(r.zonas), r.cv_tam, r.cv_px))
    return '\n'.join(linhas), porrep


def main():
    rotas = pd.read_csv('saida/resultados/rotas_vrp.csv')
    rotas = rotas[rotas.latitude.notna() & rotas.longitude.notna()]

    partes = ['AVALIACAO COMPARATIVA DA ROTEIRIZACAO',
              'Gerado a partir de saida/resultados/rotas_vrp.csv e do task_report de 28/07']

    txt1, d1 = ganho_sequenciamento(rotas)
    partes.append(txt1)
    txt2, obs, otim = baseline_observado(rotas)
    partes.append(txt2)
    txt3, porrep = equilibrio_potencial()
    partes.append(txt3)

    saida = '\n'.join(partes) + '\n'
    with open(SAIDA, 'w', encoding='utf-8') as f:
        f.write(saida)
    d1.to_csv('saida/resultados/comparativo_sequenciamento.csv', index=False)
    porrep.to_csv('saida/resultados/comparativo_equilibrio_zonas.csv', index=False)
    print(saida)
    print('\n[ok] ' + SAIDA)


if __name__ == '__main__':
    main()
