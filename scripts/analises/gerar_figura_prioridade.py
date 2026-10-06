# -*- coding: utf-8 -*-
"""
Figura: esforco de visitacao x potencial de prescricao, por faixa de prioridade.

Contrapoe, para cada faixa, a participacao no total de visitas realizadas e a
participacao no volume de prescricao do mercado. O descompasso entre as duas
barras e o achado: a faixa que concentra o maior potencial em aberto e a que
recebe menos esforco.

Estilo conforme o Manual USP/Esalq: sem grade, sem borda, sem titulo, eixos
pretos de 1,5 pt.

Saida: saida/resultados/figuras_tcc/figura_prioridade_esforco.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = 'saida/resultados/figuras_tcc'
os.makedirs(DIR, exist_ok=True)

ESCURO = '#404040'
CLARO = '#BFBFBF'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

ROTULOS = {
    '1. ALTO Px e Share ABAIXO': 'Alto volume,\nparticipação abaixo',
    '2. ALTO Px e Share ACIMA': 'Alto volume,\nparticipação acima',
    '3. MÉDIO Px e Share ABAIXO': 'Médio volume,\nparticipação abaixo',
    '4. MÉDIO Px e Share ACIMA': 'Médio volume,\nparticipação acima',
    '5. BAIXO Px e Share ABAIXO': 'Baixo volume,\nparticipação abaixo',
    '6. BAIXO Px e Share ACIMA': 'Baixo volume,\nparticipação acima',
    '7. SEM PX REGISTRADO': 'Sem prescrição\nregistrada',
}


def dados():
    t = pd.read_csv('dados_crm/2807/crm_task_report.csv', sep=';', low_memory=False)
    t['dt'] = pd.to_datetime(t.taskstart, errors='coerce')
    t = t[t.taskstate.astype(str).str.contains('Realizado', na=False)]
    t = t[(t.dt >= '2026-05-01') & (t.dt < '2026-08-01')]
    corr = pd.read_csv('saida/resultados/correspondencia_contactid.csv')
    t = t.merge(corr[['contact_id', 'medico_id']], on='contact_id', how='left')

    sc = pd.read_csv('saida/resultados/base_medicos_score.csv', low_memory=False)
    med = sc.drop_duplicates('medico_id')[['medico_id', 'classificacao', 'px_total_tri']]
    vis = t.groupby('medico_id').size().rename('visitas')
    d = med.merge(vis, on='medico_id', how='left')
    d['visitas'] = d.visitas.fillna(0)

    g = d.groupby('classificacao')
    r = pd.DataFrame({
        'visitas': g.visitas.sum() / d.visitas.sum() * 100,
        'px': g.px_total_tri.sum() / d.px_total_tri.sum() * 100,
    }).sort_index()
    return r


def main():
    r = dados()
    rot = [ROTULOS.get(k, k) for k in r.index]
    y = np.arange(len(r))[::-1]
    alt = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(y + alt / 2, r.visitas, height=alt, color=ESCURO,
            label='Participação nas visitas realizadas')
    ax.barh(y - alt / 2, r.px, height=alt, color=CLARO, edgecolor=ESCURO,
            linewidth=0.5, label='Participação no volume de prescrição')

    def num(v):
        return ('%.1f' % v).replace('.', ',')

    for yi, v in zip(y + alt / 2, r.visitas):
        ax.text(v + 0.7, yi, num(v), va='center', ha='left', fontsize=8)
    for yi, v in zip(y - alt / 2, r.px):
        ax.text(v + 0.7, yi, num(v), va='center', ha='left', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(rot, fontsize=8)
    ax.set_xlabel('Participação no total (%)')
    ax.set_xlim(0, max(r.visitas.max(), r.px.max()) * 1.14)

    ax.grid(False)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    for lado in ('left', 'bottom'):
        ax.spines[lado].set_linewidth(1.5)
        ax.spines[lado].set_color('black')
    ax.tick_params(colors='black', width=1.0)
    ax.set_title('')
    ax.set_facecolor('none')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2,
              frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.6)

    fig.patch.set_alpha(0)
    caminho = os.path.join(DIR, 'figura_prioridade_esforco.png')
    fig.savefig(caminho, transparent=False, facecolor='white')
    plt.close(fig)
    print('  [ok] %s' % caminho)
    print(r.round(1).to_string())


if __name__ == '__main__':
    main()
