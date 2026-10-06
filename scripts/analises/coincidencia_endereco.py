# -*- coding: utf-8 -*-
"""
Quanto da coincidencia de coordenadas e resultado correto e quanto e perda
de precisao.

Duas medidas, ambas sobre o cadastro e sobre as coordenadas BRUTAS do
servico aberto, nunca sobre as da base final: estas passaram pelo
deslocamento de protecao, que e distinto por medico e por isso desfaz
qualquer coincidencia real.

  1. COMPARTILHAMENTO DE ENDERECO. Proporcao de medicos cujo endereco de
     cadastro (logradouro, numero, municipio e unidade federativa) e
     identico ao de ao menos um colega. Mede concentracao real: predio
     medico, hospital, centro clinico.

  2. COINCIDENCIA NA MESMA VIA. Entre pares de medicos da mesma via com
     numeros DIFERENTES, com que frequencia a coordenada devolvida e a
     mesma. Separa os pares em que nenhum dos dois foi resolvido em nivel
     de ponto daqueles em que ambos foram. A diferenca entre as duas taxas
     isola a perda de precisao atribuivel a ausencia de numeracao predial
     na base cartografica.

Saida: saida/resultados/coincidencia_endereco.txt
"""
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONTATOS = 'dados_crm/2807/crm_contact_report.csv'
CORRESP = 'saida/resultados/correspondencia_contactid.csv'
CHECK = 'saida/resultados/regeocod_endereco_checkpoint.csv'
BASE = 'saida/resultados/base_medicos_validada.csv'
SAIDA = 'saida/resultados/coincidencia_endereco.txt'

# niveis que o servico devolve quando localiza um ponto, e nao o eixo da via
PONTO = {'building', 'amenity', 'shop', 'office', 'healthcare', 'tourism',
         'leisure', 'craft', 'club', 'emergency'}

linhas = []


def diz(t=''):
    linhas.append(t)
    print(t)


def main():
    d = pd.read_csv(CONTATOS, sep=';', low_memory=False, dtype=str)
    corr = pd.read_csv(CORRESP, dtype=str)
    d = d.merge(corr[['contact_id', 'medico_id']], on='contact_id', how='left')

    def nz(c):
        return d[c].fillna('').astype(str).str.strip().str.upper()

    d['_rua'], d['_num'] = nz('street'), nz('building')
    d['_cid'], d['_uf'] = nz('city'), nz('region')
    d['_end'] = d._rua + '|' + d._num + '|' + d._cid + '|' + d._uf
    d['_via'] = d._rua + '|' + d._cid + '|' + d._uf
    completo = (d._rua != '') & (d._num != '') & (d._cid != '')

    diz('COINCIDENCIA DE ENDERECO E DE COORDENADA')
    diz('=' * 70)
    sub = d[completo]
    g = sub._end.value_counts()
    diz('  1. COMPARTILHAMENTO DE ENDERECO NO CADASTRO')
    diz('     medicos com endereco completo        : %s' % format(len(sub), ',d'))
    diz('     dividem endereco com ao menos um     : %s (%.1f%%)'
        % (format(int(g[g > 1].sum()), ',d'), g[g > 1].sum() / len(sub) * 100))
    diz('     maior agrupamento sob um endereco    : %d' % int(g.max()))
    diz()

    ck = pd.read_csv(CHECK)
    ck = ck[ck.lat.notna()].copy()
    b = pd.read_csv(BASE, low_memory=False)
    aceitos = set(b.loc[b._geo_estrategia == 'endereco_nominatim', 'medico_id'])
    ck = ck[ck.medico_id.isin(aceitos)]
    info = d.set_index('medico_id')[['_via', '_num']]
    ck = ck.join(info, on='medico_id')
    ck['_pt'] = ck.tipo.isin(PONTO)
    ck = ck[(ck._via.notna()) & (ck._num != '') & (~ck._via.str.startswith('|'))]

    tot = {True: 0, False: 0}
    igual = {True: 0, False: 0}
    for _, s in ck.groupby('_via'):
        if len(s) < 2:
            continue
        reg = s.drop_duplicates('_num').to_dict('records')
        for i in range(len(reg)):
            for j in range(i + 1, len(reg)):
                x, y = reg[i], reg[j]
                if x['_pt'] != y['_pt']:
                    continue                      # pares mistos nao informam
                k = bool(x['_pt'])
                tot[k] += 1
                if abs(x['lat'] - y['lat']) < 1e-9 and abs(x['lon'] - y['lon']) < 1e-9:
                    igual[k] += 1

    diz('  2. MESMA VIA, NUMEROS DIFERENTES — coordenada repetida')
    for k, rot in [(False, 'nenhum resolvido em nivel de ponto'),
                   (True, 'ambos resolvidos em nivel de ponto')]:
        if tot[k]:
            diz('     %-36s %5.1f%%  (%s de %s pares)'
                % (rot, igual[k] / tot[k] * 100,
                   format(igual[k], ',d'), format(tot[k], ',d')))
        else:
            diz('     %-36s sem pares' % rot)
    diz()
    diz('  A diferenca entre as duas taxas isola a perda de precisao que decorre')
    diz('  da ausencia de numeracao predial no acervo cartografico, distinguindo-a')
    diz('  da concentracao real medida no item 1.')

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    io.open(SAIDA, 'w', encoding='utf-8').write('\n'.join(linhas) + '\n')
    print('\n[ok] %s' % SAIDA)


if __name__ == '__main__':
    main()
