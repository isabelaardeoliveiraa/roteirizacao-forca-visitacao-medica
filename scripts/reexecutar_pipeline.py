# -*- coding: utf-8 -*-
"""
Roda em sequencia as etapas que dependem da geocodificacao.

Cada uma consome a saida da anterior, e so comeca se a anterior terminar com
codigo zero. O log registra duracao e as ultimas linhas de cada etapa, de modo
que uma execucao longa possa ser conferida depois, sem acompanhamento.

Pressupoe a base de coordenadas ja validada em
saida/resultados/base_medicos_validada.csv.

O sequenciamento e a etapa longa: 688 zonas, dez segundos de busca por zona e
consultas a um servidor publico de roteamento.
"""
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ETAPAS = [
    ('05 zoneamento',      'scripts/pipeline/etapa_05_kmeans.py'),
    ('06 score',           'scripts/pipeline/etapa_06_score.py'),
    ('07 sequenciamento',  'scripts/pipeline/etapa_07_vrp.py'),
    ('avaliacao',          'scripts/analises/avaliacao_comparativa.py'),
    ('frequencia',         'scripts/analises/experimento_frequencia.py'),
    ('prioridade',         'scripts/analises/prioridade_observada.py'),
    ('sensibilidade',      'scripts/analises/sensibilidade_kmeans.py'),
    ('cv aleatorio',       'scripts/analises/cv_aleatorio.py'),
    ('visitas crm',        'scripts/analises/diagnostico_visitas_crm.py'),
]

env = dict(os.environ, PYTHONIOENCODING='utf-8')


def main():
    t0 = time.time()
    for nome, script in ETAPAS:
        print('\n' + '=' * 70)
        print('  %s  ->  %s' % (nome.upper(), script))
        print('=' * 70, flush=True)
        ini = time.time()
        r = subprocess.run([sys.executable, script], env=env,
                           capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        dur = time.time() - ini
        saida = (r.stdout or '').rstrip().split('\n')
        for l in saida[-25:]:
            print('   ' + l)
        if r.returncode != 0:
            print('\n   !! FALHOU (codigo %d) em %.1f min' % (r.returncode, dur / 60))
            print('   stderr:')
            for l in (r.stderr or '').rstrip().split('\n')[-20:]:
                print('     ' + l)
            print('\n   CADEIA INTERROMPIDA. As etapas seguintes nao rodaram.')
            return 1
        print('\n   [ok] %s em %.1f min' % (nome, dur / 60), flush=True)

    print('\n' + '=' * 70)
    print('  CADEIA COMPLETA em %.1f min' % ((time.time() - t0) / 60))
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
