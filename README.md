# Roteirização inteligente da força de visitação médica

Código do Trabalho de Conclusão de Curso do MBA em Data Science e Analytics
da USP/Esalq (2026).

O trabalho organiza a visitação de uma equipe de 109 representantes
comerciais a uma base de 15.288 médicos prescritores: localiza cada médico
a partir do endereço de cadastro, reparte cada carteira em zonas semanais
por agrupamento geoespacial e sequencia as visitas de cada semana por
otimização combinatória.

---

## Os dados não estão aqui

Este repositório contém **apenas o código**. As bases ficaram de fora por
duas razões, e nenhuma delas é negociável:

- o cadastro traz endereços de médicos identificáveis, cujo tratamento se
  submete à Lei Geral de Proteção de Dados;
- a planilha de priorização é informação comercial da empresa estudada,
  cujo nome não é divulgado no trabalho.

O código, portanto, não roda sem as bases. Ele está publicado para que o
método possa ser lido, conferido e criticado, não para ser reexecutado por
terceiros. Os nomes de arquivo e de pasta que aparecem no código referem-se
à extração usada no trabalho; quem reexecutar com as próprias bases ajusta
esses caminhos.

### Um arquivo que nunca pode ser publicado

`saida/.geomask_salt` guarda o segredo que gera o deslocamento aleatório
aplicado às coordenadas. As coordenadas da base saem deslocadas em até
300 m, de forma determinística por médico. **Quem tiver o salt e as
coordenadas deslocadas recupera as originais**, e a proteção deixa de
existir. O `.gitignore` o bloqueia; convém conferir antes de cada commit.

---

## Ordem de execução

As etapas consomem a saída da anterior e precisam rodar em sequência.

```
python scripts/pipeline/etapa_01_anonimizacao.py
python scripts/pipeline/etapa_02_diagnostico.py
python scripts/pipeline/etapa_03_correcao_cep.py
python scripts/pipeline/etapa_04_geocodificacao.py
python scripts/pipeline/etapa_04b_fallback_opencage.py
python scripts/pipeline/etapa_04d_regeocod_endereco.py
python scripts/pipeline/etapa_04c_validacao_municipio.py
python scripts/pipeline/etapa_04e_opencage_logradouro.py
python scripts/pipeline/etapa_05_kmeans.py
python scripts/pipeline/etapa_06_score.py
python scripts/pipeline/etapa_07_vrp.py
```

A etapa 07 é a longa: resolve 688 semanas com limite de dez segundos de
busca por zona e consulta um servidor de roteamento, levando cerca de uma
hora. `scripts/reexecutar_pipeline.py` encadeia da etapa 05 em diante e
interrompe na primeira falha.

A ordem das etapas 04 não acompanha a letra: a validação municipal (04c) roda
depois da regeocodificação por endereço (04d), de propósito. Validar antes
rejeitaria coordenadas que a etapa seguinte substituiria de qualquer forma.

No Windows, defina `PYTHONIOENCODING=utf-8` antes de rodar.

### O que cada etapa faz

| Etapa | O que faz |
|---|---|
| 01 | substitui identificadores pessoais por códigos determinísticos |
| 02 | diagnostica o campo de código postal em formato, repetição e geografia |
| 03 | corrige códigos recuperáveis e classifica os irrecuperáveis |
| 04 | primeira geocodificação, pelo código postal |
| 04b | contingência para os códigos que o serviço aberto não resolveu |
| 04d | regeocodificação pelo endereço completo, com número do imóvel |
| 04c | validação das coordenadas contra a mediana do município |
| 04e | contingência para os que permaneceram em nível de logradouro |
| 05 | zoneamento por k-means com restrição de tamanho |
| 06 | associação ao índice de prioridade da área comercial |
| 07 | sequenciamento das visitas por programação por restrições |

### Análises e figuras

`scripts/analises/` reúne o que sustenta os números do texto. As
principais:

| Arquivo | Pergunta que responde |
|---|---|
| `avaliacao_comparativa.py` | quanto o plano reduz de deslocamento frente à prática registrada |
| `prioridade_observada.py` | a equipe já visita mais quem tem mais potencial |
| `sensibilidade_kmeans.py` | quanto custam as duas decisões de modelagem do agrupamento |
| `coincidencia_endereco.py` | quanto da coincidência de coordenadas é real e quanto é perda de precisão |
| `experimento_frequencia.py` | para onde vai a capacidade que sobra depois de cobrir a carteira |
| `cv_aleatorio.py` | quanto do desequilíbrio entre zonas é do método e quanto é do dado |
| `diagnostico_visitas_crm.py` | o que a operação de fato faz, segundo o histórico |
| `qualidade_endereco_crm.py` | quanto o cadastro adere ao endereçamento oficial |


### Figuras

| Arquivo | O que produz |
|---|---|
| `gerar_figuras_tcc.py` | gráficos no padrão do manual: sem grade, sem borda, sem título |
| `gerar_figuras_mapa.py` | mapas cartográficos sobre malha estadual em GeoJSON local |
| `gerar_mapas_coloridos.py` | zonas e semana sequenciada sobre tiles do OpenStreetMap |
| `gerar_mapa_nacional_cor.py` | carteiras no território, uma cor por representante |
| `gerar_figuras_fluxo.py`, `gerar_figura_fluxograma_geo.py` | fluxogramas do método |
| `gerar_figura_prioridade.py` | esforço de visita confrontado ao potencial de mercado |

Todas gravam imagem em `saida/`. As coordenadas plotadas passaram pelo
deslocamento de proteção, de modo que nenhum ponto marca o consultório real.

---

## Serviços externos

| Serviço | Uso | Chave |
|---|---|---|
| ViaCEP | logradouro oficial a partir do código postal | não exige |
| Nominatim (OpenStreetMap) | endereço para coordenada | não exige; até 1 req/s |
| OpenCage | contingência quando o anterior não resolve | **exige** |
| OSRM | distância por malha viária | não exige |

A chave do OpenCage é lida da variável de ambiente, nunca do código:

```
$env:OPENCAGE_API_KEY = "<sua chave>"      # PowerShell
export OPENCAGE_API_KEY="<sua chave>"      # bash
```

O Nominatim exige identificação de quem consulta. Informe um contato em
`GEOCODER_CONTACT`; sem ele o código usa uma identificação genérica, aceita
mas menos educada com um serviço mantido por voluntários.

Os três serviços públicos impõem limite de requisições. O código respeita
o intervalo e guarda as respostas em cache, de modo que reexecuções não
repetem consultas.

---

## Instalação

```
pip install -r requirements.txt
```

Testado em Python 3.12.
