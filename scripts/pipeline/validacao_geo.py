"""
Validação geográfica de coordenadas obtidas por geocodificação.

Motivação: o Nominatim, quando não encontra o logradouro, devolve o match mais
amplo que conseguiu — inclusive o centroide do país. Sem validação, esses
resultados entram na base como se fossem endereços legítimos (foram detectados
88 CEPs apontando para (-10.33, -53.20), o centro geográfico do Brasil).

Três checagens, todas determinísticas e sem custo de API:
  1. nivel_aceitavel()  — o resultado é específico o bastante? (rua/bairro/cidade)
  2. dentro_da_uf()     — a coordenada cai no estado indicado pelo CEP?
  3. e_centroide()      — a coordenada é um centroide conhecido de país/estado?

Usado por etapa_04_geocodificacao.py e etapa_04b_fallback_opencage.py.
"""

# ── CEP → UF (faixas oficiais dos Correios) ───────────────────────────────────
FAIXAS_CEP = [
    (1000, 19999, "SP"),  (20000, 28999, "RJ"), (29000, 29999, "ES"),
    (30000, 39999, "MG"), (40000, 48999, "BA"), (49000, 49999, "SE"),
    (50000, 56999, "PE"), (57000, 57999, "AL"), (58000, 58999, "PB"),
    (59000, 59999, "RN"), (60000, 63999, "CE"), (64000, 64999, "PI"),
    (65000, 65999, "MA"), (66000, 68899, "PA"), (68900, 68999, "AP"),
    (69000, 69299, "AM"), (69300, 69389, "RR"), (69400, 69899, "AM"),
    (69900, 69999, "AC"), (70000, 72799, "DF"), (72800, 72999, "GO"),
    (73000, 73699, "DF"), (73700, 76799, "GO"), (76800, 76999, "RO"),
    (77000, 77999, "TO"), (78000, 78899, "MT"), (78900, 78999, "RO"),
    (79000, 79999, "MS"), (80000, 87999, "PR"), (88000, 89999, "SC"),
    (90000, 99999, "RS"),
]

# Bounding box por UF, com margem generosa: o objetivo é pegar erro grosseiro
# (coordenada em outro estado), não fronteira exata.
BBOX_UF = {
    "AC": (-11.2, -7.1, -74.0, -66.6), "AL": (-10.6, -8.8, -38.3, -35.1),
    "AM": (-9.9, 2.3, -73.9, -56.0),   "AP": (-1.3, 4.5, -54.9, -49.8),
    "BA": (-18.4, -8.5, -46.7, -37.3), "CE": (-7.9, -2.7, -41.5, -37.2),
    "DF": (-16.1, -15.4, -48.3, -47.3),"ES": (-21.4, -17.8, -41.9, -39.6),
    "GO": (-19.6, -12.3, -53.3, -45.9),"MA": (-10.4, -1.0, -48.8, -41.7),
    "MG": (-22.9, -14.2, -51.1, -39.8),"MS": (-24.1, -17.1, -58.2, -50.9),
    "MT": (-18.1, -7.3, -61.7, -50.2), "PA": (-9.9, 2.7, -58.9, -46.0),
    "PB": (-8.4, -6.0, -38.8, -34.7),  "PE": (-9.5, -7.2, -41.4, -34.8),
    "PI": (-10.9, -2.7, -45.9, -40.3), "PR": (-26.8, -22.4, -54.7, -48.0),
    "RJ": (-23.4, -20.7, -44.9, -40.9),"RN": (-6.9, -4.8, -38.6, -34.9),
    "RO": (-13.7, -7.9, -66.9, -59.7), "RR": (-1.6, 5.3, -64.9, -58.8),
    "RS": (-33.8, -27.0, -57.7, -49.6),"SC": (-29.4, -25.9, -53.9, -48.3),
    "SE": (-11.6, -9.4, -38.3, -36.3), "SP": (-25.4, -19.7, -53.2, -44.1),
    "TO": (-13.5, -5.1, -50.8, -45.7),
}

MARGEM_GRAUS = 0.3   # ~33 km de tolerância na fronteira

# Centroides que nunca devem ser aceitos como endereço de consultório.
CENTROIDES_PROIBIDOS = [
    (-10.33, -53.20),   # centroide do Brasil — o caso observado na base
    (-14.24, -51.93),   # outra convenção de centro do Brasil
    (0.0, 0.0),         # null island
]
TOLERANCIA_CENTROIDE = 0.05   # ~5 km

# Níveis de resultado aceitáveis (Nominatim: campo "type"/"addresstype").
# Rejeita country/state/region — amplos demais para localizar um consultório.
NIVEIS_REJEITADOS = {
    "country", "state", "region", "province", "continent",
    "administrative",   # genérico demais quando vem sozinho
}


def uf_do_cep(cep):
    """UF a partir do CEP, pelas faixas oficiais. None se não classificável."""
    try:
        prefixo = int(str(cep).replace("-", "").replace(" ", "").zfill(8)[:5])
    except (ValueError, TypeError):
        return None
    for inicio, fim, uf in FAIXAS_CEP:
        if inicio <= prefixo <= fim:
            return uf
    return None


def e_centroide(lat, lon):
    """True se a coordenada coincide com um centroide de país/estado conhecido."""
    for clat, clon in CENTROIDES_PROIBIDOS:
        if abs(lat - clat) <= TOLERANCIA_CENTROIDE and abs(lon - clon) <= TOLERANCIA_CENTROIDE:
            return True
    return False


def dentro_da_uf(lat, lon, cep):
    """True se a coordenada cai no estado indicado pelo CEP.
    Retorna True também quando não há como verificar (UF desconhecida)."""
    uf = uf_do_cep(cep)
    caixa = BBOX_UF.get(uf) if uf else None
    if not caixa:
        return True
    lat_min, lat_max, lon_min, lon_max = caixa
    return (lat_min - MARGEM_GRAUS <= lat <= lat_max + MARGEM_GRAUS and
            lon_min - MARGEM_GRAUS <= lon <= lon_max + MARGEM_GRAUS)


def nivel_aceitavel(tipo):
    """False para resultados amplos demais (país, estado, região)."""
    return str(tipo or "").strip().lower() not in NIVEIS_REJEITADOS


def validar(lat, lon, cep, tipo=None):
    """Valida uma coordenada geocodificada.

    Retorna (ok: bool, motivo: str|None). motivo preenchido só quando ok=False.
    """
    if lat is None or lon is None:
        return False, "sem_coordenada"
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False, "coordenada_invalida"

    if not (-34.0 <= lat <= 6.0 and -74.5 <= lon <= -34.0):
        return False, "fora_do_brasil"
    if e_centroide(lat, lon):
        return False, "centroide_generico"
    if not nivel_aceitavel(tipo):
        return False, f"nivel_amplo_demais({tipo})"
    if not dentro_da_uf(lat, lon, cep):
        return False, f"fora_da_uf({uf_do_cep(cep)})"
    return True, None
