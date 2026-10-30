"""Gerador de dataset sintético (determinístico por seed) de documentos de transporte.

Todos os dados são fictícios: empresas, CNPJs (com dígitos verificadores válidos, mas
gerados aleatoriamente), valores e placas. Três "estilos" de documento simulam níveis
crescentes de dificuldade:

- ``estruturado``: campos rotulados, uma linha por campo (fácil para regex);
- ``email``: texto livre em português, formatos variados (datas por extenso, toneladas...);
- ``ocr``: texto estruturado degradado (rótulos abreviados, caixa alta, confusão O/0).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

CITIES = [
    ("São Paulo", "SP"), ("Campinas", "SP"), ("Santos", "SP"), ("Curitiba", "PR"),
    ("Joinville", "SC"), ("Belo Horizonte", "MG"), ("Uberlândia", "MG"),
    ("Rio de Janeiro", "RJ"), ("Salvador", "BA"), ("Recife", "PE"), ("Fortaleza", "CE"),
    ("Goiânia", "GO"), ("Porto Alegre", "RS"), ("Manaus", "AM"), ("Cuiabá", "MT"),
]
COMPANY_STEMS = [
    "Alfa", "Boreal", "Cerrado", "Delta", "Estrela", "Fênix", "Garça", "Horizonte",
    "Ipê", "Jacarandá", "Kappa", "Litoral", "Mantiqueira", "Nordeste", "Orion",
    "Pampa", "Quartzo", "Raízes", "Sertão", "Tropical", "Umbu", "Vale Verde",
]
COMPANY_KINDS = [
    "Distribuidora de Alimentos", "Indústria Têxtil", "Comércio de Autopeças",
    "Eletrodomésticos", "Farmacêutica", "Materiais de Construção", "Agroindustrial",
    "Cosméticos", "Papelaria e Embalagens", "Equipamentos Médicos",
]
LEGAL = ["Ltda", "S.A.", "EIRELI", "ME"]
MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
          "agosto", "setembro", "outubro", "novembro", "dezembro"]
STYLES = ["estruturado", "email", "ocr"]


def _cnpj_digits(rng: random.Random) -> str:
    base = [rng.randint(0, 9) for _ in range(8)] + [0, 0, 0, 1]

    def dv(nums: list[int], weights: list[int]) -> int:
        r = sum(n * w for n, w in zip(nums, weights)) % 11
        return 0 if r < 2 else 11 - r

    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = dv(base, w1)
    d2 = dv(base + [d1], [6] + w1)
    return "".join(map(str, base + [d1, d2]))


def fmt_cnpj(d: str) -> str:
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def _plate(rng: random.Random) -> str:
    L = "ABCDEFGHJKLMNPRSTUVWXYZ"
    if rng.random() < 0.6:  # Mercosul
        return f"{''.join(rng.choice(L) for _ in range(3))}{rng.randint(0,9)}{rng.choice(L)}{rng.randint(0,99):02d}"
    return f"{''.join(rng.choice(L) for _ in range(3))}-{rng.randint(0,9999):04d}"


def _company(rng: random.Random) -> str:
    return f"{rng.choice(COMPANY_STEMS)} {rng.choice(COMPANY_KINDS)} {rng.choice(LEGAL)}"


def brl(value: float, symbol: bool = True) -> str:
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}" if symbol else s


def _num_br(value: float, decimals: int = 2) -> str:
    s = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s


@dataclass
class Doc:
    doc_id: str
    style: str
    text: str
    truth: dict


def _truth(rng: random.Random) -> dict:
    o, d = rng.sample(CITIES, 2)
    sender, receiver = _company(rng), _company(rng)
    while receiver == sender:
        receiver = _company(rng)
    return {
        "numero_documento": f"{rng.randint(1, 999999):06d}",
        "data_emissao": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "remetente_nome": sender,
        "remetente_cnpj": fmt_cnpj(_cnpj_digits(rng)),
        "destinatario_nome": receiver,
        "destinatario_cnpj": fmt_cnpj(_cnpj_digits(rng)),
        "origem": f"{o[0]}/{o[1]}",
        "destino": f"{d[0]}/{d[1]}",
        "peso_kg": round(rng.uniform(80, 18000), 1),
        "volumes": rng.randint(1, 480),
        "valor_mercadoria": round(rng.uniform(1500, 380000), 2),
        "valor_frete": round(rng.uniform(180, 14000), 2),
        # ~30% dos documentos não informam a placa: testa alucinação
        "placa_veiculo": _plate(rng) if rng.random() < 0.7 else None,
    }


def _iso_to_br(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def _render_estruturado(t: dict) -> str:
    lines = [
        "DACTE - DOCUMENTO AUXILIAR DO CONHECIMENTO DE TRANSPORTE ELETRÔNICO",
        f"Numero do documento: {t['numero_documento']}",
        f"Data de emissao: {_iso_to_br(t['data_emissao'])}",
        f"Remetente: {t['remetente_nome']}",
        f"CNPJ remetente: {t['remetente_cnpj']}",
        f"Destinatario: {t['destinatario_nome']}",
        f"CNPJ destinatario: {t['destinatario_cnpj']}",
        f"Origem: {t['origem']}",
        f"Destino: {t['destino']}",
        f"Peso bruto: {_num_br(t['peso_kg'], 1)} kg",
        f"Volumes: {t['volumes']}",
        f"Valor da mercadoria: {brl(t['valor_mercadoria'])}",
        f"Valor do frete: {brl(t['valor_frete'])}",
    ]
    if t["placa_veiculo"]:
        lines.append(f"Placa do veiculo: {t['placa_veiculo']}")
    lines.append("Documento sem valor fiscal. Consulte a chave de acesso no portal.")
    return "\n".join(lines)


def _render_email(t: dict, rng: random.Random) -> str:
    y, m, d = t["data_emissao"].split("-")
    data = rng.choice([
        f"{int(d)} de {MONTHS[int(m) - 1]} de {y}",
        f"{d}/{m}/{y}",
        f"{int(d)}/{int(m)}/{y[2:]}",
    ])
    peso = t["peso_kg"]
    if peso >= 1000 and rng.random() < 0.5:
        peso_txt = f"{_num_br(peso / 1000, 3).rstrip('0').rstrip(',')} toneladas"
        if abs(float(peso_txt.split()[0].replace(',', '.')) * 1000 - peso) > 0.05:
            peso_txt = f"{_num_br(peso, 1)} kg"
    else:
        peso_txt = f"{_num_br(peso, 1)} kg"
    placa = (
        rng.choice([
            f" O veículo responsável é a placa {t['placa_veiculo']}.",
            f" Motorista já está a caminho com o caminhão de placa {t['placa_veiculo']}.",
        ]) if t["placa_veiculo"] else rng.choice([
            " A placa do veículo ainda será confirmada pela operação.",
            "",
        ])
    )
    tpl = rng.choice([
        (
            "Prezados, bom dia!\n\nSegue o CT-e nº {n} emitido em {data}. A carga da {rem} "
            "(CNPJ {rc}) segue para a {dst_n} (CNPJ {dc}), saindo de {o} com destino a {d}. "
            "São {vol} volumes, pesando {peso}. A mercadoria está avaliada em {vm} e o frete "
            "ficou em {vf}.{placa}\n\nAtt,\nOperações"
        ),
        (
            "Olá, time!\n\nConfirmo o embarque do documento {n} (emissão: {data}). Remetente: "
            "{rem}, CNPJ {rc}. Destinatário: {dst_n}, CNPJ {dc}. Rota {o} -> {d}. Carga total: "
            "{vol} volumes / {peso}. Valor declarado da mercadoria {vm}; frete contratado {vf}."
            "{placa}\n\nObrigado!"
        ),
    ])
    return tpl.format(
        n=t["numero_documento"].lstrip("0") if rng.random() < 0.5 else t["numero_documento"],
        data=data, rem=t["remetente_nome"], rc=t["remetente_cnpj"],
        dst_n=t["destinatario_nome"], dc=t["destinatario_cnpj"], o=t["origem"].replace("/", " - "),
        d=t["destino"].replace("/", " - "), vol=t["volumes"], peso=peso_txt,
        vm=brl(t["valor_mercadoria"]), vf=brl(t["valor_frete"]), placa=placa,
    )


def _ocr_digits(s: str, rng: random.Random) -> str:
    """Confusão típica de OCR em dígitos: 0->O em ~15% dos casos (recuperável pelo contexto)."""
    return "".join("O" if ch == "0" and rng.random() < 0.15 else ch for ch in s)


def _render_ocr(t: dict, rng: random.Random) -> str:
    lines = [
        "DACTE  -  CONHECIMENTO  DE  TRANSPORTE",
        f"N DOC.:  {_ocr_digits(t['numero_documento'], rng)}   EMISSAO:  {_iso_to_br(t['data_emissao'])}",
        f"REMET.:  {t['remetente_nome'].upper()}",
        f"CNPJ REM.:  {_ocr_digits(t['remetente_cnpj'], rng)}",
        f"DEST.:  {t['destinatario_nome'].upper()}",
        f"CNPJ DEST.:  {_ocr_digits(t['destinatario_cnpj'], rng)}",
        f"ORIG: {t['origem'].upper()}      DEST: {t['destino'].upper()}",
        f"PESO BR.:  {_num_br(t['peso_kg'], 1)} KG    VOL.:  {t['volumes']}",
        f"VL. MERC.:  {brl(t['valor_mercadoria'], symbol=False)}",
        f"VL. FRETE:  {brl(t['valor_frete'], symbol=False)}",
    ]
    if t["placa_veiculo"]:
        lines.append(f"PLACA:  {t['placa_veiculo']}")
    return "\n".join(lines)


def generate(n: int, seed: int = 42, style_weights: Optional[list[float]] = None) -> list[Doc]:
    rng = random.Random(seed)
    weights = style_weights or [0.34, 0.33, 0.33]
    docs: list[Doc] = []
    for i in range(n):
        style = rng.choices(STYLES, weights=weights)[0]
        t = _truth(rng)
        if style == "estruturado":
            text = _render_estruturado(t)
        elif style == "email":
            text = _render_email(t, rng)
        else:
            text = _render_ocr(t, rng)
        docs.append(Doc(doc_id=f"doc_{seed}_{i:04d}", style=style, text=text, truth=t))
    return docs


def save_jsonl(docs: list[Doc], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Doc]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Doc(**json.loads(line)))
    return out
