"""Baseline determinístico (regex) — o "piso" contra o qual os LLMs são comparados.

Funciona bem em documentos rotulados (estilo ``estruturado``) e degrada em texto livre.
Isso é proposital: mede quanto valor o LLM realmente agrega.
"""
from __future__ import annotations

import re
from typing import Optional

_LABELS: dict[str, list[str]] = {
    "numero_documento": [r"numero\s+do\s+documento"],
    "data_emissao": [r"data\s+de\s+emissao"],
    "remetente_nome": [r"remetente"],
    "remetente_cnpj": [r"cnpj\s+remetente"],
    "destinatario_nome": [r"destinatario"],
    "destinatario_cnpj": [r"cnpj\s+destinatario"],
    "origem": [r"origem"],
    "destino": [r"destino"],
    "peso_kg": [r"peso\s+bruto"],
    "volumes": [r"volumes"],
    "valor_mercadoria": [r"valor\s+da\s+mercadoria"],
    "valor_frete": [r"valor\s+do\s+frete"],
    "placa_veiculo": [r"placa\s+do\s+veiculo"],
}


def _strip_accents(s: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _br_number(s: str) -> Optional[float]:
    m = re.search(r"\d[\d\.]*(?:,\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(".", "").replace(",", "."))
    except ValueError:
        return None


def extract(text: str) -> dict:
    """Extrai campos de um documento rotulado. Campos não encontrados ficam ``None``."""
    result: dict = {k: None for k in _LABELS}
    for raw in text.splitlines():
        line = _strip_accents(raw)
        for field, patterns in _LABELS.items():
            for pat in patterns:
                m = re.match(rf"^\s*{pat}\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
                if m:
                    # preserva o valor original (com acentos) usando o mesmo span
                    value = raw[raw.index(":") + 1:].strip()
                    result[field] = value
    # tipagem
    if result["data_emissao"]:
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", result["data_emissao"])
        result["data_emissao"] = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
    if result["peso_kg"]:
        result["peso_kg"] = _br_number(result["peso_kg"])
    for f in ("valor_mercadoria", "valor_frete"):
        if result[f]:
            result[f] = _br_number(result[f])
    if result["volumes"]:
        m = re.search(r"\d+", result["volumes"])
        result["volumes"] = int(m.group(0)) if m else None
    return result
