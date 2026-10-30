"""Parsing robusto da saída do modelo (JSON puro, cercas ```json, tags <json>)."""
from __future__ import annotations

import json
import re
from typing import Optional

_TAG = re.compile(r"<json>(.*?)</json>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[dict]:
    """Devolve o primeiro objeto JSON válido encontrado, ou ``None``."""
    if not text:
        return None
    candidates: list[str] = []
    m = _TAG.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates += [g for g in _FENCE.findall(text)]
    candidates.append(text.strip())
    # último recurso: do primeiro "{" ao último "}"
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        cand = re.sub(r"^```(?:json)?|```$", "", cand.strip()).strip()
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
