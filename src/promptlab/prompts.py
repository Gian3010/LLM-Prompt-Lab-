"""Estratégias de prompt: zero-shot, few-shot e chain-of-thought (CoT).

Cada estratégia devolve ``(system, user)``. O contrato de saída é sempre JSON com as
chaves de ``schema.FIELDS``; no CoT o modelo raciocina em ``<analise>`` e responde em
``<json>`` — isso separa o raciocínio do payload parseável.
"""
from __future__ import annotations

import json
from functools import lru_cache

from .dataset import generate
from .schema import FIELDS, schema_as_text

STRATEGIES = ["zero_shot", "few_shot", "cot"]

SYSTEM = (
    "Você é um sistema de extração de informações de documentos de transporte rodoviário "
    "de cargas no Brasil. Responda SOMENTE no formato solicitado. Nunca invente valores: "
    "se um campo não estiver no documento, use null."
)

_RULES = (
    "Regras:\n"
    "1. Use exatamente as chaves listadas, na mesma ordem.\n"
    "2. Números como números JSON (ponto decimal, sem 'R$', sem separador de milhar).\n"
    "3. Datas em ISO (AAAA-MM-DD). CNPJ no formato 00.000.000/0000-00.\n"
    "4. Corrija confusões óbvias de OCR (letra O no lugar do dígito 0).\n"
    "5. Se o peso estiver em toneladas, converta para kg.\n"
    "6. Campo ausente no documento => null (não deduza)."
)


def _task_block(doc_text: str) -> str:
    return (
        f"Extraia os campos abaixo do documento.\n\nCampos:\n{schema_as_text()}\n\n{_RULES}\n\n"
        f"=== DOCUMENTO ===\n{doc_text}\n=== FIM ==="
    )


@lru_cache(maxsize=1)
def _fewshot_examples() -> list[tuple[str, dict]]:
    """Um exemplo por estilo, de um conjunto SEPARADO (seed 999) do usado na avaliação."""
    pool = generate(60, seed=999)
    chosen: dict[str, tuple[str, dict]] = {}
    for d in pool:
        # prefere exemplos com placa informada, e um sem placa para ensinar o null
        chosen.setdefault(d.style, (d.text, d.truth))
    no_plate = next(d for d in pool if d.truth["placa_veiculo"] is None and d.style == "email")
    examples = list(chosen.values()) + [(no_plate.text, no_plate.truth)]
    return examples


def zero_shot(doc_text: str) -> tuple[str, str]:
    user = _task_block(doc_text) + "\n\nResponda apenas com um objeto JSON válido."
    return SYSTEM, user


def few_shot(doc_text: str) -> tuple[str, str]:
    shots = []
    for i, (text, truth) in enumerate(_fewshot_examples(), 1):
        ordered = {k: truth[k] for k in FIELDS}
        shots.append(
            f"### Exemplo {i}\n=== DOCUMENTO ===\n{text}\n=== FIM ===\n"
            f"Resposta:\n{json.dumps(ordered, ensure_ascii=False)}"
        )
    user = (
        "Veja exemplos resolvidos e depois extraia o documento final.\n\n"
        + "\n\n".join(shots)
        + "\n\n### Tarefa\n"
        + _task_block(doc_text)
        + "\n\nResponda apenas com um objeto JSON válido."
    )
    return SYSTEM, user


def cot(doc_text: str) -> tuple[str, str]:
    user = (
        _task_block(doc_text)
        + "\n\nPrimeiro raciocine passo a passo dentro de <analise>...</analise>: para cada campo, "
        "cite o trecho do documento que o sustenta, aponte conversões (toneladas->kg, data por "
        "extenso->ISO, correção de OCR) e indique explicitamente os campos ausentes. "
        "Depois responda o objeto JSON final dentro de <json>...</json>."
    )
    return SYSTEM, user


BUILDERS = {"zero_shot": zero_shot, "few_shot": few_shot, "cot": cot}


def build(strategy: str, doc_text: str) -> tuple[str, str]:
    try:
        return BUILDERS[strategy](doc_text)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Estratégia desconhecida: {strategy}. Use {STRATEGIES}") from exc
