"""Métricas de avaliação de extração estruturada.

Para cada documento e campo, a predição é classificada como:
- ``correct``: igual ao gabarito após normalização;
- ``missing``: gabarito tem valor e a predição é nula (omissão);
- ``wrong``: gabarito tem valor e a predição difere (erro);
- ``hallucination``: gabarito é ``null`` e a predição trouxe valor (invenção).
Quando ambos são nulos, é ``correct`` (acertar que o campo não existe também conta).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import pandas as pd

from .schema import FIELD_TYPES, FIELDS


def _strip(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().casefold()


def normalize(field: str, value: Any) -> Any:
    """Normaliza um valor para comparação justa."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"}):
        return None
    kind = FIELD_TYPES[field]
    try:
        if kind == "float":
            return round(float(str(value).replace(",", ".")) if isinstance(value, str) else float(value), 2)
        if kind == "int":
            return int(float(value))
    except (TypeError, ValueError):
        return ("__invalid__", str(value))
    if kind == "cnpj":
        return re.sub(r"\D", "", str(value))
    if field == "placa_veiculo":
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if field == "numero_documento":
        return str(value).strip().lstrip("0") or "0"
    if kind == "date":
        return str(value).strip()
    return _strip(str(value))


def classify_field(field: str, truth: Any, pred: Any) -> str:
    t, p = normalize(field, truth), normalize(field, pred)
    if t is None and p is None:
        return "correct"
    if t is None:
        return "hallucination"
    if p is None:
        return "missing"
    if FIELD_TYPES[field] == "float" and isinstance(p, float) and isinstance(t, float):
        return "correct" if abs(t - p) <= 0.01 else "wrong"
    return "correct" if t == p else "wrong"


def score_document(truth: dict, pred: Optional[dict]) -> dict[str, str]:
    """Classifica todos os campos de um documento. Predição ``None`` = JSON inválido."""
    pred = pred or {}
    return {f: classify_field(f, truth.get(f), pred.get(f)) for f in FIELDS}


def results_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Converte registros de execução (``results/*.jsonl``) em formato longo (um campo por linha).

    Cada registro de entrada precisa de: ``doc_id, style, model, strategy, truth, parsed``
    e, opcionalmente, ``latency_s, input_tokens, output_tokens, cost_usd``.
    """
    records = []
    for r in rows:
        scores = score_document(r["truth"], r.get("parsed"))
        for field, status in scores.items():
            records.append({
                "doc_id": r["doc_id"], "style": r["style"], "model": r["model"],
                "strategy": r["strategy"], "field": field, "status": status,
                "correct": status == "correct", "json_valid": r.get("parsed") is not None,
                "latency_s": r.get("latency_s"), "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"), "cost_usd": r.get("cost_usd"),
            })
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por (modelo, estratégia): acurácia por campo, exact match, JSON válido, custo, latência."""
    if df.empty:
        return pd.DataFrame()
    doc = df.groupby(["model", "strategy", "doc_id"]).agg(
        exact=("correct", "all"), json_valid=("json_valid", "first"),
        latency=("latency_s", "first"), cost=("cost_usd", "first"),
        halluc=("status", lambda s: (s == "hallucination").any()),
    ).reset_index()
    out = doc.groupby(["model", "strategy"]).agg(
        docs=("doc_id", "count"),
        exact_match=("exact", "mean"),
        json_valid=("json_valid", "mean"),
        hallucination_rate=("halluc", "mean"),
        latency_p50_s=("latency", "median"),
        cost_per_1k_docs_usd=("cost", lambda s: s.mean() * 1000 if s.notna().any() else float("nan")),
    )
    field_acc = df.groupby(["model", "strategy"])["correct"].mean().rename("field_accuracy")
    return out.join(field_acc).reset_index().sort_values(
        ["exact_match", "field_accuracy"], ascending=False
    ).reset_index(drop=True)
