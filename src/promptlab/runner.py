"""Executor de experimentos com cache em disco (retomável) e concorrência.

Cada linha de ``results/<run>.jsonl`` guarda: documento, gabarito, texto bruto, JSON parseado,
tokens, latência e custo. Reexecutar pula o que já foi calculado (chave = modelo+estratégia+doc),
o que permite estourar limites de rate sem perder o progresso.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import baseline
from .dataset import Doc
from .parsing import extract_json
from .prompts import build
from .providers import estimate_cost, make_provider


def _key(model: str, strategy: str, doc_id: str) -> str:
    return f"{model}|{strategy}|{doc_id}"


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_baseline(docs: list[Doc]) -> list[dict]:
    rows = []
    for d in docs:
        t0 = time.perf_counter()
        parsed = baseline.extract(d.text)
        rows.append({
            "doc_id": d.doc_id, "style": d.style, "model": "baseline:regex", "strategy": "regex",
            "truth": d.truth, "raw": json.dumps(parsed, ensure_ascii=False), "parsed": parsed,
            "latency_s": time.perf_counter() - t0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "error": None,
        })
    return rows


def _call(provider, model_spec: str, strategy: str, doc: Doc, retries: int = 3) -> dict:
    system, user = build(strategy, doc.text)
    last_err = None
    for attempt in range(retries):
        try:
            c = provider.complete(system, user)
            parsed = extract_json(c.text)
            return {
                "doc_id": doc.doc_id, "style": doc.style, "model": model_spec, "strategy": strategy,
                "truth": doc.truth, "raw": c.text, "parsed": parsed, "latency_s": c.latency_s,
                "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "cost_usd": estimate_cost(model_spec.partition(":")[2], c.input_tokens, c.output_tokens),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - registramos e tentamos novamente
            last_err = str(exc)
            time.sleep(2 ** attempt)
    return {
        "doc_id": doc.doc_id, "style": doc.style, "model": model_spec, "strategy": strategy,
        "truth": doc.truth, "raw": "", "parsed": None, "latency_s": None, "input_tokens": None,
        "output_tokens": None, "cost_usd": None, "error": last_err,
    }


def run_experiments(
    docs: list[Doc], models: list[str], strategies: list[str], out_path: str | Path,
    workers: int = 4,
) -> list[dict]:
    """Executa o produto cartesiano modelos x estratégias x documentos, com cache."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_results(out_path)
    done = {_key(r["model"], r["strategy"], r["doc_id"]) for r in existing if not r.get("error")}
    existing = [r for r in existing if not r.get("error")]  # erros antigos serão refeitos

    lock = threading.Lock()
    fh = out_path.open("w", encoding="utf-8")
    for r in existing:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_rows: list[dict] = []
    if "baseline" in models:
        for r in run_baseline(docs):
            if _key(r["model"], r["strategy"], r["doc_id"]) not in done:
                new_rows.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for spec in [m for m in models if m != "baseline"]:
        provider = make_provider(spec)
        jobs = [(strategy, d) for strategy in strategies for d in docs
                if _key(spec, strategy, d.doc_id) not in done]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_call, provider, spec, s, d) for s, d in jobs]
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    new_rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    fh.close()
    return existing + new_rows
