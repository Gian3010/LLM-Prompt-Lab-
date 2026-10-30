"""Camada de abstração de provedores de LLM.

Spec de modelo: ``"anthropic:<modelo>"`` ou ``"openai:<modelo>"``. Os SDKs são importados
sob demanda, então o laboratório roda offline (baseline) sem nenhuma chave configurada.

Preços (USD por 1M de tokens) são **configuráveis** via ``PRICING_JSON`` (caminho para um
JSON ``{"modelo": [preco_in, preco_out]}``). Os valores padrão abaixo são apenas um ponto
de partida e podem estar desatualizados: confira a página de preços do provedor.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> Completion: ...


DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def load_pricing() -> dict[str, tuple[float, float]]:
    pricing = dict(DEFAULT_PRICING)
    path = os.getenv("PRICING_JSON")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            pricing.update({k: tuple(v) for k, v in json.load(fh).items()})
    return pricing


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float | None:
    """Custo em USD ou ``None`` quando o preço do modelo não está configurado."""
    pricing = load_pricing()
    for key, (p_in, p_out) in pricing.items():
        if model.startswith(key):
            return in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    return None


class AnthropicProvider:
    def __init__(self, model: str):
        import anthropic  # import tardio

        self.name = f"anthropic:{model}"
        self.model = model
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> Completion:
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return Completion(text, resp.usage.input_tokens, resp.usage.output_tokens,
                          time.perf_counter() - t0)


class OpenAIProvider:
    def __init__(self, model: str):
        from openai import OpenAI  # import tardio

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI()

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> Completion:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = resp.usage
        return Completion(resp.choices[0].message.content or "", usage.prompt_tokens,
                          usage.completion_tokens, time.perf_counter() - t0)


def make_provider(spec: str) -> Provider:
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return AnthropicProvider(model)
    if kind == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"Provider inválido: {spec!r}. Use 'anthropic:<modelo>' ou 'openai:<modelo>'.")
