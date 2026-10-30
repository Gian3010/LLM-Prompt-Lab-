"""Schema alvo da extração: documento de transporte (estilo DACTE) de uma transportadora."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

# Ordem canônica dos campos. Tipos: str | cnpj | date | float | int
FIELD_TYPES: dict[str, str] = {
    "numero_documento": "str",
    "data_emissao": "date",
    "remetente_nome": "str",
    "remetente_cnpj": "cnpj",
    "destinatario_nome": "str",
    "destinatario_cnpj": "cnpj",
    "origem": "str",
    "destino": "str",
    "peso_kg": "float",
    "volumes": "int",
    "valor_mercadoria": "float",
    "valor_frete": "float",
    "placa_veiculo": "str",
}
FIELDS = list(FIELD_TYPES)

FIELD_DESCRIPTIONS = {
    "numero_documento": "número do documento, sem zeros à esquerda removidos (string de 6 dígitos)",
    "data_emissao": "data de emissão no formato ISO AAAA-MM-DD",
    "remetente_nome": "razão social do remetente",
    "remetente_cnpj": "CNPJ do remetente no formato 00.000.000/0000-00",
    "destinatario_nome": "razão social do destinatário",
    "destinatario_cnpj": "CNPJ do destinatário no formato 00.000.000/0000-00",
    "origem": "cidade de origem no formato 'Cidade/UF'",
    "destino": "cidade de destino no formato 'Cidade/UF'",
    "peso_kg": "peso bruto em quilogramas (número com ponto decimal; converta toneladas para kg)",
    "volumes": "quantidade de volumes (inteiro)",
    "valor_mercadoria": "valor da mercadoria em reais (número com ponto decimal, sem símbolo)",
    "valor_frete": "valor do frete em reais (número com ponto decimal, sem símbolo)",
    "placa_veiculo": "placa do veículo no formato AAA0A00 ou AAA-0000; null se não informada",
}


class LogisticsDoc(BaseModel):
    """Representação validada de uma extração (todos os campos opcionais)."""

    model_config = ConfigDict(extra="ignore")

    numero_documento: Optional[str] = None
    data_emissao: Optional[str] = None
    remetente_nome: Optional[str] = None
    remetente_cnpj: Optional[str] = None
    destinatario_nome: Optional[str] = None
    destinatario_cnpj: Optional[str] = None
    origem: Optional[str] = None
    destino: Optional[str] = None
    peso_kg: Optional[float] = None
    volumes: Optional[int] = None
    valor_mercadoria: Optional[float] = None
    valor_frete: Optional[float] = None
    placa_veiculo: Optional[str] = None


def schema_as_text() -> str:
    """Descrição do schema para ser embutida nos prompts."""
    lines = []
    for name in FIELDS:
        lines.append(f'- "{name}": {FIELD_DESCRIPTIONS[name]}')
    return "\n".join(lines)
