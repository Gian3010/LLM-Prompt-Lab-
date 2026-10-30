# 🧪 LLM Prompt Lab — extração estruturada de documentos de transporte

Laboratório reprodutível para **comparar modelos e técnicas de prompt** (zero-shot, few-shot, chain-of-thought) em uma tarefa de negócio: transformar documentos de transporte não estruturados em JSON validado.

> **Aviso de transparência.** A empresa (*TransLog Brasil*), os documentos, CNPJs, valores e o volume operacional descritos abaixo são **fictícios**, criados para contextualizar o projeto. Os números do baseline regex na seção de resultados foram **medidos de fato** rodando o código deste repositório. Os números de LLMs dependem das suas chaves de API e **não estão pré-preenchidos**: rode e cole a sua tabela.

---

## 1. Situação-problema (cenário fictício)

A *TransLog Brasil*, transportadora rodoviária fictícia, recebe cerca de **12.000 documentos de transporte por mês** em três formatos:

| Estilo | Origem | Dificuldade |
|---|---|---|
| `estruturado` | DACTE em texto com campos rotulados | baixa |
| `ocr` | digitalizações com rótulos abreviados, caixa alta, confusão `O`/`0` | média |
| `email` | e-mails de operação em texto livre (datas por extenso, peso em toneladas, placa às vezes ausente) | alta |

Hoje a conferência é manual: um analista leva ~**3 minutos** por documento para digitar os 13 campos no ERP. Dois problemas: custo (≈ 600 h/mês, **estimativa do cenário**) e erros de digitação em CNPJ e valores.

**Pergunta de engenharia:** qual combinação *modelo × técnica de prompt* entrega a melhor relação acurácia/custo, e o quanto um LLM realmente supera um regex?

## 2. O que este repositório demonstra

- **Dataset sintético determinístico** (`seed`) com gabarito, 3 estilos de dificuldade e campos ausentes propositais (testa *alucinação*).
- **Três estratégias de prompt** (`src/promptlab/prompts.py`): zero-shot; few-shot com exemplos de um conjunto *separado* da avaliação (sem vazamento, há teste para isso); CoT com raciocínio em `<analise>` e payload em `<json>`.
- **Camada de provedores** (Anthropic e OpenAI) com import tardio, temperatura 0, retries com backoff, cache em disco retomável e concorrência.
- **Avaliação rigorosa** (`evaluate.py`): normalização por tipo (CNPJ só dígitos, números com tolerância, acentos/caixa em texto), e taxonomia de erro — `correct`, `missing` (omissão), `wrong`, `hallucination`.
- **Baseline regex** como piso de comparação.
- **3 notebooks** executados: exploração, experimentos, análise (acurácia por campo/estilo, taxonomia de erros, custo × qualidade).
- **22 testes** (`pytest`), incluindo validade dos dígitos verificadores de CNPJ e ausência de vazamento few-shot → avaliação.

## 3. Arquitetura

```
 dataset.py ──► eval_docs.jsonl (texto + gabarito)
                       │
 prompts.py ──► (system, user) por estratégia ─┐
                                               ▼
 providers.py (Anthropic | OpenAI) ◄── runner.py (cache + threads + retries)
                                               │
 parsing.py (JSON / ```json / <json>) ◄────────┘
                       │
 evaluate.py ──► status por campo ──► summary.md + notebooks
```

```
llm-prompt-lab/
├── src/promptlab/   schema · dataset · baseline · prompts · providers · parsing · runner · evaluate
├── scripts/         generate_dataset · run_experiments · build_notebooks
├── notebooks/       01_dataset_exploration · 02_prompt_experiments · 03_results_analysis
├── tests/           test_lab.py
├── data/            eval_docs.jsonl (gerado)
└── results/         runs.jsonl · summary.md
```

## 4. Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha as chaves que for usar

make test                       # 22 testes
make baseline                   # gera dataset e roda o baseline (offline, sem custo)

# Comparação com LLMs (use os IDs de modelo vigentes do seu provedor)
python scripts/run_experiments.py \
  --models baseline anthropic:<modelo> openai:gpt-4o-mini \
  --strategies zero_shot few_shot cot --limit 60

make notebooks                  # regenera e executa os notebooks
```

Dica: comece com `--limit 20` para validar chaves e custo antes da bateria completa. Reexecutar não repete chamadas já feitas.

## 5. Resultados medidos

### Baseline regex (executado, 150 documentos, seed 42)

| Estilo | Acurácia por campo |
|---|---|
| `estruturado` | **100,0%** |
| `ocr` | 3,0% |
| `email` | 1,8% |
| **Geral** | **34,3%** (exact match de documento: 32,7%) |

**Leitura:** o regex é perfeito quando o formato é rígido e inútil fora dele. Isso justifica o LLM *apenas* para os estilos `ocr` e `email`; um roteador que usa regex para documentos rotulados e LLM para o restante reduziria o custo de inferência (hipótese a validar com seus resultados).

### LLMs (preencha com a sua execução)

Rode `python scripts/run_experiments.py ...` e copie `results/summary.md`:

| model | strategy | exact_match | field_accuracy | json_valid | hallucination_rate | cost / 1k docs (USD) |
|---|---|---|---|---|---|---|
| _seu resultado_ | | | | | | |

## 6. Impacto estimado (simulação, não medição)

Estimativa **hipotética** do cenário fictício, assumindo (a) 12.000 docs/mês, (b) 3 min/doc manual, (c) revisão humana de 15% dos documentos (os de baixa confiança) a 1 min cada:

| | Manual | Com extração por LLM + revisão |
|---|---|---|
| Tempo/mês | 600 h | 12.000 × 15% × 1 min ≈ **30 h** |
| Redução | — | **≈ 95% (estimativa)** |

Troque as premissas pelas suas e a conta muda: o valor real depende da acurácia que você medir na seção 5.

## 7. Decisões de engenharia (para entrevista)

- **Por que baseline?** Sem ele, não dá para afirmar que o LLM agrega valor.
- **Por que `null` explícito?** Mede alucinação; em produção, inventar uma placa é pior que deixar vazia.
- **Por que few-shot de outro seed?** Evita vazamento e superestimativa da acurácia.
- **Por que separar raciocínio (`<analise>`) de payload (`<json>`)?** Parsing determinístico mesmo com CoT.
- **Limitações:** dataset sintético não captura toda a sujeira de documentos reais; o avaliador não testa PDFs/imagens (apenas texto); preços variam e são configuráveis.

## 8. Próximos passos

Saída estruturada nativa (tool use / JSON schema), roteador regex → LLM por confiança, avaliação com documentos reais anonimizados, e *LLM-as-judge* para campos livres.
