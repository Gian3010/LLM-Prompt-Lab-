"""Gera os notebooks (.ipynb) a partir de células declaradas aqui.

Manter os notebooks como código-fonte facilita revisão em PR (diffs legíveis).
Uso: python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys, os, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


NB1 = [
    ("md", "# 01 · Dataset sintético de documentos de transporte\n\n"
           "**Cenário (fictício):** a *TransLog Brasil* recebe milhares de CT-e/DACTE por mês em três "
           "formatos: PDF estruturado, e-mails de operação em texto livre e digitalizações com OCR ruidoso. "
           "Este notebook explora o dataset **sintético** (gerado por `src/promptlab/dataset.py`, "
           "determinístico por seed) usado em todos os experimentos."),
    ("code", SETUP),
    ("code", "from promptlab.dataset import generate\nfrom promptlab.schema import FIELDS\n"
             "docs = generate(150, seed=42)\n"
             "df = pd.DataFrame([{**d.truth, 'style': d.style, 'doc_id': d.doc_id} for d in docs])\n"
             "df.groupby('style').size().rename('documentos').to_frame()"),
    ("md", "## Um exemplo de cada estilo\nA dificuldade cresce de `estruturado` → `ocr` → `email` (texto livre, "
           "datas por extenso, toneladas, placa ausente)."),
    ("code", "for style in ['estruturado', 'ocr', 'email']:\n"
             "    d = next(x for x in docs if x.style == style)\n"
             "    print(f'===== {style.upper()} =====\\n{d.text}\\n')"),
    ("md", "## Distribuição dos campos numéricos e campos ausentes\n"
           "~30% dos documentos **não informam a placa**: o gabarito é `null` e um modelo que inventa "
           "uma placa é contabilizado como *alucinação*."),
    ("code", "fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))\n"
             "df['peso_kg'].plot.hist(bins=20, ax=axes[0], color='#4C78A8'); axes[0].set_title('peso_kg')\n"
             "df['valor_frete'].plot.hist(bins=20, ax=axes[1], color='#F58518'); axes[1].set_title('valor_frete (R$)')\n"
             "df['placa_veiculo'].isna().map({True: 'sem placa', False: 'com placa'}).value_counts().plot.bar(ax=axes[2], color='#54A24B', rot=0)\n"
             "axes[2].set_title('placa informada?')\nplt.tight_layout(); plt.show()"),
    ("md", "## Por que um baseline regex?\nAntes de gastar tokens, medimos o *piso*: regex por rótulo. "
           "Se o regex já resolvesse tudo, LLM seria desperdício."),
    ("code", "from promptlab.baseline import extract\nfrom promptlab.evaluate import score_document\n"
             "rows = []\nfor d in docs:\n    s = score_document(d.truth, extract(d.text))\n"
             "    rows.append({'style': d.style, 'acc': sum(v == 'correct' for v in s.values()) / len(s)})\n"
             "pd.DataFrame(rows).groupby('style')['acc'].mean().round(3).to_frame('acurácia por campo (regex)')"),
]

NB2 = [
    ("md", "# 02 · Experimentos: zero-shot × few-shot × chain-of-thought\n\n"
           "Compara **estratégias de prompt** e **modelos** na mesma tarefa de extração estruturada. "
           "O notebook executa automaticamente os provedores cujas credenciais estão no ambiente "
           "(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); sem chaves, roda só o baseline. "
           "Resultados ficam em cache em `results/runs.jsonl`, então reexecutar é barato."),
    ("code", SETUP),
    ("code", "from promptlab.prompts import build, STRATEGIES\n"
             "sample = 'Numero do documento: 000123\\nPeso bruto: 1.200,5 kg'\n"
             "for s in STRATEGIES:\n    system, user = build(s, sample)\n"
             "    print(f'##### {s}  ({len(user)} caracteres de prompt)')\n"
             "    print(user[-700:] if s != 'few_shot' else user[:900] + '\\n[...]')\n    print()"),
    ("md", "## Configuração da bateria\nDefina os modelos por variável de ambiente, p.ex.:\n"
           "`LAB_MODELS=\"anthropic:<modelo> openai:gpt-4o-mini\"`. Tamanho da amostra em `LAB_N`."),
    ("code", "from promptlab.dataset import load_jsonl, generate, save_jsonl\n"
             "from promptlab.runner import run_experiments\n"
             "data_path = ROOT / 'data' / 'eval_docs.jsonl'\n"
             "if not data_path.exists():\n    save_jsonl(generate(150, seed=42), data_path)\n"
             "N = int(os.getenv('LAB_N', '150'))\n"
             "docs = load_jsonl(data_path)[:N]\n"
             "models = ['baseline'] + os.getenv('LAB_MODELS', '').split()\n"
             "print('modelos:', models, '| documentos:', len(docs))"),
    ("code", "rows = run_experiments(docs, models, STRATEGIES, ROOT / 'results' / 'runs.jsonl', workers=4)\n"
             "print(f'{len(rows)} execuções registradas')"),
    ("code", "from promptlab.evaluate import results_to_frame, summarize\n"
             "keep = {d.doc_id for d in docs}\n"
             "df = results_to_frame([r for r in rows if r['doc_id'] in keep])\n"
             "summarize(df).round(3)"),
]

NB3 = [
    ("md", "# 03 · Análise de resultados\n\nOnde cada abordagem acerta, onde erra e quanto custa. "
           "Todas as figuras são geradas a partir de `results/runs.jsonl`."),
    ("code", SETUP),
    ("code", "from promptlab.runner import load_results\nfrom promptlab.evaluate import results_to_frame, summarize\n"
             "rows = load_results(ROOT / 'results' / 'runs.jsonl')\n"
             "df = results_to_frame(rows)\n"
             "df['config'] = df['model'] + ' · ' + df['strategy']\n"
             "print(df['config'].unique())"),
    ("md", "## Acurácia por campo\nCampos como `peso_kg` (toneladas) e `placa_veiculo` (ausente → `null`) "
           "separam modelos que *entendem* o documento de modelos que só copiam padrões."),
    ("code", "pivot = df.pivot_table(index='field', columns='config', values='correct', aggfunc='mean')\n"
             "ax = pivot.plot.barh(figsize=(9, 5.5), width=0.8)\nax.set_xlim(0, 1); ax.set_xlabel('acurácia'); ax.invert_yaxis()\n"
             "ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left'); plt.tight_layout(); plt.show()"),
    ("md", "## Acurácia por estilo de documento"),
    ("code", "style = df.pivot_table(index='config', columns='style', values='correct', aggfunc='mean')[['estruturado', 'ocr', 'email']]\n"
             "style.round(3).style.background_gradient(cmap='Greens', vmin=0, vmax=1)"),
    ("md", "## Taxonomia de erros\nDistingue **omissão** (campo ficou nulo), **erro** (valor errado) e "
           "**alucinação** (inventou valor onde o gabarito é nulo). Cada tipo pede uma correção de prompt diferente."),
    ("code", "err = df[df.status != 'correct'].groupby(['config', 'status']).size().unstack(fill_value=0)\n"
             "err.plot.bar(stacked=True, figsize=(9, 4), rot=20, color=['#E45756', '#F58518', '#4C78A8'][:err.shape[1]])\n"
             "plt.ylabel('campos incorretos'); plt.tight_layout(); plt.show()"),
    ("md", "## Custo × qualidade\nSó há custo para modelos com preço configurado (`PRICING_JSON`). "
           "O ponto ideal é o canto superior esquerdo: alta acurácia, baixo custo por 1.000 documentos."),
    ("code", "s = summarize(df)\ns['config'] = s['model'] + ' · ' + s['strategy']\n"
             "paid = s.dropna(subset=['cost_per_1k_docs_usd'])\npaid = paid[paid.cost_per_1k_docs_usd > 0]\n"
             "if paid.empty:\n    print('Sem execuções pagas com preço configurado ainda — rode o notebook 02 com LAB_MODELS e PRICING_JSON.')\n"
             "else:\n    fig, ax = plt.subplots(figsize=(7, 4.5))\n    ax.scatter(paid.cost_per_1k_docs_usd, paid.exact_match, s=80)\n"
             "    for _, r in paid.iterrows():\n        ax.annotate(r.config, (r.cost_per_1k_docs_usd, r.exact_match), fontsize=8, xytext=(5, 5), textcoords='offset points')\n"
             "    ax.set_xlabel('custo por 1.000 docs (USD)'); ax.set_ylabel('exact match'); ax.set_ylim(0, 1.02); plt.show()"),
    ("md", "## Conclusões (a preencher com seus resultados)\n"
           "1. Qual estratégia de prompt trouxe maior ganho por modelo?\n"
           "2. O CoT compensa o custo extra de tokens de saída?\n"
           "3. Em qual estilo de documento o LLM mais supera o baseline?\n"
           "4. Qual a taxa de alucinação em `placa_veiculo`?"),
]

if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_dataset_exploration", NB1), ("02_prompt_experiments", NB2), ("03_results_analysis", NB3)]:
        nbf.write(nb(cells), out / f"{name}.ipynb")
        print("gerado", out / f"{name}.ipynb")
