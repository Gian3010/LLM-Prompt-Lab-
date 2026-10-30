"""Executa experimentos e gera o relatório.

Exemplos:
  python scripts/run_experiments.py --models baseline
  python scripts/run_experiments.py --models baseline anthropic:<modelo> openai:gpt-4o-mini \
      --strategies zero_shot few_shot cot --limit 60
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptlab.dataset import load_jsonl  # noqa: E402
from promptlab.evaluate import results_to_frame, summarize  # noqa: E402
from promptlab.prompts import STRATEGIES  # noqa: E402
from promptlab.runner import run_experiments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_docs.jsonl")
    ap.add_argument("--models", nargs="+", default=["baseline"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/runs.jsonl")
    ap.add_argument("--report", default="results/summary.md")
    args = ap.parse_args()

    docs = load_jsonl(args.data)
    if args.limit:
        docs = docs[: args.limit]
    rows = run_experiments(docs, args.models, args.strategies, args.out, args.workers)
    keep = {d.doc_id for d in docs}
    rows = [r for r in rows if r["doc_id"] in keep]
    df = results_to_frame(rows)
    summary = summarize(df)
    pd_opts = {"index": False, "floatfmt": ".3f"}
    try:
        md = summary.to_markdown(**pd_opts)
    except ImportError:  # tabulate ausente: fallback simples
        md = summary.to_string(index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(md + "\n", encoding="utf-8")
    print(md)

    by_style = df.groupby(["model", "strategy", "style"])["correct"].mean().unstack("style")
    print("\nAcurácia por campo, por estilo de documento:")
    print(by_style.round(3).to_string())


if __name__ == "__main__":
    main()
