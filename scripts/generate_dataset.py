"""Gera o dataset sintético de avaliação.  Uso: python scripts/generate_dataset.py --n 150"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptlab.dataset import generate, save_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/eval_docs.jsonl")
    args = ap.parse_args()
    docs = generate(args.n, seed=args.seed)
    save_jsonl(docs, args.out)
    by_style = {s: sum(d.style == s for d in docs) for s in ("estruturado", "email", "ocr")}
    print(f"{len(docs)} documentos salvos em {args.out}  {by_style}")


if __name__ == "__main__":
    main()
