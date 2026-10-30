import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from promptlab import baseline  # noqa: E402
from promptlab.dataset import generate  # noqa: E402
from promptlab.evaluate import classify_field, normalize, results_to_frame, score_document, summarize  # noqa: E402
from promptlab.parsing import extract_json  # noqa: E402
from promptlab.prompts import STRATEGIES, build  # noqa: E402
from promptlab.runner import run_baseline  # noqa: E402
from promptlab.schema import FIELDS  # noqa: E402


def test_dataset_is_deterministic():
    a, b = generate(20, seed=7), generate(20, seed=7)
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.text for d in a] != [d.text for d in generate(20, seed=8)]


def test_dataset_has_all_styles_and_missing_plates():
    docs = generate(200, seed=1)
    assert {d.style for d in docs} == {"estruturado", "email", "ocr"}
    assert any(d.truth["placa_veiculo"] is None for d in docs)
    assert all(set(d.truth) == set(FIELDS) for d in docs)


def test_cnpj_check_digits_valid():
    def valid(c: str) -> bool:
        d = [int(x) for x in c if x.isdigit()]
        w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

        def dv(nums, w):
            r = sum(n * k for n, k in zip(nums, w)) % 11
            return 0 if r < 2 else 11 - r

        return dv(d[:12], w1) == d[12] and dv(d[:13], [6] + w1) == d[13]

    for doc in generate(30, seed=3):
        assert valid(doc.truth["remetente_cnpj"]) and valid(doc.truth["destinatario_cnpj"])


def test_baseline_perfect_on_structured_docs():
    for d in [x for x in generate(100, seed=5) if x.style == "estruturado"]:
        assert all(s == "correct" for s in score_document(d.truth, baseline.extract(d.text)).values())


def test_baseline_weak_on_free_text():
    emails = [x for x in generate(100, seed=5) if x.style == "email"]
    acc = [
        sum(s == "correct" for s in score_document(d.truth, baseline.extract(d.text)).values()) / len(FIELDS)
        for d in emails
    ]
    assert sum(acc) / len(acc) < 0.5


@pytest.mark.parametrize(
    "field,truth,pred,expected",
    [
        ("peso_kg", 1234.5, "1234.5", "correct"),
        ("peso_kg", 1234.5, 1.2345, "wrong"),
        ("remetente_cnpj", "12.345.678/0001-95", "12345678000195", "correct"),
        ("placa_veiculo", None, "ABC1D23", "hallucination"),
        ("placa_veiculo", None, None, "correct"),
        ("placa_veiculo", "ABC1D23", None, "missing"),
        ("placa_veiculo", "ABC-1234", "abc1234", "correct"),
        ("numero_documento", "004512", "4512", "correct"),
        ("remetente_nome", "Ipê Cosméticos Ltda", "IPE COSMETICOS LTDA", "correct"),
        ("volumes", 12, "12", "correct"),
    ],
)
def test_classify_field(field, truth, pred, expected):
    assert classify_field(field, truth, pred) == expected


def test_invalid_number_is_wrong_not_crash():
    assert classify_field("peso_kg", 10.0, "dez quilos") == "wrong"
    assert normalize("peso_kg", "null") is None


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('texto\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json("<analise>pensando {x}</analise><json>{\"a\": 3}</json>") == {"a": 3}
    assert extract_json("sem json aqui") is None
    assert extract_json("") is None


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_prompts_contain_document_and_schema(strategy):
    system, user = build(strategy, "TEXTO-UNICO-123")
    assert "TEXTO-UNICO-123" in user and "peso_kg" in user and "null" in system
    if strategy == "few_shot":
        assert user.count("### Exemplo") == 4
    if strategy == "cot":
        assert "<analise>" in user and "<json>" in user


def test_fewshot_examples_do_not_leak_into_eval_set():
    eval_texts = {d.text for d in generate(150, seed=42)}
    from promptlab.prompts import _fewshot_examples

    assert not eval_texts & {t for t, _ in _fewshot_examples()}


def test_summary_pipeline():
    docs = generate(30, seed=2)
    df = results_to_frame(run_baseline(docs))
    s = summarize(df)
    assert len(s) == 1 and 0 <= s.loc[0, "exact_match"] <= 1 and s.loc[0, "json_valid"] == 1.0
