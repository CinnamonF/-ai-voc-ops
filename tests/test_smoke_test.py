import io

from evals.live_smoke_test import load_cases, run_cases


def test_requested_smoke_cases_are_present():
    cases = load_cases()

    assert len(cases) == 8
    assert [case["expected_subcategory"] for case in cases] == [
        "배송완료 미수령",
        "중복 결제",
        "파손",
        "개인정보/보안",
        "오배송 상품",
        "환불 지연",
        "부분 환불",
        "배송 지연",
    ]


def test_smoke_runner_reports_pass_and_accuracy():
    cases = load_cases()
    predictions = {
        case["text"]: {
            "category": case["expected_category"],
            "subcategory": case["expected_subcategory"],
            "priority": case["expected_priority"],
            "sentiment": "negative",
            "requires_human_review": case["expected_human_review"] == "true",
            "reason": "목 출력 검증을 위한 모킹 결과입니다.",
        }
        for case in cases
    }
    output = io.StringIO()

    summary = run_cases(
        cases,
        classifier=lambda text: predictions[text],
        output=output,
    )

    assert summary.passed is True
    assert summary.category_hits == 8
    assert summary.subcategory_hits == 8
    assert output.getvalue().count("PASS |") == 8
    assert "Major-category accuracy: 8/8 (100.0%)" in output.getvalue()
    assert "Subcategory accuracy: 8/8 (100.0%)" in output.getvalue()


def test_smoke_runner_continues_after_one_case_failure():
    cases = load_cases()[:2]
    calls = []
    output = io.StringIO()

    def classifier(text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("provider detail must not be printed")
        return {
            "category": cases[1]["expected_category"],
            "subcategory": cases[1]["expected_subcategory"],
            "priority": "high",
            "sentiment": "negative",
            "requires_human_review": True,
            "reason": "두 번째 케이스 결과입니다.",
        }

    summary = run_cases(cases, classifier=classifier, output=output)

    assert len(calls) == 2
    assert summary.failed_cases == 1
    assert "FAIL |" in output.getvalue()
    assert "provider detail" not in output.getvalue()
