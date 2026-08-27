import evals.export_pilot_feedback as export_pilot_feedback
from evals.export_pilot_feedback import to_review_rows


def test_pilot_feedback_exports_as_provisional_review_queue():
    rows = to_review_rows(
        [
            {
                "feedback_id": "abc",
                "message_redacted": "배송 완료인데 못 받았어요",
                "is_correct": False,
                "corrected_category": "배송",
                "corrected_subcategory": "배송완료 미수령",
                "corrected_priority": "high",
                "corrected_sentiment": "negative",
                "corrected_human_review": True,
                "feedback_note": "미수령",
                "prompt_version": "v0.1",
                "taxonomy_version": "v0.1",
                "model": "gpt-test",
            }
        ]
    )
    assert rows[0]["ticket_id"] == "PILOT-abc"
    assert rows[0]["label_status"] == "provisional"
    assert rows[0]["source_type"] == "pilot_feedback"
    assert rows[0]["human_review_gold"] == "true"


def test_export_cli_reports_missing_configuration_without_traceback(monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    exit_code = export_pilot_feedback.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "SUPABASE_URL" in captured.err
    assert "Traceback" not in captured.err
