from dataclasses import dataclass


@dataclass
class _Row:
    method: str
    config_note: str = ""


def test_select_summary_row_supports_legacy_alias_for_llada21_speed_mode():
    from aoae.experiment_utils import select_summary_row

    row = select_summary_row([_Row(method="llada21_speed_mode")], "block_smode")

    assert row.method == "llada21_speed_mode"


def test_select_summary_row_supports_modern_alias_for_legacy_quality_mode():
    from aoae.experiment_utils import select_summary_row

    row = select_summary_row([_Row(method="confidence_q_mode")], "llada21_quality_mode")

    assert row.method == "confidence_q_mode"
