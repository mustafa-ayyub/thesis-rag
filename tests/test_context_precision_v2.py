import pytest

from src.evaluation.context_precision_v2 import (
    _extract_verdicts,
    weighted_average_precision,
)


def test_weighted_average_precision_all_zero():
    assert weighted_average_precision([0.0, 0.0, 0.0]) == 0.0


def test_weighted_average_precision_all_full():
    assert weighted_average_precision([1.0, 1.0, 1.0]) == 1.0


def test_weighted_average_precision_reward_early_hits():
    early = weighted_average_precision([1.0, 0.0, 0.0])
    late = weighted_average_precision([0.0, 0.0, 1.0])
    assert early > late


def test_weighted_average_precision_partial_credit():
    score = weighted_average_precision([0.5, 0.5])
    assert 0.0 < score < 1.0


def test_extract_verdicts_json_object():
    content = '{"verdicts": [1.0, 0.5, 0.0, 0.5, 1.0]}'
    assert _extract_verdicts(content, 5) == [1.0, 0.5, 0.0, 0.5, 1.0]


def test_extract_verdicts_json_array():
    content = "[1, 0.5, 0, 0.5, 1]"
    assert _extract_verdicts(content, 5) == [1.0, 0.5, 0.0, 0.5, 1.0]


def test_extract_verdicts_code_fence():
    content = '```json\n{"verdicts": [1.0, 0.5]}\n```'
    assert _extract_verdicts(content, 2) == [1.0, 0.5]


def test_extract_verdicts_snaps_near_values():
    content = '{"verdicts": [0.98, 0.52, 0.02]}'
    assert _extract_verdicts(content, 3) == [1.0, 0.5, 0.0]


def test_extract_verdicts_rejects_wrong_length():
    content = '{"verdicts": [1.0, 0.5]}'
    with pytest.raises(ValueError):
        _extract_verdicts(content, 3)


def test_extract_verdicts_rejects_out_of_range_value():
    content = '{"verdicts": [2.0, 0.5]}'
    with pytest.raises(ValueError):
        _extract_verdicts(content, 2)


def test_extract_verdicts_rejects_missing_content():
    with pytest.raises(ValueError):
        _extract_verdicts(None, 1)


def test_extract_verdicts_ignores_trailing_square_brackets():
    content = '{"verdicts": [1.0, 0.5, 0.0]} [note]'
    assert _extract_verdicts(content, 3) == [1.0, 0.5, 0.0]


def test_extract_verdicts_finds_expected_array_with_extra_brackets():
    content = "Result [1] is [1.0, 0.5, 0.0]. Extra [2] [3]"
    assert _extract_verdicts(content, 3) == [1.0, 0.5, 0.0]


def test_extract_verdicts_json_array_with_trailing_text():
    content = "[1.0, 0.5] done [note]"
    assert _extract_verdicts(content, 2) == [1.0, 0.5]


def test_extract_verdicts_rejects_text_without_json_array():
    with pytest.raises(ValueError):
        _extract_verdicts("no json array here", 2)
