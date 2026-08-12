from __future__ import annotations

from legalqa.zero_train import (
    KNNThresholds,
    capitalize_first_letter,
    close_truncated_answer,
    conservative_reselect,
    high_confidence_train_knn,
    remove_exact_sentence_duplicates,
)


def test_high_confidence_knn_accepts_exact_and_rejects_topic_neighbor() -> None:
    train = {
        "1": {"question": "Nguyên tắc trích khấu hao tài sản cố định?", "answer": "Đáp án chính thức."},
        "2": {"question": "Nguyên tắc tính thuế thu nhập doanh nghiệp?", "answer": "Sai chủ đề."},
    }
    public = {
        "9": {"question": "Nguyên tắc trích khấu hao tài sản cố định?"},
        "10": {"question": "Nguyên tắc tính thuế thu nhập cá nhân?"},
    }
    matches, report = high_confidence_train_knn(
        train,
        public,
        KNNThresholds(combined=.99, word=.99, character=.99, margin=.10),
    )
    assert matches["9"].train_id == "1"
    assert "10" not in matches
    assert report["accepted"] == 1


def test_postprocessing_is_conservative() -> None:
    capitalized, changed = capitalize_first_letter("người lao động có quyền.")
    assert changed is True
    assert capitalized == "Người lao động có quyền."

    deduplicated, removed = remove_exact_sentence_duplicates(
        "Người lao động có quyền hưởng chế độ theo quy định. "
        "Người lao động có quyền hưởng chế độ theo quy định."
    )
    assert removed == 1
    assert deduplicated.count("Người lao động") == 1

    closed, changed = close_truncated_answer(
        "Theo quy định, đây là một câu hoàn chỉnh có nhiều từ để bảo đảm độ dài. " * 5
        + "Phần cuối bị cắt giữa chừng và không có dấu"
    )
    assert changed is True
    assert closed.endswith(".")


def test_reselect_requires_all_safety_gates() -> None:
    trace = {
        "question_id": "1",
        "selected": "extractive",
        "candidates": [
            {
                "source": "extractive",
                "answer": "fragment",
                "score": .70,
                "details": {"fluency": .70},
            },
            {
                "source": "qwen3.5-2b",
                "answer": "Câu trả lời đủ căn cứ.",
                "score": .69,
                "details": {
                    "truncated": False,
                    "evidence_support": .95,
                    "unsupported": .05,
                    "citation_precision": 1.0,
                    "numeric_precision": 1.0,
                    "fluency": .90,
                },
            },
        ],
    }
    answer, source, details = conservative_reselect(trace)
    assert answer == "Câu trả lời đủ căn cứ."
    assert source == "qwen3.5-2b"
    assert details["changed"] is True
