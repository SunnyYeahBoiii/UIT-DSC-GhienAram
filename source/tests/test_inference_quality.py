from __future__ import annotations

from pathlib import Path

import pytest

from legalqa.chunking import chunk_documents
from legalqa.data import Chunk, Document, QARecord, read_json
from legalqa.generation import Candidate, make_qwen_agentic_candidate
from legalqa.models import CrossEncoderReranker, GenerationResult, TandemRepeatStoppingCriteria
from legalqa.pipeline import generate_from_evidence
from legalqa.prompts import render_evidence_with_manifest
from legalqa.quality import (
    analyze_answer_quality,
    question_requirements,
    relational_numeric_conflicts,
    sanitize_answer,
    tandem_repeat,
)
from legalqa.retrieval import Evidence
from legalqa.verifier import score_candidates


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {"artifacts": str(tmp_path), "models": {}},
        "runtime": {},
        "corpus": {
            "max_chunk_chars": 600,
            "overlap_chars": 60,
            "min_chunk_chars": 10,
        },
        "retrieval": {},
        "routing": {},
        "generation": {
            "min_answer_words": 5,
            "target_answer_words": 40,
            "max_answer_words": 120,
            "style_floor_words": 5,
            "style_ceiling_words": 120,
            "max_new_tokens": 120,
            "simple_max_new_tokens": 120,
            "complex_max_new_tokens": 120,
            "tokens_per_vietnamese_word": 1.5,
        },
        "selection": {
            "evidence_support": 0.1,
            "query_coverage": 0.1,
            "legal_citations": 0.1,
            "numeric_faithfulness": 0.1,
            "legal_semantic": 0.0,
            "length_prior": 0.0,
            "fluency": 0.0,
            "unsupported_penalty": 0.0,
            "duplicate_penalty": 0.0,
            "fragment_penalty": 0.0,
            "truncation_penalty": 0.0,
            "retrieval_risk_penalty": 0.0,
            "intent_coverage": 0.1,
            "scope_coverage": 0.1,
            "document_citation_penalty": 0.6,
            "numeric_relation_penalty": 0.4,
            "degenerate_generation_penalty": 0.8,
            "dangling_ending_penalty": 0.2,
            "repair_minimum_quality_gain": 0.04,
            "repair_non_improvement_penalty": 1.0,
            "source_priors": {"qwen3.5-2b": 0.0, "qwen3.5-2b-repair": 0.0},
        },
    }


def _evidence(text: str, title: str = "Nghị định 06/2023/NĐ-CP") -> list[Evidence]:
    return [
        Evidence(
            Chunk("doc::s0001::w000", "doc", text, title, "Điều 2"),
            score=0.8,
            reranker_score=0.9,
        )
    ]


def test_sanitize_removes_caption_then_question_echo_and_source_slug() -> None:
    question = "Mức phạt tiền tối đa là bao nhiêu?"
    answer = (
        "Theo Điều 2, mức phạt tối đa là 50 triệu đồng. "
        "nghi-dinh-06-2023-ND-CP Mức phạt tiền tối đa là bao nhiêu? (Hình từ Internet)"
    )
    cleaned, report = sanitize_answer(question, answer)
    assert "Hình từ Internet" not in cleaned
    assert "nghi-dinh-06" not in cleaned
    assert not cleaned.endswith(question)
    assert report == {
        "caption_removed": 1,
        "source_slug_removed": 1,
        "edge_question_echo_removed": 1,
        "sanitization_applied": True,
    }
    kept_url, url_report = sanitize_answer(
        "Nộp hồ sơ ở đâu?",
        "Hồ sơ được nộp tại https://digital.business.gov.vn.",
    )
    assert "https://digital.business.gov.vn" in kept_url
    assert url_report["source_slug_removed"] == 0


def test_quality_detects_tandem_loop_and_multi_part_requirements() -> None:
    question = "Tờ khai là gì? Có mấy phương thức đăng ký?"
    loop = "quyền sử dụng tài nguyên thiên nhiên quyền sử dụng lao động " * 8
    repeat = tandem_repeat(loop)
    assert repeat["repetitions"] >= 3
    assert set(question_requirements(question)) == {"count_or_amount", "definition"}
    quality = analyze_answer_quality(
        question,
        f"Tờ khai là chứng từ. {loop}",
        "Tờ khai là chứng từ và có hai phương thức đăng ký.",
        {"truncated": True},
    )
    assert quality["degenerate_repeat"] is True
    assert "degenerate_repeat" in quality["repair_reasons"]


def test_numeric_relation_detects_lost_maximum() -> None:
    conflicts = relational_numeric_conflicts(
        "Mức cho vay là 50 triệu đồng.",
        "Mức cho vay không vượt quá 50 triệu đồng đối với mỗi hộ.",
    )
    assert conflicts and conflicts[0]["number"] == "50triệu"


def test_exact_document_number_mismatch_receives_large_penalty(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = _evidence("Điều 2 Nghị định 06/2023/NĐ-CP quy định không hạn chế số lần đăng ký.")
    supported = Candidate(
        "qwen3.5-2b",
        "Theo Điều 2 Nghị định 06/2023/NĐ-CP, không hạn chế số lần đăng ký.",
    )
    unsupported = Candidate(
        "qwen3.5-2b-repair",
        "Theo Điều 2 Nghị định 1706/2023/NĐ-CP, không hạn chế số lần đăng ký.",
    )
    ranked = score_candidates("Có giới hạn số lần đăng ký không?", evidence, [supported, unsupported], config)
    assert ranked[0].answer == supported.answer
    assert unsupported.details["document_citation_precision"] == 0.0


def test_render_evidence_uses_canonical_number_and_clause_boundary() -> None:
    evidence = _evidence(
        "Điều 2. Quy định thứ nhất có hiệu lực. Quy định thứ hai rất dài và không được cắt giữa từ."
    )
    rendered, manifest = render_evidence_with_manifest(evidence, 105)
    assert "số 06/2023/ND-CP" in rendered
    assert "văn bản doc" not in rendered
    assert manifest[0]["clipped"] is True
    assert not rendered.endswith("…")


def test_chunking_recognizes_technical_roman_and_numbered_sections(tmp_path: Path) -> None:
    document = Document(
        "270063",
        "II. CHỈ ĐỊNH\nÁp dụng khi có ổ mủ.\nIV. CHUẨN BỊ\n1. Người thực hiện\nBác sĩ chuyên khoa.",
        "Quyết định 3906/QĐ-BYT",
    )
    headings = [chunk.heading for chunk in chunk_documents([document], _config(tmp_path))]
    assert any("II. CHỈ ĐỊNH" in heading for heading in headings)
    assert any("1. Người thực hiện" in heading for heading in headings)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        result = []
        for token in text.split():
            self.tokens.append(token)
            result.append(len(self.tokens) - 1)
        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.tokens[index] for index in ids)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3


def test_query_aware_reranker_windows_find_late_relevant_section() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.tokenizer = _FakeTokenizer()
    passage = " ".join(["nội_dung_khác"] * 90 + ["học_sinh", "gian_lận", "cuộc_thi"] * 8)
    windows = reranker._query_aware_windows(
        "học_sinh gian_lận",
        passage,
        max_length=72,
        stride=12,
        maximum=2,
    )
    assert len(windows) == 2
    assert any("học_sinh" in window and "gian_lận" in window for window in windows)


def test_generation_stopper_only_fires_on_three_adjacent_token_cycles() -> None:
    torch = pytest.importorskip("torch")
    stopper = TandemRepeatStoppingCriteria(torch, prompt_length=2, minimum=8, maximum=40)
    block = list(range(10, 18))
    repeated = torch.tensor([[1, 2, *block, *block, *block]])
    assert bool(stopper(repeated, None)[0]) is True
    distinct = torch.tensor([[1, 2, *range(24)]])
    fresh = TandemRepeatStoppingCriteria(torch, prompt_length=2, minimum=8, maximum=40)
    assert bool(fresh(distinct, None)[0]) is False


def test_pipeline_repairs_only_flagged_qwen_and_selects_quality_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config["paths"]["models"]["generator"] = str(tmp_path / "generator")
    config["generation"].update(
        {
            "qwen_enabled": True,
            "extractive_enabled": False,
            "agentic_enabled": False,
            "selective_repair_enabled": True,
            "repair_max_questions": 10,
            "repair_max_new_tokens": 120,
            "compact_generated_answers": True,
            "max_context_chars": 3000,
            "max_context_tokens": 0,
        }
    )
    question = QARecord("1", "Tờ khai là gì? Có mấy phương thức đăng ký?")
    evidence = _evidence(
        "Tờ khai là chứng từ hải quan. Có hai phương thức đăng ký tờ khai."
    )

    class FakeGenerator:
        def __init__(self, _config: dict) -> None:
            self.processor = None

        def generate_with_metadata(self, _system: str, prompt: str, **_kwargs) -> GenerationResult:
            if "ĐÁP ÁN NHÁP CẦN SỬA" in prompt:
                return GenerationResult(
                    "Tờ khai là chứng từ hải quan và có hai phương thức đăng ký.",
                    18,
                    True,
                    False,
                )
            loop = "quyền sử dụng tài nguyên thiên nhiên quyền sử dụng lao động " * 8
            return GenerationResult(f"Tờ khai là chứng từ. {loop}", 120, False, True)

    monkeypatch.setattr("legalqa.pipeline.QwenTextGenerator", FakeGenerator)
    result = generate_from_evidence(
        config,
        [question],
        {"1": evidence},
        tmp_path / "predictions.json",
    )
    predictions = read_json(result["predictions"])
    assert predictions["1"]["answer"].endswith("hai phương thức đăng ký.")
    assert result["repair"] == {"eligible": 1, "generated_or_cached": 1, "failed": 0}


def test_gated_agentic_can_force_one_real_search_before_answer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["generation"].update(
        {
            "max_context_chars": 3000,
            "max_context_tokens": 0,
            "agentic_force_first_search": True,
            "agentic_max_tool_calls": 2,
            "agentic_evidence_per_call": 2,
            "agentic_max_query_chars": 240,
            "compact_generated_answers": True,
        }
    )
    calls: list[str] = []
    extra = Evidence(
        Chunk(
            "extra::s1::w0",
            "extra",
            "Có hai phương thức đăng ký tờ khai.",
            "Thông tư 01/2020/TT-BTC",
            "Điều 3",
        ),
        score=0.9,
    )

    class FinalGenerator:
        processor = None

        def generate_with_metadata(self, _system: str, _prompt: str, **_kwargs) -> GenerationResult:
            return GenerationResult(
                "Tờ khai là chứng từ và có hai phương thức đăng ký.", 20, True, False
            )

    def search(query: str, *_args) -> list[Evidence]:
        calls.append(query)
        return [extra]

    candidate = make_qwen_agentic_candidate(
        FinalGenerator(),
        "Tờ khai là gì? Có mấy phương thức đăng ký?",
        _evidence("Tờ khai là chứng từ hải quan."),
        config,
        search,
        {"route": {"route": "complex", "confidence": 0.2}},
    )
    assert len(calls) == 1
    assert candidate.details["tool_calls"][0]["forced_by_gate"] is True
    assert candidate.details["evidence_added_total"] == 1
