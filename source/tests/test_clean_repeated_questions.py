import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clean_repeated_questions",
    PROJECT / "scripts" / "clean_repeated_questions.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_removes_image_marker_download_cta_and_preceding_question() -> None:
    answer = (
        "Nội dung trả lời pháp lý đầy đủ.\n\n"
        "Bảo đảm thực hiện hợp đồng là gì? (Hình từ Internet)\n"
        "Tải trọn bộ các văn bản hiện hành: Tải về"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Bảo đảm thực hiện hợp đồng là gì?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Nội dung trả lời pháp lý đầy đủ."
    assert details["removed_image_markers"] == 1
    assert details["removed_trailing_question_block"].startswith("Bảo đảm")


def test_removes_short_question_repeated_as_final_paragraph() -> None:
    answer = "Công ty chứng khoán là doanh nghiệp được cấp phép.\n\nCông ty chứng khoán là gì?"
    status, cleaned, details = MODULE.classify_and_clean(
        "Công ty chứng khoán là gì?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Công ty chứng khoán là doanh nghiệp được cấp phép."
    assert details["removed_trailing_question_block"] == "Công ty chứng khoán là gì?"


def test_removes_caption_followed_by_download_cta() -> None:
    answer = (
        "Nội dung trả lời pháp lý.\n\n"
        "Xuất hóa đơn tạm ứng (Hình từ Internet)\n"
        "Tải trọn bộ các văn bản hiện hành: Tải về"
    )
    status, cleaned, _ = MODULE.classify_and_clean(
        "Có được xuất hóa đơn đối với khoản tạm ứng không?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Nội dung trả lời pháp lý."


def test_removes_unrelated_final_question_block() -> None:
    answer = (
        "Theo Điều 1, người lao động được hưởng chế độ theo quy định.\n\n"
        "Mức phạt giao thông mới nhất là bao nhiêu?"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Người lao động được hưởng chế độ nào?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Theo Điều 1, người lao động được hưởng chế độ theo quy định."
    assert details["removed_trailing_question_block"].startswith("Mức phạt")


def test_normalizes_unicode_and_conversion_markers() -> None:
    answer = (
        "Theo\u00a0Điều 1, nội dung pháp lý được quy định đầy đủ.\ufeff\u00ad\n"
        "Bậc 1 4.000.000 đồng++table"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Mức hưởng được quy định như thế nào?", answer, 6, 30
    )
    assert status == "clean"
    assert "\u00a0" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\u00ad" not in cleaned
    assert "++table" not in cleaned
    assert details["removed_table_markers"] == 1


def test_strips_cta_but_preserves_legal_step() -> None:
    answer = (
        "Bước 1: Người yêu cầu nộp tờ khai theo Mẫu số 01: TẢI VỀ\n"
        "Bước 2: Cơ quan có thẩm quyền tiếp nhận và giải quyết."
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Thủ tục được thực hiện như thế nào?", answer, 6, 30
    )
    assert status == "clean"
    assert "Bước 1: Người yêu cầu nộp tờ khai theo Mẫu số 01" in cleaned
    assert "TẢI VỀ" not in cleaned
    assert details["removed_download_ctas"] == 1


def test_removes_short_dangling_caption() -> None:
    answer = (
        "Theo Điều 1, đây là nội dung pháp lý đầy đủ gồm nhiều từ để vượt "
        "ngưỡng an toàn khi xóa caption ở cuối đáp án và vẫn bảo toàn câu trả lời.\n\n"
        "Bảo hiểm y tế"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Quy định về bảo hiểm y tế như thế nào?", answer, 6, 30
    )
    assert status == "clean"
    assert not cleaned.endswith("Bảo hiểm y tế")
    assert details["removed_dangling_title"] == "Bảo hiểm y tế"


def test_removes_misspelled_image_marker_and_caption() -> None:
    answer = (
        "Nội dung pháp luật đã được giải thích đầy đủ, rõ ràng.\n\n"
        "Giấy phép lao động (Hình từ Internnet)"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Giấy phép lao động được quy định thế nào?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Nội dung pháp luật đã được giải thích đầy đủ, rõ ràng."
    assert details["removed_image_markers"] == 1
    assert details["removed_image_caption"] == "Giấy phép lao động"


def test_removes_consecutive_duplicate_question_lines() -> None:
    repeated = "Thủ tục này được thực hiện cụ thể như thế nào theo quy định?"
    answer = f"Nội dung pháp lý đầy đủ.\n{repeated}\n{repeated}\nKết luận."
    status, cleaned, details = MODULE.classify_and_clean(
        "Quy định về thủ tục là gì?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned.count(repeated) == 1
    assert details["removed_duplicate_lines"] == 1


def test_removes_form_download_line_without_link_target() -> None:
    answer = (
        "Theo Điều 1, hồ sơ gồm đơn đề nghị và giấy tờ chứng minh.\n"
        "Tải mẫu đơn đề nghị mới nhất tại đây:"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Hồ sơ gồm những giấy tờ nào?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Theo Điều 1, hồ sơ gồm đơn đề nghị và giấy tờ chứng minh."
    assert details["removed_download_ctas"] == 1


def test_removes_terminal_customer_service_boilerplate() -> None:
    answer = (
        "Theo Điều 1, mức phạt tiền là 10.000.000 đồng.\n"
        "Trên đây là những thông tin liên quan mà chúng tôi cung cấp tới bạn. "
        "Trân trọng!"
    )
    status, cleaned, details = MODULE.classify_and_clean(
        "Mức phạt tiền là bao nhiêu?", answer, 6, 30
    )
    assert status == "clean"
    assert cleaned == "Theo Điều 1, mức phạt tiền là 10.000.000 đồng."
    assert details["removed_service_boilerplate"] == 1
