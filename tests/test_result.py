"""summarize 测试：verified / detect-only / not-found 三条判定分支。"""

from agent.result import summarize


def test_verified_result_wins():
    state = {
        "candidates": [{"verify_confidence": 0.9, "orig_bbox": [1, 1, 2, 2]}],
        "verified_result": [10, 20, 30, 40],
        "result_image_path": "outputs/x_result.jpg",
    }
    assert summarize(state) == {
        "found": True,
        "bbox": [10, 20, 30, 40],
        "source": "verified",
        "result_image_path": "outputs/x_result.jpg",
    }


def test_single_candidate_without_verify_is_detect_only():
    """单候选路径跳过 verify（见 agent/pipeline.py），仍算找到了。"""
    state = {
        "candidates": [{"orig_bbox": [5, 6, 7, 8], "has_waldo": True}],
        "verified_result": None,
        "result_image_path": "outputs/x_result.jpg",
    }
    result = summarize(state)
    assert result["found"] is True
    assert result["bbox"] == [5, 6, 7, 8]
    assert result["source"] == "detect-only"


def test_detect_only_falls_back_to_patch_bbox():
    """detect 没给出 patch 内精确 bbox 时退化为整块 patch。"""
    state = {
        "candidates": [{"patch_bbox": [0, 0, 256, 256]}],
        "verified_result": None,
        "result_image_path": None,
    }
    assert summarize(state)["bbox"] == [0, 0, 256, 256]


def test_verify_rejected_everything_is_not_found():
    """verify 跑过但全否决：不能退回候选，那正是 verify 判掉的误检。"""
    state = {
        "candidates": [
            {"verify_confidence": 0.0, "orig_bbox": [1, 1, 2, 2]},
            {"verify_confidence": 0.0, "orig_bbox": [3, 3, 4, 4]},
        ],
        "verified_result": None,
        "result_image_path": None,
    }
    result = summarize(state)
    assert result["found"] is False
    assert result["bbox"] is None
    assert result["source"] is None


def test_no_candidates_is_not_found():
    state = {"candidates": [], "verified_result": None, "result_image_path": None}
    assert summarize(state)["found"] is False


def test_missing_keys_do_not_crash():
    assert summarize({})["found"] is False
