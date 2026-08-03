from typing import TypedDict


class WaldoState(TypedDict):
    # ── 输入 ──────────────────────────────────────────
    original_image_path: str        # 原图路径

    # ── detect / verify 输出 ─────────────────────────
    candidates: list                # [{patch_bbox, confidence, has_waldo, verified, ...}, ...]

    # ── 最终结果 ──────────────────────────────────────
    verified_result: list | None    # [x, y, w, h]（原图坐标），未找到则 None
    result_image_path: str | None   # visualize 落盘的标注图路径，未找到则 None


def initial_state(image_path: str) -> WaldoState:
    """构造初始 State。segment 节点会把整图切成 candidates。"""
    return WaldoState(
        original_image_path=image_path,
        candidates=[],
        verified_result=None,
        result_image_path=None,
    )
