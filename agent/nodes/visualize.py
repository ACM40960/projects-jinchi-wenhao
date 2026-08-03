"""visualize 节点：将最终结果标注在原图上并保存。"""

import os

import config
from agent.state import WaldoState
from tools.visualize import visualize_result


def visualize_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, verified_result, candidates
    输出：result_image_path（标注图落盘路径；未找到 Waldo 则 None）

    策略：
    - 优先使用 verified_result
    - 若为 None（verify 全否决 / 跳过 verify 的单候选），取最佳候选的 bbox
    """
    bbox = state["verified_result"]
    if bbox is None:
        bbox = _best_candidate_bbox(state["candidates"])

    if bbox is None:
        print("[visualize] No candidate found. Waldo not located.")
        return {"result_image_path": None}

    base = os.path.splitext(os.path.basename(state["original_image_path"]))[0]
    output_path = os.path.join(config.results_dir(), f"{base}_result.jpg")

    saved = visualize_result(
        image_path=state["original_image_path"],
        bbox=bbox,
        output_path=output_path,
    )
    print(f"[visualize] Result saved → {saved}  bbox={bbox}")
    return {"result_image_path": saved}


def _best_candidate_bbox(candidates: list) -> list[int] | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c.get("verify_confidence") or c.get("confidence", 0.0))
    return best.get("orig_bbox") or best.get("patch_bbox")
