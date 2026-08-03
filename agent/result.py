"""把最终 WaldoState 压成对外结果，CLI 与 Lambda handler 共用同一份判定逻辑。"""

from agent.state import WaldoState


def summarize(state: WaldoState) -> dict:
    """从 state 提炼 {found, bbox, source, result_image_path}。

    `source` 的取值：
    - `"verified"`  —— verify 跑过并选中了某个候选。
    - `"detect-only"` —— 只有单候选，按路由跳过了 verify（见 agent/pipeline.py）。
      仍是有信心的结果：detect 的 present 二元信号精度高。
    - `None`        —— 没找到（无候选，或 verify 把候选全否决了）。
    """
    candidates = state.get("candidates") or []
    verify_ran = any("verify_confidence" in c for c in candidates)
    bbox = state.get("verified_result")

    if bbox:
        source = "verified"
    elif candidates and not verify_ran:
        best = candidates[0]
        bbox = best.get("orig_bbox") or best.get("patch_bbox")
        source = "detect-only"
    else:
        bbox = None
        source = None

    return {
        "found": bbox is not None,
        "bbox": list(bbox) if bbox else None,
        "source": source,
        "result_image_path": state.get("result_image_path"),
    }
