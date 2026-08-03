"""detect 节点：并发调用 VLM，对每个 patch 判断是否含 Waldo。"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from agent.state import WaldoState
from vision.image_utils import crop_to_pil, save_patch
from vision.segment import waldo_orig_bbox
from llm.vlm_client import get_vlm_client, BaseVLMClient, DetectResult

# ── 可调参数 ───────────────────────────────────────────────────────────
MAX_CONCURRENT = 10                  # 付费 Tier 1（~300 RPM）下提速：偶发 429 有指数退避兜底
MAX_PATCHES_PER_ITER = 80            # 每轮 patch 硬性上限，超出则随机截断
MIN_DETECT_PATCH_PX = 150            # 低于此尺寸的 patch 跳过（VLM 无法可靠识别）
# detect 用 Gemini：2026-06-15 全量复验（docs/工作日志.md）证明 gemini-3.5-flash 在
# 200px patch 上 present 二元信号最强（召回 94.4% / 误检 4.9% / 判别力 89.5），优于
# gpt-5.5（召回 88.9% / 误检 ~20%），且更快更省。⚠️ 其 confidence 彻底失效（与 present
# 矛盾率 77%、给负样本也打高分），故 detect 一律按 present(has_waldo) 过滤，绝不依赖
# confidence 排序；候选交由 verify（仍用 gpt-5.5）做精度兜底。
VLM_PROVIDER = "gemini"
VLM_MODEL = "gemini-3.5-flash"
# 旧配置（可切回作为备份）：VLM_PROVIDER="gpt4o" / VLM_MODEL="gpt-5.5"
# DETECT_MAX_TOKENS：Gemini 非推理模型不需要这么高，保留不影响（其 call 用 max_output_tokens）。
DETECT_MAX_TOKENS = 4096
# patch 落盘目录见 config.patches_dir()（本地 outputs/patches，Lambda 下 /tmp/outputs/patches）

# 限流重试
MAX_RETRIES = 4
RETRY_BASE_WAIT = 15                 # 首次等待秒数，指数退避：15→30→60→120


def detect_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, candidates（segment 节点生成，含 patch_bbox）
    输出：candidates（更新 confidence / has_waldo / waldo_bbox_in_patch / crop_path）

    流程：
    1. 把所有 patch 从原图裁剪并保存（串行，I/O 快）
    2. 并发调用 VLM（ThreadPoolExecutor）
    3. 合并结果，按 present(has_waldo) 过滤，confidence 降序仅作多候选时的稳定排序
    """
    vlm = get_vlm_client(VLM_PROVIDER, model=VLM_MODEL, max_tokens=DETECT_MAX_TOKENS)
    image_path = state["original_image_path"]
    patch_dir = config.patches_dir()

    # 1. 裁剪并保存所有 patch（超出上限时随机采样，避免系统性漏检右下角）
    candidates = list(state["candidates"])
    if len(candidates) > MAX_PATCHES_PER_ITER:
        print(f"[detect] Sampling {MAX_PATCHES_PER_ITER}/{len(candidates)} patches randomly")
        candidates = random.sample(candidates, MAX_PATCHES_PER_ITER)

    tasks: list[tuple[int, dict, str]] = []
    skipped = 0
    for i, cand in enumerate(candidates):
        pw, ph = cand["patch_bbox"][2], cand["patch_bbox"][3]
        if pw < MIN_DETECT_PATCH_PX or ph < MIN_DETECT_PATCH_PX:
            skipped += 1
            continue
        crop_path = os.path.join(patch_dir, f"patch{i}.jpg")
        patch_img = crop_to_pil(image_path, cand["patch_bbox"])
        save_patch(patch_img, crop_path)
        tasks.append((i, cand, crop_path))
    if skipped:
        print(f"[detect] Skipped {skipped} patches smaller than {MIN_DETECT_PATCH_PX}px")

    print(f"[detect] patches={len(tasks)}, workers={MAX_CONCURRENT}")

    # 2. 并发调用 VLM
    results: list[tuple[dict, str, DetectResult] | None] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        future_map = {
            pool.submit(_call_vlm, vlm, crop_path): (i, cand, crop_path)
            for i, cand, crop_path in tasks
        }
        for future in as_completed(future_map):
            i, cand, crop_path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[detect] patch {i} error: {exc}")
                result = DetectResult(has_waldo=False, confidence=0.0)
            results[i] = (cand, crop_path, result)

    # 3. 合并、过滤、排序
    updated = []
    for entry in results:
        if entry is None:
            continue
        cand, crop_path, result = entry
        wbox = _validated_bbox(result.bbox, cand["patch_bbox"])
        updated.append({
            **cand,
            "crop_path": crop_path,
            "confidence": result.confidence,
            "has_waldo": result.has_waldo,
            "waldo_bbox_in_patch": wbox,
            # 原图坐标：有精确子 bbox 则紧贴 Waldo，否则退化为整块 patch。
            # 单候选路径跳过 verify，visualize 直接用此 orig_bbox 画框。
            "orig_bbox": waldo_orig_bbox(cand["patch_bbox"], wbox),
        })

    before = len(updated)
    # Gemini confidence 失效（与 present 矛盾率 77%），只按 present(has_waldo) 二元信号过滤。
    # 不按 confidence 排序——那是假精度；保持行优先 patch 顺序，可解释且对 verify 无影响。
    updated = [c for c in updated if c["has_waldo"]]
    print(f"[detect] {len(updated)}/{before} patches with present=true")

    return {"candidates": updated}


# ── 内部工具 ───────────────────────────────────────────────────────────

def _call_vlm(vlm: BaseVLMClient, crop_path: str) -> DetectResult:
    """单次 VLM 调用，带指数退避重试（处理 429 限流）。"""
    for attempt in range(MAX_RETRIES):
        try:
            return vlm.detect(crop_path)
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                print(f"[detect] Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise


def _validated_bbox(bbox: list | None, patch_bbox: list[int]) -> list[int] | None:
    """将 VLM 返回的 patch 内 bbox 裁剪到合法范围，防止越界。"""
    if not bbox or len(bbox) != 4:
        return None
    _, _, pw, ph = patch_bbox
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, pw - 1))
    y = max(0, min(y, ph - 1))
    w = max(1, min(w, pw - x))
    h = max(1, min(h, ph - y))
    return [x, y, w, h]
