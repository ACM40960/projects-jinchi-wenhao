"""verify 节点：把全部 present 候选裁剪图一次性发给 Gemini，横向单选唯一真 Waldo。

为什么是「横向单选」而非「逐张确认」：实测在密集难图（如 2.jpg）上，逐张独立判断会
把多张候选都判 Yes、且易被红白条纹误导，再靠 confidence 排序又挑错；把所有候选摆一起
强制相对比较，Gemini 多模态能更干净地指出唯一的真 Waldo。
"""

import os

from PIL import Image

import config
from agent.state import WaldoState
from vision.image_utils import crop_to_pil, save_patch
from vision.segment import expand_bbox, waldo_orig_bbox
from llm.vlm_client import get_vlm_client

# ── 可调参数 ───────────────────────────────────────────────────────────
# 把全部 present 候选（detect 已按 has_waldo 过滤）一并送横向单选；
# VERIFY_MAX 仅作安全上限，防极端误检爆量；正常图候选数远低于它。
VERIFY_MAX = 12             # 送验证的候选数硬上限（安全阀）
PADDING_RATIO = 0.3         # 向外扩展 bbox 的比例（相对 bbox 宽/高）
MIN_VERIFY_SIZE = 120       # 发给 VLM 的 verify 图最小边长（像素）
VLM_PROVIDER = "gemini"
VLM_MODEL = "gemini-3.5-flash"
SELECT_MAX_TOKENS = 1024    # 横向单选输出含 per_image 数组，留足预算
# 特写裁剪落盘目录见 config.verify_dir()（本地 outputs/verify，Lambda 下 /tmp/outputs/verify）


def verify_node(state: WaldoState) -> dict:
    """
    输入：original_image_path, candidates（detect 已按 present=has_waldo 过滤）
    输出：candidates（更新 verified / verify_confidence / orig_bbox / verify_crop_path）
          verified_result（Gemini 横向单选选中候选的紧框，都不是则 None）

    流程：
    1. 对全部 present 候选（上限 VERIFY_MAX）裁出带 padding 的原图区域
    2. 把这些裁剪图一次性发给 Gemini 横向单选，返回唯一真 Waldo 的索引
    3. 选中索引 → 该候选 orig_bbox 作为 verified_result；choice=-1 → None
    """
    vlm = get_vlm_client(VLM_PROVIDER, model=VLM_MODEL, max_tokens=SELECT_MAX_TOKENS)
    image_path = state["original_image_path"]
    verify_dir = config.verify_dir()

    img = Image.open(image_path)
    img_w, img_h = img.size

    candidates = list(state["candidates"])
    chosen = candidates[:VERIFY_MAX]
    if not chosen:
        return {"candidates": candidates, "verified_result": None}

    # 1. 逐候选裁出带 padding 的原图区域
    crop_paths: list[str] = []
    orig_bboxes: list[list[int]] = []
    for i, cand in enumerate(chosen):
        orig_bbox = waldo_orig_bbox(cand["patch_bbox"], cand.get("waldo_bbox_in_patch"))
        padded_bbox = expand_bbox(orig_bbox, img_w, img_h, PADDING_RATIO, MIN_VERIFY_SIZE)
        crop_img = crop_to_pil(image_path, padded_bbox)
        verify_path = os.path.join(verify_dir, f"verify{i}.jpg")
        save_patch(crop_img, verify_path)
        crop_paths.append(verify_path)
        orig_bboxes.append(orig_bbox)

    # 2. 横向单选
    result = vlm.select(crop_paths)
    per_image = result.per_image or []

    # 3. 写回 verify 标记（让 main.py 仍能区分「verify 跑过但都不是」=not found）
    for i in range(len(chosen)):
        is_chosen = (i == result.choice)
        candidates[i] = {
            **chosen[i],
            "verified": is_chosen,
            "verify_confidence": result.confidence if is_chosen else 0.0,
            "orig_bbox": orig_bboxes[i],
            "verify_crop_path": crop_paths[i],
            "verify_looks_waldo": per_image[i] if i < len(per_image) else None,
        }

    verified_result = None
    if 0 <= result.choice < len(orig_bboxes):
        verified_result = orig_bboxes[result.choice]
        print(
            f"[verify] Gemini selected cand={result.choice} "
            f"conf={result.confidence:.2f} bbox={verified_result} | per_image={per_image}"
        )
    else:
        print(
            f"[verify] Gemini selected none "
            f"(choice={result.choice}) | per_image={per_image}"
        )

    return {
        "candidates": candidates,
        "verified_result": verified_result,
    }
