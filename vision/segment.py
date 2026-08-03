
def _axis_starts(start: int, length: int, tile: int, stride: int) -> list[int]:
    """一根轴上的切片起点：滑窗 + 末块贴边对齐。

    起点 start, start+stride, ... 保留所有 < (start+length-tile) 者，
    再补一个贴边起点 start+length-tile，保证覆盖到边、每块恰好 tile。
    length <= tile 时退化为单块。
    """
    if length <= tile:
        return [start]
    last = start + length - tile
    starts: list[int] = []
    x = start
    while x < last:
        starts.append(x)
        x += stride
    starts.append(last)
    return starts


def tile_region(
    region: list[int],
    tile_size: int,
    image_size: tuple[int, int],
    overlap: float = 0.15,
) -> list[dict]:
    """把一个区域切成 tile_size×tile_size 的固定尺寸滑窗 patch（末块贴边对齐）。

    Args:
        region: [x, y, w, h]，原图坐标的目标区域。
        tile_size: 每块边长（像素）。
        image_size: (img_width, img_height)，用于边界 clamp。
        overlap: 相邻 patch 重叠比例，取值 [0, 1)；stride = round(tile_size*(1-overlap))。

    Returns:
        patch 字典列表，每项 {"bbox": [x, y, w, h], "row": int, "col": int}。
        区域 > tile 时每块宽高恰为 tile_size；区域 <= tile 时单块取整边。
    """
    rx, ry, rw, rh = region
    img_w, img_h = image_size
    # 防御：region 不应超出图像；clamp 实际可用宽高，保证末块仍为固定尺寸
    rw = min(rw, img_w - rx)
    rh = min(rh, img_h - ry)
    stride = max(1, round(tile_size * (1 - overlap)))
    xs = _axis_starts(rx, rw, tile_size, stride)
    ys = _axis_starts(ry, rh, tile_size, stride)

    patches: list[dict] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            x2 = min(img_w, x + tile_size, rx + rw)
            y2 = min(img_h, y + tile_size, ry + rh)
            patches.append({"bbox": [x, y, x2 - x, y2 - y], "row": row, "col": col})
    return patches


def waldo_orig_bbox(patch_bbox: list[int], waldo_bbox_in_patch: list[int] | None) -> list[int]:
    """把 patch 内的 Waldo bbox 换算到原图坐标；无精确子 bbox 时退化为整块 patch。

    Args:
        patch_bbox: [px, py, pw, ph]，patch 在原图中的位置。
        waldo_bbox_in_patch: [wx, wy, ww, wh]，Waldo 在 patch 内的局部坐标；
            None 或空表示 detect 未给出精确位置。

    Returns:
        原图坐标 [x, y, w, h]：有子 bbox 则为 [px+wx, py+wy, ww, wh]，否则整块 patch_bbox。
    """
    if not waldo_bbox_in_patch:
        return patch_bbox
    px, py = patch_bbox[0], patch_bbox[1]
    wx, wy, ww, wh = waldo_bbox_in_patch
    return [px + wx, py + wy, ww, wh]


def expand_bbox(
    bbox: list[int],
    img_w: int,
    img_h: int,
    ratio: float,
    min_size: int,
) -> list[int]:
    """向外扩展 bbox 并保证最小边长，结果 clamp 在图像范围内。

    verify 用它给 VLM 更多上下文；handler 用它裁给页面看的 Waldo 特写图。

    Args:
        bbox: [x, y, w, h]，原图坐标。
        img_w / img_h: 原图尺寸，用于 clamp。
        ratio: 相对 bbox 宽/高向外扩展的比例。
        min_size: 扩展后的最小边长（像素）。

    Returns:
        扩展后的 [x, y, w, h]。
    """
    x, y, w, h = bbox
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    # 保证最小边长
    if x2 - x1 < min_size:
        extra = (min_size - (x2 - x1)) // 2
        x1 = max(0, x1 - extra)
        x2 = min(img_w, x2 + extra)
    if y2 - y1 < min_size:
        extra = (min_size - (y2 - y1)) // 2
        y1 = max(0, y1 - extra)
        y2 = min(img_h, y2 + extra)

    return [x1, y1, x2 - x1, y2 - y1]