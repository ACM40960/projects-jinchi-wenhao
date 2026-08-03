"""AWS Lambda 同步入口：一个函数同时提供前端页面与检测接口。

对外挂在 Lambda Function URL 上（payload format 2.0）：

- `GET  /`        → 返回 web/index.html
- `POST /detect`  → 收 base64 图片，跑 run_pipeline，返回 bbox + Waldo 特写图

为什么不回传整张标注图：Function URL 同步调用的请求与响应 payload 各有 6MB 上限，
而测试图最大已 4.1MB（base64 后约 5.5MB）。因此只回 bbox（原图坐标）与一张小的特写
图，红框由前端在自己那份原图上用 canvas 画。详见
docs/superpowers/specs/2026-08-03-lambda-deployment-design.md。
"""

import base64
import binascii
import io
import json
import os
import time
import uuid

from PIL import Image, UnidentifiedImageError

import config
from agent import run_pipeline, summarize
from vision.image_utils import crop_to_pil
from vision.segment import expand_bbox

# ── 可调参数 ───────────────────────────────────────────────────────────
CROP_PADDING_RATIO = 0.6     # 特写图相对 bbox 向外扩展的比例（比 verify 的 0.3 宽，便于人眼定位）
CROP_MIN_SIZE = 200          # 特写图最小边长（px）
CROP_MAX_SIZE = 480          # 特写图长边上限（px），控响应体积
MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 解码后原图字节上限，超出直接 400

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def lambda_handler(event, context):
    """Function URL 入口。任何未预期异常都收敛成 500，不让 Lambda 抛裸栈。"""
    method, path = _route(event)
    print(f"[handler] {method} {path}")

    try:
        if method == "OPTIONS":
            return _response(204, "", {})
        if method == "GET" and path in ("/", "", "/index.html"):
            return _html_response()
        if method == "POST" and path in ("/detect", "/"):
            return _detect_response(event)
        return _json_response(404, {"error": "not_found", "message": f"No route for {method} {path}"})
    except _BadRequest as exc:
        return _json_response(400, {"error": "bad_request", "message": str(exc)})
    except Exception as exc:                                  # noqa: BLE001 — 兜底，必须返回 HTTP 而非崩溃
        print(f"[handler] Unhandled error: {type(exc).__name__}: {exc}")
        return _json_response(500, {"error": "internal_error", "message": str(exc)})


# ── 路由处理 ───────────────────────────────────────────────────────────

def _html_response():
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    return _response(200, html, {"Content-Type": "text/html; charset=utf-8"})


def _detect_response(event):
    """收图 → 落盘 /tmp → run_pipeline → bbox + 特写图。"""
    payload = _parse_json_body(event)
    image_bytes = _decode_image_field(payload.get("image"))

    config.reset_run_dirs()
    image_path = os.path.join(config.uploads_dir(), f"{uuid.uuid4().hex}.jpg")
    img_w, img_h = _save_upload(image_bytes, image_path)

    started = time.time()
    try:
        result = summarize(run_pipeline(image_path))
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["image_size"] = [img_w, img_h]
        result["crop"] = _crop_data_url(image_path, result["bbox"], img_w, img_h)
        # 落盘路径是 Lambda 内部实现细节，不外泄给前端
        result.pop("result_image_path", None)
        print(f"[handler] found={result['found']} source={result['source']} "
              f"bbox={result['bbox']} elapsed={result['elapsed_ms']}ms")
        return _json_response(200, result)
    finally:
        # 容器会被复用，上传图不清理会一直占 /tmp（512MB 上限）
        _silent_remove(image_path)


# ── 输入解析 ───────────────────────────────────────────────────────────

class _BadRequest(Exception):
    """客户端输入问题，转成 400。"""


def _route(event) -> tuple[str, str]:
    """从 Function URL payload v2.0 取出 method 与 path。"""
    http = (event or {}).get("requestContext", {}).get("http", {})
    method = http.get("method") or (event or {}).get("httpMethod") or "GET"
    path = http.get("path") or (event or {}).get("rawPath") or "/"
    return method.upper(), path


def _parse_json_body(event) -> dict:
    body = (event or {}).get("body")
    if not body:
        raise _BadRequest("Request body is empty")
    if event.get("isBase64Encoded"):
        try:
            body = base64.b64decode(body).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise _BadRequest(f"Cannot decode base64 body: {exc}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise _BadRequest(f"Body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _BadRequest("Body must be a JSON object")
    return payload


def _decode_image_field(value) -> bytes:
    """解码 `image` 字段，容忍 `data:image/jpeg;base64,` 前缀。"""
    if not value or not isinstance(value, str):
        raise _BadRequest("Missing 'image' field (base64-encoded image)")
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise _BadRequest(f"'image' is not valid base64: {exc}") from exc
    if not raw:
        raise _BadRequest("'image' decoded to empty bytes")
    if len(raw) > MAX_IMAGE_BYTES:
        raise _BadRequest(f"Image is {len(raw)} bytes, over the {MAX_IMAGE_BYTES} limit")
    return raw


def _save_upload(image_bytes: bytes, image_path: str) -> tuple[int, int]:
    """校验并以 JPEG 落盘，返回图片尺寸。"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise _BadRequest(f"Not a readable image: {exc}") from exc
    img = img.convert("RGB")
    img.save(image_path, format="JPEG", quality=92)
    return img.size


# ── 输出构造 ───────────────────────────────────────────────────────────

def _crop_data_url(image_path: str, bbox, img_w: int, img_h: int) -> str | None:
    """裁出 Waldo 特写（带 padding、限长边），编码成 data URL。找不到则 None。"""
    if not bbox:
        return None
    padded = expand_bbox(bbox, img_w, img_h, CROP_PADDING_RATIO, CROP_MIN_SIZE)
    crop = crop_to_pil(image_path, padded)
    crop.thumbnail((CROP_MAX_SIZE, CROP_MAX_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _json_response(status: int, payload: dict):
    return _response(status, json.dumps(payload), {"Content-Type": "application/json"})


def _response(status: int, body: str, headers: dict):
    return {
        "statusCode": status,
        "headers": {**_CORS_HEADERS, **headers},
        "body": body,
    }


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
