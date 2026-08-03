"""handler 测试：路由、输入解析、错误码（打桩 run_pipeline，不调 VLM/不打网络）。"""

import base64
import io
import json

import pytest
from PIL import Image

import handler


@pytest.fixture(autouse=True)
def tmp_output_dir(tmp_path, monkeypatch):
    """把所有落盘产物导到临时目录，别污染仓库的 outputs/。"""
    monkeypatch.setenv("WALDO_OUTPUT_DIR", str(tmp_path / "outputs"))


def _image_b64(size=(64, 48), color=(200, 30, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _event(method="POST", path="/detect", body=None, is_b64=False):
    return {
        "requestContext": {"http": {"method": method, "path": path}},
        "body": body,
        "isBase64Encoded": is_b64,
    }


def _stub_pipeline(monkeypatch, state):
    monkeypatch.setattr(handler, "run_pipeline", lambda image_path: state)


def _found_state(image_path="x.jpg"):
    return {
        "original_image_path": image_path,
        "candidates": [{"verify_confidence": 0.9}],
        "verified_result": [10, 10, 20, 20],
        "result_image_path": "/tmp/outputs/x_result.jpg",
    }


# ── 路由 ───────────────────────────────────────────────────────────────

def test_get_root_returns_html():
    res = handler.lambda_handler(_event(method="GET", path="/"), None)
    assert res["statusCode"] == 200
    assert res["headers"]["Content-Type"].startswith("text/html")
    assert "<html" in res["body"].lower()


def test_unknown_route_returns_404():
    res = handler.lambda_handler(_event(method="GET", path="/nope"), None)
    assert res["statusCode"] == 404


def test_options_returns_cors_preflight():
    res = handler.lambda_handler(_event(method="OPTIONS", path="/detect"), None)
    assert res["statusCode"] == 204
    assert res["headers"]["Access-Control-Allow-Origin"] == "*"


# ── POST /detect 正常路径 ──────────────────────────────────────────────

def test_detect_returns_bbox_and_crop(monkeypatch):
    _stub_pipeline(monkeypatch, _found_state())
    res = handler.lambda_handler(_event(body=json.dumps({"image": _image_b64()})), None)

    assert res["statusCode"] == 200
    payload = json.loads(res["body"])
    assert payload["found"] is True
    assert payload["bbox"] == [10, 10, 20, 20]
    assert payload["source"] == "verified"
    assert payload["image_size"] == [64, 48]
    assert payload["crop"].startswith("data:image/jpeg;base64,")
    assert "elapsed_ms" in payload
    # 落盘路径是内部细节，不该外泄
    assert "result_image_path" not in payload


def test_detect_accepts_data_url_prefix(monkeypatch):
    _stub_pipeline(monkeypatch, _found_state())
    body = json.dumps({"image": "data:image/jpeg;base64," + _image_b64()})
    res = handler.lambda_handler(_event(body=body), None)
    assert res["statusCode"] == 200


def test_detect_accepts_base64_encoded_body(monkeypatch):
    """Function URL 在某些 Content-Type 下会把 body 再 base64 一层。"""
    _stub_pipeline(monkeypatch, _found_state())
    inner = json.dumps({"image": _image_b64()})
    body = base64.b64encode(inner.encode("utf-8")).decode("ascii")
    res = handler.lambda_handler(_event(body=body, is_b64=True), None)
    assert res["statusCode"] == 200


def test_detect_not_found_returns_null_bbox(monkeypatch):
    _stub_pipeline(monkeypatch, {
        "original_image_path": "x.jpg",
        "candidates": [],
        "verified_result": None,
        "result_image_path": None,
    })
    res = handler.lambda_handler(_event(body=json.dumps({"image": _image_b64()})), None)

    payload = json.loads(res["body"])
    assert res["statusCode"] == 200
    assert payload["found"] is False
    assert payload["bbox"] is None
    assert payload["crop"] is None


def test_detect_removes_uploaded_file(monkeypatch, tmp_path):
    """容器会被复用，上传图必须清掉，否则 /tmp 会被撑满。"""
    seen = {}

    def fake_run(image_path):
        seen["path"] = image_path
        return _found_state(image_path)

    monkeypatch.setattr(handler, "run_pipeline", fake_run)
    handler.lambda_handler(_event(body=json.dumps({"image": _image_b64()})), None)

    import os
    assert not os.path.exists(seen["path"])


# ── 错误路径 ───────────────────────────────────────────────────────────

def test_missing_image_field_returns_400():
    res = handler.lambda_handler(_event(body=json.dumps({"filename": "a.jpg"})), None)
    assert res["statusCode"] == 400
    assert json.loads(res["body"])["error"] == "bad_request"


def test_empty_body_returns_400():
    res = handler.lambda_handler(_event(body=None), None)
    assert res["statusCode"] == 400


def test_malformed_json_returns_400():
    res = handler.lambda_handler(_event(body="{not json"), None)
    assert res["statusCode"] == 400


def test_non_image_bytes_returns_400():
    junk = base64.b64encode(b"definitely not an image").decode("ascii")
    res = handler.lambda_handler(_event(body=json.dumps({"image": junk})), None)
    assert res["statusCode"] == 400


def test_oversized_image_returns_400(monkeypatch):
    monkeypatch.setattr(handler, "MAX_IMAGE_BYTES", 10)
    res = handler.lambda_handler(_event(body=json.dumps({"image": _image_b64()})), None)
    assert res["statusCode"] == 400


def test_pipeline_exception_returns_500(monkeypatch):
    def boom(image_path):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(handler, "run_pipeline", boom)
    res = handler.lambda_handler(_event(body=json.dumps({"image": _image_b64()})), None)

    assert res["statusCode"] == 500
    payload = json.loads(res["body"])
    assert payload["error"] == "internal_error"
    assert "gemini exploded" in payload["message"]
