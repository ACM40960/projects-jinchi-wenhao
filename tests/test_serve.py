"""serve.py 的 HTTP ↔ Lambda event 双向翻译测试。

真起一个 `ThreadingHTTPServer`（端口 0 由系统分配）并发真实 HTTP 请求，但把
`lambda_handler` 打桩成回显——翻译层才是这里的风险点，流水线本身有别的测试覆盖，
且绝不能在单测里打 API。
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import serve


class TestBuildEvent:
    def test_shapes_a_function_url_v2_event(self):
        event = serve.build_event("post", "/detect", '{"image":"x"}')
        assert event["requestContext"]["http"]["method"] == "POST"
        assert event["requestContext"]["http"]["path"] == "/detect"
        assert event["rawPath"] == "/detect"
        assert event["body"] == '{"image":"x"}'
        assert event["isBase64Encoded"] is False

    def test_method_is_upcased(self):
        assert serve.build_event("get", "/")["requestContext"]["http"]["method"] == "GET"

    def test_body_defaults_to_empty(self):
        assert serve.build_event("GET", "/")["body"] == ""


@pytest.fixture
def live_server(monkeypatch):
    """起一个真服务器，lambda_handler 换成回显桩。返回 (base_url, seen)。"""
    seen = {}

    def fake_handler(event, context):
        seen["event"] = event
        return {
            "statusCode": 201,                       # 非 200，用来验证状态码确实透传
            "headers": {"Content-Type": "application/json", "X-Test": "yes"},
            "body": json.dumps({"ok": True}),
        }

    monkeypatch.setattr(serve, "lambda_handler", fake_handler)

    server = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()


class TestHttpTranslation:
    def test_get_root_passes_method_and_path(self, live_server):
        base, seen = live_server
        with urllib.request.urlopen(f"{base}/") as resp:
            assert resp.status == 201
            assert resp.headers["X-Test"] == "yes"
            assert json.loads(resp.read())["ok"] is True
        assert seen["event"]["requestContext"]["http"] == {"method": "GET", "path": "/"}

    def test_post_body_round_trips(self, live_server):
        base, seen = live_server
        body = json.dumps({"image": "data:image/jpeg;base64,AAAA"}).encode()
        req = urllib.request.Request(f"{base}/detect", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 201
        event = seen["event"]
        assert event["requestContext"]["http"]["method"] == "POST"
        assert json.loads(event["body"])["image"].startswith("data:image/jpeg")

    def test_query_string_is_stripped_from_path(self, live_server):
        base, seen = live_server
        with urllib.request.urlopen(f"{base}/?cache=0"):
            pass
        # handler 按 path 精确匹配路由，带上 "?cache=0" 会掉进 404
        assert seen["event"]["requestContext"]["http"]["path"] == "/"
