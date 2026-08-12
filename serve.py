"""本地演示服务器：把 `handler.lambda_handler` 挂到一个普通 HTTP 端口上。

用途是现场演示——不装 Docker、不开 AWS 账号，`python serve.py` 就能在浏览器里
跑完整的「上传图 → 出 bbox + 特写」流程。

它只做翻译这一件事：把普通 HTTP 请求转成 Lambda Function URL 的 payload v2.0
事件，喂给 `handler.lambda_handler`，再把返回的 dict 写回 HTTP 响应。因此走的是
与 Lambda 上**完全相同**的代码路径，`web/index.html` 一个字都不用改（它从
`location.origin` 推导接口地址）。这也顺带完成了「handler 在真实 HTTP 下能不能
跑」的验证——只是用 Python 而非 Docker RIE。

用法：
    python serve.py                  # http://127.0.0.1:8000/
    python serve.py --port 9000
    python serve.py --host 0.0.0.0   # 同网段其它设备也能访问

⚠️ 单人演示专用，别同时跑两张图：`handler` 每次请求会先 `config.reset_run_dirs()`
清空 patch 目录，两个检测并发会互相删对方的中间文件。
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# 优先从 .env 文件加载环境变量（有则加载，无则跳过），与 main.py 一致
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from handler import lambda_handler

# ── 可调参数 ───────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# 本地没有 Function URL 的 6MB payload 上限，给足余量；真正的图片大小校验在
# handler 的 MAX_IMAGE_BYTES 里做。
MAX_BODY_BYTES = 32 * 1024 * 1024


def build_event(method: str, path: str, body: str = "") -> dict:
    """造一个 Function URL payload v2.0 形状的事件。

    只填 `handler._route` / `_parse_json_body` 真正会读的字段——多造无用字段
    反而会让这里和真实 Lambda 事件的差异更难看清。
    """
    return {
        "version": "2.0",
        "rawPath": path,
        "requestContext": {"http": {"method": method.upper(), "path": path}},
        "body": body,
        # 本地读到的 body 已是解码后的文本，不需要 handler 再做 base64 解码
        "isBase64Encoded": False,
    }


# ── HTTP 适配 ──────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    server_version = "WaldoDemo/1.0"
    protocol_version = "HTTP/1.1"      # 需要它才能复用连接；配合下面显式写 Content-Length

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_OPTIONS(self):
        self._dispatch("OPTIONS")

    def handle_one_request(self):
        """吞掉客户端断连异常。

        `protocol_version = HTTP/1.1` 意味着连接是持久的，浏览器用完直接关掉是
        再正常不过的行为——但基类此时正阻塞在 readline 上，会抛 ConnectionResetError
        并往 stderr 打整段 traceback。演示时控制台被这些红字刷屏，会淹没真正的错误。
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def _dispatch(self, method: str) -> None:
        body = self._read_body()
        if body is None:                # 已经回过 413，不再往下走
            return
        # 查询串不参与路由：handler 按 path 精确匹配 "/" 与 "/detect"
        path = urlsplit(self.path).path
        self._write(lambda_handler(build_event(method, path, body), None))

    def _read_body(self) -> str | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.send_error(413, f"Body is {length} bytes, over the {MAX_BODY_BYTES} limit")
            return None
        if not length:
            return ""
        return self.rfile.read(length).decode("utf-8", errors="replace")

    def _write(self, result: dict) -> None:
        payload = (result.get("body") or "").encode("utf-8")
        self.send_response(result.get("statusCode", 200))
        for key, value in (result.get("headers") or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """静音默认访问日志——handler 自己已经打了 `[handler] GET /`，两份太吵。"""


# ── 入口 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Where's Waldo 本地演示服务器")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"监听地址（默认 {DEFAULT_HOST}；设成 0.0.0.0 可让同网段设备访问）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口（默认 {DEFAULT_PORT}）")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[serve] 浏览器打开 http://{args.host}:{args.port}/ —— Ctrl-C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
