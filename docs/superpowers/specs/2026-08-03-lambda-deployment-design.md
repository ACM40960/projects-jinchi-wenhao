# 部署设计：容器镜像同步 Lambda + Function URL

- 日期：2026-08-03
- 状态：已批准，本次落地脚手架
- 上游决策：CLAUDE.md「目标架构：方案 ①」（2026-06-23 选定）

## 1. 目标

把现有 CLI 流水线（`run_pipeline`）包成一个可公开访问的网页 demo：上传一张 Where's
Waldo 图，等待，拿到 Waldo 的原图坐标 bbox 与可视结果。

约束来自 CLAUDE.md：闲时零成本（缩容到 0）、部署心智负担最小、单图慢一点可接受。

## 2. 架构

单个容器镜像 Lambda，挂 Function URL，一个函数同时提供页面与推理：

```
浏览器 ──GET  /        ──> Lambda ──> 内嵌 HTML 单页
       ──POST /detect  ──> Lambda ──> /tmp/{uuid}.jpg → run_pipeline → JSON
```

不引入 S3、DynamoDB、API Gateway、CloudFront。Function URL 没有 API Gateway 的 29s
集成超时，可同步等到 Lambda 上限 15min；单图实测约 33s（1.jpg，单候选快路径），
余量充足。

Lambda 配置：`timeout=900s`、`memory=2048MB`（内存同时决定 CPU 配额，detect 有 10
线程并发）、`AuthType=NONE`、`InvokeMode=BUFFERED`。

## 3. 请求契约

### `GET /`

返回 `web/index.html` 的内容，`Content-Type: text/html`。

### `POST /detect`

请求体 JSON（不用 multipart —— 前端 `FileReader` 直接产 base64，服务端省掉一个
multipart 解析器）：

```json
{ "image": "<base64 或 data:image/jpeg;base64,... >", "filename": "1.jpg" }
```

成功响应 `200`：

```json
{
  "found": true,
  "bbox": [1180, 640, 42, 58],
  "source": "verified",
  "crop": "data:image/jpeg;base64,...",
  "elapsed_ms": 33210,
  "image_size": [2048, 1251]
}
```

- `source`：`"verified"`（verify 跑过并选中）或 `"detect-only"`（单候选按路由跳过
  verify，见 `agent/pipeline.py`）；未找到时为 `null`。
- `bbox`：原图像素坐标 `[x, y, w, h]`；未找到时为 `null`。
- `crop`：Waldo 特写小图（长边 ≤ 480px）的 data URL，仅供页面展示。

错误响应：`400`（缺 `image` / base64 解不开 / 不是合法图片）、`404`（未知路由）、
`500`（pipeline 内部异常，body 含 `error` 与 `message`）。

### 为什么不回传整张标注图

Function URL 同步调用的**请求与响应 payload 各有 6MB 硬上限**。`original-images/14.jpg`
已有 4.1MB，base64 后约 5.5MB，上下行都会顶到限制。因此：

- **上行**：前端上传前用 canvas 重新编码（长边上限 3000px、JPEG q0.85）。测试集最大
  边长 2828px，等于**只重压缩、不降分辨率**，不伤小 Waldo 的检测能力。
- **下行**：只回 bbox（原图坐标）+ 一张 Waldo 特写小图；红框由前端在自己那份原图上
  用 canvas 画。响应从数 MB 降到几 KB。

这是对 CLAUDE.md 原文「结果以 base64 随响应回传」的修正，原因即上述 6MB 上限。

## 4. 代码改造

`agent/pipeline.py` 的编排逻辑不动。

| 改动 | 位置 | 内容 |
|---|---|---|
| 输出目录抽象 | 新增 `config.py` | `patches_dir()/verify_dir()/results_dir()/uploads_dir()`，基目录取环境变量 `WALDO_OUTPUT_DIR`（默认 `outputs`，Lambda 设 `/tmp/outputs`）。替换 `detect.py`、`verify.py`、`visualize.py` 三处硬编码常量。用函数而非模块级常量，避免 import 时把环境变量焊死 |
| 结果图路径进 state | `agent/nodes/visualize.py`、`agent/state.py` | `visualize_node` 原本返回 `{}`，结果图路径丢失。改为返回 `{"result_image_path": saved}`，`WaldoState` 加同名字段 |
| 结果提炼共用 | 新增 `agent/result.py` | `summarize(state) -> {found, bbox, source, result_image_path}`。`main.py` 里那段「verify 跑没跑过」的判断挪进来，CLI 与 handler 共用一份 |
| bbox 外扩共用 | `vision/segment.py` | `verify.py` 的私有 `_expand_bbox` 提升为 `expand_bbox`，handler 裁特写图时复用 |

`config.py` 另提供 `reset_run_dirs()`：清空 patches/verify 目录。Lambda 容器会被复用，
不清理会让上一次请求的 patch 残留在 `/tmp`（512MB 上限）。

## 5. 打包与部署

- `deploy/Dockerfile`：基于 `public.ecr.aws/lambda/python:3.12`，只拷贝运行期需要的
  模块（`agent/ llm/ vision/ tools/ web/ prompts.py config.py handler.py`）。
- `.dockerignore`：**必须排除 `yolo/`**（含 `.pt` 权重与 docx/pdf）、`original-images/`、
  `outputs/`、`.git/`、`tests/`，否则镜像白白多出几百 MB。
- `deploy/template.yaml`（AWS SAM）：一个 `AWS::Serverless::Function`，
  `PackageType: Image` + `FunctionUrlConfig`。`GOOGLE_API_KEY` 走 SAM 参数注入环境
  变量，不进镜像、不进 git。
- `deploy/samconfig.toml`：stack 名、region 等默认参数。
- `sam build && sam deploy` 自动建 ECR 仓库、推镜像、输出 Function URL。

## 6. 前端

`web/index.html`，单文件、无框架无依赖。流程：选图 → canvas 压缩 → POST → 转圈并
**累计秒数计时**（要干等 30s~数分钟，必须给反馈）→ 拿到 bbox 后在原图 canvas 上画红
框，旁边贴特写图。

## 7. 测试

`tests/test_handler.py`：打桩 `run_pipeline`，**不打真 API**。覆盖 GET / 返回 HTML、
POST 正常路径、data URL 前缀剥离、缺 `image` → 400、坏 base64 → 400、未知路由 → 404、
pipeline 抛错 → 500。

`tests/test_result.py`：`summarize` 的三条分支（verified / detect-only / not found）。

`tests/test_config.py`：`WALDO_OUTPUT_DIR` 生效且目录被创建。

现有 `test_segment.py` / `test_vlm_parse.py` / `test_pipeline.py` 不受影响。

## 8. 成本

Lambda 侧 2GB × ~40s ≈ **$0.0013/图**，相对 Gemini 的 ~$0.09/图 可忽略。无请求时
计费为 0。

## 9. 未覆盖 / 后续

- 冷启动：容器镜像 + Pillow/grpc 预计数秒到十几秒，demo 可接受，未做 Provisioned
  Concurrency。
- `AuthType: NONE` 意味着 URL 公开可调用，每次调用都花 Gemini 的钱。若要长期挂着，
  需加限流或改 `AWS_IAM`。本次不做。
- 极端大图（长边 > 3000px）由前端降采样后再上传，可能影响极小 Waldo 的召回；测试集
  无此情况。