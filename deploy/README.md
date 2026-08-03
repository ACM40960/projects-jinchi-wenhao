# 部署到 AWS Lambda

单个容器镜像 Lambda + Function URL，同步返回结果。架构与取舍见
[`docs/superpowers/specs/2026-08-03-lambda-deployment-design.md`](../docs/superpowers/specs/2026-08-03-lambda-deployment-design.md)。

## 前置条件

- Docker（构建镜像）
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- 已配好的 AWS 凭据（`aws configure`），账号需能建 Lambda / ECR / IAM 角色 / CloudFormation 栈
- 一个可用的 `GOOGLE_API_KEY`（付费 Tier 1；免费层 20 次/天，一张图就打满）

## 首次部署

```bash
cd deploy
sam build
sam deploy --guided --parameter-overrides GoogleApiKey=$GOOGLE_API_KEY
```

`--guided` 会问 stack 名与 region（默认见 `samconfig.toml`：`whereis-waldo` / `eu-west-1`），
并自动创建 ECR 仓库、推镜像。完成后输出里的 `FunctionUrl` 就是可直接在浏览器打开的地址。

> Windows PowerShell 取环境变量用 `$env:GOOGLE_API_KEY`。

## 后续更新

```bash
cd deploy
sam build && sam deploy --parameter-overrides GoogleApiKey=$GOOGLE_API_KEY
```

## 查日志

```bash
sam logs --stack-name whereis-waldo --tail
```

## 本地验证镜像（不部署）

```bash
# 在仓库根目录构建（构建上下文必须是根目录）
docker build -f deploy/Dockerfile -t waldo-lambda .

# 官方基镜像自带 Runtime Interface Emulator，9000 端口模拟 Lambda 调用
docker run --rm -p 9000:8080 -e GOOGLE_API_KEY=$GOOGLE_API_KEY waldo-lambda

# 另开一个终端：GET / 应返回前端页面 HTML
curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"requestContext":{"http":{"method":"GET","path":"/"}}}' | head -c 300
```

## 拆除

```bash
sam delete --stack-name whereis-waldo
```

ECR 里的镜像不会随栈删除，需要时到控制台或用 `aws ecr delete-repository` 手动清。

## 注意

- **`AuthType: NONE` 意味着 URL 公开可调用**，每次调用都会消耗你的 Gemini 额度
  （约 $0.09/图）。若要长期挂着，把 `template.yaml` 里改成 `AWS_IAM`，或在前面加限流。
- Lambda 侧成本约 $0.0013/图（2GB × ~40s），相对 Gemini 可忽略；无请求时为 0。
- 冷启动（容器镜像 + Pillow/grpc）预计数秒到十几秒，首次请求会明显更慢。
