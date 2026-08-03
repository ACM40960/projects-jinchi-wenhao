# Lambda 容器镜像：官方 Python 基镜像已自带 Runtime Interface Client。
# 构建上下文是仓库根目录（见 deploy/template.yaml 的 DockerContext），
# 排除项见根目录 .dockerignore —— 尤其要挡住 yolo/（含 .pt 权重）。
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# 只拷贝运行期真正需要的模块（不含 main.py / tests / original-images / yolo）
COPY handler.py config.py prompts.py ${LAMBDA_TASK_ROOT}/
COPY agent/  ${LAMBDA_TASK_ROOT}/agent/
COPY llm/    ${LAMBDA_TASK_ROOT}/llm/
COPY vision/ ${LAMBDA_TASK_ROOT}/vision/
COPY tools/  ${LAMBDA_TASK_ROOT}/tools/
COPY web/    ${LAMBDA_TASK_ROOT}/web/

# 节点的文件输出改指 /tmp —— Lambda 里只有这里可写
ENV WALDO_OUTPUT_DIR=/tmp/outputs

CMD ["handler.lambda_handler"]
