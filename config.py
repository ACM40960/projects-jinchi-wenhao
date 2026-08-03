"""运行期路径配置：所有节点的文件输出都经由这里取目录。

本地跑落在仓库的 `outputs/`；Lambda 里只有 `/tmp` 可写，部署时把环境变量
`WALDO_OUTPUT_DIR` 设成 `/tmp/outputs` 即可，节点代码无需感知自己跑在哪。

刻意用函数而非模块级常量：环境变量在测试与 Lambda 里都可能后设，import 时求值
会把值焊死。
"""

import os
import shutil

ENV_OUTPUT_DIR = "WALDO_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR = "outputs"


def output_base_dir() -> str:
    """所有运行产物的基目录。"""
    return os.environ.get(ENV_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR


def results_dir() -> str:
    """最终标注图的输出目录。"""
    return _ensure(output_base_dir())


def patches_dir() -> str:
    """detect 裁出的 patch 目录。"""
    return _ensure(os.path.join(output_base_dir(), "patches"))


def verify_dir() -> str:
    """verify 的特写裁剪目录。"""
    return _ensure(os.path.join(output_base_dir(), "verify"))


def uploads_dir() -> str:
    """handler 落盘上传图的目录。"""
    return _ensure(os.path.join(output_base_dir(), "uploads"))


def reset_run_dirs() -> None:
    """清空上一次运行的 patch / verify 产物。

    Lambda 容器会被复用，不清理的话上一次请求的裁剪图会一直留在 `/tmp`
    （上限 512MB）。本地跑也顺带避免看到上一张图的残留。
    """
    base = output_base_dir()
    for name in ("patches", "verify"):
        shutil.rmtree(os.path.join(base, name), ignore_errors=True)


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path