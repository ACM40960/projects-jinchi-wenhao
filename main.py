"""本地 runner：指定图片路径，跑核心检测流水线，打印结果。

用法：python main.py [图片路径]   # 默认 original-images/1.jpg
"""

import sys

# 优先从 .env 文件加载环境变量（有则加载，无则跳过）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import run_pipeline, summarize


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "original-images/1.jpg"
    print(f"[main] Running Waldo detection on: {image_path}")

    result = summarize(run_pipeline(image_path))

    if not result["found"]:
        print("[main] Waldo not found.")
        return

    if result["source"] == "verified":
        print(f"[main] Waldo confirmed (verified) at bbox: {result['bbox']}")
    else:
        print(f"[main] Waldo located (detect-only, verify skipped) at bbox: {result['bbox']}")
    print(f"[main] Annotated image → {result['result_image_path']}")


if __name__ == "__main__":
    main()
