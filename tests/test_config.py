"""config 测试：WALDO_OUTPUT_DIR 切换与目录创建/清理。"""

import os

import config


def test_defaults_to_repo_outputs(monkeypatch):
    monkeypatch.delenv(config.ENV_OUTPUT_DIR, raising=False)
    assert config.output_base_dir() == "outputs"


def test_env_var_redirects_all_dirs(tmp_path, monkeypatch):
    """Lambda 下把基目录指到 /tmp，节点代码无需感知。"""
    base = tmp_path / "outputs"
    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(base))

    assert config.results_dir() == str(base)
    assert config.patches_dir() == str(base / "patches")
    assert config.verify_dir() == str(base / "verify")
    assert config.uploads_dir() == str(base / "uploads")


def test_dirs_are_created(tmp_path, monkeypatch):
    base = tmp_path / "outputs"
    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(base))

    for path in (config.patches_dir(), config.verify_dir(), config.uploads_dir()):
        assert os.path.isdir(path)


def test_env_var_read_at_call_time(tmp_path, monkeypatch):
    """import 时求值会把值焊死，这里保证是调用时才读。"""
    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(tmp_path / "first"))
    assert config.output_base_dir().endswith("first")
    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(tmp_path / "second"))
    assert config.output_base_dir().endswith("second")


def test_reset_run_dirs_clears_stale_artifacts(tmp_path, monkeypatch):
    """容器复用时上一次请求的裁剪图必须清掉，否则 /tmp 越积越多。"""
    base = tmp_path / "outputs"
    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(base))

    stale = os.path.join(config.patches_dir(), "patch0.jpg")
    with open(stale, "w") as fh:
        fh.write("stale")
    result_image = os.path.join(config.results_dir(), "1_result.jpg")
    with open(result_image, "w") as fh:
        fh.write("keep")

    config.reset_run_dirs()

    assert not os.path.exists(stale)
    assert not os.path.isdir(base / "patches")
    # 只清 patch/verify，基目录里的其它东西不动
    assert os.path.exists(result_image)
