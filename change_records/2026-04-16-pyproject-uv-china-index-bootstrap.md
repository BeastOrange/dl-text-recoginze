# 配置：国内 PyPI 默认源 + 服务器引导脚本

## 变更

- `pyproject.toml`：在 `[tool.uv]` 下增加 `[[tool.uv.index]]`（清华 `pypi.tuna`，`default = true`），使 `uv lock` / `uv sync` 默认走国内镜像。
- `scripts/bootstrap_cn_server.sh`：不经 `astral.sh`，用 `pip -i` 安装 `uv` 后执行 `uv sync --extra dev --extra train-cu`。
- `uv.lock`：`uv lock` 随索引配置重新解析生成。

## 境外环境

使用官方 PyPI：`UV_DEFAULT_INDEX=https://pypi.org/simple uv sync`（或见 `CLAUDE.md`）。
