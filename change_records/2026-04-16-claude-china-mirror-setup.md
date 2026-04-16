# 文档：国内镜像下 uv 与依赖安装

## 变更

- `CLAUDE.md`：增加「服务器仅国内镜像可达」小节：用 `pip install uv -i` + `UV_DEFAULT_INDEX` / `uv sync --default-index` 替代官方 `install.sh` 与默认 PyPI。

## 原因

AutoDL 等环境无法稳定访问 `astral.sh` / `pypi.org` 时，训练与数据准备命令无法执行。
