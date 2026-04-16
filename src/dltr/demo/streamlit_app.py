from __future__ import annotations

import json
from pathlib import Path

from dltr.demo.runtime import run_paddleocr_e2e_inference
from dltr.visualization.report_index import is_mainline_report_path


def discover_report_files(reports_dir: Path) -> dict[str, list[Path]]:
    train_dir = reports_dir / "train"
    extension_dir = reports_dir / "extensions"
    eval_dir = reports_dir / "eval"
    eda_dir = reports_dir / "eda"
    return {
        "train_markdown": (
            sorted(path for path in train_dir.glob("*.md") if is_mainline_report_path(path))
            if train_dir.exists()
            else []
        ),
        "train_png": (
            sorted(path for path in train_dir.glob("*.png") if is_mainline_report_path(path))
            if train_dir.exists()
            else []
        ),
        "extension_markdown": sorted(extension_dir.glob("*.md")) if extension_dir.exists() else [],
        "extension_png": sorted(extension_dir.glob("*.png")) if extension_dir.exists() else [],
        "eval_json": sorted(eval_dir.glob("*.json")) if eval_dir.exists() else [],
        "eda_markdown": sorted(eda_dir.glob("*.md")) if eda_dir.exists() else [],
    }


def load_end_to_end_preview(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_streamlit_app() -> None:
    import streamlit as st

    root = Path.cwd()
    reports_dir = root / "reports"
    files = discover_report_files(reports_dir)

    st.set_page_config(
        page_title="DLTR 演示系统",
        page_icon="OCR",
        layout="wide",
    )
    st.title("场景文本检测与识别演示系统")
    st.caption("基于 PaddleOCR PP-OCRv5 端到端推理，同时展示检测、识别与 OCR 后结构化分析结果。")

    left, right = st.columns((1.2, 1))

    with left:
        st.subheader("运行端到端推理")
        uploaded = st.file_uploader("上传场景图片", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            if st.button("运行 OCR 流水线", type="primary"):
                try:
                    artifacts = run_paddleocr_e2e_inference(
                        image_bytes=uploaded.getvalue(),
                        project_root=root,
                    )
                    st.success("端到端推理完成（PaddleOCR PP-OCRv5 English）。")
                    st.image(str(artifacts.preview_image_path), caption="推理结果预览")
                    st.json(load_end_to_end_preview(artifacts.json_path))
                except RuntimeError as exc:
                    st.error(str(exc))
                except FileNotFoundError as exc:
                    st.error(str(exc))

        st.subheader("最近一次端到端结果")
        preview = load_end_to_end_preview(reports_dir / "eval" / "end_to_end_preview.json")
        if preview:
            st.json(preview)
        else:
            st.info("暂未发现端到端预览结果。")

        st.subheader("OCR 后规则理解示例")
        demo_report = reports_dir / "demo_assets" / "demo_preview_analysis_report.md"
        if demo_report.exists():
            st.markdown(demo_report.read_text(encoding="utf-8"))
        else:
            st.info("暂未发现 OCR 后规则理解示例报告。")

    with right:
        st.subheader("训练报告")
        if files["train_markdown"]:
            selected = st.selectbox(
                "选择训练报告",
                files["train_markdown"],
                format_func=lambda path: path.name,
            )
            st.markdown(selected.read_text(encoding="utf-8"))
        else:
            st.info("暂未发现训练报告。")

        if files["train_png"]:
            selected_png = st.selectbox(
                "选择训练曲线图",
                files["train_png"],
                format_func=lambda path: path.name,
            )
            st.image(str(selected_png), caption=selected_png.name)

        st.subheader("扩展模块报告")
        if files["extension_markdown"]:
            selected_extension = st.selectbox(
                "选择扩展报告",
                files["extension_markdown"],
                format_func=lambda path: path.name,
            )
            st.markdown(selected_extension.read_text(encoding="utf-8"))
        else:
            st.info("暂未发现扩展模块报告。")

        if files["extension_png"]:
            selected_extension_png = st.selectbox(
                "选择扩展曲线图",
                files["extension_png"],
                format_func=lambda path: path.name,
            )
            st.image(str(selected_extension_png), caption=selected_extension_png.name)

        st.subheader("数据分析报告")
        if files["eda_markdown"]:
            selected_eda = st.selectbox(
                "选择数据分析报告",
                files["eda_markdown"],
                format_func=lambda path: path.name,
            )
            st.markdown(selected_eda.read_text(encoding="utf-8"))
        else:
            st.info("暂未发现数据分析报告。")


if __name__ == "__main__":
    render_streamlit_app()
