from __future__ import annotations

import json
from pathlib import Path

from dltr.demo.runtime import resolve_demo_checkpoints, run_uploaded_inference


def discover_report_files(reports_dir: Path) -> dict[str, list[Path]]:
    train_dir = reports_dir / "train"
    eval_dir = reports_dir / "eval"
    eda_dir = reports_dir / "eda"
    return {
        "train_markdown": sorted(train_dir.glob("*.md")) if train_dir.exists() else [],
        "train_png": sorted(train_dir.glob("*.png")) if train_dir.exists() else [],
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
    st.title("中文场景文本检测与识别演示系统")
    st.caption("在一个界面中查看检测、识别、OCR 后规则理解扩展与实验报告。")

    left, right = st.columns((1.2, 1))

    with left:
        st.subheader("运行端到端推理")
        uploaded = st.file_uploader("上传场景图片", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            if st.button("运行 OCR 流水线", type="primary"):
                try:
                    checkpoints = resolve_demo_checkpoints(project_root=root)
                    artifacts = run_uploaded_inference(
                        image_bytes=uploaded.getvalue(),
                        project_root=root,
                        detector_checkpoint=checkpoints["detector"],
                        recognizer_checkpoint=checkpoints["recognizer"],
                    )
                    st.success("端到端推理完成。")
                    st.image(str(artifacts.preview_image_path), caption="推理结果预览")
                    st.json(load_end_to_end_preview(artifacts.json_path))
                except FileNotFoundError as exc:
                    st.error(str(exc))

        st.subheader("最近一次端到端结果")
        preview = load_end_to_end_preview(reports_dir / "eval" / "end_to_end_preview.json")
        if preview:
            st.json(preview)
        else:
            st.info("暂未发现端到端预览结果。")

        st.subheader("OCR 后规则理解示例")
        demo_report = reports_dir / "demo_assets" / "demo_preview_semantic_eval.md"
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
