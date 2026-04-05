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
        page_title="DLTR Demo",
        page_icon="OCR",
        layout="wide",
    )
    st.title("Chinese Scene-Text Recognition Demo")
    st.caption("Detection, recognition, semantic analysis, and experiment summaries in one place.")

    left, right = st.columns((1.2, 1))

    with left:
        st.subheader("Run End-to-End Inference")
        uploaded = st.file_uploader("Upload a scene image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            if st.button("Run OCR Pipeline", type="primary"):
                try:
                    checkpoints = resolve_demo_checkpoints(project_root=root)
                    artifacts = run_uploaded_inference(
                        image_bytes=uploaded.getvalue(),
                        project_root=root,
                        detector_checkpoint=checkpoints["detector"],
                        recognizer_checkpoint=checkpoints["recognizer"],
                    )
                    st.success("End-to-end inference completed.")
                    st.image(str(artifacts.preview_image_path), caption="Pipeline Preview")
                    st.json(load_end_to_end_preview(artifacts.json_path))
                except FileNotFoundError as exc:
                    st.error(str(exc))

        st.subheader("Latest End-to-End Preview")
        preview = load_end_to_end_preview(reports_dir / "eval" / "end_to_end_preview.json")
        if preview:
            st.json(preview)
        else:
            st.info("No end-to-end preview JSON found yet.")

        st.subheader("Semantic Demo")
        demo_report = reports_dir / "demo_assets" / "demo_preview_semantic_eval.md"
        if demo_report.exists():
            st.markdown(demo_report.read_text(encoding="utf-8"))
        else:
            st.info("No demo semantic report found yet.")

    with right:
        st.subheader("Training Reports")
        if files["train_markdown"]:
            selected = st.selectbox(
                "Select a training report",
                files["train_markdown"],
                format_func=lambda path: path.name,
            )
            st.markdown(selected.read_text(encoding="utf-8"))
        else:
            st.info("No training report markdown files found.")

        if files["train_png"]:
            selected_png = st.selectbox(
                "Select a training chart",
                files["train_png"],
                format_func=lambda path: path.name,
            )
            st.image(str(selected_png), caption=selected_png.name)

        st.subheader("EDA Reports")
        if files["eda_markdown"]:
            selected_eda = st.selectbox(
                "Select an EDA report",
                files["eda_markdown"],
                format_func=lambda path: path.name,
            )
            st.markdown(selected_eda.read_text(encoding="utf-8"))
        else:
            st.info("No EDA report markdown files found.")


if __name__ == "__main__":
    render_streamlit_app()
