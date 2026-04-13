from __future__ import annotations

from dltr.data.types import DataConfig, DatasetInventory, HardCaseMetadata, ValidationSummary


def render_eda_markdown(
    config: DataConfig,
    validation: ValidationSummary,
    inventories: dict[str, DatasetInventory],
    hardcases: dict[str, HardCaseMetadata],
) -> str:
    lines: list[str] = []
    lines.append("# Chinese Scene-Text Dataset EDA Summary")
    lines.append("")
    lines.append("## Validation Status")
    lines.append("")
    lines.append(f"- Overall status: {'PASS' if validation.ok else 'ACTION_REQUIRED'}")
    lines.append(f"- Required datasets missing: {len(validation.missing_required)}")
    lines.append(f"- Invalid dataset locations: {len(validation.invalid_locations)}")
    lines.append("")
    lines.append("## Dataset Matrix")
    lines.append("")
    lines.append("| Dataset | Required | Exists | Under data/ | Images | Labels | Label Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    validation_lookup = {item.name: item for item in validation.dataset_results}
    for spec in config.datasets:
        result = validation_lookup[spec.name]
        inventory = inventories.get(spec.name)
        if inventory is None:
            image_count = 0
            label_count = 0
            ratio = 0.0
        else:
            image_count = inventory.total_images
            label_count = inventory.total_labels
            ratio = inventory.label_presence_ratio
        lines.append(
            f"| {spec.name} | {result.required} | {result.exists} | {result.within_data_dir} | "
            f"{image_count} | {label_count} | {ratio:.3f} |"
        )

    lines.append("")
    lines.append("## Hard-Case Heuristic Snapshot")
    lines.append("")
    for dataset_name, metadata in sorted(hardcases.items()):
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append(f"- Total images scanned: {metadata.total_images}")
        lines.append(f"- Deep path ratio: {metadata.deep_path_ratio:.3f}")
        lines.append(f"- Long filename ratio: {metadata.long_name_ratio:.3f}")
        top = sorted(metadata.keyword_hit_ratio.items(), key=lambda item: item[1], reverse=True)[:3]
        if top:
            lines.append(
                "- Top keyword groups: " + ", ".join([f"{name}={value:.3f}" for name, value in top])
            )
        for recommendation in metadata.recommendations:
            lines.append(f"- Recommendation: {recommendation}")
        lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Fill missing required dataset paths before training.")
    lines.append("- Regenerate EDA after each new dataset import.")
    lines.append("- Use hard-case ratios to define targeted augmentation policies.")
    lines.append("")
    return "\n".join(lines)
