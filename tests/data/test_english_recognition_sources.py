from pathlib import Path

from dltr.data.english_recognition_sources import (
    RecognitionSourceRecord,
    load_icdar_gt_records,
    load_iiit5k_mat_records,
    load_mjsynth_records,
    load_pairs_records,
    load_svt_xml_records,
    parse_mjsynth_label_from_stem,
)


def test_parse_mjsynth_label_from_stem_supports_common_patterns() -> None:
    assert parse_mjsynth_label_from_stem("12_hello_34") == "hello"
    assert parse_mjsynth_label_from_stem("hello_12_34") == "hello"
    assert parse_mjsynth_label_from_stem("12_new_york_34") == "new_york"


def test_load_mjsynth_records_extracts_text_from_filenames(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mjsynth"
    image_path = dataset_root / "train" / "000001" / "12_coffee_shop_34.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")

    records = load_mjsynth_records(
        dataset_name="mjsynth_train",
        dataset_root=dataset_root,
        image_extensions={".jpg"},
        split="train",
    )

    assert records == [
        RecognitionSourceRecord(
            dataset="mjsynth_train",
            image_path=image_path.resolve(),
            text="coffee_shop",
            split="train",
        )
    ]


def test_load_pairs_records_reads_tab_separated_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iiit5k"
    image_path = dataset_root / "test" / "word_1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    pairs_path = dataset_root / "labels.tsv"
    pairs_path.write_text("test/word_1.png\tMANGO\nmissing.png\tSKIP\n", encoding="utf-8")

    records = load_pairs_records(
        dataset_name="iiit5k_test",
        dataset_root=dataset_root,
        pairs_path=pairs_path,
        split="test",
    )

    assert records == [
        RecognitionSourceRecord(
            dataset="iiit5k_test",
            image_path=image_path.resolve(),
            text="MANGO",
            split="test",
        )
    ]


def test_load_icdar_gt_records_supports_quoted_and_plain_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "icdar13"
    first_image = dataset_root / "images" / "word_1.jpg"
    second_image = dataset_root / "images" / "word_2.jpg"
    first_image.parent.mkdir(parents=True, exist_ok=True)
    first_image.write_bytes(b"img")
    second_image.write_bytes(b"img")
    gt_path = dataset_root / "gt.txt"
    gt_path.write_text(
        'images/word_1.jpg,"HELLO, WORLD"\nimages/word_2.jpg,COFFEE\n',
        encoding="utf-8",
    )

    records = load_icdar_gt_records(
        dataset_name="icdar13_test",
        dataset_root=dataset_root,
        gt_path=gt_path,
        split="test",
    )

    assert records == [
        RecognitionSourceRecord(
            dataset="icdar13_test",
            image_path=first_image.resolve(),
            text="HELLO, WORLD",
            split="test",
        ),
        RecognitionSourceRecord(
            dataset="icdar13_test",
            image_path=second_image.resolve(),
            text="COFFEE",
            split="test",
        ),
    ]


def test_load_iiit5k_mat_records_reads_mat_payload(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "IIIT5K"
    image_path = dataset_root / "test" / "word_1.png"
    mat_path = dataset_root / "testdata.mat"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    mat_path.write_bytes(b"mat")

    def _fake_loader(_: Path) -> dict[str, object]:
        return {
            "testdata": {
                "ImgName": ["test/word_1.png", "missing.png"],
                "GroundTruth": ["OpenAI", "SKIP"],
            }
        }

    monkeypatch.setattr(
        "dltr.data.english_recognition_sources._load_iiit5k_mat_payload",
        _fake_loader,
    )

    records = load_iiit5k_mat_records(
        dataset_name="iiit5k_test",
        dataset_root=dataset_root,
        mat_path=mat_path,
        split="test",
    )

    assert records == [
        RecognitionSourceRecord(
            dataset="iiit5k_test",
            image_path=image_path.resolve(),
            text="OpenAI",
            split="test",
        )
    ]


def test_load_svt_xml_records_crops_word_images(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    dataset_root = tmp_path / "svt1"
    image_path = dataset_root / "img" / "demo.jpg"
    crop_dir = tmp_path / "crops"
    xml_path = dataset_root / "train.xml"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 40), (0, 0, 0), -1)
    cv2.imwrite(str(image_path), image)
    xml_path.write_text(
        """
<tagset>
  <image>
    <imageName>img/demo.jpg</imageName>
    <taggedRectangles>
      <taggedRectangle x="10" y="10" width="40" height="30">
        <tag>SHOP</tag>
      </taggedRectangle>
    </taggedRectangles>
  </image>
</tagset>
""".strip(),
        encoding="utf-8",
    )

    records = load_svt_xml_records(
        dataset_name="svt_train",
        dataset_root=dataset_root,
        xml_path=xml_path,
        crop_output_dir=crop_dir,
        split="train",
    )

    assert len(records) == 1
    assert records[0].text == "SHOP"
    assert records[0].image_path.exists()
    assert records[0].image_path.parent == crop_dir
