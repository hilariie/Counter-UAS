"""Convert DUT-Anti-UAV (Pascal VOC) into Ultralytics YOLO layout.

Before:
    images/<split>/img/*.jpg
    images/<split>/xml/*.xml   (Pascal VOC, single class "UAV")

After:
    images/<split>/*.jpg
    labels/<split>/*.txt       (YOLO: class cx cy w h, normalized)
    data.yaml
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CLASSES = {"UAV": 0}


def voc_to_yolo(xml_path: Path, out_path: Path) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    rows: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        cls = CLASSES.get(name)
        if cls is None:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        # Clamp to image bounds in case of slight overflow.
        xmin, xmax = max(0.0, min(xmin, xmax)), min(w, max(xmin, xmax))
        ymin, ymax = max(0.0, min(ymin, ymax)), min(h, max(ymin, ymax))
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        if bw <= 0 or bh <= 0:
            continue
        rows.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def process_split(split: str) -> tuple[int, int, int]:
    img_src = ROOT / "images" / split / "img"
    xml_src = ROOT / "images" / split / "xml"
    img_dst = ROOT / "images" / split
    lbl_dst = ROOT / "labels" / split
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_images = n_labels = n_boxes = 0

    if xml_src.is_dir():
        for xml in xml_src.glob("*.xml"):
            n_boxes += voc_to_yolo(xml, lbl_dst / f"{xml.stem}.txt")
            n_labels += 1

    if img_src.is_dir():
        for img in img_src.iterdir():
            if img.is_file():
                shutil.move(str(img), str(img_dst / img.name))
                n_images += 1
        img_src.rmdir()

    if xml_src.is_dir():
        shutil.rmtree(xml_src)

    return n_images, n_labels, n_boxes


def write_data_yaml() -> None:
    yaml_path = ROOT / "data.yaml"
    content = (
        f"path: {ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: UAV\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"Wrote {yaml_path}")


def main() -> None:
    for split in SPLITS:
        imgs, lbls, boxes = process_split(split)
        print(f"{split}: moved {imgs} images, wrote {lbls} label files ({boxes} boxes)")
    write_data_yaml()


if __name__ == "__main__":
    main()
