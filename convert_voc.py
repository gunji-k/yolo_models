"""Convert Pascal VOC (VOCdevkit) into the flat images/targets layout.

Source layout (per year):
    VOCdevkit/VOC{2007,2012}/JPEGImages/<id>.jpg
    VOCdevkit/VOC{2007,2012}/Annotations/<id>.xml
    VOCdevkit/VOC{2007,2012}/ImageSets/Main/{trainval,test}.txt

Destination layout:
    VOC_Detection/train/images/<id>.jpg     (VOC2007 trainval + VOC2012 trainval)
    VOC_Detection/train/targets/<id>.csv
    VOC_Detection/test/images/<id>.jpg      (VOC2007 test)
    VOC_Detection/test/targets/<id>.csv

CSV format:
    object,xmin,ymin,xmax,ymax
    <class>,<int>,<int>,<int>,<int>
    ...

Objects with <difficult>1</difficult> are dropped (standard VOC convention:
they are excluded from training and ignored by the mAP evaluation protocol).
Images left with no objects after filtering are skipped entirely.
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_annotation(xml_path: Path):
    """Return list of (name, xmin, ymin, xmax, ymax) for non-difficult objects."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        if int(obj.findtext("difficult", "0")) == 1:
            continue
        bb = obj.find("bndbox")
        boxes.append((
            obj.findtext("name").strip(),
            int(float(bb.findtext("xmin"))),
            int(float(bb.findtext("ymin"))),
            int(float(bb.findtext("xmax"))),
            int(float(bb.findtext("ymax"))),
        ))
    return boxes


def read_split_ids(split_file: Path):
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def convert_split(voc_root: Path, year: str, split: str, out_images: Path, out_targets: Path,
                  limit: int | None = None):
    """Process one (year, split) pair into the destination directories."""
    year_dir = voc_root / f"VOC{year}"
    split_file = year_dir / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        print(f"  [skip] {split_file} not found", file=sys.stderr)
        return 0, 0

    ids = read_split_ids(split_file)
    if limit is not None:
        ids = ids[:limit]
    written = 0
    skipped_empty = 0

    for image_id in ids:
        xml_path = year_dir / "Annotations" / f"{image_id}.xml"
        jpg_path = year_dir / "JPEGImages" / f"{image_id}.jpg"
        if not xml_path.exists() or not jpg_path.exists():
            print(f"  [warn] missing source for {image_id}", file=sys.stderr)
            continue

        boxes = parse_annotation(xml_path)
        if not boxes:
            skipped_empty += 1
            continue

        shutil.copyfile(jpg_path, out_images / f"{image_id}.jpg")
        rows = ["object,xmin,ymin,xmax,ymax"]
        rows.extend(f"{n},{a},{b},{c},{d}" for n, a, b, c, d in boxes)
        (out_targets / f"{image_id}.csv").write_text("\n".join(rows))
        written += 1

    print(f"  VOC{year}/{split}: wrote {written}, skipped {skipped_empty} (no non-difficult objects)")
    return written, skipped_empty


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/home/imerit/data/VOCdevkit"),
                    help="Path to VOCdevkit (containing VOC2007/, VOC2012/)")
    ap.add_argument("--dst", type=Path, default=Path("/home/imerit/data/pascal_voc_dataset/VOC_Detection"),
                    help="Output root (will contain train/ and test/)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N ids per (year, split). Useful for sanity checks.")
    args = ap.parse_args()

    train_jobs = [("2007", "trainval"), ("2012", "trainval")]
    test_jobs = [("2007", "test")]

    for split, jobs in (("train", train_jobs), ("test", test_jobs)):
        out_images = args.dst / split / "images"
        out_targets = args.dst / split / "targets"
        out_images.mkdir(parents=True, exist_ok=True)
        out_targets.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] -> {args.dst / split}")
        total = 0
        for year, sp in jobs:
            written, _ = convert_split(args.src, year, sp, out_images, out_targets, args.limit)
            total += written
        print(f"[{split}] total written: {total}")


if __name__ == "__main__":
    main()
