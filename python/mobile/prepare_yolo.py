"""Convert the existing ACFR COCO splits to YOLO detection labels."""
import argparse
import json
from pathlib import Path

import yaml


def prepare(annotation_dir: Path, image_root: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    names = None
    for split in ("train", "val", "test"):
        source = json.loads((annotation_dir / f"{split}.json").read_text(encoding="utf-8"))
        categories = sorted(source["categories"], key=lambda item: item["id"])
        current_names = [item["name"] for item in categories]
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError(f"Category mapping differs in {split}")
        ids = {item["id"]: index for index, item in enumerate(categories)}
        annotations = {}
        for item in source["annotations"]:
            annotations.setdefault(item["image_id"], []).append(item)
        list_file = output / f"{split}.txt"
        paths = []
        for image in source["images"]:
            path = (image_root / image["file_name"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path.as_posix())
            label = output / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for item in annotations.get(image["id"], []):
                x, y, w, h = item["bbox"]
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid box in {split}: {item}")
                lines.append(f"{ids[item['category_id']]} {(x+w/2)/image['width']:.8f} "
                             f"{(y+h/2)/image['height']:.8f} {w/image['width']:.8f} "
                             f"{h/image['height']:.8f}")
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        list_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    # Ultralytics infers labels by replacing /images/ with /labels/. Use links/copies
    # under a conventional tree, leaving the downloaded source untouched.
    for split in ("train", "val", "test"):
        source = json.loads((annotation_dir / f"{split}.json").read_text(encoding="utf-8"))
        paths = []
        for image in source["images"]:
            src = image_root / image["file_name"]
            relative = Path(image["file_name"])
            dest = output / "images" / split / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                import shutil
                shutil.copy2(src, dest)
            paths.append(dest.resolve().as_posix())
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    config = output / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(output.resolve()),
                                      "train": "train.txt", "val": "val.txt", "test": "test.txt",
                                      "names": dict(enumerate(names))}, allow_unicode=True), encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/acfr_plum/annotations"))
    parser.add_argument("--images", type=Path, default=Path("data/acfr_plum/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/mobile_plum_dataset"))
    args = parser.parse_args()
    print(prepare(args.annotations, args.images, args.output))
