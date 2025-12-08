"""
Render Dataset - Visualisasi bounding box pada dataset
Tool untuk membuat preview visual dari dataset dengan bounding box yang sudah dilabel
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            boxes.append((int(cls), cx, cy, w, h))
    return boxes


def ensure_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_single(image_path: Path, label_path: Path, output_path: Path, palette, font):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    boxes = read_yolo_labels(label_path)

    draw = ImageDraw.Draw(image, "RGBA")
    thickness = max(2, int(min(w, h) * 0.0025))

    for idx, (cls, cx, cy, bw, bh) in enumerate(boxes):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = palette[idx % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=0, fill=color + (40,))
        tag = f"{cls}:{idx}"
        bbox = draw.textbbox((0, 0), tag, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color + (180,))
        draw.text((x1 + 2, y1 - text_h - 2), tag, fill=(0, 0, 0), font=font)

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return len(boxes)


def main():
    parser = argparse.ArgumentParser(
        description="Render seluruh bounding box dataset untuk evaluasi visual."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images/train"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/labels/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("renders/train"))
    parser.add_argument("--limit", type=int, default=None, help="Batas jumlah gambar")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Lewati gambar tanpa label supaya folder lebih ringkas",
    )
    args = parser.parse_args()

    palette = [
        (235, 64, 52),
        (52, 168, 83),
        (66, 133, 244),
        (251, 188, 5),
        (171, 71, 188),
        (0, 188, 212),
    ]
    font = ensure_font(18)

    images = list_images(args.images_dir)
    if args.limit:
        images = images[: args.limit]

    total = len(images)
    if total == 0:
        print("Tidak ada gambar ditemukan.")
        return

    rendered = 0
    skipped = 0
    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(args.images_dir)
        label_path = args.labels_dir / rel.with_suffix(".txt")
        out_path = args.output_dir / rel
        count = render_single(img_path, label_path, out_path, palette, font)
        if count == 0 and args.skip_empty:
            out_path.unlink(missing_ok=True)
            skipped += 1
        else:
            rendered += 1
        if idx % 25 == 0 or idx == total:
            print(f"[{idx}/{total}] -> rendered: {rendered}, skipped: {skipped}")

    print(f"Selesai. Preview tersimpan di: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

