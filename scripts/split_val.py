"""
Split Validation Set - Memisahkan sebagian data training menjadi validation set
"""

import os
import random
import shutil
import argparse


def split_validation(
    base_dir: str = "dataset",
    val_ratio: float = 0.1,
):
    """
    Memisahkan sebagian data training menjadi validation set
    
    Args:
        base_dir: Base directory dataset
        val_ratio: Proporsi data untuk validation (0.1 = 10%)
    """
    train_img = os.path.join(base_dir, "images/train")
    val_img = os.path.join(base_dir, "images/val")
    train_lbl = os.path.join(base_dir, "labels/train")
    val_lbl = os.path.join(base_dir, "labels/val")

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    # Ambil gambar secara acak untuk validasi
    files = [f for f in os.listdir(train_img) if f.endswith((".jpg", ".png"))]
    val_count = max(1, int(len(files) * val_ratio))

    selected = random.sample(files, val_count)
    for f in selected:
        # Move image
        shutil.move(os.path.join(train_img, f), os.path.join(val_img, f))
        # Move label
        label = f.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(os.path.join(train_lbl, label)):
            shutil.move(os.path.join(train_lbl, label), os.path.join(val_lbl, label))

    print(f"✅ Pindah {val_count} gambar ke folder val.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset menjadi train dan validation")
    parser.add_argument("--base-dir", type=str, default="dataset", help="Base dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (0.1 = 10%)")

    args = parser.parse_args()
    split_validation(base_dir=args.base_dir, val_ratio=args.val_ratio)

