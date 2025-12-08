"""
Label Review & Editor untuk dataset YOLO
Tool untuk menganalisis dan mengedit label YOLO secara interaktif
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


YOLO_LINE_ERROR = "Format label YOLO harus: <cls> <cx> <cy> <w> <h>"


def list_media_files(folder: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{YOLO_LINE_ERROR} -> {label_path}: {line}")
            cls, cx, cy, w, h = map(float, parts)
            boxes.append((int(cls), cx, cy, w, h))
    return boxes


def write_yolo_labels(label_path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(box, width, height):
    cls, cx, cy, w, h = box
    x1 = int((cx - w / 2) * width)
    y1 = int((cy - h / 2) * height)
    x2 = int((cx + w / 2) * width)
    y2 = int((cy + h / 2) * height)
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, width, height):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cls, cx / width, cy / height, w / width, h / height


def analyze_dataset(images_dir: Path, labels_dir: Path):
    img_files = list_media_files(images_dir)
    lbl_files = sorted(labels_dir.glob("*.txt"))
    img_names = {f.stem for f in img_files}
    lbl_names = {f.stem for f in lbl_files}

    missing_labels = sorted(img_names - lbl_names)
    missing_images = sorted(lbl_names - img_names)

    total_boxes = 0
    widths, heights, areas, ratios = [], [], [], []

    for label_path in lbl_files:
        boxes = read_yolo_labels(label_path)
        total_boxes += len(boxes)
        for _, _, _, w, h in boxes:
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            if h > 0:
                ratios.append(w / h)

    summary = {
        "jumlah_gambar": len(img_files),
        "jumlah_label": len(lbl_files),
        "total_bbox": total_boxes,
        "gambar_tanpa_label": missing_labels[:20],
        "label_tanpa_gambar": missing_images[:20],
        "proporsi_label": f"{len(lbl_files) / len(img_files):.2f}" if img_files else "0",
        "rata2_lebar_bbox": float(np.mean(widths)) if widths else 0.0,
        "rata2_tinggi_bbox": float(np.mean(heights)) if heights else 0.0,
        "rata2_luas_bbox": float(np.mean(areas)) if areas else 0.0,
        "median_aspect_ratio": float(np.median(ratios)) if ratios else 0.0,
    }
    return summary


class LabelEditor:
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        start_index: int = 0,
        display_scale: float = 1.0,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = list_media_files(images_dir)
        if not self.images:
            raise RuntimeError("Tidak ada gambar ditemukan di folder yang diberikan.")
        self.index = max(0, min(start_index, len(self.images) - 1))
        self.window_name = "Label Editor"
        self.selected_idx: Optional[int] = None
        self.history = {}
        self.history_limit = 100
        self.display_scale = max(0.3, min(2.5, display_scale))
        self.min_scale = 0.3
        self.max_scale = 3.0
        self.shade_boxes = True

    def current_paths(self):
        image_path = self.images[self.index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        return image_path, label_path

    def _history_key(self, image_path: Path):
        return image_path.stem

    def _copy_boxes(self, boxes):
        return [tuple(b) for b in boxes]

    def _ensure_history(self, key):
        if key not in self.history:
            self.history[key] = {"undo": [], "redo": []}

    def _push_history(self, key, boxes):
        self._ensure_history(key)
        snapshot = self._copy_boxes(boxes)
        undo_stack = self.history[key]["undo"]
        undo_stack.append(snapshot)
        if len(undo_stack) > self.history_limit:
            undo_stack.pop(0)
        self.history[key]["redo"].clear()

    def undo_action(self, key, boxes):
        self._ensure_history(key)
        hist = self.history[key]
        if not hist["undo"]:
            print("Tidak ada langkah undo.")
            return boxes
        hist["redo"].append(self._copy_boxes(boxes))
        previous = hist["undo"].pop()
        boxes = self._copy_boxes(previous)
        self.selected_idx = None if not boxes else min(self.selected_idx or 0, len(boxes) - 1)
        return boxes

    def redo_action(self, key, boxes):
        self._ensure_history(key)
        hist = self.history[key]
        if not hist["redo"]:
            print("Tidak ada langkah redo.")
            return boxes
        hist["undo"].append(self._copy_boxes(boxes))
        next_state = hist["redo"].pop()
        boxes = self._copy_boxes(next_state)
        self.selected_idx = None if not boxes else min(self.selected_idx or 0, len(boxes) - 1)
        return boxes

    def _ensure_selection(self, boxes):
        if not boxes:
            self.selected_idx = None
            return
        if self.selected_idx is None:
            self.selected_idx = 0
        self.selected_idx = max(0, min(self.selected_idx, len(boxes) - 1))

    def _pixel_step(self, image_shape):
        h, w = image_shape[:2]
        return max(1, int(min(h, w) * 0.01))

    def draw(self, image, boxes):
        canvas = image.copy()
        base = canvas.copy()
        h, w = canvas.shape[:2]
        self._ensure_selection(boxes)
        thickness = max(2, int(round(min(h, w) * 0.002)))
        for idx, box in enumerate(boxes):
            cls, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
            color = (0, 255, 255) if idx == self.selected_idx else (0, 255, 0)
            if self.shade_boxes:
                cv2.rectangle(base, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                canvas,
                f"{idx}:{cls}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        if self.shade_boxes:
            alpha = 0.25
            canvas = cv2.addWeighted(base, alpha, canvas, 1 - alpha, 0)
        info = f"{self.index + 1}/{len(self.images)} | total box: {len(boxes)}"
        cv2.putText(canvas, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        controls = "[n/p] gambar  [t] tambah  [r] redraw  [s] simpan  [u/y] undo/redo  [z/x] zoom  [h] shade  [q] keluar"
        controls2 = "[[]/]] pilih box  [d/del] hapus  [I/K/J/L atau ↑↓←→] geser  [-/=] lebar  [,/ .] tinggi"
        cv2.putText(canvas, controls, (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        cv2.putText(canvas, controls2, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        return canvas

    def refresh_view(self, image, boxes):
        canvas = self.draw(image, boxes)
        if self.display_scale != 1.0:
            canvas = cv2.resize(
                canvas,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA if self.display_scale < 1 else cv2.INTER_LINEAR,
            )
        cv2.imshow(self.window_name, canvas)

    def add_box(self, image, boxes, history_key):
        roi = cv2.selectROI(self.window_name, image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        if w == 0 or h == 0:
            return boxes
        self._push_history(history_key, boxes)
        bbox = xyxy_to_yolo(0, x, y, x + w, y + h, image.shape[1], image.shape[0])
        boxes.append(bbox)
        self.selected_idx = len(boxes) - 1
        return boxes

    def redraw_box(self, image, boxes, history_key):
        if not boxes:
            print("Tidak ada box untuk diedit.")
            return boxes
        self._ensure_selection(boxes)
        if self.selected_idx is None:
            return boxes
        roi = cv2.selectROI(self.window_name, image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        if w == 0 or h == 0:
            return boxes
        self._push_history(history_key, boxes)
        boxes[self.selected_idx] = xyxy_to_yolo(0, x, y, x + w, y + h, image.shape[1], image.shape[0])
        return boxes

    def delete_box(self, boxes, history_key):
        if not boxes:
            print("Tidak ada box untuk dihapus.")
            return boxes
        self._ensure_selection(boxes)
        if self.selected_idx is None:
            return boxes
        self._push_history(history_key, boxes)
        boxes.pop(self.selected_idx)
        if not boxes:
            self.selected_idx = None
        else:
            self.selected_idx = min(self.selected_idx, len(boxes) - 1)
        return boxes

    def save_boxes(self, label_path, boxes):
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_labels(label_path, boxes)
        print(f"✅ Label tersimpan: {label_path}")

    def adjust_scale(self, delta):
        new_scale = max(self.min_scale, min(self.max_scale, self.display_scale + delta))
        if abs(new_scale - self.display_scale) < 1e-3:
            return
        self.display_scale = new_scale
        print(f"🔍 Zoom set ke {self.display_scale:.2f}x")

    def cycle_selection(self, direction, boxes):
        if not boxes:
            self.selected_idx = None
            return
        self._ensure_selection(boxes)
        if self.selected_idx is None:
            return
        self.selected_idx = (self.selected_idx + direction) % len(boxes)

    def nudge_selected(self, boxes, image, history_key, move=(0, 0), scale=(0, 0)):
        if not boxes:
            return boxes
        self._ensure_selection(boxes)
        if self.selected_idx is None:
            return boxes

        h, w = image.shape[:2]
        cls, x1, y1, x2, y2 = yolo_to_xyxy(boxes[self.selected_idx], w, h)

        dx, dy = move
        sx, sy = scale

        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy

        if sx != 0:
            x1 -= sx
            x2 += sx
        if sy != 0:
            y1 -= sy
            y2 += sy

        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 1, min(w - 1, x2))
        y2 = max(y1 + 1, min(h - 1, y2))

        self._push_history(history_key, boxes)
        boxes[self.selected_idx] = xyxy_to_yolo(cls, x1, y1, x2, y2, w, h)
        return boxes

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while True:
            image_path, label_path = self.current_paths()
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Gagal membuka {image_path}, lanjut ke gambar berikutnya.")
                self.index = min(self.index + 1, len(self.images) - 1)
                continue
            boxes = read_yolo_labels(label_path)
            history_key = self._history_key(image_path)
            self._ensure_history(history_key)
            self.refresh_view(image, boxes)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                self.index = min(self.index + 1, len(self.images) - 1)
                self.selected_idx = None
            elif key == ord("p"):
                self.index = max(self.index - 1, 0)
                self.selected_idx = None
            elif key in (ord("]"), 93):
                self.cycle_selection(1, boxes)
            elif key in (ord("["), 91):
                self.cycle_selection(-1, boxes)
            elif key in (ord("h"), ord("H")):
                self.shade_boxes = not self.shade_boxes
                print(f"🎨 Shade boxes: {'ON' if self.shade_boxes else 'OFF'}")
                self.refresh_view(image, boxes)
                continue
            elif key in (ord("z"), ord("Z")):
                self.adjust_scale(0.1)
                self.refresh_view(image, boxes)
                continue
            elif key in (ord("x"), ord("X")):
                self.adjust_scale(-0.1)
                self.refresh_view(image, boxes)
                continue
            elif key == ord("t"):
                boxes = self.add_box(image, boxes, history_key)
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key == ord("r"):
                boxes = self.redraw_box(image, boxes, history_key)
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("d"), 8, 127, 255):
                boxes = self.delete_box(boxes, history_key)
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key == ord("s"):
                self.save_boxes(label_path, boxes)
            elif key in (ord("u"), ord("U")):
                boxes = self.undo_action(history_key, boxes)
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("y"), ord("Y")):
                boxes = self.redo_action(history_key, boxes)
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("i"), ord("I"), 82):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, move=(0, -step))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("k"), ord("K"), 84):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, move=(0, step))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("j"), ord("J"), 81):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, move=(-step, 0))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("l"), ord("L"), 83):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, move=(step, 0))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("-"), ord("_")):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, scale=(-step, 0))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key in (ord("="), ord("+")):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, scale=(step, 0))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key == ord(","):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, scale=(0, -step))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            elif key == ord("."):
                step = self._pixel_step(image.shape)
                boxes = self.nudge_selected(boxes, image, history_key, scale=(0, step))
                self.save_boxes(label_path, boxes)
                self.refresh_view(image, boxes)
            else:
                print(
                    "[n/p] gambar  [t] tambah  [r] redraw  [d/del] hapus  [s] simpan  [u/y] undo/redo\n"
                    "[[]/]] pilih box  [I/K/J/L atau ↑↓←→] geser  [-/=] ubah lebar  [,/ .] ubah tinggi."
                )

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Analisis & editor sederhana untuk label YOLO hasil auto-labeling."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images/train"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/labels/train"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--edit", action="store_true")
    args = parser.parse_args()

    if args.analyze_only or not args.edit:
        summary = analyze_dataset(args.images_dir, args.labels_dir)
        print("==== Rangkuman Dataset ====")
        for k, v in summary.items():
            print(f"{k}: {v}")

    if args.edit:
        editor = LabelEditor(
            args.images_dir, args.labels_dir, args.start_index, args.display_scale
        )
        editor.run()


if __name__ == "__main__":
    main()

