"""
Prepare Dataset - Script untuk mempersiapkan dataset dari pretrained model
Menggunakan YOLOv8 pretrained untuk auto-labeling dataset
"""

from ultralytics import YOLO
import os
import cv2
import argparse


def auto_label(
    model_path: str = "yolov8n.pt",
    source_dir: str = "dataset_original/1",
    output_img_dir: str = "dataset/images/train",
    output_lbl_dir: str = "dataset/labels/train",
    class_id: int = 0,  # 0 untuk person di COCO
):
    """
    Auto-labeling menggunakan pretrained YOLO model
    
    Args:
        model_path: Path ke pretrained model
        source_dir: Folder sumber gambar
        output_img_dir: Folder output untuk gambar
        output_lbl_dir: Folder output untuk label
        class_id: Class ID yang ingin dilabel (0 untuk person)
    """
    model = YOLO(model_path)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    count = 0
    for img_name in os.listdir(source_dir):
        if img_name.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(source_dir, img_name)
            results = model(path)
            boxes = results[0].boxes.xywhn
            cls = results[0].boxes.cls

            if len(boxes) > 0:
                label_path = os.path.join(output_lbl_dir, img_name.rsplit(".", 1)[0] + ".txt")
                with open(label_path, "w") as f:
                    for box, c in zip(boxes, cls):
                        if int(c) == class_id:
                            f.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                
                # Copy image
                img = cv2.imread(path)
                cv2.imwrite(os.path.join(output_img_dir, img_name), img)
                count += 1
                if count % 10 == 0:
                    print(f"✅ Processed {count} images...")

    print(f"✅ Selesai! {count} gambar telah dilabel.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-labeling dataset dengan YOLO pretrained")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model path")
    parser.add_argument("--source", type=str, default="dataset_original/1", help="Source image directory")
    parser.add_argument("--output-img", type=str, default="dataset/images/train", help="Output image directory")
    parser.add_argument("--output-lbl", type=str, default="dataset/labels/train", help="Output label directory")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID to label (0 for person)")

    args = parser.parse_args()
    auto_label(
        model_path=args.model,
        source_dir=args.source,
        output_img_dir=args.output_img,
        output_lbl_dir=args.output_lbl,
        class_id=args.class_id,
    )

