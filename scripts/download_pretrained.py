"""
Download Pretrained Model - Download YOLOv8 pretrained model
"""

from ultralytics import YOLO
import argparse


def download_pretrained(model_name: str = "yolov8n.pt"):
    """
    Download pretrained YOLOv8 model
    
    Args:
        model_name: Nama model (yolov8n.pt, yolov8s.pt, yolov8m.pt, dll)
    """
    print(f"📥 Downloading {model_name}...")
    model = YOLO(model_name)
    print(f"✅ Model {model_name} berhasil didownload!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLOv8 pretrained model")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
    )

    args = parser.parse_args()
    download_pretrained(args.model)

