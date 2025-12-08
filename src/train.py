"""
Training script untuk YOLOv8 Human Detection
"""

from ultralytics import YOLO
import argparse


def train(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "config/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "human_detection",
):
    """
    Melatih model YOLOv8 untuk human detection
    
    Args:
        model_name: Nama model pretrained (yolov8n.pt, yolov8s.pt, dll)
        data_yaml: Path ke file data.yaml
        epochs: Jumlah epochs
        imgsz: Ukuran gambar untuk training
        batch: Batch size
        device: Device untuk training ('0' untuk GPU, 'cpu' untuk CPU)
        project: Project directory untuk menyimpan hasil
        name: Nama experiment
    """
    # Load model
    model = YOLO(model_name)

    # Training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        save_period=10,
    )

    print(f"✅ Training selesai! Model tersimpan di: {project}/{name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Human Detection Model")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model")
    parser.add_argument("--data", type=str, default="config/data.yaml", help="Path ke data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 untuk GPU, cpu untuk CPU)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="human_detection", help="Experiment name")

    args = parser.parse_args()
    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

