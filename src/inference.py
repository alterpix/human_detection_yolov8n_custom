"""
Human Detection & Tracking dengan YOLOv8
Script untuk inference real-time menggunakan webcam atau video file
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    """Tracker untuk mengikuti objek manusia berdasarkan centroid"""
    
    def __init__(self, maxDisappeared=50, maxDistance=100):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.previousCentroids = {}

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.previousCentroids[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.previousCentroids[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                old_centroid = self.previousCentroids[objectID]
                new_centroid = inputCentroids[col]
                smooth_centroid = (
                    0.7 * np.array(old_centroid) + 0.3 * np.array(new_centroid)
                )
                self.objects[objectID] = tuple(smooth_centroid.astype(int))
                self.previousCentroids[objectID] = new_centroid
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects


def run_inference(
    model_path: str = "models/best.pt",
    source: str = "0",  # "0" untuk webcam, atau path ke video file
    confidence: float = 0.55,
    iou: float = 0.45,
    resize_width: int = 960,
    resize_height: int = 540,
):
    """
    Menjalankan inference dengan YOLO model
    
    Args:
        model_path: Path ke model weights (.pt)
        source: Source video (0 untuk webcam, atau path ke file)
        confidence: Confidence threshold
        iou: IOU threshold untuk NMS
        resize_width: Lebar frame yang diresize
        resize_height: Tinggi frame yang diresize
    """
    # Load model
    model = YOLO(model_path)
    ct = CentroidTracker(maxDisappeared=30, maxDistance=120)

    # Buka video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Gagal membuka video source")
        return

    print("✅ Inference dimulai. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_width, resize_height))
        results = model(frame, stream=True, conf=confidence, iou=iou)
        rects = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    rects.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

        # Update tracker
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            cv2.putText(
                frame,
                f"ID {objectID}",
                (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        total_people = len(objects)
        cv2.putText(
            frame,
            f"Jumlah orang: {total_people}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        cv2.imshow("Human Detection & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human Detection Inference")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path ke model weights")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 untuk webcam)")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--width", type=int, default=960, help="Resize width")
    parser.add_argument("--height", type=int, default=540, help="Resize height")

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        source=args.source,
        confidence=args.conf,
        iou=args.iou,
        resize_width=args.width,
        resize_height=args.height,
    )

