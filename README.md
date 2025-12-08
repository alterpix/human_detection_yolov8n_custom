# Human Detection dengan YOLOv8n Custom

Project untuk deteksi dan tracking manusia menggunakan YOLOv8 dengan model custom yang telah dilatih.

## 📋 Fitur

- ✅ Deteksi manusia real-time menggunakan webcam atau video file
- ✅ Tracking multiple person dengan Centroid Tracker
- ✅ Counting jumlah orang dalam frame
- ✅ Tools untuk review dan edit label dataset
- ✅ Visualisasi dataset dengan bounding box

## 🚀 Instalasi

1. Clone repository ini:
```bash
git clone <repository-url>
cd human_detection_yolov8n_custom
```

2. Buat virtual environment (disarankan):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Struktur Project

```
human_detection_yolov8n_custom/
├── src/                    # Source code utama
│   ├── inference.py       # Script untuk inference
│   └── train.py           # Script untuk training
├── utils/                  # Utility tools
│   ├── label_review.py    # Tool untuk review/edit label
│   └── render_dataset.py   # Visualisasi dataset
├── scripts/                # Helper scripts
│   ├── prepare_dataset.py  # Auto-labeling dataset
│   └── split_val.py       # Split train/val
├── config/                 # Konfigurasi
│   └── data.yaml          # Dataset configuration
├── models/                 # Model weights (tidak di-commit)
│   └── best.pt            # Model terbaik hasil training
├── dataset/               # Dataset (tidak di-commit)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🎯 Penggunaan

### 1. Inference (Deteksi Real-time)

Jalankan inference menggunakan webcam (default):
```bash
python src/inference.py --model models/best.pt --source 0
```

Atau menggunakan video file:
```bash
python src/inference.py --model models/best.pt --source path/to/video.mp4
```

**Parameter:**
- `--model`: Path ke model weights (default: `models/best.pt`)
- `--source`: Video source (0 untuk webcam, atau path ke file video)
- `--conf`: Confidence threshold (default: 0.55)
- `--iou`: IOU threshold untuk NMS (default: 0.45)
- `--width`: Resize width (default: 960)
- `--height`: Resize height (default: 540)

### 2. Training Model

Training model custom:
```bash
python src/train.py \
    --model yolov8n.pt \
    --data config/data.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

**Parameter:**
- `--model`: Pretrained model (yolov8n.pt, yolov8s.pt, dll)
- `--data`: Path ke data.yaml
- `--epochs`: Jumlah epochs
- `--batch`: Batch size
- `--device`: Device untuk training ('0' untuk GPU, 'cpu' untuk CPU)

### 3. Review & Edit Label

Analisis dataset:
```bash
python utils/label_review.py \
    --images-dir dataset/images/train \
    --labels-dir dataset/labels/train \
    --analyze-only
```

Edit label secara interaktif:
```bash
python utils/label_review.py \
    --images-dir dataset/images/train \
    --labels-dir dataset/labels/train \
    --edit
```

**Kontrol Keyboard:**
- `n/p`: Next/Previous image
- `t`: Tambah bounding box baru
- `r`: Redraw bounding box yang dipilih
- `d` atau `Del`: Hapus bounding box
- `[` / `]`: Pilih box sebelumnya/selanjutnya
- `I/K/J/L` atau `↑↓←→`: Geser box
- `-` / `=`: Ubah lebar box
- `,` / `.`: Ubah tinggi box
- `u` / `y`: Undo/Redo
- `s`: Simpan
- `z` / `x`: Zoom in/out
- `h`: Toggle shade boxes
- `q`: Keluar

### 4. Visualisasi Dataset

Render dataset dengan bounding box:
```bash
python utils/render_dataset.py \
    --images-dir dataset/images/train \
    --labels-dir dataset/labels/train \
    --output-dir renders/train
```

### 5. Prepare Dataset

Auto-labeling menggunakan pretrained model:
```bash
python scripts/prepare_dataset.py \
    --model yolov8n.pt \
    --source dataset_original/1 \
    --output-img dataset/images/train \
    --output-lbl dataset/labels/train
```

Split dataset menjadi train dan validation:
```bash
python scripts/split_val.py \
    --base-dir dataset \
    --val-ratio 0.1
```

## 📊 Dataset

Dataset harus mengikuti struktur YOLO format:
```
dataset/
├── images/
│   ├── train/    # Training images
│   └── val/       # Validation images
└── labels/
    ├── train/     # Training labels (YOLO format)
    └── val/       # Validation labels (YOLO format)
```

Format label YOLO: `<class_id> <cx> <cy> <width> <height>` (normalized)

## 🔧 Konfigurasi

Edit `config/data.yaml` untuk mengubah konfigurasi dataset:
```yaml
train: dataset/images/train
val: dataset/images/val
nc: 1
names: ['person']
```

## 📝 Catatan

- Model weights (`*.pt`) tidak di-commit ke repository karena ukurannya besar
- Dataset images dan labels juga tidak di-commit
- Gunakan Git LFS jika ingin menyimpan model weights di repository
- Pastikan GPU tersedia untuk training yang lebih cepat

## 🤝 Kontribusi

Silakan buat issue atau pull request jika ingin berkontribusi.

## 📄 Lisensi

Project ini dilisensikan di bawah [MIT License](LICENSE).

