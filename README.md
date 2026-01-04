#  HeadPoseMouse - Điều khiển chuột bằng đầu

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.20-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red)
![License](https://img.shields.io/badge/license-MIT-blue)

**Điều khiển chuột máy tính bằng chuyển động đầu, sử dụng Computer Vision và Machine Learning**

[Features](#features) • [Demo](#demo) • [Installation](#installation) • [Usage](#usage) • [Personalization](#personalization)

</div>

---

##  Features

### Core Functionality
-  **Head Tracking**: Điều khiển con trỏ chuột bằng chuyển động đầu (yaw/pitch)
-  **Blink Detection**: Click chuột bằng nháy mắt
  - Nháy trái = Left Click
  - Nháy phải = Right Click
  - Nháy 2 mắt = Toggle pause
-  **HUD Overlay**: Giao diện hiển thị real-time
  - Webcam preview với landmarks
  - Face mesh overlay
  - Detection status
-  **System Tray**: Chạy ngầm với icon trên taskbar
  - Quick toggle GUI
  - Settings
  - Exit

### Advanced Features
-  **Personalized Correction Model**: Train model riêng để cải thiện độ chính xác
  - +44% stability improvement
  - Real ground truth từ temporal averaging
  - ~1ms latency overhead
-  **Customizable Settings**: YAML config
  - Sensitivity adjustment
  - Smoothing factor
  - Blink thresholds
  - Key bindings
-  **Performance**: 30 FPS, 94.3% blink accuracy
-  **Headless Mode**: Chạy ngầm không cần cửa sổ

---

##  Demo

### Basic Usage
```
[Webcam] → [MediaPipe Face Mesh] → [Head Pose Estimation] → [Mouse Control]
                                  ↓
                          [Blink Detection] → [Click Events]
```

### With Personalized Model
```
[Webcam] → [MediaPipe] → [Correction Model*] → [Smoother Output] → [Mouse]
                                ↓
                    [Trained on your face]
                    [+44% stability]
```

---

##  Installation

### Requirements
- Python 3.11+
- Webcam
- Windows 10/11 (tested)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/HeadPoseMouse.git
cd HeadPoseMouse

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# (Optional) For personalization training
pip install torch torchvision
```

### Quick Start
```bash
# Run with default settings
python src/main.py

# Run with GUI visible
python src/main.py --show-gui

# Run without system tray
python src/main.py --no-tray
```

---

## 📖 Usage

### Basic Controls
- **Move head**: Con trỏ chuột di chuyển theo
- **Nháy mắt trái**: Left click
- **Nháy mắt phải**: Right click
- **Nháy cả 2 mắt**: Pause/Resume
- **Ctrl+Shift+H**: Toggle GUI visibility
- **Ctrl+Shift+P**: Pause/Resume tracking
- **Ctrl+Shift+Q**: Quit application

### Configuration
Edit `configs/default_config.yaml`:
```yaml
system:
  cam_width: 640
  cam_height: 480
  smoothing_factor: 8.0
  sensitivity: 1.5
  start_headless: true

gesture:
  ear_threshold: 0.20  # Blink sensitivity
  blink_consecutive_frames: 2
  ...
```

### System Tray
Khi chạy headless, icon xuất hiện trên system tray:
- **Left Click**: Show/Hide GUI
- **Right Click**: Menu
  - Toggle GUI
  - Settings (coming soon)
  - Exit

---

##  Personalization

Train model riêng để cải thiện độ chính xác cho khuôn mặt của bạn!

### Step 1: Collect Dataset
```bash
# Thu thập 150 poses (~15 phút)
python scripts/collect_personalized_dataset.py --num_poses 150
```
**Tips:**
- Thử nhiều góc nhìn (trái/phải/lên/xuống)
- Nghiêng đầu
- Gần/xa camera
- Các tư thế làm việc thường dùng

### Step 2: Train Model
```bash
# Train 50 epochs (~3 phút)
python scripts/train_correction_model.py \
    --data_dir data/personalized_dataset/landmarks \
    --epochs 50
```

### Step 3: Test Accuracy
```bash
# Test improvement
python scripts/test_personalized_accuracy.py
```

Expected result:
```
 STABILITY:
  MediaPipe thuần:     0.000806
  + Correction:        0.000449
  Cải thiện:           +44.3%
```

### Step 4: Run App
```bash
python src/main.py
# ✓ Đang chạy với correction model (độ chính xác cao hơn)
```

 [Chi tiết về Personalization](docs/PERSONALIZATION.md)

---

##  Architecture

```
src/
├── core_engine/          # Computer Vision core
│   ├── face_detector.py         # MediaPipe Face Mesh wrapper
│   ├── corrected_face_detector.py  # With correction model
│   ├── geometry_utils.py        # Head pose math
│   └── signal_filters.py        # Smoothing filters
├── input_layer/
│   └── camera.py                # Webcam interface
├── logic_layer/
│   ├── mouse_controller.py      # PyAutoGUI wrapper
│   └── gesture_recognizer.py    # Blink detection logic
├── ui_layer/
│   ├── hud_overlay.py           # OpenCV GUI
│   └── system_tray.py           # System tray handler
├── models/
│   └── correction_model.py      # Neural network architecture
└── main.py                      # Application entry point

configs/              # YAML configuration files
docs/
├── PERSONALIZATION.md           # Personalized model guide
└── experiments/                 # Jupyter notebooks
scripts/              # Training & testing scripts
```

---

##  Performance

### Benchmarks (i7-12700H, RTX 3060 Laptop)

| Metric | MediaPipe | + Correction | Improvement |
|--------|-----------|--------------|-------------|
| FPS | 30.01 | 29.99 | -0.1% |
| Latency | 3.34ms | 4.39ms | +1.05ms |
| Detection Rate | 91.5% | 93% | +1.5% |
| **Stability** | 0.000806 | 0.000449 | **+44.3%**  |
| Blink Accuracy | 94.3% | 94.3% | - |

### Model Size
- MediaPipe Face Mesh: ~5MB
- Correction Model: ~6.8MB (full) / ~500KB (lightweight)

---

##  Development

### Project Structure
```bash
# Run tests
pytest tests/

# Lint code
flake8 src/

# Format code
black src/
```

### Build Executable
```bash
# Build with PyInstaller
pyinstaller HeadPoseMouse.spec

# Output: dist/HeadPoseMouse.exe
```

### Environment Setup for Training
```bash
# Create separate venv for training (avoid conflicts)
python -m venv .venv_correction
.venv_correction\Scripts\activate
pip install torch torchvision opencv-python mediapipe numpy tqdm pyyaml
```

---
</div>
