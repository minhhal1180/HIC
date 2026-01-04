# Personalized Correction Model

## Tổng quan

HeadPoseMouse hỗ trợ **personalized correction model** để cải thiện độ chính xác và ổn định cho từng người dùng cụ thể.

## Cách hoạt động

### 1. Ground Truth Collection
Model được train với **ground truth thật** thay vì synthetic noise:

- Thu thập 40 frames liên tiếp khi giữ đầu cố định
- Loại bỏ outliers bằng MAD (Median Absolute Deviation)
- Lấy median của frames valid làm ground truth
- Input = frame có jitter tự nhiên, Target = ground truth

### 2. Model Architecture
```
LandmarkCorrectionModel:
- Input: 1404 values (468 landmarks × 3 coords)
- Encoder: 1404 → 512 → 256 → 128
- Decoder: 128 → 256 → 512 → 1404
- Output: Corrected landmarks
- Parameters: 1.77M (~6.8MB)
```

### 3. Kết quả
- **Stability improvement: +44.3%** (tested on original developer)
- Detection rate: Unchanged (100%)
- Latency overhead: ~1ms
- FPS impact: -0.1%

## Cách sử dụng

### Bước 1: Thu thập dữ liệu
```bash
# Thu thập 150 poses (khuyến nghị: 100-150)
.\.venv_correction\Scripts\python.exe scripts/collect_personalized_dataset.py --num_poses 150
```

**Hướng dẫn:**
- Mỗi pose: giữ đầu CỐ ĐỊNH trong ~1.5 giây
- Thử nhiều poses khác nhau:
  - Nhìn trái, phải, lên, xuống
  - Nghiêng đầu
  - Gần/xa camera
  - Các góc làm việc thường dùng

### Bước 2: Training
```bash
# Train model (50 epochs, ~3 phút)
.\.venv_correction\Scripts\python.exe scripts/train_correction_model.py --data_dir data/personalized_dataset/landmarks --epochs 50
```

**Kết quả:**
- Model saved: `models/best_model.pth`
- Val loss: ~0.00003
- Training time: ~3 phút

### Bước 3: Test accuracy
```bash
# Test độ cải thiện
.\.venv_correction\Scripts\python.exe scripts/test_personalized_accuracy.py
```

### Bước 4: Chạy app
App tự động load correction model nếu có:
```bash
python src/main.py
```

Console sẽ hiển thị:
```
✓ Đang chạy với correction model (độ chính xác cao hơn)
```

## Benchmark

| Metric | MediaPipe | + Correction | Improvement |
|--------|-----------|--------------|-------------|
| Stability | 0.000806 | 0.000449 | **+44.3%** ✅ |
| Detection Rate | 100% | 100% | - |
| FPS | 30.01 | 29.99 | -0.1% |
| Latency | 3.34ms | 4.39ms | +1ms |

## Dataset Requirements

### Số lượng optimal
- **Minimum**: 50 poses (đủ cover các góc cơ bản)
- **Optimal**: 100-150 poses ⭐ (sweet spot)
- **Maximum useful**: 200-300 poses (diminishing returns)

### Tại sao không cần quá nhiều?
- Train cho **1 người** → không cần diverse faces
- Ground truth method đã rất accurate
- Validation set detect overfitting sớm
- **Diversity > Quantity**: 100 poses đa dạng > 500 poses giống nhau

### Tips để tránh overfitting
- ✅ Collect poses ở nhiều góc khác nhau
- ✅ Early stopping (model tự động dừng khi val loss tăng)
- ✅ Monitor train vs val loss
- ✅ Test trên real-world usage

## So sánh với phương pháp cũ

| Aspect | Synthetic Noise ❌ | Real Ground Truth ✅ |
|--------|-------------------|---------------------|
| Training data | MediaPipe + random noise | Temporal averaging |
| Target | Original MediaPipe | Median of 40 frames |
| Result | **-233% worse!** | **+44% better!** |
| Reason | Learns to denoise artificial patterns | Fixes real MediaPipe errors |

## Files

### Scripts
- `scripts/collect_personalized_dataset.py` - Thu thập data với ground truth thật
- `scripts/train_correction_model.py` - Training pipeline
- `scripts/test_personalized_accuracy.py` - Test accuracy improvement

### Core
- `src/models/correction_model.py` - Neural network architecture
- `src/core_engine/corrected_face_detector.py` - Wrapper integrating correction

### Data
- `data/personalized_dataset/` - Dataset directory
  - `images/` - Reference images
  - `landmarks/` - Landmark files (*_input.npy, *_target.npy)
  - `split.json` - Train/test split
- `models/best_model.pth` - Trained model (ignored in git)

## Troubleshooting

### Model không load được
```
⚠ Không load được correction model: ...
→ Chạy với MediaPipe thuần
```
**Giải pháp:** Train lại model hoặc check file `models/best_model.pth` exists

### Outliers quá nhiều khi collect
```
❌ Quá nhiều outliers
```
**Nguyên nhân:** Đầu không giữ đủ cố định hoặc MediaPipe detect không ổn định

**Giải pháp:** 
- Giữ đầu cố định hơn
- Cải thiện lighting
- Thử lại pose đó

### Model không improve accuracy
**Check:**
1. Training loss có giảm không? (phải < 0.0001)
2. Val loss có converge không?
3. Test script cho kết quả gì?

**Solution:**
- Thu thập thêm data (100-150 poses)
- Đảm bảo poses đa dạng
- Re-train với epochs cao hơn

## Technical Details

### Temporal Averaging Method
```python
# 1. Collect 40 frames of fixed pose
frames = []
for i in range(40):
    landmarks = mediapipe.detect()
    frames.append(landmarks)

# 2. Filter outliers (MAD method)
median = np.median(frames, axis=0)
mad = np.median(np.abs(frames - median), axis=0)
valid_frames = frames[distances < 3 * mad]

# 3. Ground truth = median
ground_truth = np.median(valid_frames, axis=0)
```

### Training Loop
```python
# MSE Loss between corrected and ground truth
loss = MSELoss(corrected_landmarks, ground_truth)

# Adam optimizer with ReduceLROnPlateau
optimizer = Adam(lr=0.001)
scheduler = ReduceLROnPlateau(patience=5)

# Early stopping (patience=10)
if val_loss not improved for 10 epochs:
    stop training
```

## Citation

Nếu bạn sử dụng phương pháp này trong research, vui lòng cite:

```
HeadPoseMouse Personalized Correction Model
Author: [Your Name]
Year: 2026
Method: Temporal Averaging for Ground Truth Landmark Collection
```

## Future Work

- [ ] Lightweight model (13 key landmarks only, ~500KB)
- [ ] ONNX export cho faster inference
- [ ] Multi-user support (store multiple personalized models)
- [ ] Online learning (incremental training during usage)
- [ ] Cross-validation method for ground truth
