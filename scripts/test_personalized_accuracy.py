"""Test accuracy: Correction Model vs MediaPipe"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

def calculate_stability(landmarks_sequence):
    """Tính độ ổn định (thấp = tốt)"""
    if len(landmarks_sequence) < 2:
        return 0
    
    # Tính displacement giữa các frame liên tiếp
    displacements = []
    for i in range(1, len(landmarks_sequence)):
        prev = landmarks_sequence[i-1]
        curr = landmarks_sequence[i]
        
        # Euclidean distance
        displacement = np.sqrt(np.sum((curr - prev) ** 2))
        displacements.append(displacement)
    
    # Trả về std của displacement (đo jitter)
    return np.std(displacements)

def test_detector(detector, detector_name, num_frames=100):
    """Test detector"""
    print(f"\n{'='*50}\n{detector_name}\n{'='*50}")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    print(f"Collecting {num_frames} frames (HOLD HEAD FIXED)...\n")
    key_indices = [1, 33, 133, 362, 263]
    
    landmarks_sequences = {idx: [] for idx in key_indices}
    
    time.sleep(2)  # Cho user chuẩn bị
    
    success_count = 0
    
    for i in range(num_frames):
        success, frame = camera.read()
        if not success:
            continue
        
        # Detect landmarks
        if hasattr(detector, 'detect_landmarks'):
            landmarks = detector.detect_landmarks(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
        
        if landmarks is not None:
            success_count += 1
            
            # Lưu landmarks
            for idx in key_indices:
                lm = landmarks.landmark[idx]
                landmarks_sequences[idx].append(np.array([lm.x, lm.y, lm.z]))
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_frames} frames...")
    
    camera.release()
    stabilities = {idx: calculate_stability(seq) for idx, seq in landmarks_sequences.items() if len(seq) > 1}
    avg_stability = np.mean(list(stabilities.values()))
    detection_rate = (success_count / num_frames) * 100
    
    print(f"\nDetection: {detection_rate:.1f}% | Stability (low=good):")
    for idx in [1, 33, 133, 362, 263]:
        print(f"  [{idx}]: {stabilities.get(idx, 0):.6f}")
    print(f"  Avg: {avg_stability:.6f}\n")
    
    return {
        'name': detector_name,
        'detection_rate': detection_rate,
        'avg_stability': avg_stability,
        'stabilities': stabilities
    }

def main():
    print(f"\n{'='*50}\nAccuracy Test: Personalized Model\n{'='*50}")
    print("Hold head FIXED during test (~10s each)\n")
    input("Press Enter...")
    
    results = []
    try:
        from src.core_engine.face_detector import FaceMeshDetector
        print("[1/2] MediaPipe...")
        result1 = test_detector(FaceMeshDetector(), "MediaPipe", 100)
        results.append(result1)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    try:
        from src.core_engine.corrected_face_detector import CorrectedFaceDetector
        print("[2/2] Personalized...")
        result2 = test_detector(CorrectedFaceDetector('models/best_model.pth', 'full'), "Correction", 100)
        results.append(result2)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # So sánh
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("SO SÁNH KẾT QUẢ")
        print("=" * 60)
        
        base = results[0]
        improved = results[1]
        
        stability_improvement = ((base['avg_stability'] - improved['avg_stability']) / base['avg_stability']) * 100
        detection_improvement = improved['detection_rate'] - base['detection_rate']
        
        print(f"\n📊 DETECTION RATE:")
        print(f"  MediaPipe thuần:     {base['detection_rate']:.1f}%")
        print(f"  + Correction:        {improved['detection_rate']:.1f}%")
        print(f"  Chênh lệch:          {detection_improvement:+.1f}%")
        
        print(f"\n📈 STABILITY (thấp = tốt):")
        print(f"  MediaPipe thuần:     {base['avg_stability']:.6f}")
        print(f"  + Correction:        {improved['avg_stability']:.6f}")
        print(f"  Cải thiện:           {stability_improvement:+.1f}%")
        
        print(f"\n💡 KẾT LUẬN:")
        
        if stability_improvement > 10:
            print(f"  ✅ Correction model cải thiện ĐÁNG KỂ")
            print(f"     → Cursor di chuyển ổn định hơn {stability_improvement:.0f}%")
            print(f"     → Personalized training HIỆU QUẢ!")
        elif stability_improvement > 5:
            print(f"  ✅ Correction model cải thiện VỪA PHẢI")
            print(f"     → Ổn định hơn {stability_improvement:.0f}%")
            print(f"     → Đáng sử dụng")
        elif stability_improvement > 0:
            print(f"  ⚠️  Correction model cải thiện ÍT")
            print(f"     → Chỉ ổn định hơn {stability_improvement:.1f}%")
            print(f"     → Có thể do data training thiếu ground truth")
        else:
            print(f"  ❌ Correction model KHÔNG cải thiện")
            print(f"     → Thậm chí kém hơn {abs(stability_improvement):.1f}%")
            print(f"     → Data training có vấn đề!")
        
        print(f"\n📌 LƯU Ý:")
        print(f"  Model hiện tại được train trên synthetic noise")
        print(f"  (không phải real ground truth)")
        print(f"  → Cải thiện chủ yếu từ personalization,")
        print(f"     không phải từ việc sửa lỗi MediaPipe")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest bị dừng")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
