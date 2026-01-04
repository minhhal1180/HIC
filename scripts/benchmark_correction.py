"""
Benchmark để so sánh tốc độ MediaPipe thuần vs MediaPipe + Correction Model
"""

import cv2
import time
import numpy as np
import sys
from pathlib import Path
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

def benchmark_detector(detector, detector_name, num_frames=200):
    """
    Benchmark một detector
    
    Args:
        detector: Detector object (FaceMeshDetector hoặc CorrectedFaceDetector)
        detector_name: Tên hiển thị
        num_frames: Số frame để test
    """
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {detector_name}")
    print(f"{'=' * 60}\n")
    
    # Khởi tạo camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    # Warm-up (bỏ qua 10 frame đầu)
    print("Warm-up...")
    for _ in range(10):
        success, frame = camera.read()
        if success:
            if hasattr(detector, 'detect_landmarks'):
                landmarks = detector.detect_landmarks(frame)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
    
    # Đo memory trước khi test
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Benchmark
    print(f"Đang xử lý {num_frames} frames...")
    
    latencies = []
    success_count = 0
    
    start_time = time.time()
    
    for i in range(num_frames):
        success, frame = camera.read()
        if not success:
            continue
        
        # Đo latency cho frame này
        frame_start = time.time()
        
        # Gọi method phù hợp
        if hasattr(detector, 'detect_landmarks'):
            landmarks = detector.detect_landmarks(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
        
        frame_end = time.time()
        
        latency = (frame_end - frame_start) * 1000  # ms
        latencies.append(latency)
        
        if landmarks is not None:
            success_count += 1
        
        # Hiển thị progress
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_frames} frames...")
    
    end_time = time.time()
    
    # Đo memory sau test
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    # Tính toán kết quả
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    std_latency = np.std(latencies)
    
    detection_rate = (success_count / num_frames) * 100
    
    camera.release()
    
    # In kết quả
    print(f"\n{'=' * 60}")
    print(f"KẾT QUẢ: {detector_name}")
    print(f"{'=' * 60}")
    print(f"\n📊 HIỆU NĂNG:")
    print(f"  FPS:              {fps:.2f} frames/giây")
    print(f"  Thời gian/frame:  {1000/fps:.2f} ms")
    print(f"\n⏱️  LATENCY:")
    print(f"  Trung bình:       {avg_latency:.2f} ms")
    print(f"  Tối thiểu:        {min_latency:.2f} ms")
    print(f"  Tối đa:           {max_latency:.2f} ms")
    print(f"  P95:              {p95_latency:.2f} ms")
    print(f"  Độ lệch chuẩn:    {std_latency:.2f} ms")
    print(f"\n✅ DETECTION:")
    print(f"  Tỷ lệ phát hiện:  {detection_rate:.1f}%")
    print(f"\n💾 MEMORY:")
    print(f"  Sử dụng thêm:     {mem_used:.1f} MB")
    print()
    
    return {
        'name': detector_name,
        'fps': fps,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'p95_latency': p95_latency,
        'std_latency': std_latency,
        'detection_rate': detection_rate,
        'memory_mb': mem_used
    }

def compare_results(results):
    """So sánh kết quả giữa 2 detectors"""
    print(f"\n{'=' * 60}")
    print("SO SÁNH KẾT QUẢ")
    print(f"{'=' * 60}\n")
    
    base = results[0]  # MediaPipe thuần
    improved = results[1]  # MediaPipe + Correction
    
    fps_diff = ((improved['fps'] - base['fps']) / base['fps']) * 100
    latency_diff = improved['avg_latency'] - base['avg_latency']
    memory_diff = improved['memory_mb'] - base['memory_mb']
    
    print(f"📊 FPS:")
    print(f"  MediaPipe thuần:     {base['fps']:.2f} fps")
    print(f"  + Correction Model:  {improved['fps']:.2f} fps")
    print(f"  Chênh lệch:          {fps_diff:+.1f}% {'⬇️' if fps_diff < 0 else '⬆️'}")
    
    print(f"\n⏱️  LATENCY trung bình:")
    print(f"  MediaPipe thuần:     {base['avg_latency']:.2f} ms")
    print(f"  + Correction Model:  {improved['avg_latency']:.2f} ms")
    print(f"  Chênh lệch:          {latency_diff:+.2f} ms {'⬇️' if latency_diff < 0 else '⬆️'}")
    
    print(f"\n💾 MEMORY sử dụng:")
    print(f"  MediaPipe thuần:     {base['memory_mb']:.1f} MB")
    print(f"  + Correction Model:  {improved['memory_mb']:.1f} MB")
    print(f"  Chênh lệch:          {memory_diff:+.1f} MB")
    
    print(f"\n📈 TÓM TẮT:")
    if abs(fps_diff) < 5:
        print(f"  ✅ Tốc độ: Gần như không đổi ({fps_diff:+.1f}%)")
    elif fps_diff < -5:
        print(f"  ⚠️  Tốc độ: Chậm hơn {abs(fps_diff):.1f}%")
    else:
        print(f"  ✅ Tốc độ: Nhanh hơn {fps_diff:.1f}%")
    
    print(f"  ⏱️  Latency tăng: {latency_diff:.2f} ms")
    print(f"  💾 Memory tăng: {memory_diff:.1f} MB")
    
    print(f"\n💡 KẾT LUẬN:")
    if latency_diff < 5:
        print(f"  Correction model tăng latency rất ít ({latency_diff:.1f}ms)")
        print(f"  → Đáng để dùng để cải thiện độ chính xác!")
    elif latency_diff < 10:
        print(f"  Correction model tăng latency vừa phải ({latency_diff:.1f}ms)")
        print(f"  → Có thể chấp nhận được nếu cần độ chính xác cao")
    else:
        print(f"  Correction model tăng latency đáng kể ({latency_diff:.1f}ms)")
        print(f"  → Cân nhắc sử dụng lightweight model")
    print()

def main():
    print("\n" + "=" * 60)
    print("BENCHMARK: MediaPipe vs MediaPipe + Correction Model")
    print("=" * 60)
    print("\nĐo lường hiệu năng thực tế trên máy của bạn...")
    print("(Sẽ chạy 200 frames cho mỗi detector, mất ~30 giây)\n")
    
    results = []
    
    # Test 1: MediaPipe thuần
    try:
        from src.core_engine.face_detector import FaceMeshDetector
        
        print("\n[1/2] Testing MediaPipe thuần...")
        detector1 = FaceMeshDetector()
        result1 = benchmark_detector(detector1, "MediaPipe thuần", num_frames=200)
        results.append(result1)
        if hasattr(detector1, 'release'):
            detector1.release()
        
    except Exception as e:
        print(f"❌ Lỗi khi test MediaPipe thuần: {e}")
        return
    
    # Test 2: MediaPipe + Correction
    try:
        from src.core_engine.corrected_face_detector import CorrectedFaceDetector
        
        print("\n[2/2] Testing MediaPipe + Correction Model...")
        detector2 = CorrectedFaceDetector('models/best_model.pth', model_type='full')
        result2 = benchmark_detector(detector2, "MediaPipe + Correction", num_frames=200)
        results.append(result2)
        if hasattr(detector2, 'release'):
            detector2.release()
        
    except Exception as e:
        print(f"❌ Lỗi khi test Correction Model: {e}")
        print("   (Model có thể chưa được train hoặc không tìm thấy)")
        return
    
    # So sánh kết quả
    if len(results) == 2:
        compare_results(results)
    
    print("=" * 60)
    print("✅ BENCHMARK HOÀN TẤT!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark bị dừng")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
