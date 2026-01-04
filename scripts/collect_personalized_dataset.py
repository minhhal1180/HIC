import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

class RealGroundTruthCollector:
    """Thu thập ground truth từ temporal averaging"""
    
    def __init__(self, output_dir='data/personalized_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'landmarks').mkdir(exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.samples_collected = 0
    
    def collect_ground_truth_for_pose(self, camera, pose_name, frames_per_pose=40):
        """Thu thập ground truth cho 1 pose"""
        print(f"\n📸 {pose_name}")
        for i in range(3, 0, -1):
            print(f"{i}...", end='', flush=True)
            cv2.waitKey(1000)
        print(" GO!")
        
        frames = []
        landmarks_list = []
        
        # Thu thập frames
        for i in range(frames_per_pose):
            success, frame = camera.read()
            if not success:
                continue
            
            # Detect landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Convert to numpy
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in landmarks.landmark
                ], dtype=np.float32)
                
                frames.append(frame.copy())
                landmarks_list.append(landmarks_array)
                
                # Show preview
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                cv2.putText(
                    frame,
                    f"Pose: {pose_name} - {i+1}/{frames_per_pose}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    "GIU DAU CO DINH!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1)
        
        if len(landmarks_list) < frames_per_pose // 2:
            print(f" Insufficient frames ({len(landmarks_list)})")
            return None
        
        # Compute ground truth bằng MEDIAN (robust hơn mean)
        landmarks_array = np.array(landmarks_list)  # [N, 468, 3]
        
        # Lọc outliers bằng MAD (Median Absolute Deviation)
        median = np.median(landmarks_array, axis=0)  # [468, 3]
        mad = np.median(np.abs(landmarks_array - median), axis=0)  # [468, 3]
        
        # Chỉ giữ frames không phải outliers
        threshold = 3  # 3 MAD
        distances = np.abs(landmarks_array - median)
        valid_mask = np.all(distances < threshold * mad, axis=(1, 2))
        
        valid_landmarks = landmarks_array[valid_mask]
        valid_frames = [f for i, f in enumerate(frames) if valid_mask[i]]
        
        if len(valid_landmarks) < 5:
            print(" Too many outliers")
            return None
        
        # Ground truth = median của valid frames
        ground_truth = np.median(valid_landmarks, axis=0)  # [468, 3]
        
        # Chọn 1 frame đại diện (frame gần ground truth nhất)
        distances_to_gt = np.sum(np.abs(valid_landmarks - ground_truth), axis=(1, 2))
        best_frame_idx = np.argmin(distances_to_gt)
        representative_frame = valid_frames[best_frame_idx]
        
        # Tạo noisy input (chọn frame khác ground truth một chút)
        # Để model học sửa lỗi
        if len(valid_landmarks) >= 3:
            # Lấy frame khác với ground truth (không phải best frame)
            other_indices = [i for i in range(len(valid_landmarks)) if i != best_frame_idx]
            if other_indices:
                noisy_idx = np.random.choice(other_indices)
                noisy_input = valid_landmarks[noisy_idx]
            else:
                noisy_input = valid_landmarks[0]
        else:
            noisy_input = valid_landmarks[0]
        
        print(f"✓ GT created: {len(valid_landmarks)}/{len(landmarks_list)} valid")
        return representative_frame, noisy_input, ground_truth
    
    def collect_dataset(self, num_poses=20):
        """Thu thập dataset"""
        print(f"\n{'='*50}\nCollecting {num_poses} poses\n{'='*50}")
        print("Hold head FIXED during each pose (~1.5s)")
        print("Try: center, left, right, up, down, tilted, near, far\n")
        
        input("Nhấn Enter để bắt đầu...")
        
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        successful = 0
        
        pose_suggestions = [
            "Nhìn thẳng (trung tâm)",
            "Nhìn sang TRÁI",
            "Nhìn sang PHẢI",
            "Nhìn LÊN",
            "Nhìn XUỐNG",
            "Nghiêng đầu TRÁI",
            "Nghiêng đầu PHẢI",
            "Gần camera",
            "Xa camera",
            "Góc trên trái",
            "Góc trên phải",
            "Góc dưới trái",
            "Góc dưới phải",
            "Nhìn trái + lên",
            "Nhìn phải + lên",
            "Nhìn trái + xuống",
            "Nhìn phải + xuống",
            "Tư thế làm việc bình thường 1",
            "Tư thế làm việc bình thường 2",
            "Tư thế thoải mái",
        ]
        
        for i in range(num_poses):
            pose_name = pose_suggestions[i] if i < len(pose_suggestions) else f"Pose {i+1}"
            
            result = self.collect_ground_truth_for_pose(
                camera,
                pose_name,
                frames_per_pose=40
            )
            
            if result is not None:
                frame, noisy_input, ground_truth = result
                
                # Save
                sample_id = f"sample_{self.samples_collected:04d}"
                
                # Save image
                img_path = self.output_dir / 'images' / f"{sample_id}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save landmarks
                np.save(
                    self.output_dir / 'landmarks' / f"{sample_id}_input.npy",
                    noisy_input
                )
                np.save(
                    self.output_dir / 'landmarks' / f"{sample_id}_target.npy",
                    ground_truth
                )
                
                # Metadata
                metadata = {
                    'pose': pose_name,
                    'timestamp': datetime.now().isoformat(),
                    'sample_id': sample_id
                }
                with open(self.output_dir / 'landmarks' / f"{sample_id}_meta.json", 'w') as f:
                    json.dump(metadata, f)
                
                successful += 1
                self.samples_collected += 1
                
                print(f"✅ {successful}/{num_poses} saved\n")
            else:
                print("  Retry...\n")
                i -= 1
            
            if successful < num_poses:
                for j in range(3, 0, -1):
                    print(f"{j}...", end='', flush=True)
                    cv2.waitKey(1000)
                print()
        
        camera.release()
        cv2.destroyAllWindows()
        self.split_train_test()
        print(f"\n{'='*50}\nDone: {successful} samples → {self.output_dir}\n{'='*50}")
    
    def split_train_test(self, test_ratio=0.2):
        """Split train/test"""
        all_samples = sorted(list((self.output_dir / 'landmarks').glob('*_input.npy')))
        sample_ids = [f.stem.replace('_input', '') for f in all_samples]
        np.random.shuffle(sample_ids)
        split_idx = int(len(sample_ids) * (1 - test_ratio))
        split_info = {'train': sample_ids[:split_idx], 'test': sample_ids[split_idx:]}
        with open(self.output_dir / 'split.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Train/Test: {len(split_info['train'])}/{len(split_info['test'])}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Thu thập personalized dataset')
    parser.add_argument('--num_poses', type=int, default=20,
                       help='Số lượng poses (default: 20)')
    parser.add_argument('--output_dir', type=str, default='data/personalized_dataset',
                       help='Output directory')
    
    args = parser.parse_args()
    
    collector = RealGroundTruthCollector(args.output_dir)
    collector.collect_dataset(num_poses=args.num_poses)
    
    print("\n Dataset sẵn sàng!")
    print("\nBước tiếp theo:")
    print(f"  .\.venv_correction\Scripts\python.exe scripts/train_correction_model.py --data_dir {args.output_dir}/landmarks --epochs 50")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nThu thập bị dừng")
    except Exception as e:
        print(f"\nLỖI: {e}")
        import traceback
        traceback.print_exc()
