"""
Collect training dataset from webcam
No download needed - creates dataset in 15-30 minutes
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

class WebcamDatasetCollector:
    """Collect facial landmark dataset from webcam"""
    
    def __init__(self, output_dir='data/webcam_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
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
        self.target_samples = 1000
    
    def add_noise_to_landmarks(self, landmarks, noise_level=0.01):
        """
        Add random noise to landmarks to simulate MediaPipe errors
        This creates the 'input' (noisy) version
        """
        noisy = landmarks.copy()
        noise = np.random.normal(0, noise_level, landmarks.shape)
        noisy += noise
        return noisy
    
    def collect_samples(self, num_samples=1000, skip_frames=5):
        """
        Collect samples from webcam
        
        Args:Correction Model
            num_samples: Number of samples to collect (default: 1000)
            skip_frames: Save every N frames (default: 5, to get variety)
        """
        self.target_samples = num_samples
        
        print("\n" + "=" * 60)
        print("WEBCAM DATASET COLLECTION")
        print("=" * 60)
        print(f"\nTarget: {num_samples} samples")
        print("\nInstructions:")
        print("- Move your head slowly (left, right, up, down)")
        print("- Try different angles and distances")
        print("- Blink naturally")
        print("- Change lighting if possible")
        print("- Press 'q' to quit early")
        print("\nStarting in 3 seconds...")
        print()
        
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        # Set camera resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Wait 3 seconds
        cv2.waitKey(3000)
        
        frame_count = 0
        
        with tqdm(total=num_samples, desc="Collecting") as pbar:
            while self.samples_collected < num_samples:
                success, frame = camera.read()
                
                if not success:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Skip frames for variety
                if frame_count % skip_frames != 0:
                    # Still show preview
                    cv2.imshow('Dataset Collection (press q to quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Convert to numpy array
                    h, w = frame.shape[:2]
                    landmarks_array = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in landmarks.landmark
                    ], dtype=np.float32)
                    
                    # Save sample
                    sample_id = f"sample_{self.samples_collected:05d}"
                    
                    # Save image
                    img_path = self.output_dir / 'images' / f"{sample_id}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    
                    # Create training pair:
                    # - input: noisy version (simulates MediaPipe errors)
                    # - target: clean version (ground truth)
                    input_landmarks = self.add_noise_to_landmarks(landmarks_array, noise_level=0.005)
                    target_landmarks = landmarks_array
                    
                    # Save landmarks
                    np.save(
                        self.output_dir / 'landmarks' / f"{sample_id}_input.npy",
                        input_landmarks
                    )
                    np.save(
                        self.output_dir / 'landmarks' / f"{sample_id}_target.npy",
                        target_landmarks
                    )
                    
                    # Save metadata
                    metadata = {
                        'timestamp': datetime.now().isoformat(),
                        'resolution': (w, h),
                        'frame_number': frame_count
                    }
                    with open(self.output_dir / 'landmarks' / f"{sample_id}_meta.json", 'w') as f:
                        json.dump(metadata, f)
                    
                    self.samples_collected += 1
                    pbar.update(1)
                    
                    # Draw landmarks on frame for preview
                    self.mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                # Show progress on frame
                cv2.putText(
                    frame,
                    f"Collected: {self.samples_collected}/{num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Dataset Collection (press q to quit)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nCollection stopped by user")
                    break
        
        camera.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"\nCollected {self.samples_collected} samples")
        print(f"Saved to: {self.output_dir}")
        
        return self.samples_collected
    
    def split_train_test(self, test_ratio=0.2):
        """Split collected data into train/test sets"""
        print("\nSplitting into train/test sets...")
        
        # Get all samples
        all_samples = sorted(list((self.output_dir / 'landmarks').glob('*_input.npy')))
        
        # Shuffle
        np.random.shuffle(all_samples)
        
        # Split
        split_idx = int(len(all_samples) * (1 - test_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        # Save split info
        split_info = {
            'train': [s.stem.replace('_input', '') for s in train_samples],
            'test': [s.stem.replace('_input', '') for s in test_samples],
            'train_count': len(train_samples),
            'test_count': len(test_samples)
        }
        
        with open(self.output_dir / 'split.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✓ Train set: {len(train_samples)} samples")
        print(f"✓ Test set: {len(test_samples)} samples")
        print(f"✓ Split info saved to: {self.output_dir / 'split.json'}")
        
        return split_info

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training dataset from webcam')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to collect (default: 1000)')
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='Save every N frames (default: 5)')
    parser.add_argument('--output_dir', type=str, default='data/webcam_dataset',
                       help='Output directory')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Create collector
    collector = WebcamDatasetCollector(args.output_dir)
    
    # Collect samples
    collected = collector.collect_samples(
        num_samples=args.num_samples,
        skip_frames=args.skip_frames
    )
    
    if collected > 0:
        # Split train/test
        collector.split_train_test(test_ratio=args.test_ratio)
        
        print("\n✓ Dataset ready for training!")
        print("\nNext step:")
        print(f"  python scripts/train_correction_model.py --data_dir {args.output_dir}/landmarks")
    else:
        print("\nNo samples collected")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
