"""
Face detector với correction model để cải thiện độ chính xác
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.correction_model import LandmarkCorrectionModel, LightweightCorrectionModel

class CorrectedFaceDetector:
    """
    Face detector với MediaPipe + Correction Model
    
    Sử dụng:
        detector = CorrectedFaceDetector('models/best_model.pth')
        landmarks = detector.detect_landmarks(frame)
    """
    
    def __init__(self, correction_model_path=None, model_type='full'):
        """
        Args:
            correction_model_path: Đường dẫn tới trained model (.pth)
            model_type: 'full' hoặc 'lightweight'
        """
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load correction model
        self.correction_model = None
        if correction_model_path is not None:
            self.load_correction_model(correction_model_path, model_type)
    
    def load_correction_model(self, model_path, model_type='full'):
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return
        try:
            self.correction_model = LightweightCorrectionModel() if model_type == 'lightweight' else LandmarkCorrectionModel()
            checkpoint = torch.load(model_path, map_location='cpu')
            self.correction_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            self.correction_model.eval()
            print(f"✓ Loaded correction model ({model_type})")
        except Exception as e:
            print(f"ERROR: {e}")
            self.correction_model = None
    
    def detect_landmarks(self, frame):
        """Detect facial landmarks with correction"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        
        # Apply correction nếu có model
        if self.correction_model is not None:
            landmarks = self.apply_correction(landmarks)
        
        return landmarks
    
    def apply_correction(self, landmarks):
        """
        Áp dụng correction model lên landmarks
        
        Args:
            landmarks: MediaPipe landmarks object
        
        Returns:
            Corrected landmarks object
        """
        # Convert sang numpy array
        landmarks_array = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ], dtype=np.float32)
        
        # Flatten [1404]
        landmarks_flat = landmarks_array.flatten()
        
        # Convert sang tensor
        input_tensor = torch.from_numpy(landmarks_flat).unsqueeze(0)
        
        # Run correction model
        with torch.no_grad():
            corrected_flat = self.correction_model(input_tensor)
        
        # Convert về numpy
        corrected_array = corrected_flat.squeeze(0).numpy().reshape(468, 3)
        
        # Update landmarks
        for i in range(468):
            landmarks.landmark[i].x = float(corrected_array[i, 0])
            landmarks.landmark[i].y = float(corrected_array[i, 1])
            landmarks.landmark[i].z = float(corrected_array[i, 2])
        
        return landmarks
    
    def release(self):
        self.mp_face_mesh.close()
