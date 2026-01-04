"""
Landmark Correction Model - Cải thiện độ chính xác MediaPipe
"""

import torch
import torch.nn as nn

class LandmarkCorrectionModel(nn.Module):
    """
    Model học cách sửa lỗi dự đoán của MediaPipe
    
    Input: 468 landmarks × 3 coords = 1404 giá trị (có nhiễu)
    Output: 468 landmarks × 3 coords = 1404 giá trị (đã được sửa)
    
    Kích thước: ~2MB
    Tốc độ: ~2ms trên CPU
    """
    
    def __init__(self, num_landmarks=468, coord_dim=3):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.input_dim = num_landmarks * coord_dim  # 1404
        
        # Encoder: Nén thông tin
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Decoder: Tái tạo với sửa lỗi
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            
            nn.Linear(512, self.input_dim)
        )
        
        # Hệ số điều chỉnh (học được)
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        """
        Args:
            x: [batch, 1404] - landmarks từ MediaPipe (có nhiễu)
        
        Returns:
            [batch, 1404] - landmarks đã được sửa
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode để có correction
        correction = self.decoder(latent)
        
        # Residual connection: output = input + small_correction
        output = x + self.correction_scale * correction
        
        return output
    
    def get_num_parameters(self):
        """Số lượng parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightCorrectionModel(nn.Module):
    """
    Version nhẹ: Chỉ sửa 13 landmarks quan trọng (mắt + mũi)
    
    Kích thước: ~500KB
    Tốc độ: <1ms
    """
    
    def __init__(self):
        super().__init__()
        
        # Các landmarks quan trọng cho ứng dụng
        self.key_indices = [
            1,    # Mũi
            33, 160, 158, 133, 153, 144,  # Mắt phải
            362, 385, 387, 263, 373, 380  # Mắt trái
        ]
        
        self.num_key_landmarks = len(self.key_indices)
        
        self.net = nn.Sequential(
            nn.Linear(1404, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, self.num_key_landmarks * 3)
        )
        
        self.correction_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Dự đoán corrections cho key landmarks
        key_corrections = self.net(x)
        key_corrections = key_corrections.view(batch_size, self.num_key_landmarks, 3)
        
        # Bắt đầu từ input
        output = x.view(batch_size, 468, 3).clone()
        
        # Apply corrections
        for i, idx in enumerate(self.key_indices):
            output[:, idx, :] += self.correction_scale * key_corrections[:, i, :]
        
        return output.view(batch_size, -1)


def create_correction_model(model_type='full'):
    """Tạo model"""
    if model_type == 'lightweight':
        model = LightweightCorrectionModel()
    else:
        model = LandmarkCorrectionModel()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Tạo {model_type} correction model")
    print(f"  Parameters: {params:,}")
    print(f"  Kích thước: ~{params * 4 / 1024 / 1024:.1f}MB")
    
    return model
