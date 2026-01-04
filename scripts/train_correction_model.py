"""
Train correction model để cải thiện MediaPipe
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.correction_model import create_correction_model

class LandmarkCorrectionDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Đọc split info từ webcam dataset
        split_file = self.data_dir.parent / 'split.json'
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            sample_ids = split_info[split]
            self.input_files = [self.data_dir / f"{sid}_input.npy" for sid in sample_ids]
        else:
            print(f"ERROR: Không tìm thấy {split_file}")
            self.input_files = []
        
        print(f"Tìm thấy {len(self.input_files)} {split} samples")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = Path(str(input_file).replace('_input.npy', '_target.npy'))
        
        # Load data
        input_landmarks = np.load(input_file).astype(np.float32)
        target_landmarks = np.load(target_file).astype(np.float32)
        
        # Flatten
        input_flat = input_landmarks.flatten()
        target_flat = target_landmarks.flatten()
        
        return (
            torch.from_numpy(input_flat),
            torch.from_numpy(target_flat)
        )

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=1e-4, model_dir='models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.best_val_loss = float('inf')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_input)
            loss = self.criterion(output, batch_target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_input, batch_target in tqdm(self.val_loader, desc="Validating"):
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                
                output = self.model(batch_input)
                loss = self.criterion(output, batch_target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs):
        print(f"\n{'='*50}\nTraining {epochs} epochs\n{'='*50}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f}", end='')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(" ✓ Best!")
            else:
                print()
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch}.pth', epoch, val_loss)
        
        self.save_checkpoint('final_model.pth', epochs, val_loss)
        print(f"\n{'='*50}\nDone! Best val loss: {self.best_val_loss:.6f} → {self.model_dir}\n{'='*50}")
        return self.best_val_loss
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, self.model_dir / filename)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train correction model')
    parser.add_argument('--data_dir', type=str, default='data/webcam_dataset/landmarks')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'lightweight'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    train_dataset = LandmarkCorrectionDataset(args.data_dir, split='train')
    val_dataset = LandmarkCorrectionDataset(args.data_dir, split='test')
    if len(train_dataset) == 0:
        print("ERROR: No training data!")
        return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train/Val: {len(train_dataset)}/{len(val_dataset)}")
    model = create_correction_model(args.model_type)
    
    trainer = Trainer(model, train_loader, val_loader, device=args.device, lr=args.lr)
    trainer.train(args.epochs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã dừng training")
    except Exception as e:
        print(f"\nLỖI: {e}")
        import traceback
        traceback.print_exc()
