import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
import math
import time

# Configuration
CONTEXT_SIZE = 6 
EMBED_DIM = 4   
HIDDEN1 = 32
HIDDEN2 = 8
BATCH_SIZE = 4096 
LEARNING_RATE = 0.005 # RAdam usually handles LR well, but 0.01 might be high for deep narrow nets
TRAIN_TIME_SECONDS = 300 # 5 minutes

class TextDataset(Dataset):
    def __init__(self, text, context_size):
        self.text = torch.from_numpy(text).long()
        self.context_size = context_size
    
    def __len__(self):
        return len(self.text) - self.context_size
    
    def __getitem__(self, idx):
        return (
            self.text[idx : idx + self.context_size],
            self.text[idx + self.context_size]
        )

class CompressorNN(nn.Module):
    def __init__(self):
        super(CompressorNN, self).__init__()
        self.embedding = nn.Embedding(256, EMBED_DIM)
        
        # Architecture: Embed -> 32 -> 8 -> 256
        input_dim = CONTEXT_SIZE * EMBED_DIM
        self.fc1 = nn.Linear(input_dim, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc_out = nn.Linear(HIDDEN2, 256)
        
    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        
        # Leaky ReLU
        x = F.leaky_relu(self.fc1(embeds), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        logits = self.fc_out(x)
        return logits

def load_books(limit_books=32):
    files = sorted(glob.glob("books/*.txt"))[:limit_books]
    print(f"Loading {len(files)} books...")
    data = []
    for f in files:
        try:
            with open(f, "rb") as bf:
                data.append(bf.read())
        except: pass
    full_text = b"".join(data)
    import numpy as np
    arr = np.frombuffer(full_text, dtype=np.uint8)
    return arr

def train_and_evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loading
    raw_data = load_books(32)
    if len(raw_data) > 5_000_000:
        print(f"Truncating {len(raw_data)} bytes to 5MB...")
        raw_data = raw_data[:5_000_000]
        
    dataset = TextDataset(raw_data, CONTEXT_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # pin_memory disabled for MPS compat check
    
    model = CompressorNN().to(device)
    # RAdam Optimizer
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Embed(4)->H1({HIDDEN1})->H2({HIDDEN2})->Out(256). Params: {total_params}")
    print(f"Training for {TRAIN_TIME_SECONDS} seconds...")
    
    model.train()
    start_time = time.time()
    last_log_time = start_time
    step = 0
    epoch = 0
    
    # Training Loop based on Time
    while (time.time() - start_time) < TRAIN_TIME_SECONDS:
        epoch += 1
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            step += 1
            now = time.time()
            if now - last_log_time >= 5.0:
                elapsed = now - start_time
                bpc = loss.item() / math.log(2)
                # Estimate speed
                sps = step / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.0f}s] Epoch {epoch} Step {step} | Loss: {loss.item():.4f} ({bpc:.3f} bpc) | {sps:.1f} steps/s")
                last_log_time = now
            
            if (now - start_time) >= TRAIN_TIME_SECONDS:
                break
                
    print("\nTraining Finished. Calculating Compression Stats...")
    
    # Evaluate
    # Use sum for total bits
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    model.eval()
    
    total_nll = 0
    count = 0
    
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_nll += loss.item()
            count += x.size(0)
            
    total_bits = total_nll / math.log(2)
    total_bytes = total_bits / 8
    ratio = total_bytes / len(raw_data)
    bpc = total_bits / count
    
    print("="*60)
    print(f"RESULTS (32->8 Leaky RAdam)")
    print("-" * 60)
    print(f"Original Size:   {len(raw_data)} bytes")
    print(f"Compressed Size: {int(total_bytes)} bytes")
    print(f"Ratio:           {ratio:.4f}")
    print(f"Bits Per Char:   {bpc:.4f}")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate()
