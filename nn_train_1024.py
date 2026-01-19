import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import math
import time
import numpy as np

# Configuration
CONTEXT_SIZE = 6 
EMBED_DIM = 4   
HIDDEN1 = 1024
BATCH_SIZE = 4096 
LEARNING_RATE = 0.005 
TRAIN_TIME_SECONDS = 600 # 10 minutes

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

class Compressor1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, EMBED_DIM)
        input_dim = CONTEXT_SIZE * EMBED_DIM
        
        # Architecture: Embed -> 1024 -> 256
        self.fc1 = nn.Linear(input_dim, HIDDEN1)
        self.fc_out = nn.Linear(HIDDEN1, 256)
        
    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(embeds), negative_slope=0.01)
        logits = self.fc_out(x)
        return logits

def load_books(limit_books=32):
    files = sorted(glob.glob("books/*.txt"))[:limit_books]
    print(f"Loading {len(files)} books...")
    data = []
    for f in files:
        try:
            with open(f, "rb") as bf: data.append(bf.read())
        except: pass
    full = b"".join(data)
    arr = np.frombuffer(full, dtype=np.uint8)
    return arr

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    raw_data = load_books(32)
    # Use 5MB subset for consistency with previous benches, or full?
    # User said "benchmark 8 hidden... train each for 10s" then "train the 1024 one".
    # Previous full bench truncated to 5MB. I will stick to 5MB to be comparable 
    # but maybe slightly larger (10MB) to avoid overfitting in 10 mins?
    # 1024 neurons on 5MB might overfit. Let's use 10MB.
    if len(raw_data) > 10_000_000:
        print(f"Truncating {len(raw_data)} bytes to 10MB...")
        raw_data = raw_data[:10_000_000]
    else:
        print(f"Using full data: {len(raw_data)} bytes")
        
    dataset = TextDataset(raw_data, CONTEXT_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Compressor1024().to(device)
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model: Embed->1024->Out. Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Training for {TRAIN_TIME_SECONDS/60:.1f} minutes...")
    
    model.train()
    start_time = time.time()
    last_log_time = start_time
    epoch = 0
    step = 0
    
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
                sps = step / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.0f}s] Ep {epoch} Step {step} | Loss: {loss.item():.4f} ({bpc:.3f} bpc) | {sps:.1f} steps/s")
                last_log_time = now
            
            if (now - start_time) >= TRAIN_TIME_SECONDS: break

    # Final Eval
    print("\nEvaluating...")
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    total_nll = 0
    count = 0
    with torch.no_grad():
        for x,y in eval_loader:
            x,y = x.to(device), y.to(device)
            total_nll += F.cross_entropy(model(x), y, reduction='sum').item()
            count += x.size(0)
            
    total_bytes = (total_nll / math.log(2)) / 8
    ratio = total_bytes / len(raw_data)
    print("="*60)
    print(f"FINAL RESULT (1024 Hidden, 10m Train)")
    print(f"Original:   {len(raw_data)}")
    print(f"Compressed: {int(total_bytes)}")
    print(f"Ratio:      {ratio:.4f}")
    print(f"BPC:        {(total_nll/math.log(2)/count):.4f}")
    print("="*60)

if __name__ == "__main__":
    train()
