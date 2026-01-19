import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
import math
import time

CONTEXT_SIZE = 6 
EMBED_DIM = 4   
BATCH_SIZE = 4096 
LEARNING_RATE = 0.005
MAX_DATA_SIZE = 5_000_000

class TextDataset(Dataset):
    def __init__(self, text, context_size):
        self.text = torch.from_numpy(text).long()
        self.context_size = context_size
    def __len__(self):
        return len(self.text) - self.context_size
    def __getitem__(self, idx):
        return (self.text[idx : idx + self.context_size], self.text[idx + self.context_size])

class SweepNN(nn.Module):
    def __init__(self, h1, h2=0):
        super().__init__()
        self.embedding = nn.Embedding(256, EMBED_DIM)
        input_dim = CONTEXT_SIZE * EMBED_DIM
        self.h2 = h2
        
        self.fc1 = nn.Linear(input_dim, h1)
        if h2 > 0:
            self.fc2 = nn.Linear(h1, h2)
            self.fc_out = nn.Linear(h2, 256)
        else:
            self.fc_out = nn.Linear(h1, 256)
        
    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        if self.h2 > 0:
            x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        return self.fc_out(x)

def load_data():
    files = sorted(glob.glob("books/*.txt"))[:32]
    data = []
    for f in files:
        try:
            with open(f, "rb") as bf: data.append(bf.read())
        except: pass
    full = b"".join(data)
    import numpy as np
    arr = np.frombuffer(full, dtype=np.uint8)
    if len(arr) > MAX_DATA_SIZE: arr = arr[:MAX_DATA_SIZE]
    return arr

def run_sweep():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    raw_data = load_data()
    print(f"Data size: {len(raw_data)} bytes")
    
    dataset = TextDataset(raw_data, CONTEXT_SIZE)
    # Persist loader to avoid overhead? No, shuffle state needs reset usually.
    # Recreating loader is cheap compared to data copy.
    
    configs = [
        (8, 8), (4, 8), (2, 4), (8, 4), (8, 2),
        (64, 8), (128, 16), (256, 16),
        (512, 0), (1024, 0)
    ]
    
    results = []
    
    print(f"\n{'H1':<5} | {'H2':<5} | {'Params':<8} | {'Loss':<6} | {'BPC':<6} | {'Ratio':<6}")
    print("-" * 60)
    
    for h1, h2 in configs:
        model = SweepNN(h1, h2).to(device)
        optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        start = time.time()
        model.train()
        
        # Train for 10s
        while (time.time() - start) < 10.0:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                if (time.time() - start) >= 10.0: break
        
        # Evaluate
        eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)
        model.eval()
        total_nll = 0
        count = 0
        with torch.no_grad():
            for x,y in eval_loader:
                x,y = x.to(device), y.to(device)
                l = F.cross_entropy(model(x), y, reduction='sum')
                total_nll += l.item()
                count += x.size(0)
        
        avg_loss = total_nll / count
        bpc = avg_loss / math.log(2)
        ratio = (bpc / 8)
        params = sum(p.numel() for p in model.parameters())
        
        results.append((h1, h2, params, avg_loss, bpc, ratio))
        print(f"{h1:<5} | {h2:<5} | {params:<8} | {avg_loss:<6.4f} | {bpc:<6.4f} | {ratio:<6.4f}")

    print("\n" + "="*60)
    print("FINAL SUMMARY (Sorted by Best Ratio)")
    print("-" * 60)
    results.sort(key=lambda x: x[5])
    for r in results:
        print(f"H1={r[0]:<4} H2={r[1]:<4} : Ratio {r[5]:.4f} (Params: {r[2]})")

if __name__ == "__main__":
    run_sweep()
