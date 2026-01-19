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

class SingleHiddenNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(256, EMBED_DIM)
        input_dim = CONTEXT_SIZE * EMBED_DIM
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 256)
        
    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        return self.fc_out(x)

def load_data():
    files = sorted(glob.glob("books/*.txt"))[:32]
    data = []
    for f in files:
        try:
            with open(f, "rb") as bf: data.append(bf.read())
        except: pass
    full = b"".join(data)
    arr = np.frombuffer(full, dtype=np.uint8)
    if len(arr) > MAX_DATA_SIZE: arr = arr[:MAX_DATA_SIZE]
    return arr

def run_benchmark():
    # Device Selection Logic
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    raw_data = load_data()
    print(f"Data Loaded: {len(raw_data)} bytes")
    
    dataset = TextDataset(raw_data, CONTEXT_SIZE)
    # Re-create loader inside loop or reuse? Reusing might be tricky with iterators. 
    # Just creating new one is fast.
    
    hidden_sizes = [2048, 1024, 512, 256, 128, 64, 32]
    results = []
    
    print(f"\n{'Hidden':<8} | {'Params':<8} | {'Loss':<6} | {'BPC':<6} | {'Ratio':<6} | {'Speed':<8}")
    print("-" * 75)
    
    for h in hidden_sizes:
        model = SingleHiddenNN(h).to(device)
        optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        start_time = time.time()
        steps = 0
        
        model.train()
        
        # Train for 20s
        while (time.time() - start_time) < 20.0:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                steps += 1
                if (time.time() - start_time) >= 20.0: break
        
        elapsed = time.time() - start_time
        sps = steps / elapsed
        
        # Evaluate
        eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)
        model.eval()
        total_nll = 0
        count = 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                l = F.cross_entropy(model(x), y, reduction='sum')
                total_nll += l.item()
                count += x.size(0)
        
        avg_loss = total_nll / count
        bpc = avg_loss / math.log(2)
        ratio = bpc / 8.0
        params = sum(p.numel() for p in model.parameters())
        
        print(f"{h:<8} | {params:<8} | {avg_loss:<6.4f} | {bpc:<6.4f} | {ratio:<6.4f} | {sps:<8.1f}")
        results.append((h, params, ratio, bpc))

    print("\n" + "="*60)
    print("SUMMARY (Sorted by Ratio)")
    print("-" * 60)
    results.sort(key=lambda x: x[2])
    for r in results:
        print(f"Hidden={r[0]:<5} : Ratio {r[2]:.4f} (Params: {r[1]})")

if __name__ == "__main__":
    run_benchmark()
