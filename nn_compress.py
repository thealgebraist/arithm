import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import glob
import math

# Configuration
CONTEXT_SIZE = 6 # Previous contexts
EMBED_DIM = 4   # Small embedding
HIDDEN_SIZE = 16 # Requested "16 neuron" layer
BATCH_SIZE = 4096 # Large batch for speed
EPOCHS = 2
LEARNING_RATE = 0.01

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
        # Input Embedding: 256 chars -> 4 dim
        self.embedding = nn.Embedding(256, EMBED_DIM)
        
        # FFNN
        # Input: CONTEXT_SIZE * EMBED_DIM
        # Hidden: 16
        # Output: 256 logits
        input_dim = CONTEXT_SIZE * EMBED_DIM
        
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 256)
        
    def forward(self, x):
        # x: [Batch, Context]
        embeds = self.embedding(x) # [Batch, Context, Embed]
        # Flatten
        embeds = embeds.view(embeds.size(0), -1)
        
        h = F.relu(self.fc1(embeds))
        logits = self.fc2(h)
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
    # Convert to numpy uint8 then torch
    import numpy as np
    arr = np.frombuffer(full_text, dtype=np.uint8)
    return arr

def train_and_evaluate():
    # Detect Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    raw_data = load_books(32)
    # Truncate for speed if massive (e.g. > 10MB)
    if len(raw_data) > 5_000_000:
        print(f"Truncating {len(raw_data)} bytes to 5MB for fast training demo...")
        raw_data = raw_data[:5_000_000]
        
    dataset = TextDataset(raw_data, CONTEXT_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    model = CompressorNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training on {len(raw_data)} bytes...")
    
    model.train()
    for ohm in range(EPOCHS):
        total_loss = 0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            if steps % 200 == 0:
                print(f"  Epoch {ohm+1}, Step {steps}, Loss: {loss.item():.4f} ({loss.item()/math.log(2):.2f} bits/char)")

    # Evaluation (Compression Strength)
    # Ideal Compressed Size = Sum of CrossEntropy(bits)
    # We do a pass over the data (no shuffle)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    model.eval()
    
    total_nll = 0
    count = 0
    
    print("\nCalculating Compressed Size...")
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) # [B, 256]
            
            # CrossEntropy is -log(p_true)
            # Use reduction='sum'
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_nll += loss.item()
            count += x.size(0)
            
    # Total Bits = Total Nats / ln(2)
    total_bits = total_nll / math.log(2)
    total_bytes = total_bits / 8
    
    orig_bytes = len(raw_data)
    ratio = total_bytes / orig_bytes
    bpc = total_bits / count
    
    print("="*60)
    print(f"RESULTS (16-Neuron FFNN)")
    print("-" * 60)
    print(f"Original Size:   {orig_bytes} bytes")
    print(f"Compressed Size: {int(total_bytes)} bytes (Theoretical)")
    print(f"Ratio:           {ratio:.4f}")
    print(f"Bits Per Char:   {bpc:.4f}")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate()
