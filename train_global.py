import os
import struct

def train_global_model():
    # 64x64 context = 6 bits for context (shift = 2)
    counts = [[1 for _ in range(256)] for _ in range(64)]
    
    books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith(".txt")]
    print(f"Training global 64x64 model on {len(books)} books...")
    
    for book in books:
        with open(book, 'rb') as f:
            data = f.read()
            for i in range(1, len(data)):
                ctx = data[i-1] >> 2
                counts[ctx][data[i]] += 1

    # Normalize to 12-bit (4096)
    M = 4096
    model_flat = []
    
    for ctx in range(64):
        total = sum(counts[ctx])
        freqs = [1] * 256
        allocated = 256
        
        # Proportional distribution
        for sym in range(256):
            if counts[ctx][sym] > 0:
                share = (counts[ctx][sym] * (M - 256)) // total
                freqs[sym] += share
                allocated += share
        
        # fixup
        if allocated < M:
            freqs[255] += (M - allocated)
        elif allocated > M:
            excess = allocated - M
            # naive subtraction from largest? or valid ones.
            # python simplistic fixup
            for sym in range(256):
                if excess == 0: break
                if freqs[sym] > 1:
                    take = min(excess, freqs[sym] - 1)
                    freqs[sym] -= take
                    excess -= take
                    
        model_flat.extend(freqs)
        
    print("Writing static_64x64.bin...")
    with open("static_64x64.bin", "wb") as f:
        for val in model_flat:
            f.write(struct.pack('H', val))

if __name__ == "__main__":
    train_global_model()
