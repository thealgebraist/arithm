#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <array>
#include <memory> 

using namespace std;

// Split-Table Bayesian rANS Coder
// 1. Static Table (64x64) - Global Knowledge
// 2. Dynamic Table (64x64) - Local Adaptation
// Context: 6 bits (Byte >> 2)

using State = uint32_t;
constexpr uint32_t L = 1 << 16;
constexpr uint32_t M_BITS = 12;
constexpr uint32_t M = 1 << M_BITS;

struct FreqTable {
    uint16_t freq[256];
    uint16_t start[256];
};

struct DecodeSlot {
    uint8_t sym;
};

// Two models: 0=Static, 1=Dynamic
struct DualModel {
    vector<FreqTable> t_static;
    vector<FreqTable> t_dynamic;
    // Pre-mixed decoder tables (optional optimization, but we compute on fly or cache?)
    // Mixing on the fly for rANS is expensive for DECODE because we need slot->sym mapping.
    // For 64x64=4096 contexts it is cheap to precompute the MIXED table at start.
    // "Split table" implies we utilize both. The best way is to MIX them.
    // F_mix = (F_static + F_dynamic) / 2
    // Let's build a "Mixed" table set for actual coding.
    vector<FreqTable> t_mixed;
    vector<vector<DecodeSlot>> d_mixed;
    
    DualModel() {
        t_static.resize(64);
        t_dynamic.resize(64);
        t_mixed.resize(64);
        d_mixed.resize(64, vector<DecodeSlot>(M));
    }
};

DualModel model;

void load_static_model(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) {
        // Fallback: Uniform
        cerr << "Warning: static_64x64.bin not found, using uniform." << endl;
        for(int i=0; i<64; ++i) {
            for(int j=0; j<256; ++j) model.t_static[i].freq[j] = M/256;
        }
        return;
    }
    for(int i=0; i<64; ++i) {
        in.read((char*)model.t_static[i].freq, 256*2);
    }
}

void train_dynamic(const vector<uint8_t>& data) {
    vector<array<uint32_t, 256>> counts(64);
    for(auto& arr : counts) arr.fill(1);
    for(size_t i=1; i<data.size(); ++i) {
        counts[data[i-1] >> 2][data[i]]++;
    }
    
    for(int i=0; i<64; ++i) {
        uint32_t total = 0;
        for(uint32_t c : counts[i]) total += c;
        
        uint32_t allocated = 256;
        for(int j=0; j<256; ++j) {
            uint32_t f = (uint64_t)counts[i][j] * (M - 256) / total;
            model.t_dynamic[i].freq[j] = 1 + f;
            allocated += f;
        }
        // Fixup
        if (allocated < M) model.t_dynamic[i].freq[255] += (M - allocated);
        else {
            uint32_t excess = allocated - M;
            for(int k=0; k<256 && excess > 0; ++k) {
                if(model.t_dynamic[i].freq[k] > 1) {
                    uint32_t take = min(excess, (uint32_t)model.t_dynamic[i].freq[k]-1);
                    model.t_dynamic[i].freq[k] -= take;
                    excess -= take;
                }
            }
        }
    }
}

// Mix Static and Dynamic to create the Operational Table
void build_mixed_model() {
    for(int i=0; i<64; ++i) {
        uint32_t start = 0;
        // Average frequencies
        // We need to ensure sum is exactly M
        uint32_t allocated = 0;
        for(int j=0; j<256; ++j) {
            uint32_t f = (model.t_static[i].freq[j] + model.t_dynamic[i].freq[j]) / 2;
            if (f == 0) f = 1; // Safety
            model.t_mixed[i].freq[j] = f;
            allocated += f;
        }
        
        // Final Fixup for Mixed
        if (allocated < M) model.t_mixed[i].freq[255] += (M - allocated);
        else if (allocated > M) {
            uint32_t excess = allocated - M;
            for(int k=0; k<256 && excess > 0; ++k) {
                if(model.t_mixed[i].freq[k] > 1) {
                    model.t_mixed[i].freq[k]--; excess--;
                }
            }
        }

        // Build Start & Decode Map
        for(int j=0; j<256; ++j) {
            model.t_mixed[i].start[j] = start;
            uint16_t f = model.t_mixed[i].freq[j];
            for(int k=0; k<f; ++k) model.d_mixed[i][start + k].sym = j;
            start += f;
        }
    }
}

void write_dynamic_model(ostream& out) {
    // Sparse write
    for(int i=0; i<64; ++i) {
        vector<pair<uint8_t, uint16_t>> sparse;
        for(int j=0; j<256; ++j) {
            if (model.t_dynamic[i].freq[j] != 1) sparse.push_back({(uint8_t)j, model.t_dynamic[i].freq[j]});
        }
        uint16_t count = sparse.size();
        out.write((char*)&count, 2);
        for(auto& p : sparse) {
            out.write((char*)&p.first, 1);
            out.write((char*)&p.second, 2);
        }
    }
}

void read_dynamic_model(istream& in) {
    for(int i=0; i<64; ++i) {
        for(int j=0; j<256; ++j) model.t_dynamic[i].freq[j] = 1;
        uint16_t count; in.read((char*)&count, 2);
        for(int k=0; k<count; ++k) {
            uint8_t sym; uint16_t freq;
            in.read((char*)&sym, 1);
            in.read((char*)&freq, 2);
            model.t_dynamic[i].freq[sym] = freq;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    // argv[1] is mode (padding), ignored for this specific test structure 
    // but kept for compatibility with benchmark script calls
    string act = argv[2], in_p = argv[3], out_p = argv[4];

    load_static_model("static_64x64.bin");

    if (act == "c") {
        ifstream in(in_p, ios::binary);
        vector<uint8_t> data((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        if (data.empty()) return 0;

        train_dynamic(data);
        build_mixed_model();
        
        ofstream out(out_p, ios::binary);
        uint64_t sz = data.size(); out.write((char*)&sz, 8);
        write_dynamic_model(out); // Header overhead is smaller (64 contexts vs 256)

        State x = L;
        vector<uint8_t> bits;
        for (int i = (int)data.size()-1; i >= 0; --i) {
            uint8_t prev = (i > 0) ? (data[i-1] >> 2) : 0;
            uint8_t sym = data[i];
            
            uint32_t f = model.t_mixed[prev].freq[sym];
            uint32_t s = model.t_mixed[prev].start[sym];
            
            // Standard rANS Step
            uint32_t max_val = ((L >> M_BITS) << 8) * f;
            while (x >= max_val) { bits.push_back(x & 0xFF); x >>= 8; }
            x = ((x / f) << M_BITS) + (x % f) + s;
        }
        for(int i=0; i<4; ++i) { bits.push_back(x & 0xFF); x >>= 8; }
        out.write((char*)bits.data(), bits.size());
    } else {
        ifstream in(in_p, ios::binary);
        uint64_t sz; in.read((char*)&sz, 8);
        if (sz == 0) return 0;
        
        read_dynamic_model(in); // Load local stats
        build_mixed_model();    // Merge with static stats

        vector<uint8_t> bits((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        State x = 0;
        size_t pos = bits.size();
        for(int i=0; i<4; ++i) x = (x << 8) | bits[--pos];

        vector<uint8_t> res; res.reserve(sz);
        uint8_t prev = 0;
        for(uint64_t i=0; i<sz; ++i) {
            uint8_t ctx = prev >> 2;
            uint32_t slot = x & (M-1);
            
            // Fast Decode Lookup from Mixed Table
            uint8_t sym = model.d_mixed[ctx][slot].sym;
            res.push_back(sym);
            
            uint32_t f = model.t_mixed[ctx].freq[sym];
            uint32_t s = model.t_mixed[ctx].start[sym];
            
            x = f * (x >> M_BITS) + (x & (M-1)) - s;
            while (x < L && pos > 0) x = (x << 8) | bits[--pos];
            prev = sym;
        }
        ofstream out(out_p, ios::binary);
        out.write((char*)res.data(), res.size());
    }
    return 0;
}
