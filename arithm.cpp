#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <map>

using namespace std;

typedef __int128_t int128;

const uint64_t PRECISION = 32;
const uint64_t MAX_CODE = 0xFFFFFFFFULL;
const uint64_t HALF_CODE = 0x80000000ULL;
const uint64_t QUARTER_CODE = 0x40000000ULL;
const uint64_t THREE_QUARTER_CODE = 0xC0000000ULL;

struct SymbolRange {
    uint64_t low;
    uint64_t high;
    uint64_t total;
};

class BitWriter {
    ostream& out;
    uint8_t buffer = 0;
    int count = 0;
public:
    BitWriter(ostream& o) : out(o) {}
    void write(bool b) {
        if (b) buffer |= (1 << (7 - count));
        count++;
        if (count == 8) {
            out.put(buffer);
            buffer = 0;
            count = 0;
        }
    }
    void flush() {
        if (count > 0) {
            out.put(buffer);
            buffer = 0;
            count = 0;
        }
    }
};

class BitReader {
    istream& in;
    uint8_t buffer = 0;
    int count = 8;
public:
    BitReader(istream& i) : in(i) {}
    bool read() {
        if (count == 8) {
            if (!in.get((char&)buffer)) buffer = 0;
            count = 0;
        }
        bool b = (buffer >> (7 - count)) & 1;
        count++;
        return b;
    }
};

class Encoder {
    uint64_t low = 0;
    uint64_t high = MAX_CODE;
    uint64_t pending = 0;
    BitWriter& writer;

    void output_bit(bool b) {
        writer.write(b);
        while (pending > 0) {
            writer.write(!b);
            pending--;
        }
    }

public:
    Encoder(BitWriter& w) : writer(w) {}

    void encode(SymbolRange r) {
        int128 range = (int128)high - low + 1;
        uint64_t next_high = low + (uint64_t)((range * r.high) / r.total) - 1;
        uint64_t next_low = low + (uint64_t)((range * r.low) / r.total);
        high = next_high;
        low = next_low;

        while (true) {
            if (high < HALF_CODE) {
                output_bit(false);
            } else if (low >= HALF_CODE) {
                output_bit(true);
                low -= HALF_CODE;
                high -= HALF_CODE;
            } else if (low >= QUARTER_CODE && high < THREE_QUARTER_CODE) {
                pending++;
                low -= QUARTER_CODE;
                high -= QUARTER_CODE;
            } else {
                break;
            }
            low = (low << 1) & MAX_CODE;
            high = ((high << 1) | 1) & MAX_CODE;
        }
    }

    void finalize() {
        pending++;
        if (low < QUARTER_CODE) output_bit(false);
        else output_bit(true);
        writer.flush();
    }
};

class Decoder {
    uint64_t low = 0;
    uint64_t high = MAX_CODE;
    uint64_t value = 0;
    BitReader& reader;

public:
    Decoder(BitReader& r) : reader(r) {
        for (int i = 0; i < PRECISION; ++i) {
            value = (value << 1) | (reader.read() ? 1 : 0);
        }
    }

    uint64_t get_scaled_value(uint64_t total) {
        int128 range = (int128)high - low + 1;
        return (uint64_t)(((int128)value - low + 1) * total - 1) / (uint64_t)range;
    }

    void decode(SymbolRange r) {
        int128 range = (int128)high - low + 1;
        uint64_t next_high = low + (uint64_t)((range * r.high) / r.total) - 1;
        uint64_t next_low = low + (uint64_t)((range * r.low) / r.total);
        high = next_high;
        low = next_low;

        while (true) {
            if (high < HALF_CODE) {
                // MSB is 0
            } else if (low >= HALF_CODE) {
                value -= HALF_CODE;
                low -= HALF_CODE;
                high -= HALF_CODE;
            } else if (low >= QUARTER_CODE && high < THREE_QUARTER_CODE) {
                value -= QUARTER_CODE;
                low -= QUARTER_CODE;
                high -= QUARTER_CODE;
            } else {
                break;
            }
            low = (low << 1) & MAX_CODE;
            high = ((high << 1) | 1) & MAX_CODE;
            value = ((value << 1) | (reader.read() ? 1 : 0)) & MAX_CODE;
        }
    }
};

struct Model {
    uint64_t counts[257]; // 256 symbols + EOF
    uint64_t cumulative[258];
    uint64_t total;

    void build(const vector<uint8_t>& data) {
        for (int i = 0; i < 257; ++i) counts[i] = 1;
        for (uint8_t b : data) counts[b]++;
        total = 0;
        for (int i = 0; i < 257; ++i) {
            cumulative[i] = total;
            total += counts[i];
        }
        cumulative[257] = total;
    }

    SymbolRange get_range(int sym) {
        return {cumulative[sym], cumulative[sym + 1], total};
    }

    int find_symbol(uint64_t scaled) {
        // Binary search for speed
        auto it = upper_bound(cumulative, cumulative + 258, scaled);
        return (int)(distance(cumulative, it) - 1);
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " c/d input output" << endl;
        return 1;
    }

    string mode = argv[1];
    string in_path = argv[2];
    string out_path = argv[3];

    if (mode == "c") {
        ifstream in(in_path, ios::binary);
        vector<uint8_t> data((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        Model model;
        model.build(data);

        ofstream out(out_path, ios::binary);
        // Write file size and frequency table
        uint64_t size = data.size();
        out.write((char*)&size, 8);
        for (int i = 0; i < 257; ++i) {
            out.write((char*)&model.counts[i], 8);
        }

        BitWriter writer(out);
        Encoder encoder(writer);
        for (uint8_t b : data) {
            encoder.encode(model.get_range(b));
        }
        encoder.encode(model.get_range(256));
        encoder.finalize();
    } else {
        ifstream in(in_path, ios::binary);
        uint64_t original_size;
        in.read((char*)&original_size, 8);
        Model model;
        for (int i = 0; i < 257; ++i) {
            in.read((char*)&model.counts[i], 8);
        }
        model.total = 0;
        for (int i = 0; i < 257; ++i) {
            model.cumulative[i] = model.total;
            model.total += model.counts[i];
        }
        model.cumulative[257] = model.total;

        BitReader reader(in);
        Decoder decoder(reader);
        ofstream out(out_path, ios::binary);
        for (uint64_t i = 0; i < original_size; ++i) {
            uint64_t scaled = decoder.get_scaled_value(model.total);
            int sym = model.find_symbol(scaled);
            if (sym >= 256) break;
            out.put((uint8_t)sym);
            decoder.decode(model.get_range(sym));
        }
    }

    return 0;
}
