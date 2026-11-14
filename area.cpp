#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct Tap {
    int source;
    float weight;
};
struct Table {
    std::vector<Tap> taps;
    std::vector<int> offsets;
};

static Table make_table(int input_size, int output_size) {
    const double scale = 1.0 / (double(output_size) / input_size);
    Table table;
    table.taps.reserve(input_size + output_size);
    table.offsets.reserve(output_size + 1);
    for (int i = 0; i < output_size; ++i) {
        table.offsets.push_back(int(table.taps.size()));
        const double start = i * scale;
        const double end = start + scale;
        const double span = std::min(scale, input_size - start);
        const int last = std::min(int(std::floor(end)), input_size - 1);
        const int first = std::min(int(std::ceil(start)), last);
        if (first - start > 1e-3)
            table.taps.push_back({first - 1, float((first - start) / span)});
        for (int j = first; j < last; ++j)
            table.taps.push_back({j, float(1.0 / span)});
        if (end - last > 1e-3)
            table.taps.push_back({last, float(std::min({end - last, 1.0, span}) / span)});
    }
    table.offsets.push_back(int(table.taps.size()));
    return table;
}

template<bool fused>
static inline float accumulate(float sum, float value, float weight) {
    if constexpr (fused)
        return std::fma(value, weight, sum);
    return sum + value * weight;
}

template<bool grouped, bool reuse, bool fused_h, bool fused_v>
static void run(const uint8_t* input, intptr_t stride, int ow, int oh, int channels,
                uint8_t* output, const Table& x, const Table& y) {
    const int width = ow * channels;
    std::vector<float> buffers(width * 2);
    float* row = buffers.data();
    float* sum = row + width;
    int previous_source_row = -1;
    for (int dy = 0; dy < oh; ++dy) {
        for (int yi = y.offsets[dy]; yi < y.offsets[dy+1]; ++yi) {
            const auto [sy, wy] = y.taps[yi];
            if (!reuse || sy != previous_source_row) {
                const uint8_t* src = input + sy * stride;
                if constexpr (grouped) {
                    for (int dx = 0; dx < ow; ++dx) {
                        const int begin = x.offsets[dx], end = x.offsets[dx+1];
                        for (int c = 0; c < channels; ++c) {
                            float v = src[x.taps[begin].source * channels + c] * x.taps[begin].weight;
                            for (int i = begin + 1; i < end; ++i)
                                v = accumulate<fused_h>(v, src[x.taps[i].source * channels + c], x.taps[i].weight);
                            row[dx * channels + c] = v;
                        }
                    }
                } else {
                    std::fill_n(row, width, 0.0f);
                    for (int dx = 0; dx < ow; ++dx) {
                        for (int i = x.offsets[dx]; i < x.offsets[dx+1]; ++i) {
                            const auto [sx, wx] = x.taps[i];
                            for (int c = 0; c < channels; ++c) {
                                const int k = dx * channels + c;
                                row[k] = accumulate<fused_h>(row[k], src[sx * channels + c], wx);
                            }
                        }
                    }
                }
                previous_source_row = sy;
            }
            if (yi == y.offsets[dy]) {
                for (int i = 0; i < width; ++i)
                    sum[i] = row[i] * wy;
            } else {
                for (int i = 0; i < width; ++i)
                    sum[i] = accumulate<fused_v>(sum[i], row[i], wy);
            }
        }
        for (int i = 0; i < width; ++i)
            output[dy * width + i] = uint8_t(std::nearbyint(sum[i]));
    }
}

extern "C" int area(const uint8_t* in, int ih, int iw, intptr_t stride, int oh, int ow, int channels,
                    uint8_t* out, int mode) {
    const Table x = make_table(iw, ow), y = make_table(ih, oh);
    switch (mode) {
    case 0: run<false,false,false,false>(in,stride,ow,oh,channels,out,x,y); break;
    case 1: run<false,true,false,false>(in,stride,ow,oh,channels,out,x,y); break;
    case 2: run<true,true,false,false>(in,stride,ow,oh,channels,out,x,y); break;
    case 3: run<true,true,true,false>(in,stride,ow,oh,channels,out,x,y); break;
    case 4: run<true,true,true,true>(in,stride,ow,oh,channels,out,x,y); break;
    case 5: run<true,true,false,true>(in,stride,ow,oh,channels,out,x,y); break;
    }
    return int(x.taps.size() + y.taps.size());
}
