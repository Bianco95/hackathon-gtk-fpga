// Vitis HLS kernel "pairedges", factorized pair MLP (hybrid_inference, phase 9b). Same interface and output
// as pairedges_kernel.cpp (see there for the memory layout and the candidate packing).
//
// The pair head is fc1 (64 -> 64) + ReLU + fc2 (64 -> 2) on concat(emb[i], emb[j]). fc1 is linear before
// the ReLU, so its accumulator splits into a part of hit i and a part of hit j:
//     acc1[o] = b1[o] + sum_k<32 x_i[k] W1[k][o] + sum_k<32 x_j[k] W1[32+k][o] = U[i][o] + V[j][o]
// U (bias folded in) and V are computed once per hit; a pair costs 64 additions + fc2. The products and the
// sums are those of hls4ml's dense_latency (product cast to accum_t, sums in accum_t: ap_fixed WRAP
// arithmetic, exact and order-independent), followed by the same casts, so the logits are bit-identical to
// the hls4ml core on the same pair. Types and weights come from the generated firmware (parameters.h,
// names mapped by pairhead_names.h).

#include <ap_int.h>
#include <stdint.h>

#include "myproject.h"    // defines.h: input_t, result_t, the layer types
#include "parameters.h"   // layer configs and weights

// Names of the hls4ml objects used below. hls4ml numbers layers by position, which differs between frontends
// (Keras: fc1 = 2, relu = 4, fc2 = 5; PyTorch: fc1 = 2, relu = 3, fc2 = 4), so a generated pairhead_names.h maps
// them; without it, the Keras-path names of the study's first build.
#if __has_include("pairhead_names.h")
#include "pairhead_names.h"
#else
#define PH_FC1_CONFIG config2
#define PH_FC2_CONFIG config5
#define PH_FC1_OUT_T layer2_t
#define PH_RELU_OUT_T layer4_t
#define PH_W1 w2
#define PH_B1 b2
#define PH_W2 w5
#define PH_B2 b5
#define PH_W1_FILE "w2.txt"
#define PH_B1_FILE "b2.txt"
#define PH_W2_FILE "w5.txt"
#define PH_B2_FILE "b5.txt"
#endif

#ifndef PE_N_MAX
#define PE_N_MAX 256
#endif
#ifndef PE_KP
#define PE_KP 2           // fc1 inputs per clock in the per-hit precompute (64 * 2 * PE_KP multipliers)
#endif
#define PE_D 32           // embedding size; the pair-head input is 2 * PE_D
#define PE_H 64           // fc1 width
#define PE_FLOATS_PER_WORD 16
#define PE_WORDS_PER_HIT (PE_D / PE_FLOATS_PER_WORD)
#define PE_CAND_PER_WORD 8
#define PE_K_MAX (PE_N_MAX * (PE_N_MAX + 1) / 2)

typedef ap_uint<512> word_t;
typedef ap_uint<64> cand_t;
typedef PH_FC1_CONFIG::accum_t acc1_t;
typedef PH_FC2_CONFIG::accum_t acc2_t;
typedef PH_FC1_CONFIG::weight_t w1_t;
typedef PH_FC1_CONFIG::bias_t b1_t;
typedef PH_FC2_CONFIG::weight_t w2_t;
typedef PH_FC2_CONFIG::bias_t b2_t;
static_assert(PH_FC1_CONFIG::n_in == 2 * PE_D && PH_FC1_CONFIG::n_out == PE_H && PH_FC2_CONFIG::n_in == PE_H &&
              PH_FC2_CONFIG::n_out == 2,
              "pair head shape");

static void load_weights() {
#ifndef __SYNTHESIS__
    static bool loaded = false;
    if (!loaded) {
        nnet::load_weights_from_txt<w1_t, 4096>(PH_W1, PH_W1_FILE);
        nnet::load_weights_from_txt<b1_t, 64>(PH_B1, PH_B1_FILE);
        nnet::load_weights_from_txt<w2_t, 128>(PH_W2, PH_W2_FILE);
        nnet::load_weights_from_txt<b2_t, 2>(PH_B2, PH_B2_FILE);
        loaded = true;
    }
#endif
}

static void load_event(const word_t* emb, int hit_base, int n, input_t E[PE_N_MAX][PE_D]) {
LoadHits:
    for (int w = 0; w < n * PE_WORDS_PER_HIT; w++) {
        #pragma HLS PIPELINE II = 1
        #pragma HLS LOOP_TRIPCOUNT min = 2 max = 2 * PE_N_MAX
        word_t v = emb[hit_base * PE_WORDS_PER_HIT + w];
        int h = w / PE_WORDS_PER_HIT, f0 = (w % PE_WORDS_PER_HIT) * PE_FLOATS_PER_WORD;
        for (int f = 0; f < PE_FLOATS_PER_WORD; f++) {
            #pragma HLS UNROLL
            uint32_t bits = v.range(32 * f + 31, 32 * f);
            union {
                uint32_t u;
                float x;
            } c;
            c.u = bits;
            E[h][f0 + f] = (input_t)c.x;   // float -> ap_fixed with input_t's rounding / saturation
        }
    }
}

// U[h][o] = b1[o] + sum_k x_h[k] W1[k][o],  V[h][o] = sum_k x_h[k] W1[32 + k][o]   (k < 32)
static void precompute(int n, const input_t E[PE_N_MAX][PE_D], acc1_t U[PE_N_MAX][PE_H], acc1_t V[PE_N_MAX][PE_H]) {
Hits:
    for (int h = 0; h < n; h++) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = PE_N_MAX
        acc1_t u[PE_H], v[PE_H];
        #pragma HLS ARRAY_PARTITION variable = u complete
        #pragma HLS ARRAY_PARTITION variable = v complete
        for (int o = 0; o < PE_H; o++) {
            #pragma HLS UNROLL
            u[o] = (acc1_t)PH_B1[o];
            v[o] = 0;
        }
    Inputs:
        for (int kb = 0; kb < PE_D; kb += PE_KP) {
            #pragma HLS PIPELINE II = 1
            for (int kk = 0; kk < PE_KP; kk++) {
                #pragma HLS UNROLL
                input_t x = E[h][kb + kk];
                for (int o = 0; o < PE_H; o++) {
                    #pragma HLS UNROLL
                    u[o] += (acc1_t)PH_FC1_CONFIG::template product<input_t, w1_t>::product(x, PH_W1[(kb + kk) * PE_H + o]);
                    v[o] += (acc1_t)PH_FC1_CONFIG::template product<input_t, w1_t>::product(x, PH_W1[(PE_D + kb + kk) * PE_H + o]);
                }
            }
        }
        for (int o = 0; o < PE_H; o++) {
            #pragma HLS UNROLL
            U[h][o] = u[o];
            V[h][o] = v[o];
        }
    }
}

static int pairs(int n, int thr_raw, const acc1_t U[PE_N_MAX][PE_H], const acc1_t V[PE_N_MAX][PE_H], cand_t C[PE_K_MAX]) {
    int k_total = n * (n + 1) / 2;
    int i = 0, j = 0, cnt = 0;
PairLoop:
    for (int k = 0; k < k_total; k++) {
        #pragma HLS PIPELINE II = 1
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = PE_K_MAX
        PH_RELU_OUT_T hid[PE_H];
        #pragma HLS ARRAY_PARTITION variable = hid complete
        for (int o = 0; o < PE_H; o++) {
            #pragma HLS UNROLL
            acc1_t a = U[i][o] + V[j][o];
            PH_FC1_OUT_T z = nnet::cast<input_t, PH_FC1_OUT_T, PH_FC1_CONFIG>(a);   // fc1 output (hls4ml cast)
            hid[o] = z > 0 ? (PH_RELU_OUT_T)z : (PH_RELU_OUT_T)0;                  // relu (hls4ml relu)
        }
        result_t y[2];
        for (int c = 0; c < 2; c++) {
            #pragma HLS UNROLL
            acc2_t a = (acc2_t)PH_B2[c];
            for (int o = 0; o < PE_H; o++) {
                #pragma HLS UNROLL
                a += (acc2_t)PH_FC2_CONFIG::template product<PH_RELU_OUT_T, w2_t>::product(hid[o], PH_W2[o * 2 + c]);
            }
            y[c] = nnet::cast<PH_RELU_OUT_T, result_t, PH_FC2_CONFIG>(a);
        }
        ap_int<result_t::width> r0 = y[0].range(), r1 = y[1].range();   // raw bits of the logits
        ap_int<16> l0 = r0, l1 = r1;                                      // sign-extended
        if ((int)l1 - (int)l0 > thr_raw) {
            cand_t c = 0;
            c.range(15, 0) = i;
            c.range(31, 16) = j;
            c.range(47, 32) = l0;
            c.range(63, 48) = l1;
            C[cnt] = c;
            cnt++;
        }
        if (j == n - 1) {
            i++;
            j = i;
        } else {
            j++;
        }
    }
    return cnt;
}

static int write_event(int word_base, int cnt, const cand_t C[PE_K_MAX], word_t* out) {
    int words = (cnt + PE_CAND_PER_WORD - 1) / PE_CAND_PER_WORD;
WriteCand:
    for (int w = 0; w < words; w++) {
        #pragma HLS PIPELINE II = 1
        #pragma HLS LOOP_TRIPCOUNT min = 0 max = PE_K_MAX / PE_CAND_PER_WORD
        word_t v = 0;
        for (int c = 0; c < PE_CAND_PER_WORD; c++) {
            #pragma HLS UNROLL
            int idx = w * PE_CAND_PER_WORD + c;
            v.range(64 * c + 63, 64 * c) = (idx < cnt) ? C[idx] : (cand_t)0;
        }
        out[word_base + w] = v;
    }
    return words;
}

extern "C" void pairedges(const word_t* emb, const int* n_hits, int n_events, int thr_raw, word_t* out, int* n_cand) {
    #pragma HLS INTERFACE m_axi port = emb offset = slave bundle = gmem0 max_read_burst_length = 64 depth = 4096
    #pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem0 max_write_burst_length = 64 depth = 65536
    #pragma HLS INTERFACE m_axi port = n_hits offset = slave bundle = gmem1 depth = 64
    #pragma HLS INTERFACE m_axi port = n_cand offset = slave bundle = gmem1 depth = 64
    #pragma HLS ARRAY_PARTITION variable = PH_W1 complete
    #pragma HLS ARRAY_PARTITION variable = PH_B1 complete
    #pragma HLS ARRAY_PARTITION variable = PH_W2 complete
    #pragma HLS ARRAY_PARTITION variable = PH_B2 complete
    load_weights();

    static input_t E[PE_N_MAX][PE_D];
    #pragma HLS ARRAY_PARTITION variable = E complete dim = 2
    static acc1_t U[PE_N_MAX][PE_H], V[PE_N_MAX][PE_H];
    #pragma HLS ARRAY_PARTITION variable = U complete dim = 2
    #pragma HLS ARRAY_PARTITION variable = V complete dim = 2
    static cand_t C[PE_K_MAX];
    #pragma HLS ARRAY_PARTITION variable = C cyclic factor = PE_CAND_PER_WORD
    #pragma HLS BIND_STORAGE variable = C type = ram_2p impl = uram

    int hit_base = 0, word_base = 0;
Events:
    for (int e = 0; e < n_events; e++) {
        #pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
        int n = n_hits[e];
        if (n > PE_N_MAX) n = PE_N_MAX;   // host checks N <= PE_N_MAX; never read out of bounds
        int cnt = 0;
        if (n > 0) {
            load_event(emb, hit_base, n, E);
            precompute(n, E, U, V);
            cnt = pairs(n, thr_raw, U, V, C);
            word_base += write_event(word_base, cnt, C, out);
        }
        n_cand[e] = cnt;
        hit_base += n_hits[e];
    }
}

// ---------------------------------------------------------------------------------------------------
// C simulation entry point for the host (ctypes): plain arrays in, same packing out.
#ifndef __SYNTHESIS__
#include <vector>
extern "C" int pairedges_csim(const float* emb, const int* n_hits, int n_events, int thr_raw, uint64_t* out,
                              int* n_cand) {
    long hits = 0, kmax_words = 0;
    for (int e = 0; e < n_events; e++) {
        if (n_hits[e] > PE_N_MAX || n_hits[e] < 0) return -1;
        hits += n_hits[e];
        long k = (long)n_hits[e] * (n_hits[e] + 1) / 2;
        kmax_words += (k + PE_CAND_PER_WORD - 1) / PE_CAND_PER_WORD;
    }
    std::vector<word_t> e_words(hits * PE_WORDS_PER_HIT + 1), o_words(kmax_words + 1);
    for (long w = 0; w < hits * PE_WORDS_PER_HIT; w++)
        for (int f = 0; f < PE_FLOATS_PER_WORD; f++) {
            union {
                float x;
                uint32_t u;
            } c;
            c.x = emb[w * PE_FLOATS_PER_WORD + f];
            e_words[w].range(32 * f + 31, 32 * f) = c.u;
        }
    pairedges(e_words.data(), n_hits, n_events, thr_raw, o_words.data(), n_cand);
    for (long w = 0; w < kmax_words; w++)
        for (int c = 0; c < PE_CAND_PER_WORD; c++) out[w * PE_CAND_PER_WORD + c] = o_words[w].range(64 * c + 63, 64 * c).to_uint64();
    return 0;
}
#endif
