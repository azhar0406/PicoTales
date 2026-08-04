/*
 * runq4_neo_host.c - Host-side C99 runner for SPEC_Q4_NEO v4 GPT-Neo models.
 *
 * Build: cc -std=c99 -Wall -Wextra -O2 -o runq4_neo_host firmware/runq4_neo_host.c -lm
 * Run:   ./runq4_neo_host --model models/tinystories-instruct-3M --max-tokens 30 < prompt.txt
 *        ./runq4_neo_host models/tinystories_3M_neo_k16384.bin "prompt" -n 30 -t 0
 */

#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEQ_LEN 512
#define MAX_TOK_LEN 256
#define MAX_PATH_LEN 512
#define MAGIC_Q4 0x51344C4Du
#define VERSION_Q4 1
#define VERSION_EXTRA_NEO_MIXED 4

typedef struct {
    int32_t vocab_size;
    int32_t max_token_len;
    uint8_t *token_data;
    uint32_t *token_offsets;
    uint16_t *token_lens;
    uint16_t *merge_left;
    uint16_t *merge_right;
    uint16_t *merge_out;
    int32_t merge_count;
} Tokenizer;

typedef struct {
    uint32_t magic;
    int32_t version;
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
    int32_t group_size;
    uint32_t flags;
    float rope_theta;
    float layer_norm_eps;
    int32_t version_extra;
    int32_t eos_token_id;
    int32_t wpe_rows;
} ModelHeader;

Tokenizer g_tok;
static uint8_t *g_model_buf;
static size_t g_model_size;
static ModelHeader g_hdr;
int g_dim, g_hidden_dim, g_n_layers, g_n_heads, g_kv_dim, g_head_dim;
int g_vocab_size, g_seq_len, g_group_size, g_eos_token_id, g_wpe_rows;
static int g_kv_f32;
static int g_logits_f32;

static float **ln1_w, **ln1_b, **ln2_w, **ln2_b;
static float *lnf_w, *lnf_b;
static float *out_b, *fc_b, *proj_b;
static uint8_t *wte_q, *wpe_q;
static uint16_t *wte_s, *wpe_s;
static int g_all_q8;
static int8_t **wq_q, **wk_q, **wv_q, **wo_q, **fc_q, **proj_q;
static uint16_t **wq_s, **wk_s, **wv_s, **wo_s, **fc_s, **proj_s;

static int8_t *k_cache, *v_cache;
static float *k_cache_f32, *v_cache_f32;
static float *k_scales, *v_scales;
static float *x, *xb, *xb2, *q_buf, *k_buf, *v_buf, *att, *hb;
static float *logits_f;
static uint16_t *logits_h;
static size_t heap_model, heap_tok_data, heap_tok_meta, heap_ptrs, heap_kv, heap_state;

static float fp16_to_f32(uint16_t h) {
    unsigned sign = (h >> 15) & 1u;
    unsigned exponent = (h >> 10) & 0x1Fu;
    unsigned mantissa = h & 0x3FFu;
    if (exponent == 0) {
        float val = ldexpf((float)mantissa, -24);
        return sign ? -val : val;
    }
    if (exponent == 0x1F) return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
    {
        float val = ldexpf((float)(mantissa | 0x400u), (int)exponent - 25);
        return sign ? -val : val;
    }
}

static uint16_t f32_to_fp16(float f) {
    union { float f; uint32_t u; } v;
    uint32_t sign, exp, mant;
    v.f = f;
    sign = (v.u >> 16) & 0x8000u;
    exp = (v.u >> 23) & 0xFFu;
    mant = v.u & 0x7FFFFFu;
    if (exp == 0xFFu) return (uint16_t)(sign | (mant ? 0x7E00u : 0x7C00u));
    if (exp > 142u) return (uint16_t)(sign | 0x7C00u);
    if (exp < 113u) {
        if (exp < 103u) return (uint16_t)sign;
        mant |= 0x800000u;
        mant >>= (126u - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    return (uint16_t)(sign | ((exp - 112u) << 10) | ((mant + 0x1000u) >> 13));
}

static void *xcalloc(size_t n, size_t sz, const char *name) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "calloc failed for %s (%zu bytes)\n", name, n * sz);
        printf("ALLOC FAIL: %s needs %zu bytes\n", name, n * sz);
        exit(1);
    }
    return p;
}

static uint8_t *load_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    size_t sz;
    if (!f) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); exit(1); }
    fseek(f, 0, SEEK_END);
    sz = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (uint8_t *)malloc(sz);
    if (!buf) { fprintf(stderr, "malloc failed for %s\n", path); exit(1); }
    if (fread(buf, 1, sz, f) != sz) { fprintf(stderr, "short read: %s\n", path); exit(1); }
    fclose(f);
    *out_size = sz;
    return buf;
}

static uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static int32_t rd_i32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static float rd_f32(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

static int tok_load(Tokenizer *tok, const char *path) {
    size_t sz, total_bytes, bytes_after_lens;
    uint8_t *buf = load_file(path, &sz);
    const uint8_t *p;
    size_t off_table, merge_off;
    tok->vocab_size = (int32_t)rd_u32(buf);
    tok->max_token_len = (int32_t)rd_u32(buf + 4);
    tok->merge_left = tok->merge_right = tok->merge_out = NULL;
    tok->merge_count = 0;
    if (tok->vocab_size <= 0 || tok->max_token_len <= 0) return -1;
    tok->token_lens = (uint16_t *)xcalloc((size_t)tok->vocab_size, sizeof(uint16_t), "token_lens");
    tok->token_offsets = (uint32_t *)xcalloc((size_t)tok->vocab_size, sizeof(uint32_t), "token_offsets");
    heap_tok_meta += (size_t)tok->vocab_size * (sizeof(uint16_t) + sizeof(uint32_t));
    p = buf + 8;
    total_bytes = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        tok->token_lens[i] = len;
        tok->token_offsets[i] = (uint32_t)total_bytes;
        total_bytes += len;
    }
    bytes_after_lens = sz - 8u - (size_t)tok->vocab_size * 2u;
    if (bytes_after_lens < total_bytes) return -1;
    tok->token_data = (uint8_t *)malloc(total_bytes ? total_bytes : 1u);
    if (!tok->token_data) return -1;
    memcpy(tok->token_data, p, total_bytes);
    heap_tok_data += total_bytes;
    {
        /* Appended offsets table starts at the next multiple of 4 after
         * token_bytes (zero-padded by the exporter); mirror the device path. */
        off_table = 8u + (size_t)tok->vocab_size * 2u + total_bytes;
        off_table = (off_table + 3u) & ~(size_t)3u;
        if (off_table + (size_t)tok->vocab_size * 4u <= sz) {
            const uint8_t *q = buf + off_table;
            for (int i = 0; i < tok->vocab_size; i++) tok->token_offsets[i] = rd_u32(q + (size_t)i * 4u);
        }
        merge_off = off_table + (size_t)tok->vocab_size * 4u;
        if (merge_off + 8u <= sz && rd_u32(buf + merge_off) == 0x31455042u) {
            uint32_t count = rd_u32(buf + merge_off + 4u);
            const uint8_t *q = buf + merge_off + 8u;
            if (merge_off + 8u + (size_t)count * 6u <= sz) {
                tok->merge_left = (uint16_t *)xcalloc(count ? count : 1u, sizeof(uint16_t), "merge_left");
                tok->merge_right = (uint16_t *)xcalloc(count ? count : 1u, sizeof(uint16_t), "merge_right");
                tok->merge_out = (uint16_t *)xcalloc(count ? count : 1u, sizeof(uint16_t), "merge_out");
                tok->merge_count = (int32_t)count;
                heap_tok_meta += (size_t)count * 3u * sizeof(uint16_t);
                memcpy(tok->merge_left, q, (size_t)count * 2u);
                memcpy(tok->merge_right, q + (size_t)count * 2u, (size_t)count * 2u);
                memcpy(tok->merge_out, q + (size_t)count * 4u, (size_t)count * 2u);
            }
        }
    }
    free(buf);
    return 0;
}

static const uint8_t *tok_str(const Tokenizer *tok, int token_id) {
    return tok->token_data + tok->token_offsets[token_id];
}

static int token_bytes_equal(const Tokenizer *tok, int token_id, const uint8_t *bytes, int len) {
    return token_id >= 0 && token_id < tok->vocab_size && tok->token_lens[token_id] == len &&
           memcmp(tok_str(tok, token_id), bytes, (size_t)len) == 0;
}

static int token_id_for_bytes(const Tokenizer *tok, const uint8_t *bytes, int len) {
    for (int tid = 0; tid < tok->vocab_size; tid++) {
        if (token_bytes_equal(tok, tid, bytes, len)) return tid;
    }
    return -1;
}

static int initial_symbol_id(const Tokenizer *tok, unsigned char c, int leading_space) {
    uint8_t buf[2];
    if (leading_space) { buf[0] = 0xC4; buf[1] = 0xA0; return token_id_for_bytes(tok, buf, 2); }
    if (c == '\n' || c == '\r') { buf[0] = 0xC4; buf[1] = 0x8A; return token_id_for_bytes(tok, buf, 2); }
    buf[0] = c;
    return token_id_for_bytes(tok, buf, 1);
}

static int find_merge_rank(const Tokenizer *tok, int left, int right, int *out_token) {
    for (int i = 0; i < tok->merge_count; i++) {
        if (tok->merge_left[i] == (uint16_t)left && tok->merge_right[i] == (uint16_t)right) {
            *out_token = tok->merge_out[i];
            return i;
        }
    }
    return -1;
}

static int bpe_merge_symbols(const Tokenizer *tok, int *symbols, int n_symbols, int *tokens, int max_tokens) {
    if (tok->merge_count <= 0) {
        int n = n_symbols < max_tokens ? n_symbols : max_tokens;
        for (int i = 0; i < n; i++) tokens[i] = symbols[i];
        return n;
    }
    while (n_symbols > 1) {
        int best_rank = 0x7fffffff, best_pos = -1, best_out = -1;
        for (int i = 0; i < n_symbols - 1; i++) {
            int out = -1;
            int rank = find_merge_rank(tok, symbols[i], symbols[i + 1], &out);
            if (rank >= 0 && rank < best_rank) { best_rank = rank; best_pos = i; best_out = out; }
        }
        if (best_pos < 0) break;
        symbols[best_pos] = best_out;
        for (int j = best_pos + 1; j < n_symbols - 1; j++) symbols[j] = symbols[j + 1];
        n_symbols--;
    }
    if (n_symbols > max_tokens) n_symbols = max_tokens;
    for (int i = 0; i < n_symbols; i++) tokens[i] = symbols[i];
    return n_symbols;
}

static int bpe_emit_piece(const Tokenizer *tok, const char *text, int start, int end, int leading_space, int *tokens, int max_tokens) {
    int symbols[MAX_TOK_LEN], n = 0;
    if (leading_space) {
        int id = initial_symbol_id(tok, ' ', 1);
        if (id >= 0 && n < MAX_TOK_LEN) symbols[n++] = id;
    }
    for (int i = start; i < end && n < MAX_TOK_LEN; i++) {
        unsigned char c = (unsigned char)text[i];
        int id = initial_symbol_id(tok, c, 0);
        if (id >= 0) symbols[n++] = id;
    }
    return bpe_merge_symbols(tok, symbols, n, tokens, max_tokens);
}

static int starts_contraction(const char *text, int pos, int len, int *out_len) {
    if (pos >= len || text[pos] != '\'') return 0;
    if (pos + 2 <= len && (text[pos + 1] == 's' || text[pos + 1] == 't' || text[pos + 1] == 'm' || text[pos + 1] == 'd')) { *out_len = 2; return 1; }
    if (pos + 3 <= len && (!strncmp(text + pos + 1, "re", 2) || !strncmp(text + pos + 1, "ve", 2) || !strncmp(text + pos + 1, "ll", 2))) { *out_len = 3; return 1; }
    return 0;
}

static int bpe_encode_core(const Tokenizer *tok, const char *text, int *tokens, int max_tokens, int initial_space) {
    int n = 0, len = (int)strlen(text), i = 0;
    while (i < len && n < max_tokens) {
        int leading_space = 0, start, clen;
        unsigned char c = (unsigned char)text[i];
        if (c == ' ' || c == '\t') {
            int j = i;
            while (j < len && (text[j] == ' ' || text[j] == '\t')) j++;
            if (j >= len) {
                for (; i < j && n < max_tokens; i++) {
                    int id = initial_symbol_id(tok, ' ', 1);
                    if (id >= 0) tokens[n++] = id;
                }
                break;
            }
            leading_space = 1;
            i = j;
            c = (unsigned char)text[i];
        } else if (initial_space) {
            leading_space = 1;
            initial_space = 0;
        }
        if (c == '\n' || c == '\r') {
            int id = initial_symbol_id(tok, c, 0);
            if (id >= 0) tokens[n++] = id;
            i++;
            continue;
        }
        start = i;
        if (starts_contraction(text, i, len, &clen)) {
            i += clen;
        } else if (isalpha(c)) {
            while (i < len && isalpha((unsigned char)text[i])) i++;
        } else if (isdigit(c)) {
            while (i < len && isdigit((unsigned char)text[i])) i++;
        } else {
            while (i < len) {
                unsigned char d = (unsigned char)text[i];
                if (d == ' ' || d == '\t' || d == '\n' || d == '\r' || isalpha(d) || isdigit(d)) break;
                if (starts_contraction(text, i, len, &clen)) break;
                i++;
            }
        }
        n += bpe_emit_piece(tok, text, start, i, leading_space, tokens + n, max_tokens - n);
    }
    return n;
}

static int bpe_encode_ex(const Tokenizer *tok, const char *text, int *tokens, int max_tokens) {
    return bpe_encode_core(tok, text, tokens, max_tokens, 0);
}

static int bpe_encode_prefix_all(const Tokenizer *tok, const char *text, int *tokens, int max_tokens) {
    return bpe_encode_core(tok, text, tokens, max_tokens, 1);
}

static int bpe_decode_id(const Tokenizer *tok, int token_id, char *out, int max_len) {
    int out_len = 0;
    int tlen = tok->token_lens[token_id];
    const uint8_t *tstr = tok_str(tok, token_id);
    for (int i = 0; i < tlen && out_len < max_len - 1; i++) {
        uint8_t b = tstr[i];
        if (b == 0xC4 && i + 1 < tlen) {
            uint8_t b2 = tstr[++i];
            out[out_len++] = (b2 == 0xA0) ? ' ' : (b2 == 0x8A ? '\n' : (char)b2);
        } else if (b <= 0x7F) {
            out[out_len++] = (char)b;
        }
    }
    out[out_len] = '\0';
    return out_len;
}

static int padded_in(int in_ch) {
    return (in_ch + g_group_size - 1) / g_group_size * g_group_size;
}

static void read_q4_tensor(uint8_t **q, uint16_t **s, uint8_t *base, size_t *off, int out_ch, int in_ch) {
    int pin = padded_in(in_ch);
    size_t packed = (size_t)out_ch * (size_t)pin / 2u;
    size_t scales = (size_t)out_ch * (size_t)(pin / g_group_size) * 2u;
    *q = base + *off; *off += packed;
    *s = (uint16_t *)(void *)(base + *off); *off += scales;
}

static void read_q8_tensor(int8_t **q, uint16_t **s, uint8_t *base, size_t *off, int out_ch, int in_ch) {
    int pin = padded_in(in_ch);
    size_t qbytes = (size_t)out_ch * (size_t)pin;
    size_t scales = (size_t)out_ch * (size_t)(pin / g_group_size) * 2u;
    *q = (int8_t *)(void *)(base + *off); *off += qbytes;
    *s = (uint16_t *)(void *)(base + *off); *off += scales;
}

static void dequant_q4_row(float *out, const uint8_t *wq, const uint16_t *ws, int row, int in_ch) {
    int pin = padded_in(in_ch), n_groups = pin / g_group_size;
    const uint8_t *row_q = wq + (size_t)row * (size_t)pin / 2u;
    for (int g = 0; g < n_groups; g++) {
        float scale = fp16_to_f32(ws[row * n_groups + g]);
        int jmax = in_ch - g * g_group_size;
        if (jmax > g_group_size) jmax = g_group_size;
        for (int j = 0; j < jmax; j++) {
            uint8_t byte = row_q[g * g_group_size / 2 + j / 2];
            uint8_t nib = (j & 1) ? (byte >> 4) : (byte & 0xFu);
            out[g * g_group_size + j] = (float)((int)nib - 8) * scale;
        }
    }
}

static void dequant_q8_row(float *out, const int8_t *wq, const uint16_t *ws, int row, int in_ch) {
    int pin = padded_in(in_ch), n_groups = pin / g_group_size;
    const int8_t *row_q = wq + (size_t)row * (size_t)pin;
    for (int g = 0; g < n_groups; g++) {
        float scale = fp16_to_f32(ws[row * n_groups + g]);
        int jmax = in_ch - g * g_group_size;
        if (jmax > g_group_size) jmax = g_group_size;
        for (int j = 0; j < jmax; j++) {
            out[g * g_group_size + j] = (float)row_q[g * g_group_size + j] * scale;
        }
    }
}

static void matmul_q8(float *out, const float *xv, const int8_t *wq, const uint16_t *ws, int out_ch, int in_ch) {
    int pin = padded_in(in_ch), n_groups = pin / g_group_size;
    for (int o = 0; o < out_ch; o++) {
        const int8_t *row_q = wq + (size_t)o * (size_t)pin;
        float sum = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            float scale = fp16_to_f32(ws[o * n_groups + g]);
            int jmax = in_ch - g * g_group_size;
            if (jmax > g_group_size) jmax = g_group_size;
            for (int j = 0; j < jmax; j++) sum += ((float)row_q[g * g_group_size + j] * scale) * xv[g * g_group_size + j];
        }
        out[o] = sum;
    }
}

static void layernorm(float *out, const float *in, const float *weight, const float *bias, int n) {
    float mean = 0.0f, var = 0.0f, inv;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    inv = 1.0f / sqrtf(var / (float)n + g_hdr.layer_norm_eps);
    for (int i = 0; i < n; i++) out[i] = (in[i] - mean) * inv * weight[i] + bias[i];
}

static float gelu_new(float v) {
    const float c = 0.7978845608028654f;
    return 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

static int sample_argmax_h(const uint16_t *logits, int n) {
    int best = 0;
    float best_v = fp16_to_f32(logits[0]);
    for (int i = 1; i < n; i++) { float v = fp16_to_f32(logits[i]); if (v > best_v) { best = i; best_v = v; } }
    return best;
}

static int sample_temperature_h(uint16_t *logits, int n, float temp) {
    float max_v = fp16_to_f32(logits[0]), sum = 0.0f, r, acc = 0.0f;
    for (int i = 1; i < n; i++) { float v = fp16_to_f32(logits[i]); if (v > max_v) max_v = v; }
    for (int i = 0; i < n; i++) {
        float p = expf((fp16_to_f32(logits[i]) - max_v) / temp);
        logits[i] = f32_to_fp16(p);
        sum += p;
    }
    r = ((float)rand() / (float)RAND_MAX) * sum;
    for (int i = 0; i < n; i++) { acc += fp16_to_f32(logits[i]); if (r <= acc) return i; }
    return n - 1;
}

static int sample_argmax_f(const float *logits, int n) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > best_v) { best = i; best_v = logits[i]; }
    return best;
}

static int sample_temperature_f(float *logits, int n, float temp) {
    float max_v = logits[0], sum = 0.0f, r, acc = 0.0f;
    for (int i = 1; i < n; i++) if (logits[i] > max_v) max_v = logits[i];
    for (int i = 0; i < n; i++) { logits[i] = expf((logits[i] - max_v) / temp); sum += logits[i]; }
    r = ((float)rand() / (float)RAND_MAX) * sum;
    for (int i = 0; i < n; i++) { acc += logits[i]; if (r <= acc) return i; }
    return n - 1;
}

void model_clear_kv(void) {
    size_t kv_elems = (size_t)g_n_layers * (size_t)g_seq_len * (size_t)g_kv_dim;
    if (k_cache) memset(k_cache, 0, kv_elems * sizeof(int8_t));
    if (v_cache) memset(v_cache, 0, kv_elems * sizeof(int8_t));
    if (k_cache_f32) memset(k_cache_f32, 0, kv_elems * sizeof(float));
    if (v_cache_f32) memset(v_cache_f32, 0, kv_elems * sizeof(float));
    if (k_scales) memset(k_scales, 0, (size_t)g_n_layers * (size_t)g_seq_len * sizeof(float));
    if (v_scales) memset(v_scales, 0, (size_t)g_n_layers * (size_t)g_seq_len * sizeof(float));
}

int model_forward_step(int token, int pos, float temperature) {
    if (token < 0 || token >= g_vocab_size) { fprintf(stderr, "token out of range: %d\n", token); exit(1); }
    if (pos < 0 || pos >= g_seq_len) { fprintf(stderr, "pos out of range: %d\n", pos); exit(1); }

    if (!g_all_q8) {
        dequant_q4_row(x, wte_q, wte_s, token, g_dim);
    } else {
        dequant_q8_row(x, (const int8_t *)(void *)wte_q, wte_s, token, g_dim);
    }
    if (pos < g_wpe_rows) {
        if (!g_all_q8) {
            dequant_q4_row(xb, wpe_q, wpe_s, pos, g_dim);
        } else {
            dequant_q8_row(xb, (const int8_t *)(void *)wpe_q, wpe_s, pos, g_dim);
        }
        for (int i = 0; i < g_dim; i++) x[i] += xb[i];
    }

    for (int l = 0; l < g_n_layers; l++) {
        layernorm(xb, x, ln1_w[l], ln1_b[l], g_dim);
        matmul_q8(q_buf, xb, wq_q[l], wq_s[l], g_dim, g_dim);
        matmul_q8(k_buf, xb, wk_q[l], wk_s[l], g_dim, g_dim);
        matmul_q8(v_buf, xb, wv_q[l], wv_s[l], g_dim, g_dim);

        if (g_kv_f32) {
            float *kc = k_cache_f32 + ((size_t)l * g_seq_len + pos) * g_kv_dim;
            float *vc = v_cache_f32 + ((size_t)l * g_seq_len + pos) * g_kv_dim;
            for (int i = 0; i < g_kv_dim; i++) { kc[i] = k_buf[i]; vc[i] = v_buf[i]; }
        } else {
            for (int which = 0; which < 2; which++) {
                float *src = which == 0 ? k_buf : v_buf;
                float amax = 0.0f, scale;
                int8_t *dst = (which == 0 ? k_cache : v_cache) + ((size_t)l * g_seq_len + pos) * g_kv_dim;
                float *sc = (which == 0 ? k_scales : v_scales);
                for (int i = 0; i < g_kv_dim; i++) { float av = fabsf(src[i]); if (av > amax) amax = av; }
                if (amax < 1e-12f) amax = 1.0f;
                scale = amax / 127.0f;
                sc[l * g_seq_len + pos] = scale;
                for (int i = 0; i < g_kv_dim; i++) {
                    int iv = (int)roundf(src[i] / scale);
                    if (iv < -127) iv = -127; else if (iv > 127) iv = 127;
                    dst[i] = (int8_t)iv;
                }
            }
        }

        for (int h = 0; h < g_n_heads; h++) {
            float *qh = q_buf + h * g_head_dim;
            float max_score = -INFINITY, sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float score = 0.0f;
                if (g_kv_f32) {
                    float *kc = k_cache_f32 + ((size_t)l * g_seq_len + t) * g_kv_dim + h * g_head_dim;
                    for (int d = 0; d < g_head_dim; d++) score += qh[d] * kc[d];
                } else {
                    int8_t *kc = k_cache + ((size_t)l * g_seq_len + t) * g_kv_dim + h * g_head_dim;
                    float ks = k_scales[l * g_seq_len + t];
                    for (int d = 0; d < g_head_dim; d++) score += qh[d] * ((float)kc[d] * ks);
                }
                att[h * g_seq_len + t] = score;
                if (score > max_score) max_score = score;
            }
            for (int t = 0; t <= pos; t++) { float e = expf(att[h * g_seq_len + t] - max_score); att[h * g_seq_len + t] = e; sum += e; }
            for (int d = 0; d < g_head_dim; d++) xb[h * g_head_dim + d] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float a = att[h * g_seq_len + t] / sum;
                if (g_kv_f32) {
                    float *vc = v_cache_f32 + ((size_t)l * g_seq_len + t) * g_kv_dim + h * g_head_dim;
                    for (int d = 0; d < g_head_dim; d++) xb[h * g_head_dim + d] += a * vc[d];
                } else {
                    int8_t *vc = v_cache + ((size_t)l * g_seq_len + t) * g_kv_dim + h * g_head_dim;
                    float vs = v_scales[l * g_seq_len + t];
                    for (int d = 0; d < g_head_dim; d++) xb[h * g_head_dim + d] += a * ((float)vc[d] * vs);
                }
            }
        }

        matmul_q8(xb2, xb, wo_q[l], wo_s[l], g_dim, g_dim);
        for (int i = 0; i < g_dim; i++) x[i] += xb2[i] + out_b[l * g_dim + i];

        layernorm(xb, x, ln2_w[l], ln2_b[l], g_dim);
        matmul_q8(hb, xb, fc_q[l], fc_s[l], g_hidden_dim, g_dim);
        for (int i = 0; i < g_hidden_dim; i++) hb[i] = gelu_new(hb[i] + fc_b[l * g_hidden_dim + i]);
        matmul_q8(xb, hb, proj_q[l], proj_s[l], g_dim, g_hidden_dim);
        for (int i = 0; i < g_dim; i++) x[i] += xb[i] + proj_b[l * g_dim + i];
    }

    layernorm(xb, x, lnf_w, lnf_b, g_dim);
    if (!g_all_q8) {
        for (int v = 0; v < g_vocab_size; v++) {
            float sum = 0.0f;
            int pin = padded_in(g_dim), n_groups = pin / g_group_size;
            const uint8_t *row_q = wte_q + (size_t)v * (size_t)pin / 2u;
            for (int g = 0; g < n_groups; g++) {
                float scale = fp16_to_f32(wte_s[v * n_groups + g]);
                for (int j = 0; j < g_group_size && g * g_group_size + j < g_dim; j++) {
                    uint8_t byte = row_q[g * g_group_size / 2 + j / 2];
                    uint8_t nib = (j & 1) ? (byte >> 4) : (byte & 0xFu);
                    sum += ((float)((int)nib - 8) * scale) * xb[g * g_group_size + j];
                }
            }
            if (g_logits_f32) logits_f[v] = sum;
            else logits_h[v] = f32_to_fp16(sum);
        }
    } else {
        for (int v = 0; v < g_vocab_size; v++) {
            float sum = 0.0f;
            int pin = padded_in(g_dim), n_groups = pin / g_group_size;
            const int8_t *row_q = (const int8_t *)(void *)wte_q + (size_t)v * (size_t)pin;
            for (int g = 0; g < n_groups; g++) {
                float scale = fp16_to_f32(wte_s[v * n_groups + g]);
                for (int j = 0; j < g_group_size && g * g_group_size + j < g_dim; j++) {
                    sum += ((float)row_q[g * g_group_size + j] * scale) * xb[g * g_group_size + j];
                }
            }
            if (g_logits_f32) logits_f[v] = sum;
            else logits_h[v] = f32_to_fp16(sum);
        }
    }
    if (g_logits_f32) {
        return (temperature <= 0.0f || temperature < 1e-6f) ? sample_argmax_f(logits_f, g_vocab_size) : sample_temperature_f(logits_f, g_vocab_size, temperature);
    }
    return (temperature <= 0.0f || temperature < 1e-6f) ? sample_argmax_h(logits_h, g_vocab_size) : sample_temperature_h(logits_h, g_vocab_size, temperature);
}

static void parse_model_data(uint8_t *data, size_t model_size, int owns_buffer) {
    size_t off = 64, qoff = 0;
    uint8_t *qbase;
    g_model_buf = owns_buffer ? data : NULL;
    g_model_size = model_size;
    heap_model = g_model_size;
    g_hdr.magic = rd_u32(data + 0);
    g_hdr.version = rd_i32(data + 4);
    g_hdr.dim = rd_i32(data + 8);
    g_hdr.hidden_dim = rd_i32(data + 12);
    g_hdr.n_layers = rd_i32(data + 16);
    g_hdr.n_heads = rd_i32(data + 20);
    g_hdr.n_kv_heads = rd_i32(data + 24);
    g_hdr.vocab_size = rd_i32(data + 28);
    g_hdr.seq_len = rd_i32(data + 32);
    g_hdr.group_size = rd_i32(data + 36);
    g_hdr.flags = rd_u32(data + 40);
    g_hdr.rope_theta = rd_f32(data + 44);
    g_hdr.layer_norm_eps = rd_f32(data + 48);
    g_hdr.version_extra = rd_i32(data + 52);
    g_hdr.eos_token_id = rd_i32(data + 56);
    g_hdr.wpe_rows = rd_i32(data + 60);
    if (g_hdr.magic != MAGIC_Q4 || g_hdr.version != VERSION_Q4
        || (g_hdr.version_extra != VERSION_EXTRA_NEO_MIXED && g_hdr.version_extra != 5)) {
        fprintf(stderr, "Bad Neo model header: magic=0x%08X version=%d extra=%d\n", g_hdr.magic, g_hdr.version, g_hdr.version_extra);
        exit(1);
    }
    g_all_q8 = (g_hdr.version_extra == 5);
    if (g_all_q8) {
        fprintf(stderr, "Info: ALL-Q8 mode (wte/wpe/attn/mlp all Q8g64)\n");
    }
    if (g_hdr.group_size != 64 || g_hdr.n_kv_heads != g_hdr.n_heads) { fprintf(stderr, "Unsupported Neo header\n"); exit(1); }
    g_dim = g_hdr.dim; g_hidden_dim = g_hdr.hidden_dim; g_n_layers = g_hdr.n_layers; g_n_heads = g_hdr.n_heads;
    g_vocab_size = g_hdr.vocab_size;     g_seq_len = g_hdr.seq_len;
#ifdef NEO_SEQ_CAP
    if (g_seq_len > NEO_SEQ_CAP) g_seq_len = NEO_SEQ_CAP;
#endif
    g_group_size = g_hdr.group_size;
    g_eos_token_id = g_hdr.eos_token_id; g_wpe_rows = g_hdr.wpe_rows;
    g_head_dim = g_dim / g_n_heads; g_kv_dim = g_dim;

    ln1_w = (float **)xcalloc((size_t)g_n_layers * 4u, sizeof(float *), "ln ptrs");
    ln1_b = ln1_w + g_n_layers;
    ln2_w = ln1_b + g_n_layers;
    ln2_b = ln2_w + g_n_layers;
    heap_ptrs += (size_t)g_n_layers * 4u * sizeof(float *);
    for (int l = 0; l < g_n_layers; l++) {
        ln1_w[l] = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
        ln1_b[l] = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
    }
    for (int l = 0; l < g_n_layers; l++) {
        ln2_w[l] = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
        ln2_b[l] = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
    }
    lnf_w = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
    lnf_b = (float *)(void *)(data + off); off += (size_t)g_dim * 4u;
    out_b = (float *)(void *)(data + off); off += (size_t)g_n_layers * g_dim * 4u;
    fc_b = (float *)(void *)(data + off); off += (size_t)g_n_layers * g_hidden_dim * 4u;
    proj_b = (float *)(void *)(data + off); off += (size_t)g_n_layers * g_dim * 4u;

    qbase = data + off;
    if (!g_all_q8) {
        read_q4_tensor(&wte_q, &wte_s, qbase, &qoff, g_vocab_size, g_dim);
        read_q4_tensor(&wpe_q, &wpe_s, qbase, &qoff, g_wpe_rows, g_dim);
    } else {
        int8_t *wte_q8; read_q8_tensor(&wte_q8, &wte_s, qbase, &qoff, g_vocab_size, g_dim); wte_q = (uint8_t *)(void *)wte_q8;
        int8_t *wpe_q8; read_q8_tensor(&wpe_q8, &wpe_s, qbase, &qoff, g_wpe_rows, g_dim); wpe_q = (uint8_t *)(void *)wpe_q8;
    }
    wq_q = (int8_t **)xcalloc((size_t)g_n_layers * 6u, sizeof(int8_t *), "q8 ptrs");
    wq_s = (uint16_t **)xcalloc((size_t)g_n_layers * 6u, sizeof(uint16_t *), "scale ptrs");
    wk_q = wq_q + g_n_layers; wv_q = wk_q + g_n_layers; wo_q = wv_q + g_n_layers; fc_q = wo_q + g_n_layers; proj_q = fc_q + g_n_layers;
    wk_s = wq_s + g_n_layers; wv_s = wk_s + g_n_layers; wo_s = wv_s + g_n_layers; fc_s = wo_s + g_n_layers; proj_s = fc_s + g_n_layers;
    heap_ptrs += (size_t)g_n_layers * 6u * (sizeof(int8_t *) + sizeof(uint16_t *));
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&wq_q[l], &wq_s[l], qbase, &qoff, g_dim, g_dim);
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&wk_q[l], &wk_s[l], qbase, &qoff, g_dim, g_dim);
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&wv_q[l], &wv_s[l], qbase, &qoff, g_dim, g_dim);
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&wo_q[l], &wo_s[l], qbase, &qoff, g_dim, g_dim);
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&fc_q[l], &fc_s[l], qbase, &qoff, g_hidden_dim, g_dim);
    for (int l = 0; l < g_n_layers; l++) read_q8_tensor(&proj_q[l], &proj_s[l], qbase, &qoff, g_dim, g_hidden_dim);
    if (off + qoff != g_model_size) fprintf(stderr, "Warning: parsed %zu bytes, file has %zu\n", off + qoff, g_model_size);
}

static void parse_model(const char *path) {
    size_t model_size;
    uint8_t *data = load_file(path, &model_size);
    parse_model_data(data, model_size, 1);
}

static void alloc_runtime(void) {
    size_t kv_elems = (size_t)g_n_layers * (size_t)g_seq_len * (size_t)g_kv_dim;
    size_t scale_elems = (size_t)g_n_layers * (size_t)g_seq_len;
    if (g_kv_f32) {
        k_cache_f32 = (float *)xcalloc(kv_elems, sizeof(float), "k_cache_f32");
        v_cache_f32 = (float *)xcalloc(kv_elems, sizeof(float), "v_cache_f32");
        heap_kv = 2u * kv_elems * sizeof(float);
    } else {
        k_cache = (int8_t *)xcalloc(kv_elems, sizeof(int8_t), "k_cache");
        v_cache = (int8_t *)xcalloc(kv_elems, sizeof(int8_t), "v_cache");
        k_scales = (float *)xcalloc(scale_elems, sizeof(float), "k_scales");
        v_scales = (float *)xcalloc(scale_elems, sizeof(float), "v_scales");
        heap_kv = 2u * kv_elems * sizeof(int8_t) + 2u * scale_elems * sizeof(float);
    }
    x = (float *)xcalloc((size_t)g_dim, sizeof(float), "x");
    xb = (float *)xcalloc((size_t)(g_hidden_dim > g_dim ? g_hidden_dim : g_dim), sizeof(float), "xb");
    xb2 = (float *)xcalloc((size_t)g_dim, sizeof(float), "xb2");
    q_buf = (float *)xcalloc((size_t)g_dim, sizeof(float), "q");
    k_buf = (float *)xcalloc((size_t)g_dim, sizeof(float), "k");
    v_buf = (float *)xcalloc((size_t)g_dim, sizeof(float), "v");
    att = (float *)xcalloc((size_t)g_n_heads * (size_t)g_seq_len, sizeof(float), "att");
    hb = (float *)xcalloc((size_t)g_hidden_dim, sizeof(float), "hb");
    if (g_logits_f32) logits_f = (float *)xcalloc((size_t)g_vocab_size, sizeof(float), "logits_f");
    else logits_h = (uint16_t *)xcalloc((size_t)g_vocab_size, sizeof(uint16_t), "logits_h");
    heap_state = ((size_t)g_dim * 5u + (size_t)g_hidden_dim * 2u + (size_t)g_n_heads * (size_t)g_seq_len) * sizeof(float)
        + (size_t)g_vocab_size * (g_logits_f32 ? sizeof(float) : sizeof(uint16_t));
}

static void print_heap_table(void) {
    /* Tokenizer heap counts toward the target total whenever it is nonzero.
     * On the device flash path (XIP zero-copy tokenizer) both are 0; they are
     * only nonzero when the host file loader (tok_load) is used. */
    size_t target_runtime = heap_ptrs + heap_kv + heap_state + heap_tok_data + heap_tok_meta;
    size_t host_runtime = target_runtime;
    printf("=== runq4_neo_host ===\n");
    printf("Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d eos=%d wpe=%d\n", g_dim, g_hidden_dim, g_n_layers, g_n_heads, g_vocab_size, g_seq_len, g_eos_token_id, g_wpe_rows);
    printf("Heap accounting:\n");
    printf("  tensor_ptrs:     %zu bytes\n", heap_ptrs);
    printf("  kv_cache_%s:    %zu bytes\n", g_kv_f32 ? "f32 " : "int8", heap_kv);
    printf("  runstate:        %zu bytes\n", heap_state);
    printf("  logits_storage:  %s\n", g_logits_f32 ? "fp32 validation mode" : "fp16");
    printf("  target_sram:     %zu bytes (%.1f KB)\n", target_runtime, target_runtime / 1024.0);
    printf("  host_tok_data:   %zu bytes (host-file-loader-only; XIP zero-copy on device)\n", heap_tok_data);
    printf("  host_tok_meta:   %zu bytes (host-file-loader-only; XIP zero-copy on device)\n", heap_tok_meta);
    printf("  host_malloc:     %zu bytes (%.1f KB, excludes model file)\n", host_runtime, host_runtime / 1024.0);
    printf("  host_model_buf:  %zu bytes (not target SRAM; flash/XIP on device)\n", heap_model);
}

size_t model_target_sram_bytes(void) {
    /* Include tokenizer heap if any was allocated (0 on the flash/XIP path;
     * nonzero only under the host file loader) so the boot gate is honest. */
    return heap_ptrs + heap_kv + heap_state + heap_tok_data + heap_tok_meta;
}

void model_print_heap_table(void) {
    print_heap_table();
}

int tok_decode_id(int token_id, char *out, int max_len) {
    return bpe_decode_id(&g_tok, token_id, out, max_len);
}

int tok_encode_text(const char *text, int *tokens, int max_tokens) {
    return bpe_encode_ex(&g_tok, text, tokens, max_tokens);
}

int tok_encode_text_prefix_all(const char *text, int *tokens, int max_tokens) {
    return bpe_encode_prefix_all(&g_tok, text, tokens, max_tokens);
}

int model_eos_token_id(void) {
    return g_eos_token_id;
}

int load_model_standalone(const char *model_path, const char *tok_path, FILE *log) {
    (void)log;
    if (tok_load(&g_tok, tok_path) != 0) return -1;
    parse_model(model_path);
    if (g_tok.vocab_size != g_vocab_size) return -2;
    alloc_runtime();
    return 0;
}

static int tok_load_from_data(const uint8_t *tok_data, size_t tok_sz) {
    /* XIP zero-copy: token_lens, token_data, and token_offsets all point
     * directly into the flash-resident sidecar. No heap is used, so
     * heap_tok_meta/heap_tok_data stay 0 on this path. Nothing may free
     * these pointers (tok_free is host-main-only, compiled out by NO_MAIN). */
    const uint8_t *p = tok_data + 8;
    size_t total_bytes = 0, off_table, merge_off;
    if (tok_sz < 8) return -1;
    g_tok.vocab_size = (int32_t)rd_u32(tok_data);
    g_tok.max_token_len = (int32_t)rd_u32(tok_data + 4);
    if (g_tok.vocab_size <= 0 || g_tok.max_token_len <= 0) return -1;
    g_tok.merge_left = g_tok.merge_right = g_tok.merge_out = NULL;
    g_tok.merge_count = 0;
    if (tok_sz < 8u + (size_t)g_tok.vocab_size * 2u) return -1;
    g_tok.token_lens = (uint16_t *)(void *)p; /* offset 8: 2-aligned, entries stay 2-aligned */
    for (int i = 0; i < g_tok.vocab_size; i++) {
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        total_bytes += len;
    }
    if ((size_t)(p - tok_data) + total_bytes > tok_sz) return -1;
    g_tok.token_data = (uint8_t *)(void *)p;
    /* Appended offsets table starts at the next multiple of 4 after token_bytes. */
    off_table = 8u + (size_t)g_tok.vocab_size * 2u + total_bytes;
    off_table = (off_table + 3u) & ~(size_t)3u;
    if (off_table + (size_t)g_tok.vocab_size * 4u > tok_sz) return -1; /* device requires the table */
    g_tok.token_offsets = (uint32_t *)(void *)(tok_data + off_table);
    merge_off = off_table + (size_t)g_tok.vocab_size * 4u;
    if (merge_off + 8u <= tok_sz && rd_u32(tok_data + merge_off) == 0x31455042u) {
        uint32_t count = rd_u32(tok_data + merge_off + 4u);
        if (merge_off + 8u + (size_t)count * 6u <= tok_sz) {
            const uint8_t *q = tok_data + merge_off + 8u;
            g_tok.merge_left = (uint16_t *)(void *)q;
            g_tok.merge_right = (uint16_t *)(void *)(q + (size_t)count * 2u);
            g_tok.merge_out = (uint16_t *)(void *)(q + (size_t)count * 4u);
            g_tok.merge_count = (int32_t)count;
        }
    }
    return 0;
}

int load_model_from_flash(const uint8_t *model_data, size_t model_sz,
                          const uint8_t *tok_data, size_t tok_sz, FILE *log) {
    (void)log;
    if (tok_load_from_data(tok_data, tok_sz) != 0) return -1;
    parse_model_data((uint8_t *)(void *)model_data, model_sz, 0);
    if (g_tok.vocab_size != g_vocab_size) return -2;
    alloc_runtime();
    return 0;
}

/* Deferred-allocation variant for firmware that time-shares SRAM between an
 * audio front-end and the LLM: parse headers/pointers only, then call
 * model_alloc_runtime()/model_free_runtime() around each generation phase. */
int load_model_from_flash_noalloc(const uint8_t *model_data, size_t model_sz,
                                  const uint8_t *tok_data, size_t tok_sz, FILE *log) {
    (void)log;
    if (tok_load_from_data(tok_data, tok_sz) != 0) return -1;
    parse_model_data((uint8_t *)(void *)model_data, model_sz, 0);
    if (g_tok.vocab_size != g_vocab_size) return -2;
    return 0;
}

void model_alloc_runtime(void) {
    alloc_runtime();
}

void model_free_runtime(void) {
    free(k_cache); k_cache = NULL;
    free(v_cache); v_cache = NULL;
    free(k_cache_f32); k_cache_f32 = NULL;
    free(v_cache_f32); v_cache_f32 = NULL;
    free(k_scales); k_scales = NULL;
    free(v_scales); v_scales = NULL;
    free(x); x = NULL;
    free(xb); xb = NULL;
    free(xb2); xb2 = NULL;
    free(q_buf); q_buf = NULL;
    free(k_buf); k_buf = NULL;
    free(v_buf); v_buf = NULL;
    free(att); att = NULL;
    free(hb); hb = NULL;
    free(logits_f); logits_f = NULL;
    free(logits_h); logits_h = NULL;
    heap_kv = 0;
    heap_state = 0;
}

#ifndef NO_MAIN
static void tok_free(Tokenizer *tok) {
    free(tok->token_data);
    free(tok->token_offsets);
    free(tok->token_lens);
    free(tok->merge_left);
    free(tok->merge_right);
    free(tok->merge_out);
}

static void infer_paths(const char *model_arg, char *model_path, char *tok_path) {
    if (strstr(model_arg, ".bin")) {
        snprintf(model_path, MAX_PATH_LEN, "%s", model_arg);
    } else if (strstr(model_arg, "1M")) {
        snprintf(model_path, MAX_PATH_LEN, "models/tinystories_1M_neo_k16384.bin");
    } else {
        snprintf(model_path, MAX_PATH_LEN, "models/tinystories_3M_neo_k16384.bin");
    }
    snprintf(tok_path, MAX_PATH_LEN, "%.*s_tok.bin", (int)(strlen(model_path) - (strstr(model_path, ".bin") ? 4 : 0)), model_path);
}

static void read_stdin_prompt(char *buf, size_t cap) {
    size_t n = fread(buf, 1, cap - 1, stdin);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) n--;
    buf[n] = '\0';
}

int main(int argc, char **argv) {
    const char *model_arg = NULL, *prompt_arg = NULL;
    char model_path[MAX_PATH_LEN], tok_path[MAX_PATH_LEN], prompt[4096];
    int max_tokens = 30, encode_only = 0, quiet_ids = 0, debug_first = 0;
    float temperature = 0.0f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) { model_arg = argv[++i]; quiet_ids = 1; }
        else if ((strcmp(argv[i], "--max-tokens") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if ((strcmp(argv[i], "--temperature") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--encode") == 0) encode_only = 1;
        else if (strcmp(argv[i], "--kv-f32") == 0) g_kv_f32 = 1;
        else if (strcmp(argv[i], "--logits-f32") == 0) g_logits_f32 = 1;
#ifdef NEO_DEBUG
        else if (strcmp(argv[i], "--debug-first") == 0) debug_first = 1;
#endif
        else if (!model_arg) model_arg = argv[i];
        else if (!prompt_arg) prompt_arg = argv[i];
    }
    if (getenv("NEO_KV_F32")) g_kv_f32 = 1;
    if (getenv("NEO_LOGITS_F32")) g_logits_f32 = 1;
    if (!model_arg) {
        fprintf(stderr, "Usage: %s [--model model_dir|model.bin] [prompt] [-n steps] [-t temp] [--encode]\n", argv[0]);
        return 1;
    }
    infer_paths(model_arg, model_path, tok_path);
    if (tok_load(&g_tok, tok_path) != 0) { fprintf(stderr, "Failed to load tokenizer %s\n", tok_path); return 1; }
    parse_model(model_path);
    if (g_tok.vocab_size != g_vocab_size) { fprintf(stderr, "Tokenizer/model vocab mismatch %d vs %d\n", g_tok.vocab_size, g_vocab_size); return 1; }
    alloc_runtime();
    if (prompt_arg) snprintf(prompt, sizeof(prompt), "%s", prompt_arg); else read_stdin_prompt(prompt, sizeof(prompt));
    if (!quiet_ids) print_heap_table();

    int tokens[MAX_SEQ_LEN];
    int n_prompt = bpe_encode_ex(&g_tok, prompt, tokens, g_seq_len - 1);
    if (n_prompt <= 0) { fprintf(stderr, "Failed to encode prompt\n"); return 1; }
    if (encode_only) {
        for (int i = 0; i < n_prompt; i++) printf("%d%s", tokens[i], (i + 1 == n_prompt) ? "\n" : " ");
        return 0;
    }
    model_clear_kv();
    for (int i = 0; i < n_prompt - 1; i++) (void)model_forward_step(tokens[i], i, 0.0f);
    int next = tokens[n_prompt - 1];
    for (int s = 0; s < max_tokens && n_prompt - 1 + s < g_seq_len; s++) {
        char decoded[MAX_TOK_LEN];
        next = model_forward_step(next, n_prompt - 1 + s, temperature);
        if (debug_first && s == 0 && g_logits_f32) {
#ifdef NEO_DEBUG
            fprintf(stderr, "DEBUG first-token top-5 logits:\n");
            for (int dbg = 0; dbg < 5; dbg++) {
                float max_v = -1e30f; int max_i = -1;
                for (int vi = 0; vi < g_vocab_size; vi++) {
                    if (logits_f[vi] > max_v) { max_v = logits_f[vi]; max_i = vi; }
                }
                fprintf(stderr, "  token=%d logit=%.4f\n", max_i, max_v);
                logits_f[max_i] = -1e30f;
            }
            exit(0);
#else
            fprintf(stderr, "DEBUG: --debug-first requires NEO_DEBUG build\n");
            exit(1);
#endif
        }
        if (quiet_ids) printf("%d\n", next);
        else { bpe_decode_id(&g_tok, next, decoded, sizeof(decoded)); printf("%s", decoded); fflush(stdout); }
        if (next == g_eos_token_id) break;
    }
    if (!quiet_ids) printf("\n");
    tok_free(&g_tok);
    free(g_model_buf);
    free(ln1_w);
    free(wq_q); free(wq_s);
    free(k_cache); free(v_cache); free(k_cache_f32); free(v_cache_f32); free(k_scales); free(v_scales);
    free(x); free(xb); free(xb2); free(q_buf); free(k_buf); free(v_buf); free(att); free(hb); free(logits_h); free(logits_f);
    return 0;
}
#endif
