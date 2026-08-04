#!/usr/bin/env python3
"""Apply the embedded-target patches to vendored tinysr/ (idempotent).

Patches:
1. full_reco.c: line-buffer stdout (piped results stream immediately).
2. tinysr.c: iterative FFT with heap-allocated scratch/twiddle tables plus
   heap hamming/DCT tables (upstream's recursive VLA FFT needs ~12KB stack;
   RP2040 has a 4KB stack bank, and static tables would cost ~11KB of bss).
   Adds tinysr_free_scratch() so firmware can reclaim ~13KB between phases.
"""
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---- 1. full_reco.c ----
path = "tinysr/apps/full_reco.c"
src = open(path).read()
if "setvbuf" not in src:
    anchor = '\tfprintf(stderr, "Allocating context.\\n");'
    assert anchor in src, "full_reco anchor missing"
    src = src.replace(anchor, "\tsetvbuf(stdout, NULL, _IOLBF, 0);\n" + anchor, 1)
    open(path, "w").write(src)
    print("patched full_reco.c")
else:
    print("full_reco.c already patched")

# ---- 2. tinysr.c: utterance-detection sensitivity ----
# Upstream needs log-energy +5.0 (nat-log, ~22 dB) over the noise floor to
# open an utterance — that demands shouting on a far-field breadboard mic.
# 4.0 (~17 dB) detects a normal voice; the DTW confidence gate downstream
# rejects the extra junk this lets through. Applied to BOTH host tools and
# firmware so enrollment and device segmentation stay identical.
path = "tinysr/tinysr.c"
src = open(path).read()
old = "#define UTTERANCE_START_ENERGY_THRESHOLD 5.0"
new = "#define UTTERANCE_START_ENERGY_THRESHOLD 4.0"
if new in src:
    print("tinysr.c threshold already patched")
else:
    assert old in src, "threshold anchor missing"
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print("patched tinysr.c utterance threshold 5.0 -> 4.0")

# ---- 3. tinysr.c: harden the utterance-copy mallocs (OOM = skip word, not
# hard fault). Requires PICO_MALLOC_PANIC=0 on the firmware target, else the
# SDK panics inside malloc before these checks can run.
src = open(path).read()
if "EMBEDDED: skip utterance on OOM" in src:
    print("tinysr.c utterance-copy already hardened")
else:
    old = """\t\t\tfeature_vector_t* utterance_fvs = malloc(sizeof(feature_vector_t) * utterance_length);
\t\t\ti = 0;"""
    new = """\t\t\tfeature_vector_t* utterance_fvs = malloc(sizeof(feature_vector_t) * utterance_length);
\t\t\tif (utterance_fvs == NULL) { /* EMBEDDED: skip utterance on OOM */
\t\t\t\tctx->utterance_start = NULL;
\t\t\t\tctx->utterance_state = 0;
\t\t\t\tcontinue;
\t\t\t}
\t\t\ti = 0;"""
    assert old in src, "utterance_fvs anchor missing"
    src = src.replace(old, new, 1)
    old = """\t\t\tutterance_t* utterance = malloc(sizeof(utterance_t));
\t\t\tutterance->length = utterance_length;"""
    new = """\t\t\tutterance_t* utterance = malloc(sizeof(utterance_t));
\t\t\tif (utterance == NULL) { /* EMBEDDED: skip utterance on OOM */
\t\t\t\tfree(utterance_fvs);
\t\t\t\tctx->utterance_start = NULL;
\t\t\t\tctx->utterance_state = 0;
\t\t\t\tcontinue;
\t\t\t}
\t\t\tutterance->length = utterance_length;"""
    assert old in src, "utterance obj anchor missing"
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print("patched tinysr.c utterance-copy OOM hardening")

# ---- 4. tinysr.c: embedded-target patches ----
src = open(path).read()
if "EMBEDDED PATCH" in src:
    print("tinysr.c already patched")
    sys.exit(0)

# 2a. heap-table registry at file scope
registry = """
// EMBEDDED PATCH: DSP scratch/tables live on the heap so memory-constrained
// firmware can release them between recognition phases (they re-allocate
// lazily on next use). ~13KB total.
static float *tinysr_tab_re, *tinysr_tab_im, *tinysr_tab_twc, *tinysr_tab_tws;
static float *tinysr_tab_ham, *tinysr_tab_dct;
static int tinysr_tab_tw_ready, tinysr_tab_ham_ready, tinysr_tab_dct_ready;

void tinysr_free_scratch(void) {
\tfree(tinysr_tab_re); tinysr_tab_re = NULL;
\tfree(tinysr_tab_im); tinysr_tab_im = NULL;
\tfree(tinysr_tab_twc); tinysr_tab_twc = NULL;
\tfree(tinysr_tab_tws); tinysr_tab_tws = NULL;
\tfree(tinysr_tab_ham); tinysr_tab_ham = NULL;
\tfree(tinysr_tab_dct); tinysr_tab_dct = NULL;
\ttinysr_tab_tw_ready = tinysr_tab_ham_ready = tinysr_tab_dct_ready = 0;
}

"""
anchor = "void list_append_back(list_t* list, void* datum) {"
assert anchor in src
src = src.replace(anchor, registry + anchor, 1)

# 2b. hamming window table
old = """\t// Hamming window. (ES 201 108 4.2.7)
\tfor (i = 0; i < FRAME_LENGTH; i++)
\t\tctx->temp_buffer[i] *= 0.54 - 0.46 * cosf((PI2 * i)/(FRAME_LENGTH-1));"""
new = """\t// Hamming window. (ES 201 108 4.2.7) — heap table, computed once.
\tif (!tinysr_tab_ham_ready) {
\t\tif (tinysr_tab_ham == NULL) tinysr_tab_ham = malloc(sizeof(float) * FRAME_LENGTH);
\t\tfor (i = 0; i < FRAME_LENGTH; i++)
\t\t\ttinysr_tab_ham[i] = 0.54 - 0.46 * cosf((PI2 * i)/(FRAME_LENGTH-1));
\t\ttinysr_tab_ham_ready = 1;
\t}
\tfor (i = 0; i < FRAME_LENGTH; i++)
\t\tctx->temp_buffer[i] *= tinysr_tab_ham[i];"""
assert old in src
src = src.replace(old, new, 1)

# 2c. DCT basis table
old = """\tfloat cepstrum[13] = {0};
\tfor (i = 0; i < 13; i++) {
\t\t// Compute the discrete cosine transform (DCT) the naive way.
\t\t// XXX: Again notice that I'm zero indexing: filter_bank[j] contains what the spec calls f_(j+1).
\t\t// This is why it's (j + 0.5) rather than (j - 0.5) like in the spec in the upcoming expression.
\t\tint j;
\t\tfor (j = 0; j < 23; j++)
\t\t\tcepstrum[i] += filter_bank[j] * cosf(PI * i * (j + 0.5) / 23.0);
\t}"""
new = """\tif (!tinysr_tab_dct_ready) {
\t\tint j;
\t\tif (tinysr_tab_dct == NULL) tinysr_tab_dct = malloc(sizeof(float) * 13 * 23);
\t\tfor (i = 0; i < 13; i++)
\t\t\tfor (j = 0; j < 23; j++)
\t\t\t\ttinysr_tab_dct[i * 23 + j] = cosf(PI * i * (j + 0.5) / 23.0);
\t\ttinysr_tab_dct_ready = 1;
\t}
\tfloat cepstrum[13] = {0};
\tfor (i = 0; i < 13; i++) {
\t\t// Compute the discrete cosine transform (DCT) the naive way.
\t\t// XXX: Again notice that I'm zero indexing: filter_bank[j] contains what the spec calls f_(j+1).
\t\t// This is why it's (j + 0.5) rather than (j - 0.5) like in the spec in the upcoming expression.
\t\tint j;
\t\tfor (j = 0; j < 23; j++)
\t\t\tcepstrum[i] += filter_bank[j] * tinysr_tab_dct[i * 23 + j];
\t}"""
assert old in src
src = src.replace(old, new, 1)

# 2d. iterative FFT
old = """// Computes the absolute value of the FFT of an array of data.
void tinysr_abs_fft(float* array, int length) {
\tassert(length > 0 && !(length & (length - 1))); // tinysr_abs_fft: array length must be power of two
\tint i;
\t// Allocate an array of zero imaginary parts.
\tfloat imag[length];
\tfor (i = 0; i < length; i++)
\t\timag[i] = 0.0;
\t// Take the FFT.
\ttinysr_fft_dit(array, imag, length, 1, array, imag);
\t// Take the absolute value of each entry.
\tfor (i = 0; i < length; i++)
\t\tarray[i] = sqrtf(array[i]*array[i] + imag[i]*imag[i]);
}"""
new = """// Computes the absolute value of the FFT of an array of data.
// EMBEDDED PATCH: iterative in-place radix-2 DIT with heap scratch/twiddle
// tables (upstream recursive VLA version needs ~12KB stack). Identical DFT.
void tinysr_abs_fft(float* array, int length) {
\tassert(length > 0 && !(length & (length - 1))); // tinysr_abs_fft: array length must be power of two
\tassert(length <= FFT_LENGTH);
\tint i, k, len, b;
\tif (tinysr_tab_re == NULL) tinysr_tab_re = malloc(sizeof(float) * FFT_LENGTH);
\tif (tinysr_tab_im == NULL) tinysr_tab_im = malloc(sizeof(float) * FFT_LENGTH);
\tfloat* re = tinysr_tab_re;
\tfloat* im = tinysr_tab_im;
\tif (!tinysr_tab_tw_ready) {
\t\tint off = 0;
\t\tif (tinysr_tab_twc == NULL) tinysr_tab_twc = malloc(sizeof(float) * FFT_LENGTH);
\t\tif (tinysr_tab_tws == NULL) tinysr_tab_tws = malloc(sizeof(float) * FFT_LENGTH);
\t\tfor (len = 2; len <= FFT_LENGTH; len <<= 1) {
\t\t\tfloat ang = -PI2 / (float)len;
\t\t\tfor (k = 0; k < len / 2; k++) {
\t\t\t\ttinysr_tab_twc[off + k] = cosf(ang * k);
\t\t\t\ttinysr_tab_tws[off + k] = sinf(ang * k);
\t\t\t}
\t\t\toff += len / 2;
\t\t}
\t\ttinysr_tab_tw_ready = 1;
\t}
\t// Bit-reversal permutation copy.
\tfor (i = 0; i < length; i++) {
\t\tunsigned r = 0;
\t\tfor (b = 1; b < length; b <<= 1) {
\t\t\tr <<= 1;
\t\t\tif (i & b) r |= 1;
\t\t}
\t\tre[r] = array[i];
\t\tim[r] = 0.0f;
\t}
\t// Iterative butterflies.
\tint tw_off = 0;
\tfor (len = 2; len <= length; len <<= 1) {
\t\tint half = len / 2;
\t\tfor (i = 0; i < length; i += len) {
\t\t\tfor (k = 0; k < half; k++) {
\t\t\t\tfloat c = tinysr_tab_twc[tw_off + k], s = tinysr_tab_tws[tw_off + k];
\t\t\t\tfloat vr = re[i + k + half] * c - im[i + k + half] * s;
\t\t\t\tfloat vi = re[i + k + half] * s + im[i + k + half] * c;
\t\t\t\tfloat ur = re[i + k], ui = im[i + k];
\t\t\t\tre[i + k] = ur + vr;
\t\t\t\tim[i + k] = ui + vi;
\t\t\t\tre[i + k + half] = ur - vr;
\t\t\t\tim[i + k + half] = ui - vi;
\t\t\t}
\t\t}
\t\ttw_off += half;
\t}
\tfor (i = 0; i < length; i++)
\t\tarray[i] = sqrtf(re[i]*re[i] + im[i]*im[i]);
}"""
assert old in src
src = src.replace(old, new, 1)


# 2e. gaussian quadratic form via symmetry (M is symmetric: ~2x fewer mults,
# matters a lot on soft-float MCUs)
old = """\tfloat log_likelihood = gauss->log_likelihood_offset;
\t// Then, compute the quadratic form cepstrum^T cepstrum_inverse_covariance cepstrum.
\tfor (i = 0; i < 13; i++)
\t\tfor (j = 0; j < 13; j++)
\t\t\tlog_likelihood -= 0.5 * cepstrum[i] * gauss->cepstrum_inverse_covariance[j + i*13] * cepstrum[j]; 
\treturn log_likelihood;"""
new = """\tfloat log_likelihood = gauss->log_likelihood_offset;
\t// Quadratic form via symmetry: sum_i M_ii d_i^2 + 2*sum_{i<j} M_ij d_i d_j.
\tfloat acc = 0.0f;
\tfor (i = 0; i < 13; i++) {
\t\tacc += 0.5f * gauss->cepstrum_inverse_covariance[i + i*13] * cepstrum[i] * cepstrum[i];
\t\tfloat rowacc = 0.0f;
\t\tfor (j = i + 1; j < 13; j++)
\t\t\trowacc += gauss->cepstrum_inverse_covariance[j + i*13] * cepstrum[j];
\t\tacc += cepstrum[i] * rowacc;
\t}
\tlog_likelihood -= acc;
\treturn log_likelihood;"""
assert old in src
src = src.replace(old, new, 1)


# 2f. survive malloc failure on constrained targets: drop data instead of
# writing through NULL (a hard fault that froze the voice firmware)
old = """\tfeature_vector_t* fv = malloc(sizeof(feature_vector_t));
\tfv->log_energy = log_energy;"""
new = """\tfeature_vector_t* fv = malloc(sizeof(feature_vector_t));
\tif (fv == NULL) return; /* EMBEDDED: drop frame under memory pressure */
\tfv->log_energy = log_energy;"""
assert old in src
src = src.replace(old, new, 1)

old = """\tlist_node_t* tail = malloc(sizeof(list_node_t));
\ttail->datum = datum;"""
new = """\tlist_node_t* tail = malloc(sizeof(list_node_t));
\tif (tail == NULL) { list->length--; return; } /* EMBEDDED: drop on OOM */
\ttail->datum = datum;"""
assert old in src
src = src.replace(old, new, 1)

open(path, "w").write(src)
print("patched tinysr.c")
