/*
 * main_voice.c — standalone voice-to-story firmware for Shrike-lite.
 *
 * Speak exactly 3 vocabulary words -> ChinnaLLM writes a story about them on
 * the OLED. No host computer involved. USB serial mirrors everything.
 *
 * Memory time-share (264 KB SRAM):
 *   LISTEN phase heap: TinySR model (~143 KB) + utterance buffer + mic ring
 *   GENERATE phase heap: LLM KV cache + runstate (~232 KB)
 * Each phase frees everything before the other allocates.
 *
 * Controls: BOOTSEL button = clear slots (listening) / abort story (generating).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "pico/time.h"
#include "hardware/vreg.h"
#include "hardware/clocks.h"
#include "hardware/pio.h"
#include "hardware/irq.h"
#include "hardware/sync.h"
#include "hardware/structs/ioqspi.h"
#include "hardware/structs/sio.h"

#include <malloc.h>

#include "inmp441.pio.h"
#include "ssd1306.h"
#include "template_ids_neo.h"
#include "tinysr.h"

extern void tinysr_free_scratch(void); /* embedded patch: frees DSP tables */

static void log_heap(const char *tag) {
    struct mallinfo mi = mallinfo();
    printf("heap[%s]: in_use=%d free=%d\n", tag, mi.uordblks, mi.fordblks);
}

/* ---- LLM engine API (runq4_neo_host.c) ---- */
extern int load_model_from_flash_noalloc(const uint8_t *model_data, size_t model_sz,
                                         const uint8_t *tok_data, size_t tok_sz, FILE *log);
extern void model_alloc_runtime(void);
extern void model_free_runtime(void);
extern int model_forward_step(int token, int pos, float temp);
extern void model_clear_kv(void);
extern int tok_decode_id(int token_id, char *out, int max_len);
extern int tok_encode_text_prefix_all(const char *text, int *tokens, int max_tokens);
extern int model_eos_token_id(void);
extern int g_dim, g_n_layers, g_n_heads, g_seq_len;

extern const uint8_t model_data_start[];
extern const uint8_t model_data_end[];
extern const uint8_t tokenizer_data_start[];
extern const uint8_t tokenizer_data_end[];
extern const uint8_t tinysr_model_start[];
extern const uint8_t tinysr_model_end[];

/* ---- tuning ---- */
#define PIN_SCK 10 /* WS = SCK+1, SD below */
#define PIN_SD 16
#define FRAME 160                 /* 10 ms VAD frames at 16 kHz */
#define PREROLL_FRAMES 12         /* frames kept before utterance trigger */
#define START_FRAMES 4            /* loud frames in a row to trigger */
#define STOP_FRAMES 10            /* quiet frames in a row to end */
#define DROP_TAIL_FRAMES 7        /* quiet frames trimmed from the end */
#define START_DB 4.5f             /* nat-log energy over floor to be "loud" (~20 dB) */
#define STOP_DB 3.5f
#define UTTER_MAX_S 1.2f
#define UTTER_MAX_SAMPLES ((int)(16000 * UTTER_MAX_S))
#define RING_SIZE 2048            /* mic ring, int16 samples (0.128 s) */
#define SCORE_ACCEPT (-50.0f)     /* winner must score at least this */
#define SCORE_MARGIN 3.5f         /* ...and beat the runner-up by this */
/* second tier: a landslide winner is trusted even at a weaker absolute score
 * (cross-normalization gives some words systematically lower scores — a true
 * "dragon" hit at -57.8 beat the runner-up by 38.7 and was still rejected) */
#define SCORE_ACCEPT_WIDE (-70.0f)
#define SCORE_MARGIN_WIDE 15.0f
#define MIN_WORD_FRAMES 30        /* reject sub-0.3s blips outright */
#define GEN_TEMP 0.8f

/* ---- mic capture (PIO + IRQ ring) ---- */
static PIO g_pio = pio0;
static uint g_sm;
static uint g_pio_offset;
static int16_t *ring;             /* heap: LISTEN phase only */
static volatile uint32_t ring_head, ring_tail;
static volatile uint32_t stream_phase, mic_phase_sel;

static inline int16_t cook_raw(uint32_t word) {
    int32_t s = (int32_t)(word << 1);
    return (int16_t)(s >> 16);
}

/* 4x digital gain (the INMP441 barely tickles 16-bit range at speaking
 * distance — this recovers two real bits of signal) followed by a ~20 Hz
 * one-pole DC-block. The mic's DC offset + infrasonic rumble used to inflate
 * the VAD noise floor, which is why only shouting cleared the threshold.
 * Sub-64 Hz never enters the mel filterbank and TinySR does per-utterance
 * cepstral mean normalization, so both gain and filter are invisible to the
 * trained model — no re-enrollment needed. */
static int32_t hp_y, hp_px;
static inline int16_t cook(uint32_t word) {
    int32_t s = (int32_t)(word << 1) >> 14; /* 4x vs the old >>16 */
    hp_y += s - hp_px - (hp_y >> 7);        /* y[n] = x[n]-x[n-1] + (127/128)y[n-1] */
    hp_px = s;
    int32_t y = hp_y;
    if (y > 32767) y = 32767;
    else if (y < -32768) y = -32768;
    return (int16_t)y;
}

static void pio_rx_irq(void) {
    while (!pio_sm_is_rx_fifo_empty(g_pio, g_sm)) {
        uint32_t w = pio_sm_get(g_pio, g_sm);
        uint32_t ph = stream_phase++;
        if (!ring || (ph & 1) != mic_phase_sel) continue;
        uint32_t next = (ring_head + 1) & (RING_SIZE - 1);
        if (next == ring_tail) continue; /* overflow: drop */
        ring[ring_head] = cook(w);
        ring_head = next;
    }
}

static void mic_hw_init(void) {
    g_pio_offset = pio_add_program(g_pio, &inmp441_program);
    g_sm = pio_claim_unused_sm(g_pio, true);
    pio_gpio_init(g_pio, PIN_SCK);
    pio_gpio_init(g_pio, PIN_SCK + 1);
    pio_gpio_init(g_pio, PIN_SD);
    pio_sm_set_consecutive_pindirs(g_pio, g_sm, PIN_SCK, 2, true);
    pio_sm_set_consecutive_pindirs(g_pio, g_sm, PIN_SD, 1, false);
    gpio_pull_down(PIN_SD);
    pio_sm_config c = inmp441_program_get_default_config(g_pio_offset);
    sm_config_set_sideset_pins(&c, PIN_SCK);
    sm_config_set_in_pins(&c, PIN_SD);
    sm_config_set_in_shift(&c, false, true, 32);
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);
    /* 200 MHz sys clock -> BCLK 1.024 MHz -> fs 16 kHz (2.048 MHz PIO clk) */
    sm_config_set_clkdiv_int_frac(&c, 97, 168);
    pio_sm_init(g_pio, g_sm, g_pio_offset, &c);
    irq_set_exclusive_handler(PIO0_IRQ_0, pio_rx_irq);
}

/* lock onto the half-frame that carries the mic (continuous counter — do NOT
 * clear the FIFO afterwards, that randomizes the channel mapping) */
static void mic_start_listening(void) {
    ring_head = ring_tail = 0;
    pio_sm_clear_fifos(g_pio, g_sm);
    pio_sm_set_enabled(g_pio, g_sm, true);
    uint64_t sum[2] = {0, 0};
    uint32_t ph = 0;
    for (int i = 0; i < 8000; i++) {
        while (pio_sm_is_rx_fifo_empty(g_pio, g_sm)) tight_loop_contents();
        /* raw path: channel calibration must see the DC settling ramp that
         * distinguishes the live half-frame; the DC-block would erase it */
        int16_t s = cook_raw(pio_sm_get(g_pio, g_sm));
        sum[ph & 1] += (uint64_t)abs((int)s);
        ph++;
    }
    mic_phase_sel = (sum[1] > sum[0]) ? 1 : 0;
    stream_phase = ph;
    pio_set_irq0_source_enabled(g_pio, pis_sm0_rx_fifo_not_empty + g_sm, true);
    irq_set_enabled(PIO0_IRQ_0, true);
    /* INMP441 outputs a ~1s DC settling ramp after its clocks start; if it
     * reaches the energy detector it triggers a ghost utterance (which then
     * scores as some vocabulary word — the infamous phantom "cat"). Discard it. */
    sleep_ms(1200);
    irq_set_enabled(PIO0_IRQ_0, false);
    ring_head = ring_tail = 0;
    irq_set_enabled(PIO0_IRQ_0, true);
}

static void mic_stop(void) {
    irq_set_enabled(PIO0_IRQ_0, false);
    pio_set_irq0_source_enabled(g_pio, pis_sm0_rx_fifo_not_empty + g_sm, false);
    pio_sm_set_enabled(g_pio, g_sm, false);
}

/* blocking read of one 10 ms frame from the ring */
static void ring_read_frame(int16_t *dst) {
    for (int i = 0; i < FRAME; i++) {
        while (ring_tail == ring_head) tight_loop_contents();
        dst[i] = ring[ring_tail];
        ring_tail = (ring_tail + 1) & (RING_SIZE - 1);
    }
}

/* ---- BOOTSEL button as an input (standard RP2040 trick) ---- */
static bool __no_inline_not_in_flash_func(bootsel_pressed)(void) {
    const uint CS_PIN_INDEX = 1;
    uint32_t flags = save_and_disable_interrupts();
    hw_write_masked(&ioqspi_hw->io[CS_PIN_INDEX].ctrl,
                    GPIO_OVERRIDE_LOW << IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_LSB,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_BITS);
    for (volatile int i = 0; i < 1000; ++i) tight_loop_contents();
    bool pressed = !(sio_hw->gpio_hi_in & (1u << CS_PIN_INDEX));
    hw_write_masked(&ioqspi_hw->io[CS_PIN_INDEX].ctrl,
                    GPIO_OVERRIDE_NORMAL << IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_LSB,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_BITS);
    restore_interrupts(flags);
    return pressed;
}

/* ---- TinySR model loader from flash (upstream loader needs fopen) ---- */
static int load_tinysr_model_mem(tinysr_ctx_t *ctx, const uint8_t *p, size_t sz) {
    size_t off = 0;
    int entries = 0;
    while (off + 4 <= sz) {
        uint32_t nl;
        memcpy(&nl, p + off, 4); off += 4;
        if (nl > 64 || off + nl + 12 > sz) return -1;
        recog_entry_t *re = malloc(sizeof(*re));
        if (!re) return -1;
        re->index = ctx->recog_entry_list.length;
        re->name = malloc(nl + 1);
        memcpy(re->name, p + off, nl); re->name[nl] = '\0'; off += nl;
        memcpy(&re->ll_offset, p + off, 4); off += 4;
        memcpy(&re->ll_slope, p + off, 4); off += 4;
        uint32_t tl;
        memcpy(&tl, p + off, 4); off += 4;
        if (tl == 0 || tl > 512 || off + (size_t)tl * 732 > sz) return -1;
        re->model_template_length = (int)tl;
        re->model_template = malloc(sizeof(gaussian_t) * tl);
        if (!re->model_template) return -1;
        for (uint32_t i = 0; i < tl; i++) {
            gaussian_t *g = &re->model_template[i];
            memcpy(&g->log_likelihood_offset, p + off, 4); off += 4;
            memcpy(g->cepstrum_mean, p + off, 52); off += 52;
            memcpy(g->cepstrum_inverse_covariance, p + off, 676); off += 676;
        }
        list_append_back(&ctx->recog_entry_list, re);
        entries++;
    }
    ctx->word_names = malloc(sizeof(char *) * entries);
    int i = 0;
    for (list_node_t *n = ctx->recog_entry_list.head; n; n = n->next)
        ctx->word_names[i++] = ((recog_entry_t *)n->datum)->name;
    return entries;
}

/* ---- UI helpers ---- */
static void ui_text(const char *s) {
    printf("%s", s);
    oled_print(s);
}

static char slot_words[3][24];
static int n_slots;

static void show_slots(const char *status) {
    oled_clear();
    oled_print("speak 3 words:\n");
    char line[26];
    for (int i = 0; i < 3; i++) {
        snprintf(line, sizeof(line), " %d: %s\n", i + 1, i < n_slots ? slot_words[i] : "_");
        oled_print(line);
    }
    if (status && status[0]) {
        oled_print("\n");
        oled_print(status);
        oled_print("\n");
    }
    printf("SLOTS: [%s] [%s] [%s] status=%s\n",
           n_slots > 0 ? slot_words[0] : "_",
           n_slots > 1 ? slot_words[1] : "_",
           n_slots > 2 ? slot_words[2] : "_",
           status ? status : "");
}

/* ---- LISTEN phase: fill 3 slots by voice ---- */
static void listen_for_words(void) {
    ring = malloc(RING_SIZE * sizeof(int16_t));
    int16_t *utter = malloc(UTTER_MAX_SAMPLES * sizeof(int16_t));
    int16_t *preroll = malloc(PREROLL_FRAMES * FRAME * sizeof(int16_t));
    int16_t *frame = malloc(FRAME * sizeof(int16_t));
    tinysr_ctx_t *ctx = tinysr_allocate_context();
    ctx->input_sample_rate = 16000;
    ctx->utterance_mode = TINYSR_MODE_FREE_RUNNING; /* device re-segments like the host */
    int n_words_model = load_tinysr_model_mem(ctx, tinysr_model_start,
                                              (size_t)(tinysr_model_end - tinysr_model_start));
    printf("tinysr: %d vocabulary words loaded\n", n_words_model);
    if (!ring || !utter || !preroll || !frame || n_words_model <= 0) {
        ui_text("VOICE INIT FAILED\n");
        while (1) tight_loop_contents();
    }

    n_slots = 0;
    log_heap("listen-phase");
    show_slots("listening...");
    mic_start_listening();

    float noise_floor = 100.0f;
    int excite = 0, bore = 0, in_utter = 0;
    int utter_n = 0;
    int preroll_fill = 0;
    int last_capture_n = 0;

    while (n_slots < 3) {
        ring_read_frame(frame);

        if (bootsel_pressed()) {
            n_slots = 0;
            show_slots("cleared! speak...");
            sleep_ms(400);
            continue;
        }
        {
            int hc = getchar_timeout_us(0);
            if ((hc == 'd' || hc == 'D') && last_capture_n > 0) {
                printf("REC_LEN %d\nREC_START\n", last_capture_n);
                fwrite(utter, sizeof(int16_t), (size_t)last_capture_n, stdout);
                fflush(stdout);
                printf("\nREC_END\n");
            }
        }

        /* frame log-energy + slow-rise fast-fall noise floor (TinySR-style) */
        float energy = 2e-22f;
        for (int i = 0; i < FRAME; i++)
            energy += (float)frame[i] * (float)frame[i];
        float le = logf(energy);
        if (le < noise_floor) noise_floor = le;
        else if (!in_utter) noise_floor = 0.995f * noise_floor + 0.005f * le;

        /* live level bar: your voice vs the trigger threshold (tick mark) */
        {
            static int meter_div;
            if ((++meter_div & 7) == 0) {
                int level = (int)((le - 10.0f) * (128.0f / 12.0f));
                int marker = (int)((noise_floor + START_DB - 10.0f) * (128.0f / 12.0f));
                if (marker < 0) marker = 0;
                if (marker > 127) marker = 127;
                oled_meter(level, marker);
            }
        }

        int loud = le > noise_floor + START_DB;
        int quiet = le < noise_floor + STOP_DB;
        excite = loud ? excite + 1 : 0;
        bore = quiet ? bore + 1 : 0;

        if (!in_utter) {
            /* keep rolling preroll */
            if (preroll_fill < PREROLL_FRAMES) {
                memcpy(preroll + preroll_fill * FRAME, frame, FRAME * sizeof(int16_t));
                preroll_fill++;
            } else {
                memmove(preroll, preroll + FRAME, (PREROLL_FRAMES - 1) * FRAME * sizeof(int16_t));
                memcpy(preroll + (PREROLL_FRAMES - 1) * FRAME, frame, FRAME * sizeof(int16_t));
            }
            if (excite >= START_FRAMES) {
                in_utter = 1;
                utter_n = 0;
                memcpy(utter, preroll, preroll_fill * FRAME * sizeof(int16_t));
                utter_n = preroll_fill * FRAME;
                printf("utterance start (floor=%.1f)\n", noise_floor);
            }
        } else {
            if (utter_n + FRAME <= UTTER_MAX_SAMPLES) {
                memcpy(utter + utter_n, frame, FRAME * sizeof(int16_t));
                utter_n += FRAME;
            }
            if (bore >= STOP_FRAMES || utter_n + FRAME > UTTER_MAX_SAMPLES) {
                in_utter = 0;
                excite = bore = 0;
                preroll_fill = 0;
                if (utter_n < FRAME * MIN_WORD_FRAMES) { /* too short: ignore blip */
                    printf("utterance too short (%d), ignored\n", utter_n);
                    continue;
                }
                show_slots("thinking...");
                /* mic keeps running during the crunch (restarting it would
                 * emit a fresh DC transient = ghost utterance); the ring just
                 * overflows harmlessly and is discarded afterwards */
                absolute_time_t t0 = get_absolute_time();
                /* feed the whole capture (with its trailing quiet) and let
                 * TinySR's own detector cut the word out — identical
                 * segmentation to the host pipeline the model was trained on.
                 * Pad with ambient preroll audio so the utterance closes
                 * without cratering the noise-floor estimate. */
                tinysr_feed_input(ctx, utter, utter_n);
                {
                    int16_t padf[FRAME];
                    for (int i = 0; i < FRAME; i++) padf[i] = (int16_t)(preroll[i] / 2);
                    for (int pad = 0; pad < STOP_FRAMES + 8; pad++)
                        tinysr_feed_input(ctx, padf, FRAME);
                }
                tinysr_detect_utterances(ctx);
                /* keep the longest detected utterance (the word), drop others */
                utterance_t *u = NULL;
                while (ctx->utterance_list.length) {
                    utterance_t *cand = (utterance_t *)list_pop_front(&ctx->utterance_list);
                    if (u == NULL || cand->length > u->length) {
                        if (u) { free(u->feature_vectors); free(u); }
                        u = cand;
                    } else {
                        free(cand->feature_vectors);
                        free(cand);
                    }
                }
                if (u) printf("segmented: %d frames\n", u->length);
                if (u && u->length < 28) { /* re-segmented blip (a normal-pace
                                            * "cat" segments to ~33 frames) */
                    free(u->feature_vectors);
                    free(u);
                    u = NULL;
                }
                int widx = -1;
                float score = -1e30f, second = -1e30f;
                const char *wname = "?";
                if (u) {
                    for (list_node_t *nd = ctx->recog_entry_list.head; nd; nd = nd->next) {
                        recog_entry_t *re = (recog_entry_t *)nd->datum;
                        float sc = compute_dynamic_time_warping(re, u);
                        printf("  score %-10s %.1f\n", re->name, sc);
                        if (sc > score) {
                            second = score;
                            score = sc;
                            widx = re->index;
                            wname = re->name;
                        } else if (sc > second) {
                            second = sc;
                        }
                    }
                    free(u->feature_vectors);
                    free(u);
                }
                /* purge detector state: an unclosed utterance would pin every
                 * frame in memory forever (this exact leak OOM-crashed v3) */
                while (ctx->fv_list.length) free(list_pop_front(&ctx->fv_list));
                ctx->current_fv = NULL;
                ctx->utterance_start = NULL;
                ctx->utterance_state = 0;
                ctx->excitement = ctx->boredom = 0.0f;
                while (ctx->utterance_list.length) {
                    utterance_t *uu = (utterance_t *)list_pop_front(&ctx->utterance_list);
                    free(uu->feature_vectors);
                    free(uu);
                }
                last_capture_n = utter_n;
                int64_t ms = absolute_time_diff_us(t0, get_absolute_time()) / 1000;
                printf("HEARD: %s score=%.1f margin=%.1f (%d samples, %lld ms)\n",
                       wname, score, score - second, utter_n, ms);
                int duplicate = 0;
                for (int si = 0; si < n_slots; si++)
                    if (strcmp(slot_words[si], wname) == 0) duplicate = 1;
                float margin = score - second;
                int confident = (widx >= 0 &&
                                 ((score >= SCORE_ACCEPT && margin >= SCORE_MARGIN) ||
                                  (score >= SCORE_ACCEPT_WIDE && margin >= SCORE_MARGIN_WIDE)));
                if (confident && duplicate) {
                    char msg[40];
                    snprintf(msg, sizeof(msg), "already have %s!", wname);
                    show_slots(msg);
                } else if (confident) {
                    snprintf(slot_words[n_slots], sizeof(slot_words[0]), "%s", wname);
                    n_slots++;
                    show_slots(n_slots < 3 ? "listening..." : "got them all!");
                } else {
                    show_slots("say that again?");
                }
                /* discard everything the mic heard while we were thinking */
                irq_set_enabled(PIO0_IRQ_0, false);
                ring_head = ring_tail = 0;
                irq_set_enabled(PIO0_IRQ_0, true);
            }
        }
    }

    mic_stop();
    tinysr_free_context(ctx);
    tinysr_free_scratch();
    free(ring); ring = NULL;
    free(utter);
    free(preroll);
    free(frame);
    log_heap("after-listen-cleanup");
}

/* ---- GENERATE phase: 3 words -> story on OLED ---- */
static void append_ids(int *dst, int *n, int max_n, const int *src, int src_n) {
    for (int i = 0; i < src_n && *n < max_n; i++) dst[(*n)++] = src[i];
}

static void generate_story(void) {
    log_heap("before-llm-alloc");
    model_alloc_runtime();
    model_clear_kv();
    log_heap("after-llm-alloc");

    char req[128];
    snprintf(req, sizeof(req), "Write a story about a %s, a %s and a %s.",
             slot_words[0], slot_words[1], slot_words[2]);
    printf("PROMPT: %s\n", req);

    int prompt[192];
    int n = 0;
    append_ids(prompt, &n, 192, NEO_TEMPLATE_CHAT_SYSTEM, NEO_TEMPLATE_CHAT_SYSTEM_LEN);
    n += tok_encode_text_prefix_all("You write simple, clear short stories for young children.",
                                    prompt + n, 192 - n);
    append_ids(prompt, &n, 192, NEO_TEMPLATE_CHAT_USER, NEO_TEMPLATE_CHAT_USER_LEN);
    n += tok_encode_text_prefix_all(req, prompt + n, 192 - n);
    append_ids(prompt, &n, 192, NEO_TEMPLATE_CHAT_ASSISTANT, NEO_TEMPLATE_CHAT_ASSISTANT_LEN);

    int max_gen = g_seq_len - n;
    if (n <= 0 || max_gen <= 0) {
        ui_text("[prompt too long]\n");
        model_free_runtime();
        return;
    }

    oled_clear();
    {
        char hdr[80];
        snprintf(hdr, sizeof(hdr), "%s + %s + %s\n", slot_words[0], slot_words[1], slot_words[2]);
        ui_text(hdr);
    }

    int eos = model_eos_token_id();
    char piece[256];
    int aborted = 0;
    int total_steps = (n - 1) + max_gen;
    int step = 0;
    for (int i = 0; i < n - 1; i++) {
        (void)model_forward_step(prompt[i], i, 0.0f);
        step++;
        oled_meter(step * 128 / total_steps, -1); /* progress bar */
        if (bootsel_pressed()) { aborted = 1; break; }
    }
    if (!aborted) {
        int next = prompt[n - 1];
        int gen = 0;
        for (int pos = n - 1; gen < max_gen && pos < g_seq_len; pos++) {
            next = model_forward_step(next, pos, GEN_TEMP);
            if (next == eos) break;
            tok_decode_id(next, piece, sizeof(piece));
            ui_text(piece);
            gen++;
            step++;
            oled_meter(step * 128 / total_steps, -1);
            if (bootsel_pressed()) { aborted = 1; break; }
        }
        ui_text("\n");
    }
    ui_text(aborted ? "\n[stopped]\n" : "\n[the end]\n");
    model_free_runtime();
}

int main(void) {
    vreg_set_voltage(VREG_VOLTAGE_1_15);
    sleep_ms(10);
    set_sys_clock_khz(200000, true);
    stdio_init_all();
    stdio_set_translate_crlf(&stdio_usb, false);
    sleep_ms(2500);
    printf("\n=== shrike_voice_story boot ===\n");

    int oled_up = oled_init();
    printf("OLED: %s\n", oled_up ? "OK" : "MISSING");
    oled_print("voice story machine\nloading model...\n");

    if (load_model_from_flash_noalloc(model_data_start,
                                      (size_t)(model_data_end - model_data_start),
                                      tokenizer_data_start,
                                      (size_t)(tokenizer_data_end - tokenizer_data_start),
                                      NULL) != 0) {
        printf("LLM load FAILED\n");
        oled_print("MODEL LOAD FAILED\n");
        while (1) tight_loop_contents();
    }
    printf("LLM: dim=%d L=%d H=%d seq=%d\n", g_dim, g_n_layers, g_n_heads, g_seq_len);

    mic_hw_init();
    srand((unsigned int)time_us_64());

    while (1) {
        listen_for_words();
        generate_story();
        oled_print("\nBOOTSEL = new story\n");
        printf("story done — press BOOTSEL or wait 8s for new round\n");
        absolute_time_t until = make_timeout_time_ms(8000);
        while (absolute_time_diff_us(get_absolute_time(), until) > 0) {
            if (bootsel_pressed()) break;
            sleep_ms(50);
        }
    }
}
