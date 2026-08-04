/*
 * main_chat_neo.c - TinyStories-Instruct chat firmware for SPEC_Q4_NEO v4.
 *
 * Host build: cc -std=c99 -Wall -Wextra -O2 -DNO_MAIN -o shrike_chat_neo \
 *             firmware/main_chat_neo.c firmware/runq4_neo_host.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "template_ids_neo.h"

#ifdef PICO
#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/vreg.h"
#include "hardware/clocks.h"
#include "ssd1306.h"
extern const uint8_t model_data_start[];
extern const uint8_t model_data_end[];
extern const uint8_t tokenizer_data_start[];
extern const uint8_t tokenizer_data_end[];
#endif

/* mirror text to USB serial and (on device) the OLED */
static void ui_text(const char *s) {
    printf("%s", s);
#ifdef PICO
    oled_print(s);
#endif
}

extern int load_model_standalone(const char *mp, const char *tp, FILE *log);
extern int load_model_from_flash(const uint8_t *model_data, size_t model_sz,
                                 const uint8_t *tok_data, size_t tok_sz, FILE *log);
extern int model_forward_step(int token, int pos, float temp);
extern void model_clear_kv(void);
extern int tok_decode_id(int token_id, char *out, int max_len);
extern int tok_encode_text_prefix_all(const char *text, int *tokens, int max_tokens);
extern size_t model_target_sram_bytes(void);
extern void model_print_heap_table(void);
extern int model_eos_token_id(void);
extern int g_dim, g_n_layers, g_n_heads, g_seq_len;

#define MAX_LINE 512
#define MAX_PROMPT 192
#define HEAP_GATE_BYTES 240000u

static float temp = 0.0f;
static int total_gen = 0;
static int total_replies = 0;
static char words_field[MAX_LINE] = "";

static void strip_nl(char *s) {
    int l = (int)strlen(s);
    while (l > 0 && (s[l - 1] == '\n' || s[l - 1] == '\r')) s[--l] = '\0';
}

#ifdef PICO
static int read_line_pico(char *buf, int bufsize) {
    int pos = 0;
    while (pos < bufsize - 1) {
        int ch = getchar_timeout_us(0);
        if (ch == PICO_ERROR_TIMEOUT) continue;
        if (ch == '\r' || ch == '\n') {
            putchar('\r'); putchar('\n');
            buf[pos] = '\0';
            if (ch == '\r') (void)getchar_timeout_us(10000);
            return pos;
        }
        if (ch == '\b' || ch == 127) {
            if (pos > 0) { pos--; putchar('\b'); putchar(' '); putchar('\b'); }
            continue;
        }
        if (ch >= 32 && ch < 127) { buf[pos++] = (char)ch; putchar(ch); }
    }
    buf[pos] = '\0';
    return pos;
}
#endif

static void append_ids(int *dst, int *n, int max_n, const int *src, int src_n) {
    for (int i = 0; i < src_n && *n < max_n; i++) dst[(*n)++] = src[i];
}

static int build_prompt_ids(const char *summary, const char *words, int *tokens, int max_tokens) {
    int n = 0;
#ifdef NEO_CHAT_MODE
    (void)words;
    append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_CHAT_SYSTEM, NEO_TEMPLATE_CHAT_SYSTEM_LEN);
    n += tok_encode_text_prefix_all("You write simple, clear short stories for young children.", tokens + n, max_tokens - n);
    append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_CHAT_USER, NEO_TEMPLATE_CHAT_USER_LEN);
    n += tok_encode_text_prefix_all(summary, tokens + n, max_tokens - n);
    append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_CHAT_ASSISTANT, NEO_TEMPLATE_CHAT_ASSISTANT_LEN);
#else
    append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_SUMMARY, NEO_TEMPLATE_SUMMARY_LEN);
    n += tok_encode_text_prefix_all(summary, tokens + n, max_tokens - n);
    if (words && words[0]) {
        append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_WORDS, NEO_TEMPLATE_WORDS_LEN);
        n += tok_encode_text_prefix_all(words, tokens + n, max_tokens - n);
    }
    append_ids(tokens, &n, max_tokens, NEO_TEMPLATE_STORY, NEO_TEMPLATE_STORY_LEN);
#endif
    return n;
}

static int generate_story(const int *prompt, int n_prompt, int *reply, int max_reply) {
    int next;
    int n = 0;
    int eos = model_eos_token_id();
    char piece[256];
    model_clear_kv();
    for (int i = 0; i < n_prompt - 1; i++) {
        (void)model_forward_step(prompt[i], i, 0.0f);
        if ((i & 7) == 7) ui_text("."); /* prefill progress */
    }
    ui_text("\n");
    next = prompt[n_prompt - 1];
    for (int pos = n_prompt - 1; n < max_reply && pos < g_seq_len; pos++) {
        next = model_forward_step(next, pos, temp);
        if (next == eos) break;
        reply[n++] = next;
        tok_decode_id(next, piece, sizeof(piece));
        ui_text(piece); /* stream each token as it decodes */
#ifndef PICO
        fflush(stdout);
#endif
    }
    return n;
}

static void print_decoded(const int *tokens, int n) {
    char buf[256];
    for (int i = 0; i < n; i++) {
        int len = tok_decode_id(tokens[i], buf, sizeof(buf));
        (void)len;
        printf("%s", buf);
    }
}

static void print_stats(void) {
    size_t used = model_target_sram_bytes();
    size_t free_heap = (used < HEAP_GATE_BYTES) ? (HEAP_GATE_BYTES - used) : 0;
    printf("Replies: %d, generated: %d, temp: %.2f\n", total_replies, total_gen, temp);
    printf("Words: %s\n", words_field[0] ? words_field : "[none]");
    printf("Target heap: %u gate, %u used, %u free\n", (unsigned)HEAP_GATE_BYTES, (unsigned)used, (unsigned)free_heap);
}

static void chat_turn(const char *inp) {
    if (inp[0] == '/') {
        if (!strncmp(inp, "/help", 5)) {
            printf("/help /clear /reset /stats /temp /words /quit\n");
#ifdef NEO_CHAT_MODE
            printf("Enter a user request; /words is ignored in chat mode.\n");
#else
            printf("Enter a summary; optional /words comma,list adds a Words field.\n");
#endif
        } else if (!strncmp(inp, "/clear", 6)) {
            model_clear_kv();
            printf("[cleared]\n");
        } else if (!strncmp(inp, "/reset", 6)) {
            words_field[0] = '\0'; total_gen = 0; total_replies = 0; temp = 0.0f; model_clear_kv();
            printf("[reset]\n");
        } else if (!strncmp(inp, "/stats", 6)) {
            print_stats();
        } else if (!strncmp(inp, "/temp", 5)) {
            const char *a = inp + 5; while (*a == ' ') a++;
            if (*a) { float t = (float)atof(a); if (t >= 0.0f && t <= 1.5f) temp = t; }
            printf("temp: %.2f\n", temp);
        } else if (!strncmp(inp, "/words", 6)) {
            const char *a = inp + 6; while (*a == ' ') a++;
            snprintf(words_field, sizeof(words_field), "%s", a);
            printf("[words: %s]\n", words_field[0] ? words_field : "none");
        } else {
            printf("[unknown, /help]\n");
        }
        return;
    }

    int prompt[MAX_PROMPT];
    int reply[MAX_PROMPT];
    int n_prompt = build_prompt_ids(inp, words_field, prompt, MAX_PROMPT);
    int max_gen = g_seq_len - n_prompt;
    if (n_prompt <= 0 || max_gen <= 0) { printf("[prompt too long]\n"); return; }
    if (max_gen > MAX_PROMPT) max_gen = MAX_PROMPT;
#ifdef PICO
    oled_clear();
#endif
    printf("You: %s\n", inp);
#ifdef PICO
    { char hdr[64]; snprintf(hdr, sizeof(hdr), "You: %.40s\n", inp); oled_print(hdr); }
#endif
#ifdef NEO_CHAT_MODE
    ui_text("Assistant:");
#else
    ui_text("Story:");
#endif
    int n_reply = generate_story(prompt, n_prompt, reply, max_gen);
    (void)print_decoded;
    printf("\n\n");
    total_gen += n_reply;
    total_replies++;
}

int main(int argc, char **argv) {
#ifdef PICO
    (void)argc; (void)argv;
    hw_set_bits(&vreg_and_chip_reset_hw->vreg, VREG_AND_CHIP_RESET_VREG_VSEL_BITS);
    set_sys_clock_khz(200000, true);
    stdio_init_all();
    sleep_ms(2000);
    printf("\n=== shrike_chat_neo Pico boot ===\n");
    {
        int oled_up = oled_init();
        printf("OLED: %s\n", oled_up ? "OK (SDA=IO8 SCL=IO9)" : "not found, serial only");
        if (oled_up) oled_print("ChinnaLLM-2.7M\nRP2040 200MHz\nloading model...\n");
    }
    if (load_model_from_flash(model_data_start, (size_t)(model_data_end - model_data_start),
                              tokenizer_data_start, (size_t)(tokenizer_data_end - tokenizer_data_start), NULL) != 0) {
        printf("Model load FAILED\n");
        oled_print("MODEL LOAD FAILED\n");
        while (1) tight_loop_contents();
    }
    oled_print("model loaded.\ntype a prompt over\nUSB serial to begin\n");
    srand((unsigned int)time_us_64());
#else
    if (argc < 3) { fprintf(stderr, "Usage: %s <model> <tok>\n", argv[0]); return 1; }
    if (load_model_standalone(argv[1], argv[2], stderr) != 0) { fprintf(stderr, "Load fail\n"); return 1; }
    srand((unsigned int)time(NULL));
#endif
    printf("=== shrike_chat_neo — dim=%d L=%d H=%d seq=%d ===\n", g_dim, g_n_layers, g_n_heads, g_seq_len);
    model_print_heap_table();
    if (model_target_sram_bytes() > HEAP_GATE_BYTES) printf("HEAP GATE FAIL\n");
    else printf("HEAP GATE PASS <= %u\n", (unsigned)HEAP_GATE_BYTES);
    printf("Type /help for commands.\n\n");

    char line[MAX_LINE];
#ifdef PICO
    while (read_line_pico(line, MAX_LINE) >= 0) {
#else
    while (fgets(line, MAX_LINE, stdin)) {
        strip_nl(line);
#endif
        if (strlen(line) == 0) continue;
        if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
        chat_turn(line);
    }
    printf("Goodbye.\n");
    return 0;
}
