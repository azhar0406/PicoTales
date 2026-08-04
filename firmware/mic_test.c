/*
 * mic_test.c — INMP441 capture test for Shrike-lite.
 *
 * Wiring: SCK=IO10, WS=IO11, SD=IO16, L/R=GND, VDD=3V3.
 *
 * Idle: prints a level-meter line once per second (proves live audio).
 * On 'r' from the host: records 3 s, picks the channel that carries energy,
 * prints stats, then streams the 96,000-byte mono int16 capture as raw
 * binary between REC_START/REC_END markers (tools/record_mic.py saves WAV).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/pio.h"
#include "hardware/irq.h"
#include "inmp441.pio.h"
#include "ssd1306.h"

#define PIN_SCK 10 /* WS must be SCK+1 */
#define PIN_SD 16
#define FS 16000
#define REC_SECONDS 3
#define NSAMP (FS * REC_SECONDS)

static int16_t chan_a[NSAMP];
static int16_t chan_b[NSAMP];

static PIO g_pio = pio0;
static uint g_sm;

static inline int16_t cook_raw(uint32_t word) {
    /* undo the 1-bit I2S MSB delay, keep top 16 of the 24-bit sample */
    int32_t s = (int32_t)(word << 1);
    return (int16_t)(s >> 16);
}

/* IDENTICAL to main_voice.c cook(): 4x digital gain + ~20 Hz DC-block.
 * Enrollment audio must pass through the same front-end the voice firmware
 * hears, or the trained templates won't match live captures. */
static int32_t hp_y, hp_px;
static inline int16_t cook(uint32_t word) {
    int32_t s = (int32_t)(word << 1) >> 14; /* 4x vs raw >>16 */
    hp_y += s - hp_px - (hp_y >> 7);        /* y[n] = x[n]-x[n-1] + (127/128)y[n-1] */
    hp_px = s;
    int32_t y = hp_y;
    if (y > 32767) y = 32767;
    else if (y < -32768) y = -32768;
    return (int16_t)y;
}

static void mic_start(void) {
    uint offset = pio_add_program(g_pio, &inmp441_program);
    g_sm = pio_claim_unused_sm(g_pio, true);

    pio_gpio_init(g_pio, PIN_SCK);
    pio_gpio_init(g_pio, PIN_SCK + 1);
    pio_gpio_init(g_pio, PIN_SD);
    pio_sm_set_consecutive_pindirs(g_pio, g_sm, PIN_SCK, 2, true);
    pio_sm_set_consecutive_pindirs(g_pio, g_sm, PIN_SD, 1, false);
    gpio_pull_down(PIN_SD);

    pio_sm_config c = inmp441_program_get_default_config(offset);
    sm_config_set_sideset_pins(&c, PIN_SCK);
    sm_config_set_in_pins(&c, PIN_SD);
    sm_config_set_in_shift(&c, false /* shift left, MSB first */, true /* autopush */, 32);
    sm_config_set_fifo_join(&c, PIO_FIFO_JOIN_RX);
    sm_config_set_clkdiv_int_frac(&c, 61, 9); /* 125 MHz -> BCLK 1.024 MHz -> fs 16 kHz */

    pio_sm_init(g_pio, g_sm, offset, &c);
    pio_sm_clear_fifos(g_pio, g_sm);
    pio_sm_set_enabled(g_pio, g_sm, true);
}

/* ---- streaming mode: IRQ drains the PIO FIFO into a ring buffer so USB
 * writes can never cause sample drops ---- */
#define RING_SIZE 16384 /* int16 samples (1 s); power of two */
static volatile int16_t ring[RING_SIZE];
static volatile uint32_t ring_head, ring_tail;
static volatile uint32_t ring_drops;
static volatile uint32_t stream_phase;
static volatile uint32_t mic_phase_sel; /* which half-frame carries the mic */

static void pio_rx_irq(void) {
    while (!pio_sm_is_rx_fifo_empty(g_pio, g_sm)) {
        uint32_t w = pio_sm_get(g_pio, g_sm);
        uint32_t ph = stream_phase++;
        if ((ph & 1) != mic_phase_sel) continue;
        uint32_t next = (ring_head + 1) & (RING_SIZE - 1);
        if (next == ring_tail) { ring_drops++; continue; }
        ring[ring_head] = cook(w);
        ring_head = next;
    }
}

/* Listen briefly and lock onto whichever half-frame carries energy.
 * CRITICAL: no FIFO clear afterwards — the returned word counter must stay
 * continuous into the streaming IRQ, otherwise the parity/channel mapping
 * randomizes (that bug silently streamed the dead channel 50% of the time). */
static uint32_t calibrate_phase(void) {
    uint64_t sum[2] = {0, 0};
    uint32_t ph = 0;
    pio_sm_clear_fifos(g_pio, g_sm);
    for (int i = 0; i < 8000; i++) {
        while (pio_sm_is_rx_fifo_empty(g_pio, g_sm)) tight_loop_contents();
        int16_t s = cook_raw(pio_sm_get(g_pio, g_sm));
        sum[ph & 1] += (uint64_t)abs((int)s);
        ph++;
    }
    mic_phase_sel = (sum[1] > sum[0]) ? 1 : 0;
    return ph; /* counter continues into streaming, FIFO NOT cleared */
}

static void stream_until_key(void) {
    uint32_t ph = calibrate_phase();
    ring_head = ring_tail = ring_drops = 0;
    stream_phase = ph; /* continuous counter keeps channel mapping valid */

    pio_set_irq0_source_enabled(g_pio, pis_sm0_rx_fifo_not_empty + g_sm, true);
    irq_set_exclusive_handler(PIO0_IRQ_0, pio_rx_irq);
    irq_set_enabled(PIO0_IRQ_0, true);

    printf("STREAM_START\n");
    fflush(stdout);
    static uint8_t out[512];
    char text[64];
    int text_n = -1; /* >=0 while collecting a 't' display line */
    while (1) {
        int stop = 0;
        int ch = getchar_timeout_us(0);
        if (ch != PICO_ERROR_TIMEOUT) {
            if (text_n >= 0) {
                /* collecting display text until newline */
                if (ch == '\n' || ch == '\r' || text_n >= (int)sizeof(text) - 2) {
                    text[text_n] = '\0';
                    oled_print(text);
                    oled_print("\n");
                    text_n = -1;
                } else {
                    text[text_n++] = (char)ch;
                }
            } else if (ch == 't') {
                text_n = 0; /* display-text command: does NOT stop the stream */
            } else {
                stop = 1;
            }
        }
        uint32_t avail = (ring_head - ring_tail) & (RING_SIZE - 1);
        if (avail >= 256 || (stop && avail > 0)) {
            uint32_t n = avail > 256 ? 256 : avail;
            for (uint32_t i = 0; i < n; i++) {
                int16_t s = ring[(ring_tail + i) & (RING_SIZE - 1)];
                out[i * 2] = (uint8_t)(s & 0xFF);
                out[i * 2 + 1] = (uint8_t)((s >> 8) & 0xFF);
            }
            fwrite(out, 1, n * 2, stdout);
            fflush(stdout);
            ring_tail = (ring_tail + n) & (RING_SIZE - 1);
        }
        if (stop) break;
        tight_loop_contents();
    }

    irq_set_enabled(PIO0_IRQ_0, false);
    pio_set_irq0_source_enabled(g_pio, pis_sm0_rx_fifo_not_empty + g_sm, false);
    printf("\nSTREAM_END drops=%lu\n", (unsigned long)ring_drops);
    pio_sm_clear_fifos(g_pio, g_sm);
}

static float rms16(const int16_t *x, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)x[i] * (double)x[i];
    return (float)sqrt(acc / (double)n);
}

static int16_t peak16(const int16_t *x, int n) {
    int16_t p = 0;
    for (int i = 0; i < n; i++) { int16_t a = (int16_t)abs(x[i]); if (a > p) p = a; }
    return p;
}

int main(void) {
    stdio_init_all();
    stdio_set_translate_crlf(&stdio_usb, false); /* raw binary passthrough */
    sleep_ms(3000);
    printf("\n=== shrike INMP441 capture test ===\n");
    printf("SCK=IO%d WS=IO%d SD=IO%d fs=%d\n", PIN_SCK, PIN_SCK + 1, PIN_SD, FS);
    printf("Cmds: 'r' record 3s, 's' stream (in-stream: 't<text>\\n' shows on OLED, other=stop).\n");

    if (oled_init()) {
        oled_print("mic + display ready\nsay the magic words\n");
        printf("OLED: OK\n");
    } else {
        printf("OLED: not found\n");
    }

    mic_start();

    uint32_t phase = 0;
    uint64_t acc_a = 0, acc_b = 0;
    uint32_t nacc = 0;
    absolute_time_t next_meter = make_timeout_time_ms(1000);

    while (1) {
        /* drain the FIFO continuously so the meter always reflects live audio */
        while (!pio_sm_is_rx_fifo_empty(g_pio, g_sm)) {
            uint32_t w = pio_sm_get(g_pio, g_sm);
            int16_t s = cook(w);
            if (phase & 1) acc_b += (uint64_t)((int32_t)s * (int32_t)s);
            else acc_a += (uint64_t)((int32_t)s * (int32_t)s);
            phase++;
            nacc++;
        }

        if (absolute_time_diff_us(next_meter, get_absolute_time()) > 0) {
            if (nacc > 0) {
                float ra = (float)sqrt((double)acc_a / (double)(nacc / 2 + 1));
                float rb = (float)sqrt((double)acc_b / (double)(nacc / 2 + 1));
                printf("level: A=%.0f B=%.0f (words=%lu)\n", ra, rb, (unsigned long)nacc);
            } else {
                printf("level: NO DATA FROM MIC (check SD/SCK/WS wiring)\n");
            }
            acc_a = acc_b = 0;
            nacc = 0;
            next_meter = make_timeout_time_ms(100); /* 10 Hz for live wire-wiggling */
        }

        int ch = getchar_timeout_us(0);
        if (ch == 's' || ch == 'S') {
            stream_until_key();
            phase = 0;
            acc_a = acc_b = 0;
            nacc = 0;
            continue;
        }
        if (ch == 'r' || ch == 'R') {
            printf("RECORDING %d s...\n", REC_SECONDS);
            pio_sm_clear_fifos(g_pio, g_sm);
            phase = 0;
            uint32_t n = 0;
            while (n < NSAMP) {
                while (pio_sm_is_rx_fifo_empty(g_pio, g_sm)) tight_loop_contents();
                uint32_t w = pio_sm_get(g_pio, g_sm);
                int16_t s = cook(w);
                if (phase & 1) chan_b[n] = s; else chan_a[n] = s;
                if (phase & 1) n++; /* advance after storing both halves */
                phase++;
            }

            float ra = rms16(chan_a, NSAMP), rb = rms16(chan_b, NSAMP);
            const int16_t *mono = (ra >= rb) ? chan_a : chan_b;
            printf("stats: rmsA=%.0f rmsB=%.0f peak=%d chosen=%c\n",
                   ra, rb, (int)peak16(mono, NSAMP), (ra >= rb) ? 'A' : 'B');
            printf("REC_START\n");
            fwrite(mono, sizeof(int16_t), NSAMP, stdout);
            fflush(stdout);
            printf("\nREC_END\n");
        }
    }
}
