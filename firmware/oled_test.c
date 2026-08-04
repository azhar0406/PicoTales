/*
 * oled_test.c — SSD1306 wiring self-test for Shrike-lite.
 *
 * Bit-bangs open-drain I2C across every plausible free GPIO pair, probes for
 * an SSD1306 at 0x3C/0x3D, then initializes the panel and draws a test
 * pattern plus a text banner reporting the detected SDA/SCL pins. Serial
 * (USB CDC) mirrors everything, so tools/capture_from_boot.py can verify a
 * flash without watching the screen.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "pico/time.h"

#define I2C_HALF_US 4 /* ~125 kHz */

static int g_sda = -1, g_scl = -1;

/* ---- open-drain bit-bang primitives ---- */
static inline void pin_release(int pin) { gpio_set_dir(pin, GPIO_IN); }
static inline void pin_drive_low(int pin) { gpio_put(pin, 0); gpio_set_dir(pin, GPIO_OUT); }
static inline int pin_read(int pin) { return gpio_get(pin); }
static inline void qdelay(void) { sleep_us(I2C_HALF_US); }

static void bb_init_pins(int sda, int scl) {
    g_sda = sda; g_scl = scl;
    gpio_init(sda); gpio_init(scl);
    gpio_pull_up(sda); gpio_pull_up(scl);
    pin_release(sda); pin_release(scl);
    sleep_us(50);
}

static void bb_start(void) {
    pin_release(g_sda); pin_release(g_scl); qdelay();
    pin_drive_low(g_sda); qdelay();
    pin_drive_low(g_scl); qdelay();
}

static void bb_stop(void) {
    pin_drive_low(g_sda); qdelay();
    pin_release(g_scl); qdelay();
    pin_release(g_sda); qdelay();
}

/* returns 0 on ACK, 1 on NACK */
static int bb_write_byte(uint8_t b) {
    for (int i = 7; i >= 0; i--) {
        if ((b >> i) & 1) pin_release(g_sda); else pin_drive_low(g_sda);
        qdelay();
        pin_release(g_scl); qdelay();
        pin_drive_low(g_scl); qdelay();
    }
    pin_release(g_sda); qdelay();
    pin_release(g_scl); qdelay();
    int nack = pin_read(g_sda);
    pin_drive_low(g_scl); qdelay();
    return nack;
}

/* probe address; 1 if a device ACKs */
static int bb_probe(uint8_t addr7) {
    bb_start();
    int nack = bb_write_byte((uint8_t)(addr7 << 1));
    bb_stop();
    return !nack;
}

/* bus must idle high through the pull-ups, else something is shorted/miswired */
static int bb_bus_ok(void) {
    return pin_read(g_sda) && pin_read(g_scl);
}

/* ---- SSD1306 ---- */
static uint8_t g_addr = 0x3C;
static uint8_t fb[1024]; /* 128x64 / 8 */

static int oled_cmd(uint8_t c) {
    bb_start();
    int n = bb_write_byte((uint8_t)(g_addr << 1));
    n |= bb_write_byte(0x00);
    n |= bb_write_byte(c);
    bb_stop();
    return n; /* 0 = all ACKed */
}

static int oled_cmd_verbose(uint8_t c) {
    int nack = oled_cmd(c);
    printf("  cmd 0x%02X -> %s\n", c, nack ? "NACK" : "ack");
    return nack;
}

static void oled_init_common(void) {
    static const uint8_t seq[] = {
        0xAE,       /* display off */
        0xD5, 0x80, /* clock */
        0xA8, 0x3F, /* multiplex 64 */
        0xD3, 0x00, /* display offset */
        0x40,       /* start line 0 */
        0xA1, 0xC8, /* segment/scan remap */
        0xDA, 0x12, /* com pins */
        0x81, 0xCF, /* contrast */
        0xD9, 0xF1, /* precharge */
        0xDB, 0x40, /* vcom */
        0xA4,       /* resume from RAM */
        0xA6,       /* normal (non-inverted) */
    };
    for (unsigned i = 0; i < sizeof(seq); i++) oled_cmd_verbose(seq[i]);
}

static void oled_pump_ssd1306(void) {
    printf("  charge pump (SSD1306 style: 0x8D 0x14)\n");
    oled_cmd_verbose(0x8D); oled_cmd_verbose(0x14);
}

static void oled_pump_sh1106(void) {
    printf("  charge pump (SH1106 style: 0xAD 0x8B, pump voltage 0x32)\n");
    oled_cmd_verbose(0xAD); oled_cmd_verbose(0x8B);
    oled_cmd_verbose(0x32); /* pump voltage 8.0V */
}

/* SSD1306 horizontal-addressing flush */
static void oled_flush_ssd1306(void) {
    oled_cmd(0x20); oled_cmd(0x00);                 /* horizontal mode */
    oled_cmd(0x21); oled_cmd(0x00); oled_cmd(0x7F); /* columns 0..127 */
    oled_cmd(0x22); oled_cmd(0x00); oled_cmd(0x07); /* pages 0..7 */
    bb_start();
    bb_write_byte((uint8_t)(g_addr << 1));
    bb_write_byte(0x40);
    for (int i = 0; i < 1024; i++) bb_write_byte(fb[i]);
    bb_stop();
}

/* SH1106 page-by-page flush (132-col RAM, panel at columns 2..129) */
static void oled_flush_sh1106(void) {
    for (int page = 0; page < 8; page++) {
        oled_cmd((uint8_t)(0xB0 | page));
        oled_cmd(0x02); /* column low nibble = 2 */
        oled_cmd(0x10); /* column high nibble = 0 */
        bb_start();
        bb_write_byte((uint8_t)(g_addr << 1));
        bb_write_byte(0x40);
        for (int i = 0; i < 128; i++) bb_write_byte(fb[page * 128 + i]);
        bb_stop();
    }
}

/* 5x7 font, ASCII 32..127 (classic GLCD font, public domain) */
static const uint8_t font5x7[96][5] = {
    {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},
    {0x14,0x7F,0x14,0x7F,0x14},{0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},
    {0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},{0x00,0x1C,0x22,0x41,0x00},
    {0x00,0x41,0x22,0x1C,0x00},{0x14,0x08,0x3E,0x08,0x14},{0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},
    {0x20,0x10,0x08,0x04,0x02},{0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},{0x18,0x14,0x12,0x7F,0x10},
    {0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},
    {0x00,0x56,0x36,0x00,0x00},{0x08,0x14,0x22,0x41,0x00},{0x14,0x14,0x14,0x14,0x14},
    {0x00,0x41,0x22,0x14,0x08},{0x02,0x01,0x51,0x09,0x06},{0x32,0x49,0x79,0x41,0x3E},
    {0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x09,0x01},
    {0x3E,0x41,0x49,0x49,0x7A},{0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},
    {0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},{0x7F,0x40,0x40,0x40,0x40},
    {0x7F,0x02,0x0C,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31},{0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},
    {0x1F,0x20,0x40,0x20,0x1F},{0x3F,0x40,0x38,0x40,0x3F},{0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07},{0x61,0x51,0x49,0x45,0x43},{0x00,0x7F,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20},{0x00,0x41,0x41,0x7F,0x00},{0x04,0x02,0x01,0x02,0x04},
    {0x40,0x40,0x40,0x40,0x40},{0x00,0x01,0x02,0x04,0x00},{0x20,0x54,0x54,0x54,0x78},
    {0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},{0x38,0x44,0x44,0x48,0x7F},
    {0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x0C,0x52,0x52,0x52,0x3E},
    {0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},
    {0x7F,0x10,0x28,0x44,0x00},{0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},
    {0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},{0x7C,0x14,0x14,0x14,0x08},
    {0x08,0x14,0x14,0x18,0x7C},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},
    {0x3C,0x40,0x30,0x40,0x3C},{0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},
    {0x44,0x64,0x54,0x4C,0x44},{0x00,0x08,0x36,0x41,0x00},{0x00,0x00,0x7F,0x00,0x00},
    {0x00,0x41,0x36,0x08,0x00},{0x08,0x08,0x2A,0x1C,0x08},{0x08,0x1C,0x2A,0x08,0x08},
};

static void fb_text(int col, int page, const char *s) {
    int x = col;
    for (; *s && x <= 122; s++) {
        unsigned char c = (unsigned char)*s;
        if (c < 32 || c > 127) c = '?';
        const uint8_t *g = font5x7[c - 32];
        for (int i = 0; i < 5; i++) fb[page * 128 + x + i] = g[i];
        fb[page * 128 + x + 5] = 0x00;
        x += 6;
    }
}

/* candidate pins: free header GPIOs (0-3,12-15 are FPGA; 4 is LED) */
static const int kPins[] = {5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20};
#define NPINS ((int)(sizeof(kPins) / sizeof(kPins[0])))

static int scan_for_oled(void) {
    for (int a = 0; a < NPINS; a++) {
        for (int b = 0; b < NPINS; b++) {
            if (a == b) continue;
            bb_init_pins(kPins[a], kPins[b]);
            if (!bb_bus_ok()) continue;
            for (int addr = 0x3C; addr <= 0x3D; addr++) {
                if (bb_probe((uint8_t)addr)) {
                    /* require a repeatable ACK to reject floating-pin ghosts */
                    if (bb_probe((uint8_t)addr) && bb_probe((uint8_t)addr)) {
                        g_addr = (uint8_t)addr;
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

#define LED_PIN 4

int main(void) {
    stdio_init_all();
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    sleep_ms(3000);
    printf("\n=== shrike OLED wiring self-test ===\n");
    printf("LED blinking = OLED not detected; LED SOLID = OLED responding on I2C.\n");

rescan:
    {
        int found = 0;
        int led = 0;
        for (int attempt = 1; !found; attempt++) {
            gpio_put(LED_PIN, led = !led); /* blink while searching */
            printf("scan attempt %d...\n", attempt);
            found = scan_for_oled();
            if (!found) {
                if (attempt % 5 == 1)
                    printf("NO I2C DEVICE FOUND. Check: VDD->3.3V, GND->GND, SDA/SCL on free IO pins.\n");
                sleep_ms(400);
            }
        }
    }
    gpio_put(LED_PIN, 1); /* solid = connected */

    printf("FOUND: addr=0x%02X SDA=IO%d SCL=IO%d\n", g_addr, g_sda, g_scl);

    printf("init: common commands\n");
    oled_init_common();
    printf("init: enabling BOTH charge pump styles\n");
    oled_pump_ssd1306();
    oled_pump_sh1106();
    printf("init: display ON (0xAF)\n");
    oled_cmd_verbose(0xAF);
    sleep_ms(120);

    /* prepare banner in framebuffer */
    memset(fb, 0, sizeof(fb));
    for (int x = 0; x < 128; x++) { fb[x] |= 0x01; fb[7 * 128 + x] |= 0x80; }
    char line[32];
    fb_text(4, 1, "SHRIKE OLED OK");
    snprintf(line, sizeof(line), "SDA=IO%d SCL=IO%d", g_sda, g_scl);
    fb_text(4, 3, line);
    snprintf(line, sizeof(line), "ADDR=0x%02X", g_addr);
    fb_text(4, 4, line);
    fb_text(4, 6, "ready for stories!");

    /* diagnostic cycle: each phase announced on serial */
    while (1) {
        printf("RE-INIT with per-command ACK check:\n");
        oled_init_common();
        oled_pump_ssd1306();
        oled_pump_sh1106();
        oled_cmd_verbose(0xAF);
        sleep_ms(120);

        printf("PHASE 1: ALL-PIXELS-ON (0xA5) for 3s -- panel must light fully if it works at all\n");
        oled_cmd_verbose(0xA5);
        sleep_ms(3000);

        printf("PHASE 2: resume RAM (0xA4), SSD1306-style banner flush, 3s\n");
        oled_cmd(0xA4);
        oled_flush_ssd1306();
        sleep_ms(3000);

        printf("PHASE 3: SH1106-style banner flush, 3s\n");
        oled_flush_sh1106();
        sleep_ms(3000);

        printf("PHASE 4: inverted (0xA7) 1.5s then normal (0xA6) 1.5s\n");
        oled_cmd(0xA7);
        sleep_ms(1500);
        oled_cmd(0xA6);
        sleep_ms(1500);

        printf("alive: addr=0x%02X SDA=IO%d SCL=IO%d (cycle repeats)\n", g_addr, g_sda, g_scl);

        /* connection watchdog: if the OLED stops ACKing, go back to scanning */
        bb_init_pins(g_sda, g_scl);
        if (!bb_probe(g_addr)) {
            printf("LOST CONNECTION to OLED (no ACK) — back to scanning, LED blinks.\n");
            gpio_put(LED_PIN, 0);
            goto rescan;
        }
    }
}
