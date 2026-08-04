/*
 * ssd1306.h — minimal SSD1306 128x64 I2C driver for Shrike-lite chat firmware.
 * Wiring: SDA=GP8, SCL=GP9 (I2C0), addr 0x3C. Text mode: 21 cols x 8 rows,
 * auto word-wrap and scroll.
 */
#ifndef SSD1306_H
#define SSD1306_H

/* returns 1 if the display responded and was initialized, 0 otherwise;
 * all other calls are safe no-ops when init failed (headless operation) */
int oled_init(void);
void oled_clear(void);
void oled_print(const char *s); /* streams text, wraps, scrolls, flushes */
void oled_meter(int level, int marker); /* live bar on bottom row, fast partial flush */

#endif
