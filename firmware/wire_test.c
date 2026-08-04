/*
 * wire_test.c — board / jumper-wire / breadboard continuity tester.
 *
 * Heartbeat: user LED (GPIO4) blinks 1 Hz => board + firmware alive.
 * Bridge any two monitored IO pins with a wire (directly or through
 * breadboard rows) => LED goes SOLID and serial prints which pins.
 * Remove the bridge => LED returns to blinking.
 *
 * Monitored pins: IO5..IO11, IO16..IO20 (all free header pins).
 */

#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/time.h"

#define LED_PIN 4

static const int kPins[] = {5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20};
#define NPINS ((int)(sizeof(kPins) / sizeof(kPins[0])))

/* connection matrix from the previous scan, for change reporting */
static char was_connected[NPINS][NPINS];

int main(void) {
    stdio_init_all();

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    for (int i = 0; i < NPINS; i++) {
        gpio_init(kPins[i]);
        gpio_set_dir(kPins[i], GPIO_IN);
        gpio_pull_up(kPins[i]);
    }

    sleep_ms(3000);
    printf("\n=== shrike wire/breadboard continuity tester ===\n");
    printf("Monitored pins: IO5-IO11, IO16-IO20\n");
    printf("Bridge any two with a wire -> LED solid + report here.\n\n");

    absolute_time_t next_blink = get_absolute_time();
    int led_state = 0;

    while (1) {
        int n_connected = 0;

        for (int a = 0; a < NPINS; a++) {
            /* drive pin A low, see which other pins follow */
            gpio_set_dir(kPins[a], GPIO_OUT);
            gpio_put(kPins[a], 0);
            sleep_us(50);
            for (int b = 0; b < NPINS; b++) {
                if (b == a) continue;
                int connected = !gpio_get(kPins[b]);
                /* report each pair once (a<b) */
                if (b > a) {
                    if (connected && !was_connected[a][b]) {
                        printf("CONNECT: IO%d <-> IO%d\n", kPins[a], kPins[b]);
                    } else if (!connected && was_connected[a][b]) {
                        printf("LOST:    IO%d <-> IO%d\n", kPins[a], kPins[b]);
                    }
                    was_connected[a][b] = (char)connected;
                }
                if (connected) n_connected++;
            }
            gpio_set_dir(kPins[a], GPIO_IN); /* release */
            sleep_us(50);
        }

        if (n_connected > 0) {
            gpio_put(LED_PIN, 1); /* solid = continuity detected */
        } else if (absolute_time_diff_us(next_blink, get_absolute_time()) > 0) {
            led_state = !led_state;
            gpio_put(LED_PIN, led_state);
            next_blink = make_timeout_time_ms(500); /* 1 Hz heartbeat */
        }

        sleep_ms(50);
    }
}
