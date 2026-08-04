# PicoTales 🎤🐉📺

**A standalone voice→AI storyteller on a $4 RP2040.**

Speak three words — PicoTales recognizes them with on-device speech recognition,
then a 2.7M-parameter LLM ([ChinnaLLM](https://github.com/azhar0406/ChinnaLLM),
trained from scratch — *chinna* means "small" in Tamil) writes a children's
story about them, streamed word-by-word to a tiny OLED.

No internet. No cloud. No Raspberry Pi. **264 KB of RAM.**

> 🎬 *demo video / GIF here*

## How it fits: one stage, two actors

An RP2040 has 264 KB of RAM. Speech recognition needs ~150 KB. The LLM's
KV-cache and activations need ~205 KB. They cannot coexist — so they
**time-share the same RAM**, like actors swapping on one stage:

```
LISTEN phase                     GENERATE phase
┌──────────────────────┐         ┌──────────────────────┐
│ TinySR model  143 KB │  free   │ KV cache      ~180 KB│
│ mic ring + VAD ~50 KB│ ──────► │ activations    ~25 KB│
└──────────────────────┘  alloc  └──────────────────────┘
      hears 3 words                writes the story
```

The 2.7 MB of LLM weights never touch RAM at all — they stream directly from
flash (XIP). Recognition is speaker-dependent template matching (MFCC + DTW via
[TinySR](https://github.com/petersn/tinysr)): you enroll each vocabulary word
in your own voice, which is how the whole recognizer fits in ~105 KB.

## Hardware (~$8 total)

| Part | Price | Connection |
|---|---|---|
| RP2040 board (tested: Vicharak Shrike-lite) | ~$4 | USB |
| INMP441 I2S microphone | ~$2 | see below |
| SSD1306 0.96" I2C OLED | ~$2 | see below |

```
INMP441 mic                     RP2040
    VDD ──────────────────────── 3V3
    GND ──────────────────────── GND
    SCK ──────────────────────── GP10   (I2S bit clock)
    WS  ──────────────────────── GP11   (must be SCK pin + 1)
    SD  ──────────────────────── GP16
    L/R ──[any resistor]──────── GND    (select LEFT channel — do not float!)

SSD1306 OLED
    VCC ──────────────────────── 3V3
    GND ──────────────────────── GND
    SDA ──────────────────────── GP8    (I2C0)
    SCL ──────────────────────── GP9
```

Both modules are 3.3 V. A 10,000 mAh power bank runs the whole machine for
days (enable its low-current mode if it has one).

## Build

Prereqs: [pico-sdk](https://github.com/raspberrypi/pico-sdk), `arm-none-eabi-gcc`,
CMake ≥ 3.13, Python 3 + numpy.

```sh
git clone https://github.com/azhar0406/PicoTales
cd PicoTales
./tools/tinysr_setup.sh          # vendors + patches the TinySR recognizer

export PICO_SDK_PATH=/path/to/pico-sdk
cmake -S firmware -B build
cmake --build build -j8
```

Flash `build/picotales_voice_story.uf2` by holding BOOTSEL while plugging in,
then copying the UF2 to the `RPI-RP2` drive.

### Enroll YOUR voice (required)

The included `models/tinysr_3word.model` is trained on the author's voice —
it will hear *you* poorly. Enroll your own (2 minutes per word):

```sh
# flash build/picotales_mic_test.uf2 first, then per word:
./tools/enroll_word.sh cat        # say "cat" ~15 times, normal voice,
./tools/enroll_word.sh dragon     #   from where you'll use the machine
./tools/enroll_word.sh princess

python3 tools/tinysr_model_gen.py --eval-holdout 3 \
    tinysr_data/cat tinysr_data/dragon tinysr_data/princess \
    models/tinysr_3word.model

cmake --build build -j8           # rebuild picks up the new model
```

Any three storybook words work (dog, king, robot, star, moon, bear…) — the
LLM was trained on children's stories, so stick to its world.

## Using it

1. OLED shows `speak 3 words` with a live level meter — the tick mark is the
   trigger threshold; your voice should push past it.
2. Say your words one at a time; each fills a slot (~2 s of thinking per word).
3. Third slot filled → progress bar → the story streams to the OLED.
4. BOOTSEL button = clear slots / abort / new story.

Diagnostic firmwares: `picotales_oled_test` (display + I2C scan),
`picotales_wire_test` (continuity), `picotales_mic_test` (level meter +
enrollment streamer), `picotales_chat` (serial chat with the LLM, no mic).

## Specs

| | |
|---|---|
| MCU | RP2040 (2× Cortex-M0+, no FPU), 200 MHz overclock |
| RAM | 264 KB, time-shared between listen/generate phases |
| LLM | ChinnaLLM-2.7M: GPT-Neo arch, 8L × 128d, 8K vocab, Q8, trained from scratch |
| STT | TinySR (MFCC + DTW), speaker-dependent, ~105 KB model, 2–4 s/word |
| Mic front-end | PIO I2S @ 16 kHz, 4× digital gain, 20 Hz DC-block |
| Story speed | ~0.16 tok/s (every word earned honestly) |

## Credits

- [TinySR](https://github.com/petersn/tinysr) by petersn (public domain) —
  patched for embedded use by `tools/patch_tinysr.py` (iterative FFT, heap DSP
  tables, OOM hardening)
- [pico-sdk](https://github.com/raspberrypi/pico-sdk) (BSD-3)
- LLM inference loosely descended from [llama2.c](https://github.com/karpathy/llama2.c)
  by Andrej Karpathy (MIT), heavily rewritten for GPT-Neo + Q8 + XIP streaming

## License

MIT — see [LICENSE](LICENSE).
