#!/usr/bin/env python3
"""Live terminal VU meter for shrike_mic_test firmware.
Run in its own terminal: python3 tools/mic_meter.py
Bars update ~10x/s; BEEPS when the mic first produces data. Ctrl-C to exit."""
import glob, math, os, re, sys, termios, time, tty

devs = glob.glob("/dev/cu.usbmodem*")
if not devs:
    sys.exit("no usbmodem device — is the board plugged in?")
fd = os.open(devs[0], os.O_RDWR | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd)
a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)
print(f"listening on {devs[0]} — wiggle wires; bars + beep when mic is alive. Ctrl-C quits.")

BARW = 36
LINE = re.compile(rb"level: A=(\d+) B=(\d+)")


def bar(v):
    n = 0 if v <= 0 else int(min(BARW, BARW * math.log10(v + 1) / math.log10(32768)))
    return "█" * n + "·" * (BARW - n)


buf = b""
alive = False
last_nonzero = 0.0
try:
    while True:
        try:
            c = os.read(fd, 4096)
        except (BlockingIOError, OSError):
            c = b""
        if c:
            buf += c
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                m = LINE.search(line)
                if not m:
                    if b"NO DATA" in line:
                        sys.stdout.write("\r\x1b[K PIO/clock problem: NO DATA line from firmware")
                        sys.stdout.flush()
                    continue
                va, vb = int(m.group(1)), int(m.group(2))
                if (va or vb) and not alive:
                    alive = True
                    sys.stdout.write("\a")  # beep on first life
                if va or vb:
                    last_nonzero = time.time()
                if alive and time.time() - last_nonzero > 2.0:
                    alive = False  # contact lost again
                status = "\x1b[32mSIGNAL!\x1b[0m" if (va or vb) else "\x1b[31mdead   \x1b[0m"
                sys.stdout.write(
                    f"\r\x1b[K {status}  A|{bar(va)}| {va:5d}   B|{bar(vb)}| {vb:5d}"
                )
                sys.stdout.flush()
        else:
            time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    os.close(fd)
    print("\nbye")
