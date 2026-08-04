#!/usr/bin/env python3
"""Stream raw s16le 16 kHz mono audio from shrike_mic_test to stdout.
Usage: mic_stream.py [seconds]   (default 20; Ctrl-C also stops cleanly)
Progress/level goes to stderr; ONLY raw audio bytes go to stdout, so pipe it:
    python3 tools/mic_stream.py 30 | tinysr/apps/store_utters data/cat
"""
import glob, math, os, struct, sys, termios, time, tty

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

devs = glob.glob("/dev/cu.usbmodem*")
if not devs:
    sys.exit("no usbmodem device")
fd = os.open(devs[0], os.O_RDWR | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd)
a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)
sys.stderr.write(f"[mic_stream] opened {devs[0]}, streaming {secs:.0f}s\n")


def read_avail():
    try:
        return os.read(fd, 65536)
    except (BlockingIOError, OSError):
        return b""


os.write(fd, b"s")
buf = b""
t0 = time.time()
while b"STREAM_START\n" not in buf:
    if time.time() - t0 > 10:
        sys.exit("timeout waiting for STREAM_START")
    c = read_avail()
    if c:
        buf += c
    else:
        time.sleep(0.02)
_, _, data = buf.partition(b"STREAM_START\n")

out = sys.stdout.buffer
total = 0
level_acc = 0
level_n = 0
next_meter = time.time() + 1.0
t0 = time.time()
try:
    while time.time() - t0 < secs:
        c = read_avail()
        if c:
            data += c
        # keep a partial-sample remainder so int16 alignment survives chunking
        emit = data[: len(data) - (len(data) % 2)]
        data = data[len(emit):]
        if emit:
            out.write(emit)
            out.flush()
            total += len(emit)
            vals = struct.unpack(f"<{len(emit)//2}h", emit)
            level_acc += sum(v * v for v in vals)
            level_n += len(vals)
        if time.time() >= next_meter:
            rms = math.sqrt(level_acc / max(1, level_n))
            bars = "#" * min(30, int(rms / 100))
            sys.stderr.write(f"[mic_stream] {total//2} samples rms={rms:6.0f} |{bars}\n")
            level_acc = level_n = 0
            next_meter = time.time() + 1.0
        if not c:
            time.sleep(0.005)
except KeyboardInterrupt:
    pass

os.write(fd, b"x")  # stop streaming
time.sleep(0.3)
tail = read_avail()
os.close(fd)
m = tail[tail.find(b"STREAM_END"):].split(b"\n")[0] if b"STREAM_END" in tail else b"(no end marker)"
sys.stderr.write(f"[mic_stream] done: {total//2} samples ({total/32000.0:.1f}s) {m.decode(errors='replace')}\n")
