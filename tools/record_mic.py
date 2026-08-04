#!/usr/bin/env python3
"""Record from the shrike_mic_test firmware and save a playable WAV.
Usage: record_mic.py <out.wav> [monitor_seconds_before_record]
Stdlib-only. Sends 'r', parses REC_START/REC_END, writes 16 kHz mono WAV."""
import glob, os, struct, sys, termios, time, tty

FS = 16000
REC_BYTES = FS * 3 * 2  # 3 s of int16

out_path = sys.argv[1]
monitor_s = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

devs = glob.glob("/dev/cu.usbmodem*")
if not devs:
    sys.exit("no usbmodem device")
fd = os.open(devs[0], os.O_RDWR | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd)
a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)
sys.stderr.write(f"opened {devs[0]}\n")


def read_avail():
    try:
        return os.read(fd, 65536)
    except (BlockingIOError, OSError):
        return b""


# show the level meter briefly so the console log proves live audio
t0 = time.time()
while time.time() - t0 < monitor_s:
    c = read_avail()
    if c:
        sys.stderr.write(c.decode("utf-8", "replace"))
    time.sleep(0.05)

os.write(fd, b"r")
buf = b""
t0 = time.time()
while b"REC_START\n" not in buf:
    if time.time() - t0 > 15:
        sys.exit("timeout waiting for REC_START")
    c = read_avail()
    if c:
        buf += c
    else:
        time.sleep(0.02)

head, _, rest = buf.partition(b"REC_START\n")
sys.stderr.write(head.decode("utf-8", "replace"))
data = rest
t0 = time.time()
while len(data) < REC_BYTES:
    if time.time() - t0 > 60:
        sys.exit(f"timeout: got {len(data)}/{REC_BYTES} bytes")
    c = read_avail()
    if c:
        data += c
    else:
        time.sleep(0.005)

pcm = data[:REC_BYTES]
tail = data[REC_BYTES:]
t0 = time.time()
while b"REC_END" not in tail and time.time() - t0 < 5:
    tail += read_avail()
    time.sleep(0.02)
os.close(fd)

with open(out_path, "wb") as f:
    f.write(b"RIFF")
    f.write(struct.pack("<I", 36 + len(pcm)))
    f.write(b"WAVEfmt ")
    f.write(struct.pack("<IHHIIHH", 16, 1, 1, FS, FS * 2, 2, 16))
    f.write(b"data")
    f.write(struct.pack("<I", len(pcm)))
    f.write(pcm)

n = len(pcm) // 2
vals = struct.unpack(f"<{n}h", pcm)
peak = max(abs(v) for v in vals)
rms = (sum(v * v for v in vals) / n) ** 0.5
print(f"\nsaved {out_path}: {n} samples, peak={peak}, rms={rms:.0f}")
print("end_marker_seen" if b"REC_END" in tail else "WARNING: no REC_END marker")
