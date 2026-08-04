#!/usr/bin/env python3
"""Fetch the voice firmware's last captured utterance as a WAV for analysis.
Usage: dump_capture.py <out.wav>   (sends 'd', parses REC_LEN/REC_START/REC_END)"""
import glob, os, struct, sys, termios, time, tty

out_path = sys.argv[1]
devs = glob.glob("/dev/cu.usbmodem*")
if not devs:
    sys.exit("no usbmodem device")
fd = os.open(devs[0], os.O_RDWR | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd)
a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)


def read_avail():
    try:
        return os.read(fd, 65536)
    except (BlockingIOError, OSError):
        return b""


time.sleep(0.3)
read_avail()  # drain
os.write(fd, b"d")
buf = b""
t0 = time.time()
while b"REC_START\n" not in buf:
    if time.time() - t0 > 8:
        sys.exit("timeout (is the board in listening mode with a capture available?)")
    buf += read_avail()
    time.sleep(0.02)
head, _, data = buf.partition(b"REC_START\n")
import re
m = re.search(rb"REC_LEN (\d+)", head)
if not m:
    sys.exit("no REC_LEN header")
nsamp = int(m.group(1))
need = nsamp * 2
t0 = time.time()
while len(data) < need:
    if time.time() - t0 > 30:
        sys.exit(f"timeout: {len(data)}/{need} bytes")
    c = read_avail()
    if c:
        data += c
    else:
        time.sleep(0.005)
os.close(fd)
pcm = data[:need]
with open(out_path, "wb") as f:
    f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt ")
    f.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
    f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)
print(f"saved {out_path}: {nsamp} samples ({nsamp/16000.0:.2f}s)")
