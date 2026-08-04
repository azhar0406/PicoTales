#!/usr/bin/env python3
"""Live word recognition with OLED echo.
Streams board audio into TinySR's full_reco; every recognized word is printed
here AND sent back to the board's OLED display.
Usage: recognize_live.py [seconds=30] [model=models/tinysr_3word.model]
"""
import glob, os, re, select, subprocess, sys, termios, time, tty

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
model = sys.argv[2] if len(sys.argv) > 2 else "models/tinysr_3word.model"

devs = glob.glob("/dev/cu.usbmodem*")
if not devs:
    sys.exit("no usbmodem device")
fd = os.open(devs[0], os.O_RDWR | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd)
a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)
print(f"[reco] opened {devs[0]}, listening {secs:.0f}s — speak!")

reco = subprocess.Popen(
    ["./tinysr/apps/full_reco", model],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
os.set_blocking(reco.stdout.fileno(), False)

WORD = re.compile(rb"=== (\S+) \(([-\d.]+)\)")


def read_port():
    try:
        return os.read(fd, 65536)
    except (BlockingIOError, OSError):
        return b""


os.write(fd, b"s")
buf = b""
t0 = time.time()
while b"STREAM_START\n" not in buf:
    if time.time() - t0 > 10:
        sys.exit("timeout waiting for STREAM_START (is another program on the port?)")
    buf += read_port()
    time.sleep(0.02)
_, _, audio = buf.partition(b"STREAM_START\n")

n_words = 0
reco_buf = b""
t0 = time.time()
try:
    while time.time() - t0 < secs:
        chunk = read_port()
        if chunk:
            audio += chunk
        emit = audio[: len(audio) - (len(audio) % 2)]
        audio = audio[len(emit):]
        if emit:
            reco.stdin.write(emit)
            reco.stdin.flush()
        try:
            out = reco.stdout.read()
        except (BlockingIOError, TypeError):
            out = None
        if out:
            reco_buf += out
            while b"\n" in reco_buf:
                line, _, reco_buf = reco_buf.partition(b"\n")
                m = WORD.search(line)
                if m:
                    word = m.group(1).decode()
                    score = float(m.group(2))
                    n_words += 1
                    print(f"[reco] HEARD: {word}  (score {score:.1f})")
                    os.write(fd, b"t> " + word.encode() + b"\n")
                elif b"Utterance" in line:
                    print(f"[reco] {line.decode(errors='replace')}")
        if not chunk:
            time.sleep(0.005)
except KeyboardInterrupt:
    pass

os.write(fd, b"x")
time.sleep(0.3)
read_port()
reco.stdin.close()
reco.wait(timeout=5)
# drain anything still buffered in the recognizer at exit
os.set_blocking(reco.stdout.fileno(), True)
tail = reco.stdout.read() or b""
for line in (reco_buf + tail).split(b"\n"):
    m = WORD.search(line)
    if m:
        word = m.group(1).decode()
        n_words += 1
        print(f"[reco] HEARD (late): {word}  (score {float(m.group(2)):.1f})")
        os.write(fd, b"t> " + word.encode() + b"\n")
time.sleep(0.3)
os.close(fd)
print(f"[reco] done — {n_words} words recognized")
