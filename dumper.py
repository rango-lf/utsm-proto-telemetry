import argparse
import datetime
import sys
import time

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run: python -m pip install pyserial")
    sys.exit(1)

if not hasattr(serial, "Serial"):
    print("ERROR: The imported 'serial' package is not pyserial.")
    print("This usually happens when a conflicting package named 'serial' is installed.")
    print("Fix it by uninstalling the wrong package and installing pyserial:")
    print("  python -m pip uninstall serial")
    print("  python -m pip install pyserial")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Dump telemetry from a serial device.")
parser.add_argument("--port", default="COM13", help="Serial port (Windows COM port or /dev/ttyUSBx)")
parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
parser.add_argument("--idle-timeout", type=float, default=5.0, help="Seconds to wait after last data before exiting")
parser.add_argument("--binary", action="store_true", help="Save raw binary output instead of text")
parser.add_argument("--command", default="d", help="Command to send to trigger the dump")
args = parser.parse_args()

filename = f"telemetry_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{'.bin' if args.binary else '.csv'}"

with serial.Serial(args.port, args.baud, timeout=1) as ser:
    if args.command:
        ser.write(args.command.encode("ascii", errors="ignore"))
        ser.flush()
    with open(filename, "wb" if args.binary else "w", newline="") as f:
        print(f"Saving to {filename}. Waiting for full dump... (idle timeout {args.idle_timeout}s)")
        last_data = time.monotonic()
        buffer = b""
        while True:
            chunk = ser.read(1024)
            if chunk:
                last_data = time.monotonic()
                if args.binary:
                    f.write(chunk)
                else:
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        text = line.decode("utf-8", errors="ignore").rstrip("\r")
                        print(text)
                        f.write(text + "\n")
            elif time.monotonic() - last_data >= args.idle_timeout:
                if not args.binary and buffer:
                    text = buffer.decode("utf-8", errors="ignore").rstrip("\r")
                    print(text)
                    f.write(text + "\n")
                break
print("Done!")