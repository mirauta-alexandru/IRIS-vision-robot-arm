# Setup

## Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

```bash
export GEMINI_API_KEY="your-key-here"
```

## Serial Port

The default serial port is configured in `files/iris_bridge.py`:

```python
SERIAL_PORT = "/dev/tty.usbserial-0001"
BAUD_RATE = 115200
```

Update it to match your ESP32.

## Start The Bridge

```bash
python files/iris_bridge.py
```

Open:

```text
http://localhost:8765/live
```

Other useful pages:

```text
http://localhost:8765/
http://localhost:8765/video_feed
```

## Hardware Checklist

- ESP32 connected over USB serial.
- PCA9685 connected through I2C.
- Servo power connected to a stable 5V supply.
- Common ground between ESP32, PCA9685, and power supply.
- Camera fixed rigidly above the workspace.
- ChArUco board printed flat and measured correctly.
