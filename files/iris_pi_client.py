"""Authenticated TCP client for the IRIS Raspberry Pi PCA9685 controller."""

from __future__ import annotations

import json
import socket
import threading
from typing import Any


class PiServoClient:
    def __init__(self, host: str, port: int, token: str, timeout: float = 1.0):
        if not token:
            raise ValueError("IRIS_TOKEN is required for the Pi controller.")
        self.host = host
        self.port = int(port)
        self.token = token
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None
        self._reader: Any = None
        self._lock = threading.Lock()

    def _close_unlocked(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
            self._reader = None
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _connect_unlocked(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._reader = sock.makefile("rb")

    def _request(self, command: str, **values: Any) -> dict[str, Any]:
        payload = {"command": command, "token": self.token, **values}
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            for attempt in range(2):
                try:
                    self._connect_unlocked()
                    assert self._socket is not None and self._reader is not None
                    self._socket.sendall(encoded)
                    line = self._reader.readline(4097)
                    if not line or len(line) > 4096:
                        raise ConnectionError("Pi controller returned no valid response.")
                    result = json.loads(line.decode("utf-8"))
                    if not isinstance(result, dict):
                        raise ValueError("Pi controller response was not an object.")
                    return result
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self._close_unlocked()
                    if attempt == 1:
                        return {"status": "error", "hardware_moved": False, "error": str(exc)}
            return {"status": "error", "hardware_moved": False, "error": "Connection failed."}

    def ping(self) -> dict[str, Any]:
        return self._request("PING")

    def set_angle(self, channel: int, angle: float) -> dict[str, Any]:
        return self._request("SET_ANGLE", channel=int(channel), angle=float(angle))

    def set_pulse(self, channel: int, pulse_us: float) -> dict[str, Any]:
        return self._request("SET_PULSE", channel=int(channel), pulse_us=float(pulse_us))

    def stop(self) -> dict[str, Any]:
        return self._request("STOP")
