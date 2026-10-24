"""A host of the live sync for the tests, as EMtools 1.6 serves it: the WebSocket handshake, the
text frames (the client's masked, the host's not), every message received kept, messages sent to
every client."""
from __future__ import annotations

import base64
import hashlib
import json
import select
import socket
import struct
import threading

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeHost:
    def __init__(self, port: int):
        self.port = port
        self.received: list[dict] = []
        self.clients: list[socket.socket] = []
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", port))
        self._srv.listen(4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buffers = {}
        while not self._stop.is_set():
            socks = [self._srv] + self.clients
            try:
                ready, _, _ = select.select(socks, [], [], 0.05)
            except (OSError, ValueError):
                break
            for s in ready:
                if s is self._srv:
                    try:
                        conn, _ = self._srv.accept()
                    except OSError:
                        continue
                    self._handshake(conn)
                    self.clients.append(conn)
                    buffers[conn] = bytearray()
                    continue
                try:
                    data = s.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    self.clients.remove(s)
                    continue
                buffers[s] += data
                self._frames(buffers[s])

    @staticmethod
    def _handshake(conn):
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(1024)
        key = next(ln.split(b":", 1)[1].strip() for ln in data.split(b"\r\n") if ln.lower().startswith(b"sec-websocket-key"))
        accept = base64.b64encode(hashlib.sha1(key + _GUID.encode()).digest())
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")

    def _frames(self, buf):
        while len(buf) >= 2:
            opcode, n, off = buf[0] & 0x0F, buf[1] & 0x7F, 2
            assert buf[1] & 0x80, "a client frame is masked"
            if n == 126:
                n, off = struct.unpack(">H", buf[2:4])[0], 4
            elif n == 127:
                n, off = struct.unpack(">Q", buf[2:10])[0], 10
            if len(buf) < off + 4 + n:
                return
            key = buf[off:off + 4]
            data = bytes(b ^ key[i % 4] for i, b in enumerate(buf[off + 4:off + 4 + n]))
            del buf[:off + 4 + n]
            if opcode == 0x1:
                self.received.append(json.loads(data.decode()))

    def send(self, message, raw: bool = False):
        text = message if raw else json.dumps(message)
        payload = text.encode()
        n = len(payload)
        head = bytearray([0x81])
        if n < 126:
            head.append(n)
        elif n < 65536:
            head.append(126)
            head += struct.pack(">H", n)
        else:
            head.append(127)
            head += struct.pack(">Q", n)
        for c in list(self.clients):
            c.sendall(bytes(head) + payload)

    def kinds(self):
        return [m.get("type") for m in self.received]

    def close(self):
        self._stop.set()
        for c in self.clients:
            try:
                c.close()
            except OSError:
                pass
        self._srv.close()
        self._thread.join(2)
