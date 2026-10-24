"""The live sync with EMtools 1.6: the selection shared between Blender and the program.

EMtools 1.6 is the HOST of the live sync of the Extended Matrix (ADR-002 of EMStudio and EMtools):
Blender serves a WebSocket on ``127.0.0.1:8788`` (the port is a preference of EMtools) once the
sync is started in its panel, and EMStudio connects to it as a client. The program connects the
same way, as one more client, and speaks the same wire, version 2: every message is

    {"v": 2, "type": KIND, "source": "pyarchinit-photogrammetry", "payload": {...}}

* ``client_info`` (sent on connecting): who this is and which matrix it has open (``graph_id``,
  ``graph_name``), so EMtools can say when the two are not looking at the same study;
* ``request_snapshot`` -> ``snapshot`` (``{"doc": em.json, "host": ...}``): the graph EMtools holds;
* ``select`` (``{"node_id", "node_ids"}``), both ways: a proxy picked in Blender selects its unit
  here, a unit picked here selects and frames its proxy in Blender (EMtools must *accept* the
  selection, its default);
* ``host_info``: what EMtools says of itself (tool, document, whether it accepts commands).

Graph edits (``op``) and commands are EMtools' and EMStudio's business: this client never sends
them. The socket is read in a thread; ``on_message(kind, payload)`` and ``on_state(state, text)``
are called from that thread (the plugin passes them on to the interface thread). Nothing outside
the standard library: the packaged program has no ``websockets``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import select
import socket
import struct
import threading
import time

WIRE = 2
SOURCE = "pyarchinit-photogrammetry"
HOST = "127.0.0.1"
PORT = 8788
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WireError(ValueError):
    """A message this client cannot read, with the reason."""


def envelope(kind: str, payload: dict | None = None, source: str = SOURCE, **routing) -> dict:
    message = {"v": WIRE, "type": kind, "source": source}
    message.update({k: v for k, v in routing.items() if v is not None})
    message["payload"] = dict(payload or {})
    return message


def read(message) -> tuple[str, dict]:
    """``(kind, payload)`` of a message of wire 2; another version is refused, not half read."""
    if not isinstance(message, dict):
        raise WireError("a wire message is a JSON object")
    if message.get("v") != WIRE:
        raise WireError(f"wire v{message.get('v')!r}, this client speaks v{WIRE}")
    kind = str(message.get("type") or "")
    if not kind:
        raise WireError("a wire message says its type")
    payload = message.get("payload") or {}
    if not isinstance(payload, dict):
        raise WireError(f"the payload of {kind!r} is not an object")
    return kind, payload


def _xor(data: bytes, key: bytes) -> bytes:
    """``data`` masked (or unmasked) with the four bytes of ``key``, as whole integers: fast also on
    the snapshot of a large matrix."""
    n = len(data)
    if not n:
        return b""
    pad = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(pad, "big")).to_bytes(n, "big")


class WebSocket:
    """The client side of RFC 6455, text frames only (what the wire uses)."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buffer = bytearray()
        self._partial: tuple[int, bytearray] | None = None  # a fragmented message being joined

    @classmethod
    def connect(cls, host: str = HOST, port: int = PORT, timeout: float = 3.0) -> "WebSocket":
        sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        try:
            sock.sendall(request.encode("latin1"))
            reply = b""
            while b"\r\n\r\n" not in reply:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionError("the host closed the connection during the handshake")
                reply += chunk
                if len(reply) > 16384:
                    raise ConnectionError("not a WebSocket host")
            head, _, rest = reply.partition(b"\r\n\r\n")
            lines = head.decode("latin1").split("\r\n")
            if " 101 " not in lines[0] + " ":
                raise ConnectionError(f"not a WebSocket host: {lines[0]}")
            headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
            expected = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
            if headers.get("sec-websocket-accept") != expected:
                raise ConnectionError("the host answered the handshake wrongly")
        except BaseException:
            sock.close()
            raise
        sock.settimeout(None)
        ws = cls(sock)
        ws.buffer += rest
        return ws

    def _frame(self, opcode: int, data: bytes) -> bytes:
        mask = os.urandom(4)  # a client masks every frame
        n = len(data)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        return bytes(head) + mask + _xor(data, mask)

    def send_text(self, text: str) -> None:
        self.sock.sendall(self._frame(0x1, text.encode("utf-8")))

    def _pong(self, data: bytes) -> None:
        self.sock.sendall(self._frame(0xA, data))

    def frames(self):
        """The complete frames in the buffer: ``(opcode, payload)``; continuation frames are joined."""
        while True:
            buf = self.buffer
            if len(buf) < 2:
                return
            opcode, n, offset = buf[0] & 0x0F, buf[1] & 0x7F, 2
            if n == 126:
                if len(buf) < 4:
                    return
                n, offset = struct.unpack(">H", buf[2:4])[0], 4
            elif n == 127:
                if len(buf) < 10:
                    return
                n, offset = struct.unpack(">Q", buf[2:10])[0], 10
            masked = buf[1] & 0x80
            if masked:
                offset += 4
            if len(buf) < offset + n:
                return
            data = bytes(buf[offset:offset + n])
            if masked:
                data = _xor(data, bytes(buf[offset - 4:offset]))
            fin = buf[0] & 0x80
            del buf[:offset + n]
            if opcode in (0x1, 0x2) and not fin:
                self._partial = (opcode, bytearray(data))
                continue
            if opcode == 0x0:
                if self._partial is not None:
                    self._partial[1].extend(data)
                    if fin:
                        first, message = self._partial
                        self._partial = None
                        yield first, bytes(message)
                continue
            yield opcode, data

    def receive(self, timeout: float) -> list[str] | None:
        """The text messages that arrived within ``timeout`` seconds; None when the host closed."""
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if not ready:
            return []
        chunk = self.sock.recv(65536)
        if not chunk:
            return None
        self.buffer += chunk
        texts = []
        for opcode, data in self.frames():
            if opcode == 0x1:
                texts.append(data.decode("utf-8", "replace"))
            elif opcode == 0x8:
                return None
            elif opcode == 0x9:
                self._pong(data)
        return texts

    def close(self) -> None:
        try:
            self.sock.sendall(self._frame(0x8, b""))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class SyncClient:
    """A client of the EMtools host, connected in a thread and connected again when the host goes
    away and comes back (the sync started later, Blender restarted)."""

    def __init__(self, port: int = PORT, host: str = HOST, on_message=None, on_state=None, info=None,
                 retry: float = 3.0):
        self.host, self.port, self.retry = host, int(port), float(retry)
        self.on_message = on_message or (lambda kind, payload: None)
        self.on_state = on_state or (lambda state, text: None)
        self.info = info or (lambda: {})  # the client_info payload, asked at every connection
        self._ws: WebSocket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_selection: tuple = ()
        self.state = "stopped"

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ppg-emtools-sync", daemon=True)
        self._thread.start()

    def stop(self, wait: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            ws, self._ws = self._ws, None
        if ws is not None:
            ws.close()
        if self._thread is not None:
            self._thread.join(wait)
        self._set("stopped", "")

    def _set(self, state: str, text: str) -> None:
        if state != self.state or text:
            self.state = state
            try:
                self.on_state(state, text)
            except Exception:  # noqa: BLE001 - the thread must not die of a callback
                pass

    def send(self, kind: str, payload: dict | None = None) -> bool:
        """One message to the host; False when not connected."""
        text = json.dumps(envelope(kind, payload), ensure_ascii=False)
        with self._lock:
            ws = self._ws
            if ws is None:
                return False
            try:
                ws.send_text(text)
                return True
            except OSError:
                return False

    def select(self, node_ids, active: str | None = None) -> bool:
        """The units selected here, to Blender; nothing when it is the selection that came from there."""
        ids = tuple(str(n) for n in node_ids if n)
        if not ids or ids == self.last_selection:
            return False
        body = {"node_id": str(active or ids[0])}
        if len(ids) > 1:
            body["node_ids"] = list(ids)
        sent = self.send("select", body)
        if sent:
            self.last_selection = ids
        return sent

    def request_snapshot(self) -> bool:
        return self.send("request_snapshot", {})

    def _run(self) -> None:
        while not self._stop.is_set():
            self._set("connecting", f"{self.host}:{self.port}")
            try:
                ws = WebSocket.connect(self.host, self.port)
            except OSError as exc:
                self._set("waiting", str(exc))
                self._stop.wait(self.retry)
                continue
            with self._lock:
                self._ws = ws
            self._set("connected", f"{self.host}:{self.port}")
            try:
                info = {"tool": "pyarchinit-photogrammetry", **(self.info() or {})}
            except Exception:  # noqa: BLE001
                info = {"tool": "pyarchinit-photogrammetry"}
            self.send("client_info", info)
            self.request_snapshot()
            reason = ""
            while not self._stop.is_set():
                try:
                    texts = ws.receive(0.25)
                except OSError as exc:
                    texts, reason = None, str(exc)
                if texts is None:
                    break
                for text in texts:
                    self._deliver(text)
            with self._lock:
                if self._ws is ws:
                    self._ws = None
            ws.close()
            if not self._stop.is_set():
                self._set("waiting", reason or "the host closed the connection")
                self._stop.wait(self.retry)

    def _deliver(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        if isinstance(message, dict) and message.get("source") == SOURCE:
            return  # an echo of our own
        try:
            kind, payload = read(message)
        except WireError as exc:
            self._set(self.state, f"refused: {exc}")
            return
        if kind == "select":
            ids = payload.get("node_ids") or ([payload["node_id"]] if payload.get("node_id") else [])
            self.last_selection = tuple(str(n) for n in ids)
        try:
            self.on_message(kind, payload)
        except Exception:  # noqa: BLE001 - a callback that fails does not close the link
            pass


def wait_for(predicate, timeout: float = 5.0, step: float = 0.02) -> bool:
    """``predicate()`` true within ``timeout`` seconds (for the tests and the command line)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


__all__ = ["WIRE", "SOURCE", "HOST", "PORT", "WireError", "envelope", "read", "WebSocket", "SyncClient", "wait_for"]
