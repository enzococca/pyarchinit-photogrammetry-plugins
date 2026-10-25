"""The worker (:mod:`.worker`) seen from the program: started once for a run, the models loaded once,
then one request per tile. Cancelling stops it at once, also in the middle of a tile (the worker is
killed: a tile on the processor takes tens of seconds); a worker that closes or fails comes back as
:class:`WorkerError` with its last lines.

The process starts as every process of the program does: the absolute path of its Python, the file
descriptors inherited and no working directory (posix_spawn on macOS), without the variables of the
program's Python.
"""
from __future__ import annotations

import pickle
import queue
import struct
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from . import environment
from .texts import t

WORKER = Path(__file__).with_name("worker.py")
POLL = 0.2  # seconds between two looks at the cancel button while the worker works
DETECTORS = {"base": "IDEA-Research/grounding-dino-base", "tiny": "IDEA-Research/grounding-dino-tiny"}
SEGMENTERS = {"tiny": "facebook/sam2.1-hiera-tiny", "small": "facebook/sam2.1-hiera-small",
              "base_plus": "facebook/sam2.1-hiera-base-plus", "large": "facebook/sam2.1-hiera-large"}
# the models of each choice of the pane: (Grounding DINO, SAM 2.1)
MODELS = {"normal": ("base", "small"), "fast": ("tiny", "tiny"), "precise": ("base", "large")}


class WorkerError(RuntimeError):
    """The worker failed: ``details`` holds its traceback and its last lines."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details


def unpack(obj: dict) -> tuple[int, int, np.ndarray]:
    """The mask of an object of the worker: ``(row, col, boolean array)`` in the tile."""
    r0, c0, h, w, bits = obj["mask"]
    mask = np.unpackbits(np.frombuffer(bits, np.uint8))[: h * w].reshape(h, w).astype(bool)
    return r0, c0, mask


class Segmenter:
    """The models in their process: :meth:`segment` an RGB tile with a text, :meth:`close` at the end
    (also as a context manager). ``engine``: ``"open"`` (Grounding DINO + SAM 2.1); ``detector`` and
    ``segmenter``: ``base``/``tiny`` and ``tiny``/``small``/``base_plus``/``large``, or the name of a
    model on Hugging Face. ``python``, ``script``: another Python or worker (the tests)."""

    def __init__(self, engine: str = "open", detector: str = "base", segmenter: str = "small", device: str | None = None,
                 token: str | None = None, progress=None, python: Path | None = None, script: Path | None = None):
        python = Path(python) if python is not None else environment.worker_python()
        script = Path(script).absolute() if script is not None else WORKER.absolute()
        request = {"op": "init", "engine": engine, "detector": DETECTORS.get(detector, detector),
                   "segmenter": SEGMENTERS.get(segmenter, segmenter), "device": device, "token": token}
        try:
            reply = self._start(python, script, request, progress, {})
        except WorkerError as exc:
            if not download_failed(str(exc)):
                raise
            # the weights broke off while they came through Hugging Face's Xet storage: once more by plain
            # HTTP, which takes up the files already in the cache
            reply = self._start(python, script, request, progress, {"HF_HUB_DISABLE_XET": "1"})
        self.engine, self.device, self.versions = reply["engine"], reply["device"], reply.get("versions") or {}

    def _start(self, python: Path, script: Path, request: dict, progress, extra: dict) -> dict:
        """The worker started and its models loaded: the reply to ``init``."""
        env = environment.clean_environment()
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU, the processor for the few operations it lacks
        env.update(extra)
        self._proc = subprocess.Popen([str(python), "-u", str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, env=env, close_fds=False)
        self._said: deque = deque(maxlen=60)
        self._replies: queue.Queue = queue.Queue()
        # each thread bound to its own process and queue: those of a worker started again never mix
        threading.Thread(target=self._drain, args=(self._proc, self._said), name="samgeo worker stderr",
                         daemon=True).start()
        threading.Thread(target=self._pump, args=(self._proc, self._replies), name="samgeo worker replies",
                         daemon=True).start()
        try:
            return self._call(request, progress)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _drain(proc, said) -> None:
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    said.append(line[-300:])
        except (OSError, ValueError):
            pass

    @staticmethod
    def _pump(proc, replies) -> None:
        """The replies of the worker into the queue; None when its output ends, a reply with
        ``broken`` when it cannot be read (the worker is then stopped by :meth:`_call`)."""
        stream = proc.stdout
        try:
            while True:
                head = stream.read(8)
                if len(head) < 8:
                    break
                (n,) = struct.unpack(">Q", head)
                replies.put(pickle.loads(stream.read(n)))
        except Exception as exc:  # noqa: BLE001 - the pipe closed under it, or a reply not readable here
            replies.put({"ok": False, "broken": True, "error": f"{type(exc).__name__}: {exc}"})
            return
        replies.put(None)

    def _call(self, request: dict, progress=None) -> dict:
        from pyarchinit_photogrammetry.core.progress import as_progress

        progress = as_progress(progress)
        data = pickle.dumps(request, protocol=4)
        try:
            self._proc.stdin.write(struct.pack(">Q", len(data)) + data)
            self._proc.stdin.flush()
        except OSError:
            pass  # closed: the reply says why
        while True:
            try:
                reply = self._replies.get(timeout=POLL)
                break
            except queue.Empty:
                try:
                    progress.check()  # the user may cancel while a tile runs
                except BaseException:
                    self.close(wait=False)  # at once, not at the end of the tile
                    raise
        if reply is None:
            try:
                code = self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # its output closed, the process not: it goes
                self.close()
                code = self._proc.returncode
            raise WorkerError(t("Il processo dei modelli si è chiuso inaspettatamente (codice {0}).",
                                "The process of the models closed unexpectedly (code {0}).").format(code)
                              + _hint("\n".join(self._said)), "\n".join(self._said))
        if reply.get("broken"):  # the conversation cannot go on
            self.close()
        if not reply.get("ok"):
            text = str(reply.get("error") or "?")
            raise WorkerError(t("Modelli: {0}", "Models: {0}").format(text) + _hint(text + "\n" + "\n".join(self._said)),
                              str(reply.get("traceback") or "") + "\n" + "\n".join(self._said))
        return reply

    def segment(self, image: np.ndarray, prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25,
                progress=None) -> list[dict]:
        """The objects of ``prompt`` in an RGB tile (uint8, ``(H, W, 3)``): dicts ``box``, ``score``,
        ``quality``, ``label`` and ``mask`` (see :func:`unpack`)."""
        return self._call({"op": "segment", "image": np.ascontiguousarray(image[..., :3], dtype=np.uint8),
                           "prompt": prompt, "box_threshold": float(box_threshold),
                           "text_threshold": float(text_threshold)}, progress)["objects"]

    def close(self, wait: bool = True) -> None:
        """The worker asked to quit (``wait``: up to 5 s for it to finish what it does), else killed."""
        proc = self._proc
        if wait and proc.poll() is None:
            try:
                data = pickle.dumps({"op": "quit"}, protocol=4)
                proc.stdin.write(struct.pack(">Q", len(data)) + data)
                proc.stdin.flush()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - it goes anyway
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


DOWNLOAD_ERRORS = ("cas client", "file reconstruction", "error decoding response body", "xet", "connection reset",
                   "incompleteread", "chunkedencodingerror")


def download_failed(text: str) -> bool:
    """The text of a failure says the download of the weights broke off."""
    low = text.lower()
    return any(k in low for k in DOWNLOAD_ERRORS)


def _hint(text: str) -> str:
    """A suggestion for a failure the user can do something about."""
    low = text.lower()
    if download_failed(text):
        return "\n" + t("Il download dei pesi da Hugging Face si è interrotto: riprova, riprende da dove era arrivato. "
                        "Se si ripete, avvia il programma con HF_HUB_DISABLE_XET=1.",
                        "The download of the weights from Hugging Face broke off: try again, it goes on from where it "
                        "stopped. If it happens again, start the program with HF_HUB_DISABLE_XET=1.")
    if any(k in low for k in ("connection", "resolve", "offline", "timed out", "max retries", "huggingface.co")):
        return "\n" + t("I pesi dei modelli si scaricano da Hugging Face al primo uso: serve la connessione a internet.",
                        "The weights of the models are downloaded from Hugging Face at the first use: an internet "
                        "connection is needed.")
    if "out of memory" in low or "cannot allocate" in low:
        return "\n" + t("Memoria esaurita: usa una risoluzione più grossolana, un'area più piccola o i modelli più "
                        "piccoli.", "Out of memory: use a coarser resolution, a smaller area or the smaller models.")
    if "no module named" in low:
        return "\n" + t("All'ambiente manca un pacchetto: installa di nuovo i modelli.",
                        "The environment lacks a package: install the models again.")
    return ""
