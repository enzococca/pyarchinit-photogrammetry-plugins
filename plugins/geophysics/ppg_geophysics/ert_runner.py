"""The pyGIMLi worker in a process of its own: the job written beside its input, the messages read
while it runs (the progress bar moves, cancelling kills it at once, even inside a long inversion),
the result or the reason of a failure.

The process starts as every process of the program does: the absolute path of its Python, the file
descriptors inherited and no working directory (posix_spawn on macOS), without the variables of the
program's Python and with a home folder of the plugin (:func:`.ert_environment.worker_environment`).
A run that does not finish leaves no result: the worker writes the result under another name and
renames it at the end; after a failure or a cancel the runner removes both."""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from . import ert_environment as environment
from .texts import t

WORKER = Path(__file__).with_name("ert_worker.py")
PREFIX = "@ppg-geophysics "
POLL = 0.2
KEEP_LINES = 80
PHASES = {"start": ("avvio", "starting"), "load": ("lettura", "reading"), "data": ("dati", "data"),
          "factors": ("fattori geometrici sulla topografia", "geometric factors on the topography"),
          "mesh": ("maglia", "mesh"), "invert": ("inversione", "inversion"),
          "doi": ("profondità d'indagine (due inversioni in più)", "depth of investigation (two more inversions)"),
          "forward": ("modellazione diretta", "forward modelling"), "save": ("scrittura", "writing"),
          "done": ("fatto", "done")}


class WorkerError(RuntimeError):
    def __init__(self, message: str, phase: str | None = None, output=(), details: str = ""):
        super().__init__(message)
        self.phase, self.output, self.details = phase, list(output), details


def parse(line: str) -> dict | None:
    i = line.find(PREFIX)
    if i < 0:
        return None
    try:
        msg = json.loads(line[i + len(PREFIX):])
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def phase_text(name: str) -> str:
    return "pyGIMLi: " + t(*PHASES.get(str(name), (str(name), str(name))))


def hint_for(text: str, output=()) -> str:
    low = (text + "\n" + "\n".join(output)).lower()
    if "memoryerror" in low or "unable to allocate" in low or "bad_alloc" in low:
        return t("Memoria insufficiente: una maglia più grossolana (cella massima più grande, profondità minore).",
                 "Not enough memory: a coarser mesh (larger largest cell, smaller depth).")
    if "no module named" in low and "pygimli" in low:
        return t("pyGIMLi non c'è più nell'ambiente: installa di nuovo pyGIMLi.",
                 "pyGIMLi is no longer in the environment: install pyGIMLi again.")
    if "usable readings" in low or "fewer than 4 readings" in low:
        return t("Troppe letture tolte dai filtri: allenta i filtri.",
                 "Too many readings removed by the filters: loosen the filters.")
    return ""


def run(job: dict, folder: Path, progress=None, python: Path | None = None, script: Path | None = None,
        timeout: float | None = None) -> dict:
    """Run ``job`` (see :mod:`.ert_worker`) and return its result message; ``folder`` holds the job
    file. Raises :class:`WorkerError` (also when the worker runs longer than ``timeout`` seconds: it
    is stopped), :class:`.ert_environment.GimliMissing` without a Python for it, ``Cancelled`` when the
    user stops it; in every one of those cases no result is left."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    python = Path(python).absolute() if python is not None else environment.worker_python()
    script = Path(script).absolute() if script is not None else WORKER.absolute()
    folder = Path(folder).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    output = Path(job["output"]).absolute()
    job = {**job, "output": str(output)}
    job_path = folder / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    progress(0.0, phase_text("start"))
    proc = subprocess.Popen([str(python), "-I", "-u", str(script), str(job_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=environment.worker_environment(), close_fds=False)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    reader = threading.Thread(target=pump, name="ert worker output", daemon=True)
    reader.start()
    said: deque = deque(maxlen=KEEP_LINES)
    result = error = hello = last = None
    finished = False
    started = time.monotonic()

    def check_time() -> None:
        if timeout is not None and time.monotonic() - started > float(timeout):
            raise WorkerError(t("pyGIMLi fermato dopo {0:.0f} s senza finire (limite di tempo).",
                                "pyGIMLi stopped after {0:.0f} s without finishing (time limit).")
                              .format(float(timeout)), (last or {}).get("phase"), said, "\n".join(said))
    try:
        try:
            while True:
                try:
                    line = lines.get(timeout=POLL)
                except queue.Empty:
                    progress.check()
                    check_time()
                    continue
                if line is None:
                    break
                progress.check()
                check_time()
                msg = parse(line)
                if msg is None:
                    if line.strip():
                        said.append(line)
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    last = msg
                    progress(min(max(float(msg.get("fraction", 0.0)), 0.0), 1.0), phase_text(msg.get("phase", "")))
                elif kind == "hello":
                    hello = msg
                elif kind == "result":
                    result = msg.get("result")
                elif kind == "error":
                    error = msg
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            reader.join(timeout=5)
            try:
                proc.stdout.close()
            except OSError:
                pass
        if error is not None:
            text = str(error.get("error") or "").strip()
            message = f"{phase_text(error.get('phase') or 'start')}: {text}"
            hint = hint_for(text, said)
            details = str(error.get("traceback") or "") + ("\n" + "\n".join(said) if said else "")
            raise WorkerError(message + ("\n" + hint if hint else ""), error.get("phase"), said, details)
        if result is None or code != 0:
            where = t(" durante «{0}»", " during '{0}'").format(phase_text(last.get("phase"))) if last else ""
            hint = hint_for("", said)
            raise WorkerError(t("pyGIMLi si è chiuso inaspettatamente{0} (codice {1}): nessun risultato è stato "
                                "scritto.", "pyGIMLi closed unexpectedly{0} (code {1}): no result was written.")
                              .format(where, code) + ("\n" + hint if hint else ""), (last or {}).get("phase"),
                              said, "\n".join(said))
        finished = True
    finally:
        if not finished:
            for p in (output, output.with_name(output.name + ".partial.npz")):
                p.unlink(missing_ok=True)
    if hello is not None:
        result.setdefault("versions", hello.get("versions", {}))
    return result


def load(result: dict) -> dict:
    """The arrays of a result."""
    import numpy as np

    with np.load(result["output"], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
