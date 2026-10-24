"""The worker in a process of its own: the job written beside the mesh, the messages read while
it runs (the progress bar moves, cancelling stops it at once, even inside a long filter), the
result or the reason of a failure - a filter that refused the mesh, or MeshLab that crashed.

The process starts as every process of the program does: the absolute path of its Python, the
file descriptors inherited and no working directory (posix_spawn on macOS: a fork after SpatiaLite
was loaded can die in the child), without the variables of the program's Python.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path

from . import environment
from .texts import PHASES, t, title

WORKER = Path(__file__).with_name("worker.py")
PREFIX = "@ppg-mesh-tools "
POLL = 0.2  # seconds between two looks at the cancel button while the worker says nothing
KEEP_LINES = 80  # lines of MeshLab's own output kept for the report of a failure


class WorkerError(RuntimeError):
    """The worker failed: ``phase`` and ``op`` say where, ``output`` holds MeshLab's last lines,
    ``details`` the traceback of the worker."""

    def __init__(self, message: str, phase: str | None = None, op: str | None = None, output=(), details: str = ""):
        super().__init__(message)
        self.phase, self.op, self.output, self.details = phase, op, list(output), details


def parse(line: str) -> dict | None:
    """The message of a line of the worker, None for MeshLab's own lines."""
    i = line.find(PREFIX)
    if i < 0:
        return None
    try:
        msg = json.loads(line[i + len(PREFIX):])
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def phase_text(msg: dict) -> str:
    """What the worker is doing, in the language of the interface."""
    phase = str(msg.get("phase") or "")
    if phase == "run":
        name = title(str(msg.get("op", "")))
        steps = int(msg.get("steps", 1) or 1)
        return "MeshLab: " + (t(*PHASES["run"]).format(name, msg.get("step", 1), steps) if steps > 1 else name)
    return "MeshLab: " + t(*PHASES.get(phase, (phase, phase)))


def hint_for(text: str, output=()) -> str:
    """A suggestion for a failure the user can do something about. ``output``: MeshLab's own lines
    (pymeshlab prints there the modules it could not load, with the library each one lacked)."""
    low = text.lower()
    said = "\n".join(output)
    libs = environment.missing_libraries(text + "\n" + said)
    no_modules = bool(libs) or "unknown format for load" in low or (
        "has no attribute" in low and any(k in low for k in ("'meshing_", "'apply_", "'get_", "'compute_")))
    if no_modules:  # MeshLab's plugins did not load: on Linux a system library is missing
        if environment.is_linux():
            return environment.linux_hint(libs)
        return t("MeshLab non ha caricato i suoi moduli{0}: installa di nuovo MeshLab (pymeshlab).",
                 "MeshLab did not load its modules{0}: install MeshLab (pymeshlab) again.").format(
            f" ({', '.join(libs)})" if libs else "")
    if "manifold" in low:
        return t("Il modello ha spigoli o vertici non-manifold: usa prima «Ripara la topologia».",
                 "The model has non-manifold edges or vertices: use 'Repair the topology' first.")
    if "no module named" in low and "pymeshlab" in low:
        return t("pymeshlab non c'è più nell'ambiente: installa di nuovo MeshLab.",
                 "pymeshlab is no longer in the environment: install MeshLab again.")
    return ""


def _crash_text(code: int, last: dict | None) -> str:
    where = ""
    if last and last.get("phase") == "run":
        where = t(" durante «{0}»", " during '{0}'").format(title(str(last.get("op", ""))))
    elif last and last.get("phase"):
        where = t(" durante {0}", " during {0}").format(t(*PHASES.get(str(last["phase"]), ("", ""))))
    return t("MeshLab si è chiuso inaspettatamente{0} (codice {1}): il modello non è cambiato. Un modello con difetti "
             "di topologia può farlo: prova prima «Ripara la topologia».",
             "MeshLab closed unexpectedly{0} (code {1}): the model has not changed. A model with topology defects "
             "can make it: try 'Repair the topology' first.").format(where, code)


def run(job: dict, folder: Path, progress=None, python: Path | None = None, script: Path | None = None) -> dict:
    """Run ``job`` (see :mod:`.worker`) with ``python`` (the one of :func:`environment.worker_python`
    by default) and return its result. ``folder`` holds the job file. Raises :class:`WorkerError`,
    :class:`environment.MeshLabMissing` without a Python for it, ``Cancelled`` when the user stops it."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    python = Path(python) if python is not None else environment.worker_python()
    script = Path(script).absolute() if script is not None else WORKER.absolute()
    folder = Path(folder).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    job_path = folder / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    progress(0.0, phase_text({"phase": "start"}))
    proc = subprocess.Popen([str(python), "-u", str(script), str(job_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=environment.clean_environment(), close_fds=False)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw.rstrip("\r\n"))
        except (OSError, ValueError):  # the pipe closed under it
            pass
        finally:
            lines.put(None)

    reader = threading.Thread(target=pump, name="mesh-tools worker output", daemon=True)
    reader.start()
    output: deque = deque(maxlen=KEEP_LINES)
    result = error = hello = last = None
    try:
        while True:
            try:
                line = lines.get(timeout=POLL)
            except queue.Empty:
                progress.check()  # the user may cancel while a filter runs and the worker says nothing
                continue
            if line is None:
                break
            progress.check()  # also while MeshLab fills the output with its own lines
            msg = parse(line)
            if msg is None:
                if line.strip():
                    output.append(line)
                continue
            kind = msg.get("type")
            if kind == "progress":
                last = msg
                progress(min(max(float(msg.get("fraction", 0.0)), 0.0), 1.0), phase_text(msg))
            elif kind == "hello":
                hello = msg
            elif kind == "result":
                result = msg.get("result")
            elif kind == "error":
                error = msg
        code = proc.wait()
    except BaseException:  # cancelled (or the program stopping): the worker goes at once
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
        where = title(str(error["op"])) if error.get("op") else t(*PHASES.get(str(error.get("phase")), ("MeshLab", "MeshLab")))
        message = f"MeshLab ({where}): {text}"
        hint = hint_for(text, output)
        details = str(error.get("traceback") or "") + ("\n" + "\n".join(output) if output else "")
        raise WorkerError(message + ("\n" + hint if hint else ""), error.get("phase"), error.get("op"), output, details)
    if result is None or code != 0:
        raise WorkerError(_crash_text(code, last), (last or {}).get("phase"), (last or {}).get("op"), output,
                          "\n".join(output))
    if hello is not None:
        result.setdefault("version", hello.get("version", ""))
    return result
